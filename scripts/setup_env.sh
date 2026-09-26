#!/usr/bin/env bash
# Create the redox conda env with a backend-matched torch (cuda | rocm | cpu).
# Usage: ./setup_env.sh [ENV_NAME]  |  BACKEND=cpu ./setup_env.sh
set -euo pipefail

ENV_NAME="${1:-redox}"
PY_VER="${PY_VER:-3.11}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND="${BACKEND:-auto}"
ROCM_HOME="${ROCM_HOME:-/opt/rocm}"

if command -v mamba >/dev/null 2>&1; then
    CONDA="$(command -v mamba)"
elif command -v conda >/dev/null 2>&1; then
    CONDA="$(command -v conda)"
else
    echo "ERROR: conda or mamba is required." >&2
    exit 1
fi

# Auto-detect backend unless BACKEND is set explicitly.
if [ "$BACKEND" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 &&
       nvidia-smi >/dev/null 2>&1; then
        BACKEND="cuda"
    elif command -v rocminfo >/dev/null 2>&1 &&
         rocminfo >/dev/null 2>&1; then
        BACKEND="rocm"
    elif command -v rocm-smi >/dev/null 2>&1 &&
         rocm-smi >/dev/null 2>&1; then
        BACKEND="rocm"
    else
        BACKEND="cpu"
    fi
fi

case "$BACKEND" in
    cuda|rocm|cpu)
        ;;
    *)
        echo "ERROR: BACKEND must be cuda, rocm, or cpu." >&2
        exit 1
        ;;
esac

echo ">> conda: $CONDA"
echo ">> env: $ENV_NAME"
echo ">> python: $PY_VER"
echo ">> backend: $BACKEND"

# Create env, upgrade pip, install backend-agnostic deps first.
"$CONDA" create -n "$ENV_NAME" "python=$PY_VER" -y

conda run -n "$ENV_NAME" python -m pip install --upgrade pip

conda run -n "$ENV_NAME" pip install \
    -r "$HERE/requirements.txt"

case "$BACKEND" in

    cuda)
        # Parse driver CUDA version so torch wheels match the driver.
        CUDA_VERSION="$(
            nvidia-smi 2>/dev/null |
            sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' |
            head -n1
        )"

        case "$CUDA_VERSION" in
            12.4*) CUDA_TAG="cu124" ;;
            12.6*) CUDA_TAG="cu126" ;;
            12.8*) CUDA_TAG="cu128" ;;
            12.9*) CUDA_TAG="cu129" ;;
            *)
                echo "ERROR: unsupported NVIDIA CUDA version: $CUDA_VERSION" >&2
                echo "Set BACKEND=cpu or use a supported CUDA version." >&2
                exit 1
                ;;
        esac

        # torch must come from the CUDA-specific index or cuda stays False.
        conda run -n "$ENV_NAME" pip install \
            "torch==2.8.0" \
            --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

        # GPU DFT backend (SMD on CUDA).
        conda run -n "$ENV_NAME" pip install \
            gpu4pyscf-cuda12x \
            cutensor-cu12
        ;;

    rocm)
        # ROCm HIP torch + CuPy for AMD GPUs.
        if [ ! -d "$ROCM_HOME" ]; then
            echo "ERROR: ROCm not found at $ROCM_HOME" >&2
            exit 1
        fi

        export ROCM_HOME

        ROCM_ARCH="${ROCM_ARCH:-$(
            rocminfo 2>/dev/null |
            sed -n 's/.*Name:[[:space:]]*\(gfx[0-9a-z]*\).*/\1/p' |
            head -n1
        )}"

        if [ -z "$ROCM_ARCH" ]; then
            echo "ERROR: could not determine AMD GPU architecture." >&2
            echo "Set ROCM_ARCH manually, e.g. ROCM_ARCH=gfx1100." >&2
            exit 1
        fi

        conda run -n "$ENV_NAME" pip install \
            "torch==2.8.0" \
            --index-url "https://download.pytorch.org/whl/rocm6.4"

        conda run -n "$ENV_NAME" env \
            ROCM_HOME="$ROCM_HOME" \
            CUPY_INSTALL_USE_HIP=1 \
            HCC_AMDGPU_TARGET="$ROCM_ARCH" \
            pip install \
            amd-cupy \
            --extra-index-url https://pypi.amd.com/simple
        ;;

    cpu)
        # CPU-only torch; DFT falls back to pyscf.
        conda run -n "$ENV_NAME" pip install \
            "torch==2.8.0" \
            --index-url "https://download.pytorch.org/whl/cpu"
        ;;

esac

# Sanity-probe torch visibility before the full check.
conda run -n "$ENV_NAME" python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("HIP/ROCm version:", getattr(torch.version, "hip", None))

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
PY

if [ "$BACKEND" = "rocm" ]; then
    # Verify CuPy/HIP matmul on AMD GPU.
    conda run -n "$ENV_NAME" env \
        ROCM_HOME="$ROCM_HOME" \
        python - <<'PY'
import cupy as cp

n = cp.cuda.runtime.getDeviceCount()
print("CuPy:", cp.__version__)
print("AMD GPU count:", n)

for i in range(n):
    props = cp.cuda.runtime.getDeviceProperties(i)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    print(f"GPU {i}:", name)

x = cp.arange(16, dtype=cp.float64).reshape(4, 4)
y = x @ x

print("CuPy ROCm test: OK")
print(y)
PY
fi

# Fail fast: full env smoke test (HERE is scripts/, so no extra scripts/ prefix).
conda run -n "$ENV_NAME" python "$HERE/check_env.py"

cat <<EOF

Done.

Activate with:
  source \$(conda info --base)/etc/profile.d/conda.sh
  conda activate $ENV_NAME

Backend:
  $BACKEND

EOF