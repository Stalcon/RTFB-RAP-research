import importlib.util
import os
import shutil
from typing import Literal, Optional

BackendName = Literal["cpu", "cuda", "rocm"]


def resolve_backend() -> BackendName:
    requested = os.getenv("BACKEND", "auto").strip().lower()
    if requested in ("cpu", "cuda", "rocm"):
        return requested  # type: ignore[return-value]
    if requested != "auto":
        raise ValueError(f"BACKEND must be cpu, cuda, rocm, or auto (got {requested!r})")
    try:
        import torch

        if torch.cuda.is_available():
            if getattr(torch.version, "hip", None):
                return "rocm"
            return "cuda"
    except Exception:
        pass
    return "cpu"


def configure_threads(uma_threads: int = 2, dft_threads: int = 4) -> dict:
    uma = str(max(1, int(uma_threads)))
    dft = str(max(1, int(dft_threads)))
    os.environ.setdefault("OMP_NUM_THREADS", dft)
    os.environ.setdefault("MKL_NUM_THREADS", dft)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", dft)
    return {
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
        "OPENBLAS_NUM_THREADS": os.environ["OPENBLAS_NUM_THREADS"],
        "uma_threads": uma,
        "dft_threads": dft,
    }


def describe() -> dict:
    backend = resolve_backend()
    info: dict = {"backend": backend}
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["hip_build"] = getattr(torch.version, "hip", None)
        info["n_gpus"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if info["n_gpus"]:
            info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(info["n_gpus"])]
        else:
            info["gpu_names"] = []
    except Exception as e:
        info["torch_version"] = None
        info["torch_error"] = str(e)[:120]
        info["n_gpus"] = 0
        info["gpu_names"] = []
    try:
        info["has_gpu4pyscf"] = importlib.util.find_spec("gpu4pyscf.dft") is not None
    except Exception:
        info["has_gpu4pyscf"] = False
    try:
        info["has_cupy"] = importlib.util.find_spec("cupy") is not None
    except Exception:
        info["has_cupy"] = False
    info["nvidia_smi"] = shutil.which("nvidia-smi")
    info["rocm_smi"] = shutil.which("rocm-smi")
    info["rocminfo"] = shutil.which("rocminfo")
    info["dft_backend_name"] = get_dft_backend_name(backend)
    return info


def get_uma_device(backend: Optional[str] = None):
    backend = backend or resolve_backend()
    try:
        import torch
    except Exception:
        return "cpu"
    if backend in ("cuda", "rocm") and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_dft_backend_name(backend: Optional[str] = None) -> str:
    backend = (backend or resolve_backend()).strip().lower()
    if backend == "cuda":
        return "gpu4pyscf"
    if backend == "rocm":
        return "rocm_hip"
    return "pyscf"


def get_dft_module(backend: Optional[str] = None):
    import importlib

    backend = (backend or resolve_backend()).strip().lower()
    if backend == "cuda":
        try:
            return importlib.import_module("gpu4pyscf.dft")
        except Exception as e:
            raise ImportError(f"BACKEND=cuda but gpu4pyscf.dft is not importable: {e}") from e
    if backend == "rocm":
        return importlib.import_module("core.rocm_dft")
    return importlib.import_module("pyscf.dft")


def get_geomopt_fn():
    from pyscf.geomopt.geometric_solver import optimize

    return optimize


def get_cupy_module(backend: Optional[str] = None):
    import importlib

    backend = (backend or resolve_backend()).strip().lower()
    if backend != "rocm":
        return None
    try:
        return importlib.import_module("cupy")
    except Exception:
        return None
