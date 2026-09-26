import importlib
import os
import shutil
import sys

BACKEND = os.getenv("BACKEND", "auto")
REQUIRED = [
    "torch",
    "ase",
    "rdkit",
    "pyscf",
    "geometric",
    "numpy",
    "pandas",
    "matplotlib",
    "huggingface_hub",
    "openpyxl",
]
FAIRCHEM_CANDIDATES = ["fairchem.core", "fairchem"]


def check_module(name):
    try:
        mod = importlib.import_module(name)
        print(f"  [ok] {name:16s} {getattr(mod, '__version__', '?')}")
        return True
    except Exception as e:
        print(f"  [MISS] {name:16s} {str(e)[:100]}")
        return False


def main():
    print(f"BACKEND={BACKEND}")
    ok = True

    for m in REQUIRED:
        if not check_module(m):
            ok = False

    fairchem_ok = False
    for m in FAIRCHEM_CANDIDATES:
        try:
            mod = importlib.import_module(m)
            print(f"  [ok] {m:16s} {getattr(mod, '__version__', '?')}")
            fairchem_ok = True
            break
        except Exception:
            continue
    if not fairchem_ok:
        ok = False
        print("  [MISS] fairchem-core    not importable as fairchem.core or fairchem")

    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"  torch {torch.__version__} cuda={cuda} devices={torch.cuda.device_count()}")
        print(f"  cuda build={torch.version.cuda} hip={getattr(torch.version, 'hip', None)}")
        if cuda:
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        elif BACKEND == "cuda":
            print("  [MISS] BACKEND=cuda but torch.cuda is not available")
            ok = False
    except Exception as e:
        ok = False
        print(f"  [MISS] torch probe failed: {e}")

    if BACKEND in ("cuda", "auto"):
        try:
            importlib.import_module("gpu4pyscf.dft")
            print("  [ok] gpu4pyscf.dft      (GPU DFT backend)")
        except Exception as e:
            msg = f"  [..] gpu4pyscf.dft unavailable: {str(e)[:80]}"
            if BACKEND == "cuda":
                print(msg.replace("[..]", "[MISS]"))
                ok = False
            else:
                print(msg + " (ok on cpu)")

    if BACKEND == "rocm":
        if not check_module("cupy"):
            ok = False
        else:
            try:
                import cupy as cp

                n = cp.cuda.runtime.getDeviceCount()
                print(f"  CuPy {cp.__version__} AMD GPUs={n}")
                x = cp.arange(4, dtype=cp.float64).reshape(2, 2)
                _ = (x @ x).tolist()
                print("  [ok] cupy matmul")
            except Exception as e:
                ok = False
                print(f"  [MISS] cupy ROCm test failed: {str(e)[:120]}")

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        m = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        if AllChem.EmbedMolecule(m, randomSeed=0xF00D) != 0:
            raise RuntimeError("EmbedMolecule returned nonzero")
        print("  [ok] rdkit embed (CCO)")
    except Exception as e:
        ok = False
        print(f"  [MISS] rdkit embed failed: {str(e)[:120]}")

    try:
        from pyscf import dft, gto

        mol = gto.M(
            atom="O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586",
            basis="sto-3g",
            verbose=0,
        )
        mf = dft.RKS(mol).density_fit().SMD()
        mf.with_solvent.solvent = "water"
        mf.verbose = 0
        mf.max_cycle = 50
        e = float(mf.kernel())
        if not mf.converged:
            raise RuntimeError("SCF did not converge")
        print(f"  [ok] pyscf SMD water SP ({e:.4f} Ha)")
    except Exception as e:
        ok = False
        print(f"  [MISS] pyscf SMD water SP failed: {str(e)[:140]}")

    try:
        token = None
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if fairchem_ok and token:
            print("  [ok] UMA load path (fairchem importable, HF token present)")
        else:
            print("  [..] UMA weights skipped (no HF token) — run huggingface-cli login later")
    except Exception as e:
        print(f"  [..] UMA check skipped: {str(e)[:80]}")

    xtb_bin = shutil.which("xtb")
    print(f"  xtb binary: {xtb_bin or 'MISSING'}")
    try:
        importlib.import_module("xtb")
        print("  [ok] xtb python package")
    except Exception:
        print("  [..] xtb python package not importable (ok if xtb binary present)")

    print("ENV OK" if ok else "ENV INCOMPLETE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
