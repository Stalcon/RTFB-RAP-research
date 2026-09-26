from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
LIBRARY = ROOT / "library"
CALCS = ROOT / "calcs"
UMA = CALCS / "uma"
DFT = CALCS / "dft"
RESULTS = ROOT / "results"

HARTREE_EV = 27.2114
EV_KJ = 96.485
KT_EV = 0.02569


def read_manifest() -> list[dict]:
    with (LIBRARY / "manifest.csv").open() as f:
        return list(csv.DictReader(f))


def read_result(gid: str, state: str, root: Path = DFT) -> dict | None:
    p = Path(root) / gid / state / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_xyz(atoms, path: Path, comment: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(atoms)), comment]
    for s, p in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}")
    path.write_text("\n".join(lines) + "\n")
