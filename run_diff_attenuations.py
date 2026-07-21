"""
Compare X-ray attenuation across 5 elements (nearly transparent -> iron).

Materials (low -> high attenuation at ~80 keV), via setElement:
  Li, Be, Al, Ti, Fe

Rotations: every (rx, ry, rz) with angles in {0, 45, 90} -> 3^3 = 27 poses.
Total projections: 5 x 27 = 135.

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u run_diff_attenuations.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from contaminant_simulator import generate_contaminant_projection  # noqa: E402

MESH_PATH = ROOT / "data" / "welsh-dragon-small.stl"
MATERIALS = ("Li", "Be", "Al", "Ti", "Fe")
ANGLES = (0, 45, 90)

OUT_DIR = ROOT / "output" / "diff_attenuations"
PNG_ALL_DIR = ROOT / "output" / "pngs_5attenuations_04590"
PNG_ZERO_DIR = ROOT / "output" / "pngs_attenuations"


def _run_id(material: str, rx: int, ry: int, rz: int) -> str:
    return f"{material}_r_{rx:03d}_{ry:03d}_{rz:03d}"


def _save_preview(path: Path, projection: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(projection, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_run(
    material: str,
    rotation: tuple[int, int, int],
    projection: np.ndarray,
) -> None:
    rx, ry, rz = rotation
    run_id = _run_id(material, rx, ry, rz)
    run_dir = OUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    PNG_ALL_DIR.mkdir(parents=True, exist_ok=True)
    PNG_ZERO_DIR.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "projection.npy", projection)
    (run_dir / "stats.txt").write_text(
        "\n".join(
            [
                f"material: {material}",
                f"rotation: ({rx}, {ry}, {rz})",
                f"shape: {projection.shape}",
                f"dtype: {projection.dtype}",
                f"min: {np.nanmin(projection)}",
                f"max: {np.nanmax(projection)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    preview = run_dir / "preview.png"
    _save_preview(preview, projection, f"{material}  ({rx}, {ry}, {rz})")

    png_name = f"{run_id}.png"
    shutil.copy2(preview, PNG_ALL_DIR / png_name)
    if rotation == (0, 0, 0):
        shutil.copy2(preview, PNG_ZERO_DIR / png_name)


def _complete(material: str, rotation: tuple[int, int, int]) -> bool:
    rx, ry, rz = rotation
    run_id = _run_id(material, rx, ry, rz)
    return (OUT_DIR / run_id / "projection.npy").is_file() and (
        PNG_ALL_DIR / f"{run_id}.png"
    ).is_file()


def main() -> None:
    if not MESH_PATH.is_file():
        raise FileNotFoundError(
            f"{MESH_PATH}\nRun first_xray_simulation.py first to obtain the dragon STL."
        )

    jobs = [
        (mat, (rx, ry, rz))
        for mat in MATERIALS
        for rx in ANGLES
        for ry in ANGLES
        for rz in ANGLES
    ]
    n = len(jobs)

    print(f"Mesh:      {MESH_PATH}")
    print(f"Materials: {MATERIALS}  (low -> high attenuation)")
    print(f"Angles:    {ANGLES}  -> {len(ANGLES)**3} rotations each")
    print(f"Total:     {n} projections")

    t0 = time.perf_counter()
    skipped = 0
    for i, (material, rotation) in enumerate(jobs, start=1):
        run_id = _run_id(material, *rotation)
        if _complete(material, rotation):
            skipped += 1
            if i == 1 or i % 25 == 0 or i == n:
                print(f"[{i}/{n}] {run_id}  skip (exists)")
            continue

        if i == 1 or i % 25 == 0 or i == n:
            elapsed = time.perf_counter() - t0
            computed = i - skipped
            rate = computed / elapsed if elapsed > 0 and computed > 0 else 0.0
            eta = (n - i + 1) / rate if rate > 0 else float("nan")
            print(f"[{i}/{n}] {run_id}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

        projection = generate_contaminant_projection(
            mesh_path=MESH_PATH,
            rotation=(float(rotation[0]), float(rotation[1]), float(rotation[2])),
            material=material,
            verbose=False,
        )
        _save_run(material, rotation, projection)

    print(f"\nDone in {time.perf_counter() - t0:.1f}s  ({n} jobs, {skipped} skipped)")


if __name__ == "__main__":
    main()
