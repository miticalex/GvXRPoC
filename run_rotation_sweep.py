"""
Rotation sweeps for generate_contaminant_projection (material = Fe).

1) Axis sweeps — 30 projections (10 per axis):
     (0,0,0)..(0,0,90),  (0,0,0)..(0,90,0),  (0,0,0)..(90,0,0)
   Folders: output/rotation_sweep_axes/
   PNGs:    output/png00-90_axes/

2) Full grid — every (rx, ry, rz) with angles in {0,10,...,90}:
     10 x 10 x 10 = 1000 poses
   Folders: output/rotation_sweep_000909090/
   PNGs:    output/png000909090/

Usage:
    .\\.venv\\Scripts\\python.exe -u run_rotation_sweep.py
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
from gvxrPython3 import gvxr  # noqa: E402

MESH_PATH = ROOT / "data" / "welsh-dragon-small.stl"
MATERIAL = "Fe"
ANGLES = list(range(0, 100, 10))

AXES_DIR = ROOT / "output" / "rotation_sweep_axes"
AXES_PNG_DIR = ROOT / "output" / "png00-90_axes"
GRID_DIR = ROOT / "output" / "rotation_sweep_000909090"
GRID_PNG_DIR = ROOT / "output" / "png000909090"


def _save_preview(path: Path, projection: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(projection, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_run(
    run_dir: Path,
    png_dir: Path,
    png_name: str,
    rotation: tuple[float, float, float],
    projection: np.ndarray,
    *,
    save_tiff: bool = True,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "projection.npy", projection)
    (run_dir / "stats.txt").write_text(
        "\n".join(
            [
                f"rotation: {rotation}",
                f"material: {MATERIAL}",
                f"shape: {projection.shape}",
                f"dtype: {projection.dtype}",
                f"min: {np.nanmin(projection)}",
                f"max: {np.nanmax(projection)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if save_tiff:
        gvxr.saveLastXRayImage(str(run_dir / "projection.tif"))

    preview = run_dir / "preview.png"
    rx, ry, rz = rotation
    _save_preview(preview, projection, f"Fe  ({rx:g}, {ry:g}, {rz:g})")
    shutil.copy2(preview, png_dir / png_name)


def _run_complete(run_dir: Path, png_dir: Path, png_name: str) -> bool:
    return (run_dir / "projection.npy").is_file() and (png_dir / png_name).is_file()


def _project(rotation: tuple[float, float, float]) -> np.ndarray:
    return generate_contaminant_projection(
        mesh_path=MESH_PATH,
        rotation=rotation,
        material=MATERIAL,
        verbose=False,
    )


def run_axis_sweeps() -> None:
    print("\n========== AXIS SWEEPS (30) ==========")
    jobs: list[tuple[str, tuple[float, float, float]]] = []
    for a in ANGLES:
        jobs.append((f"rx_{a:03d}", (float(a), 0.0, 0.0)))
    for a in ANGLES:
        jobs.append((f"ry_{a:03d}", (0.0, float(a), 0.0)))
    for a in ANGLES:
        jobs.append((f"rz_{a:03d}", (0.0, 0.0, float(a))))

    t0 = time.perf_counter()
    for i, (run_id, rotation) in enumerate(jobs, start=1):
        run_dir = AXES_DIR / run_id
        png_name = f"{run_id}.png"
        if _run_complete(run_dir, AXES_PNG_DIR, png_name):
            print(f"[{i}/{len(jobs)}] {run_id}  skip (exists)")
            continue
        print(f"[{i}/{len(jobs)}] {run_id}  rotation={rotation}")
        projection = _project(rotation)
        _save_run(run_dir, AXES_PNG_DIR, png_name, rotation, projection, save_tiff=True)
    print(f"Axis sweeps done in {time.perf_counter() - t0:.1f}s")


def run_full_grid() -> None:
    n = len(ANGLES) ** 3
    print("\n========== FULL GRID ==========")
    print(f"Total poses: {n}")
    t0 = time.perf_counter()
    done = 0
    skipped = 0
    for rx in ANGLES:
        for ry in ANGLES:
            for rz in ANGLES:
                done += 1
                run_id = f"r_{rx:03d}_{ry:03d}_{rz:03d}"
                rotation = (float(rx), float(ry), float(rz))
                run_dir = GRID_DIR / run_id
                png_name = f"{run_id}.png"
                if _run_complete(run_dir, GRID_PNG_DIR, png_name):
                    skipped += 1
                    if done == 1 or done % 50 == 0 or done == n:
                        print(f"[{done}/{n}] {run_id}  skip (exists)")
                    continue
                if done == 1 or done % 50 == 0 or done == n:
                    elapsed = time.perf_counter() - t0
                    computed = done - skipped
                    rate = computed / elapsed if elapsed > 0 and computed > 0 else 0.0
                    eta = (n - done + 1) / rate if rate > 0 else float("nan")
                    print(f"[{done}/{n}] {run_id}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
                projection = _project(rotation)
                _save_run(
                    run_dir, GRID_PNG_DIR, png_name, rotation, projection, save_tiff=False
                )
    print(
        f"Full grid done in {time.perf_counter() - t0:.1f}s  ({n} poses, {skipped} skipped)"
    )


def main() -> None:
    if not MESH_PATH.is_file():
        raise FileNotFoundError(MESH_PATH)
    print(f"Mesh:     {MESH_PATH}")
    print(f"Material: {MATERIAL}")
    run_axis_sweeps()
    run_full_grid()
    print("\nAll sweeps finished.")


if __name__ == "__main__":
    main()
