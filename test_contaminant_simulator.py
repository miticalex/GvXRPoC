"""
Smoke-test generate_contaminant_projection and write comparison artefacts.

Usage:
    .\\.venv\\Scripts\\python.exe -u test_contaminant_simulator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from contaminant_simulator import generate_contaminant_projection  # noqa: E402
from gvxrPython3 import gvxr  # noqa: E402

MESH_PATH = ROOT / "data" / "welsh-dragon-small.stl"
OUTPUT_DIR = ROOT / "output" / "contaminant_simulator_test"
FIRST_XRAY_NPY = ROOT / "output" / "first_xray_image" / "raw_x-ray_image.npy"


def _save_bundle(name: str, projection: np.ndarray) -> Path:
    out = OUTPUT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "projection.npy", projection)
    np.savetxt(out / "first_2_rows.txt", projection[:2], fmt="%.6g")
    gvxr.saveLastXRayImage(str(out / "projection.tif"))
    (out / "stats.txt").write_text(
        "\n".join(
            [
                f"shape: {projection.shape}",
                f"dtype: {projection.dtype}",
                f"min: {np.nanmin(projection)}",
                f"max: {np.nanmax(projection)}",
                f"mean: {np.nanmean(projection)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    preview = out / "preview.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(projection, cmap="gray")
    ax.set_title(f"contaminant_simulator: {name}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    print(f"Saved bundle -> {out}")
    return out


def _compare(a: np.ndarray, b: np.ndarray, label: str) -> None:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    print(f"Compare [{label}]")
    print(f"  identical: {np.array_equal(a, b)}")
    print(f"  max|diff|: {diff.max()}")
    print(f"  mean|diff|: {diff.mean()}")


def main() -> None:
    if not MESH_PATH.is_file():
        raise FileNotFoundError(MESH_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    print("\n=== Run A: rotation=(0,0,0), material=Fe ===")
    proj_a = generate_contaminant_projection(MESH_PATH, (0.0, 0.0, 0.0), "Fe")
    _save_bundle("A_identity_Fe", proj_a)

    print("\n=== Run A2: same as A (repeat) ===")
    proj_a2 = generate_contaminant_projection(MESH_PATH, (0.0, 0.0, 0.0), "Fe")
    _save_bundle("A2_identity_Fe_repeat", proj_a2)
    _compare(proj_a, proj_a2, "A vs A2 (should match)")

    print("\n=== Run B: rotation=(0,10,10), material=Fe ===")
    proj_b = generate_contaminant_projection(MESH_PATH, (0.0, 10.0, 10.0), "Fe")
    _save_bundle("B_rot_0_10_10_Fe", proj_b)
    _compare(proj_a, proj_b, "A vs B (should differ)")

    if FIRST_XRAY_NPY.is_file():
        first = np.load(FIRST_XRAY_NPY)
        print("\n=== Compare to first_xray (Ti alloy) ===")
        _compare(proj_a, first, "A (Fe) vs first_xray (Ti)")
    print("\nDone.")


if __name__ == "__main__":
    main()
