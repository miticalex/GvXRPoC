"""
Composite Li dragon + first two random Li polyhedra onto a fish-fillet X-ray.

Placements (row, col on foodscan):
  - Li dragon:           centre + 300 px right
  - random shape_00:     centre
  - random shape_01:     centre - 300 px left

Meshes for shape_00 / shape_01 come from:
  C:\\Projects\\DiraTech\\GVxRPoC\\data\\random_li_polyhedra
Projections are simulated into this repo's output/random_li_polyhedra if missing.
Export folder is unchanged: output/fillet_contaminant_composite

Usage:
    .\\.venv\\Scripts\\python.exe -u test_fillet_contaminant_composite.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from contaminant_simulator import generate_contaminant_projection  # noqa: E402
from fillet_contaminant_compositor import (  # noqa: E402
    composite_contaminants_on_foodscan,
)

FOODSCAN = (
    ROOT
    / "assets"
    / "fish_fillets"
    / "2026-02-20_121244.829435_23490_TE_RAW.tif"
)
DRAGON = ROOT / "output" / "diff_attenuations" / "Li_r_000_000_000"
# User-specified mesh source (1st / 2nd random Li polyhedra)
MESH_DIR = Path(r"C:\Projects\DiraTech\GVxRPoC\data\random_li_polyhedra")
POLY_OUT = ROOT / "output" / "random_li_polyhedra"
OUTPUT_DIR = ROOT / "output" / "fillet_contaminant_composite"
SCALE = 1.0


def _ensure_poly_projection(shape_id: str, *, force: bool = False) -> Path:
    """Simulate projection.npy for a polyhedron STL (optionally force re-sim)."""
    out_dir = POLY_OUT / shape_id
    npy_path = out_dir / "projection.npy"
    if npy_path.is_file() and not force:
        return out_dir

    mesh = MESH_DIR / f"{shape_id}.stl"
    if not mesh.is_file():
        raise FileNotFoundError(mesh)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Simulating {shape_id} from {mesh} ...")
    img = generate_contaminant_projection(
        mesh_path=mesh,
        rotation=(0.0, 0.0, 0.0),
        material="Li",
        verbose=True,
    )
    np.save(npy_path, img)
    print(
        f"  saved {npy_path}  min={float(img.min()):.2f} max={float(img.max()):.2f}"
    )
    return out_dir


def main() -> None:
    if not FOODSCAN.is_file():
        raise FileNotFoundError(FOODSCAN)
    if not (DRAGON / "projection.npy").is_file():
        raise FileNotFoundError(DRAGON / "projection.npy")

    # Re-use cached projections unless deleted
    shape_00 = _ensure_poly_projection("shape_00", force=False)
    shape_01 = _ensure_poly_projection("shape_01", force=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(FOODSCAN) as im:
        h, w = np.array(im).shape[:2]
    center = (h / 2.0, w / 2.0)
    dragon_pos = (center[0], center[1] + 300.0)
    shape00_pos = center
    shape01_pos = (center[0], center[1] - 300.0)

    # Polyhedra native detector size (~60 mm / ~140 px)
    poly_scale = SCALE
    contaminants = [
        {"path": DRAGON, "position": dragon_pos, "scale": SCALE},
        {"path": shape_00, "position": shape00_pos, "scale": poly_scale},
        {"path": shape_01, "position": shape01_pos, "scale": poly_scale},
    ]

    print(f"Foodscan: {FOODSCAN}")
    print(f"Dragon:   {DRAGON}  @ centre+300 right  scale={SCALE}")
    print(f"shape_00: {shape_00}  @ centre  scale={poly_scale}")
    print(f"shape_01: {shape_01}  @ centre-300 left  scale={poly_scale}")

    composite, meta = composite_contaminants_on_foodscan(
        foodscan_path=FOODSCAN,
        contaminants=contaminants,
    )

    npy_path = OUTPUT_DIR / "composite.npy"
    tif_path = OUTPUT_DIR / "composite.tif"
    png_path = OUTPUT_DIR / "composite_preview.png"
    diff_path = OUTPUT_DIR / "composite_diff_preview.png"
    meta_path = OUTPUT_DIR / "meta.json"

    np.save(npy_path, composite)
    Image.fromarray(composite).save(tif_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fish = np.array(Image.open(FOODSCAN)).astype(np.float64)
    comp = composite.astype(np.float64)
    vmin = float(np.min(fish))
    vmax = float(np.max(fish))
    diff = fish - comp

    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    axes[0].imshow(fish, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Fish fillet (original)")
    axes[0].axis("off")
    axes[1].imshow(comp, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title(
        "Fillet + dragon (+300R) + shape_00 (centre) + shape_01 (-300L)"
    )
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (r, c), title in zip(
        axes,
        [shape01_pos, shape00_pos, dragon_pos],
        ["shape_01 (-300L)", "shape_00 (centre)", "dragon (+300R)"],
    ):
        half = 220
        r0, r1 = int(r - half), int(r + half)
        c0, c1 = int(c - half), int(c + half)
        crop = diff[max(0, r0) : min(h, r1), max(0, c0) : min(w, c1)]
        ax.imshow(crop, cmap="magma", vmin=0, vmax=max(1.0, float(diff.max())))
        ax.set_title(f"{title}\nmax drop={float(crop.max()):.0f}")
        ax.axis("off")
    fig.suptitle("Intensity drop (original - composite), zoomed")
    fig.tight_layout()
    fig.savefig(diff_path, dpi=140)
    plt.close(fig)

    print(f"Display window (shared): vmin={vmin:.0f}, vmax={vmax:.0f}")
    print(f"Saved: {npy_path}")
    print(f"Saved: {tif_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {diff_path}")
    print(
        f"Composite dtype={composite.dtype} "
        f"min={composite.min()} max={composite.max()}"
    )


if __name__ == "__main__":
    main()
