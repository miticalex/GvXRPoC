"""
Test compositing a Li dragon contaminant onto a fish-fillet X-ray TIFF.

Model: I_out = I_fish * (I_c / 80)^strength

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

from fillet_contaminant_compositor import (  # noqa: E402
    composite_contaminant_on_foodscan,
)

FOODSCAN = (
    ROOT
    / "assets"
    / "fish_fillets"
    / "2026-02-20_121244.829435_23490_TE_RAW.tif"
)
CONTAMINANT = ROOT / "output" / "diff_attenuations" / "Li_r_000_000_000"
OUTPUT_DIR = ROOT / "output" / "fillet_contaminant_composite"
SCALE = 1.0


def main() -> None:
    if not FOODSCAN.is_file():
        raise FileNotFoundError(FOODSCAN)
    if not (CONTAMINANT / "projection.npy").is_file():
        raise FileNotFoundError(CONTAMINANT / "projection.npy")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with Image.open(FOODSCAN) as im:
        h, w = np.array(im).shape[:2]
    position = (h / 2.0, w / 2.0 + 300.0)

    print(f"Foodscan:    {FOODSCAN}")
    print(f"Contaminant: {CONTAMINANT}")
    print(f"Position:    row={position[0]:.1f}, col={position[1]:.1f}  (centre + 300 px right)")
    print(f"Scale:       {SCALE}")

    composite, meta = composite_contaminant_on_foodscan(
        foodscan_path=FOODSCAN,
        contaminant_path=CONTAMINANT,
        position=position,
        scale=SCALE,
    )

    npy_path = OUTPUT_DIR / "composite.npy"
    tif_path = OUTPUT_DIR / "composite.tif"
    png_path = OUTPUT_DIR / "composite_preview.png"
    meta_path = OUTPUT_DIR / "meta.json"

    np.save(npy_path, composite)
    Image.fromarray(composite).save(tif_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fish = np.array(Image.open(FOODSCAN))
    vmin = float(np.min(fish))
    vmax = float(np.max(fish))
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].imshow(fish, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Fish fillet (original)")
    axes[0].axis("off")
    axes[1].imshow(composite, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Fillet + Li dragon (centre+300px right, scale=1)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    print(f"Display window (shared): vmin={vmin:.0f}, vmax={vmax:.0f}")
    print(f"Saved: {npy_path}")
    print(f"Saved: {tif_path}")
    print(f"Saved: {png_path}")
    print(f"Composite dtype={composite.dtype} min={composite.min()} max={composite.max()}")


if __name__ == "__main__":
    main()
