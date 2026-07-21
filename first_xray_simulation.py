"""
First X-ray simulation with gVirtualXRay.

Mirrors tutorials/first_xray_simulation.ipynb as a plain Python script.
Six steps: OpenGL context → source → spectrum → detector → sample → compute image.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
from gvxrPython3 import gvxr

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "first_xray_image"
MESH_NAME = "welsh-dragon-small.stl"
MESH_PATH = DATA_DIR / MESH_NAME

MESH_URL = (
    "https://raw.githubusercontent.com/effepivi/gvxr-demos/"
    "main/training-course/input_data/welsh-dragon-small.stl"
)

SAMPLE_ID = "Dragon"


def ensure_mesh() -> Path:
    """Make sure data/welsh-dragon-small.stl exists; return its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if MESH_PATH.is_file():
        print(f"Using existing mesh: {MESH_PATH}")
        return MESH_PATH

    packaged = Path(gvxr.__file__).resolve().parent / MESH_NAME
    if packaged.is_file():
        print(f"Copying mesh from gvxr package:\n  {packaged}\n  -> {MESH_PATH}")
        shutil.copy2(packaged, MESH_PATH)
        return MESH_PATH

    print(f"Downloading mesh from:\n  {MESH_URL}")
    response = requests.get(MESH_URL, timeout=60)
    response.raise_for_status()
    MESH_PATH.write_bytes(response.content)
    print(f"Saved mesh to: {MESH_PATH}")
    return MESH_PATH


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh_path = ensure_mesh()

    gvxr.useLogFile(str(OUTPUT_DIR / "gvxr.log"))

    print("1. Create OpenGL context")
    gvxr.createOpenGLContext()

    print("2. Set X-ray source")
    gvxr.setSourcePosition(-40.0, 0.0, 0.0, "cm")
    gvxr.usePointSource()

    print("3. Set spectrum (monochromatic)")
    gvxr.setMonoChromaticPerPixelAtSDD(80.0, "keV", 1000)

    print("4. Set detector")
    gvxr.setDetectorPosition(10.0, 0.0, 0.0, "cm")
    gvxr.setDetectorUpVector(0, 0, -1)
    gvxr.setDetectorNumberOfPixels(640, 320)
    gvxr.setDetectorPixelSize(0.5, 0.5, "mm")

    print(f"5. Load sample from {mesh_path}")
    gvxr.loadMeshFile(SAMPLE_ID, str(mesh_path), "mm")
    gvxr.moveToCentre(SAMPLE_ID)
    gvxr.setMixture(SAMPLE_ID, "Ti90Al6V4")
    gvxr.setDensity(SAMPLE_ID, 4.43, "g/cm3")

    print("6. Compute X-ray image")
    projection = gvxr.computeXRayImage()
    projection_array = np.asarray(projection, dtype=np.float32)

    print("first 2 rows:", projection_array[:2])
    print("type:", type(projection))
    print("shape:", projection_array.shape)
    print("dtype:", projection_array.dtype)
    print("min:", np.nanmin(projection_array))
    print("max:", np.nanmax(projection_array))
    print("contains NaN:", np.isnan(projection_array).any())
    print("contains infinity:", np.isinf(projection_array).any())

    rows_txt_path = OUTPUT_DIR / "first_2_rows.txt"
    np.savetxt(rows_txt_path, projection_array[:2], fmt="%.6g")
    print(f"Saved first 2 rows: {rows_txt_path}")

    npy_path = OUTPUT_DIR / "raw_x-ray_image.npy"
    np.save(npy_path, projection_array)
    print(f"Saved NPY: {npy_path}")

    tiff_path = OUTPUT_DIR / "raw_x-ray_image.tif"
    gvxr.saveLastXRayImage(str(tiff_path))
    print(f"Saved TIFF: {tiff_path}")

    preview_path = OUTPUT_DIR / "preview.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(projection_array, cmap="gray")
    ax.set_title("First simulated X-ray image (Welsh dragon, Ti90Al6V4)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    print(f"Saved preview: {preview_path}")

    print(
        f"\nDone. Image shape: {projection_array.shape}, "
        f"min={projection_array.min():.4g}, max={projection_array.max():.4g}"
    )

    gvxr.destroy()


if __name__ == "__main__":
    main()
