"""
Generate 10 random 20-triangle polyhedra (Li) and simulate X-ray projections.

Meshes:
  - Topology: icosahedron = 12 vertices, exactly 20 triangular faces
  - Shape: random radial vertex jitter (can look convex or dented/concave)
  - Size: axis-aligned bounding box no larger than 60 x 60 x 60 mm
  - Material: Li (lithium) via generate_contaminant_projection
  - Beam / detector / spectrum: same as contaminant_simulator / notebook demo
  - Rotation: (0, 0, 0) — shape variation only

Outputs:
  data/random_li_polyhedra/shape_XX.stl
  output/random_li_polyhedra/shape_XX/   (npy, preview, stats)
  output/pngs_random_li_polyhedra/shape_XX.png

Usage (from repo root):
    .\\.venv\\Scripts\\python.exe -u run_random_be_polyhedra.py
"""

from __future__ import annotations

import shutil
import struct
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from contaminant_simulator import generate_contaminant_projection  # noqa: E402

MESH_DIR = ROOT / "data" / "random_li_polyhedra"
OUT_DIR = ROOT / "output" / "random_li_polyhedra"
PNG_DIR = ROOT / "output" / "pngs_random_li_polyhedra"

N_SHAPES = 10
MAX_EXTENT_MM = 60.0
MATERIAL = "Li"
ROTATION = (0.0, 0.0, 0.0)
RNG_SEED = 42


def _icosahedron_unit() -> tuple[np.ndarray, np.ndarray]:
    """Regular icosahedron: 12 verts, 20 triangular faces, roughly unit size."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1.0, phi, 0.0],
            [1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [1.0, -phi, 0.0],
            [0.0, -1.0, phi],
            [0.0, 1.0, phi],
            [0.0, -1.0, -phi],
            [0.0, 1.0, -phi],
            [phi, 0.0, -1.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Project onto unit sphere
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )
    return verts, faces


def _orient_faces_outward(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip any triangle whose normal points toward the mesh centroid."""
    centroid = verts.mean(axis=0)
    out = faces.copy()
    for i, (a, b, c) in enumerate(faces):
        n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        face_center = (verts[a] + verts[b] + verts[c]) / 3.0
        if np.dot(n, face_center - centroid) < 0.0:
            out[i] = [a, c, b]
    return out


def random_20face_polyhedron(
    rng: np.random.Generator,
    max_extent_mm: float = MAX_EXTENT_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build one random 20-face triangle mesh inside a max_extent_mm cube.

    Radial jitter in [0.35, 1.55] can pull vertices inward enough for
    dented / non-convex silhouettes while keeping 20 triangles.
    """
    verts, faces = _icosahedron_unit()

    # Independent radial scales per vertex (irregular / possibly concave look)
    scales = rng.uniform(0.35, 1.55, size=(verts.shape[0], 1))
    verts = verts * scales

    # Small tangential noise
    verts += rng.normal(0.0, 0.08, size=verts.shape)

    faces = _orient_faces_outward(verts, faces)

    # Centre at origin, then scale so AABB fits in max_extent_mm^3
    verts = verts - verts.mean(axis=0)
    extent = verts.max(axis=0) - verts.min(axis=0)
    longest = float(np.max(extent))
    if longest < 1e-9:
        raise RuntimeError("Degenerate mesh generated")
    # Use 98% of the budget so we stay strictly under the limit after float noise
    verts *= (0.98 * max_extent_mm) / longest

    extent = verts.max(axis=0) - verts.min(axis=0)
    assert np.all(extent <= max_extent_mm + 1e-6)

    return verts.astype(np.float64), faces


def write_binary_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary STL (mm coordinates)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_faces = int(faces.shape[0])
    with path.open("wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", n_faces))
        for a, b, c in faces:
            p0, p1, p2 = verts[a], verts[b], verts[c]
            normal = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            else:
                normal = np.zeros(3)
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *p0))
            f.write(struct.pack("<3f", *p1))
            f.write(struct.pack("<3f", *p2))
            f.write(struct.pack("<H", 0))


def _save_preview(path: Path, projection: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(projection, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _bbox_mm(verts: np.ndarray) -> np.ndarray:
    return verts.max(axis=0) - verts.min(axis=0)


def main() -> None:
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    print(f"Generating {N_SHAPES} random 20-face polyhedra (Li)")
    print(f"Max AABB: {MAX_EXTENT_MM} x {MAX_EXTENT_MM} x {MAX_EXTENT_MM} mm")
    print(f"Meshes:   {MESH_DIR}")
    print(f"Runs:     {OUT_DIR}")
    print(f"PNGs:     {PNG_DIR}")

    t0 = time.perf_counter()
    for i in range(N_SHAPES):
        name = f"shape_{i:02d}"
        stl_path = MESH_DIR / f"{name}.stl"
        run_dir = OUT_DIR / name
        png_name = f"{name}.png"

        verts, faces = random_20face_polyhedron(rng)
        write_binary_stl(stl_path, verts, faces)
        bb = _bbox_mm(verts)
        print(
            f"[{i + 1}/{N_SHAPES}] {name}: "
            f"{faces.shape[0]} faces, bbox_mm=({bb[0]:.2f}, {bb[1]:.2f}, {bb[2]:.2f})"
        )

        projection = generate_contaminant_projection(
            mesh_path=stl_path,
            rotation=ROTATION,
            material=MATERIAL,
            verbose=False,
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        np.save(run_dir / "projection.npy", projection)
        (run_dir / "stats.txt").write_text(
            "\n".join(
                [
                    f"mesh: {stl_path}",
                    f"material: {MATERIAL}",
                    f"rotation: {ROTATION}",
                    f"faces: {int(faces.shape[0])}",
                    f"bbox_mm: ({bb[0]:.6g}, {bb[1]:.6g}, {bb[2]:.6g})",
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
        _save_preview(preview, projection, f"Li  {name}  rot={ROTATION}")
        shutil.copy2(preview, PNG_DIR / png_name)

    print(f"\nDone in {time.perf_counter() - t0:.1f}s")
    print(f"  {MESH_DIR}")
    print(f"  {OUT_DIR}")
    print(f"  {PNG_DIR}")


if __name__ == "__main__":
    main()
