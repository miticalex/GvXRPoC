"""
Composite a simulated contaminant projection onto a real food X-ray (fillet) scan.

Does not modify gVXR / contaminant_simulator behaviour. Physics model:

    I_out = I_fish * T_c^strength

where
    T_c = I_contaminant / I0_contaminant   (transmission factor in (0, 1])

I_fish already contains the fillet's Beer-Lambert attenuation (from the line TIFF).
We do not re-estimate fish mu; the radiograph *is* the fish attenuation map.

Contaminant should be a float .npy from gVXR (preferred over PNG).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import cv2
import numpy as np

# Open-beam intensity used in notebook / contaminant_simulator monochromatic sims
DEFAULT_CONTAMINANT_I0 = 80.0

PathLike = Union[str, Path]


def _resolve_contaminant_npy(contaminant_path: PathLike) -> Path:
    """Accept a .npy file or a run folder containing projection.npy."""
    path = Path(contaminant_path)
    if path.is_dir():
        candidate = path / "projection.npy"
        if not candidate.is_file():
            raise FileNotFoundError(f"No projection.npy in folder: {path}")
        return candidate
    if path.suffix.lower() != ".npy":
        raise ValueError(
            f"Contaminant must be a .npy file or a folder with projection.npy, got: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_foodscan(foodscan_path: PathLike) -> np.ndarray:
    """Load a food-line radiograph (TIFF/PNG/…) as 2D float64."""
    path = Path(foodscan_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    # Prefer tifffile-free path: Pillow handles I;16 industrial TIFFs
    from PIL import Image

    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float64)


def _transmission_from_contaminant(
    intensity: np.ndarray,
    open_beam: float,
    strength: float,
) -> np.ndarray:
    """
    Convert contaminant intensity image to a transmission multiplier.

    T = (I / I0)^strength, clipped to (eps, 1].
    strength=1 is pure extra optical depth; >1 exaggerates the contaminant.
    """
    if open_beam <= 0:
        raise ValueError("open_beam must be > 0")
    eps = 1e-8
    t = np.clip(intensity.astype(np.float64) / float(open_beam), eps, 1.0)
    if strength != 1.0:
        t = np.power(t, float(strength))
    return t


def _resize_transmission(t: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if abs(scale - 1.0) < 1e-12:
        return t
    h, w = t.shape
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # Linear is fine for a smooth transmission field
    return cv2.resize(t, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _paste_transmission(
    canvas_shape: tuple[int, int],
    transmission: np.ndarray,
    position_rc: Sequence[float],
) -> np.ndarray:
    """
    Build a full-frame transmission map (default 1) and paste ``transmission``
    centred at ``position_rc = (row, col)`` on the foodscan grid.
    """
    if len(position_rc) != 2:
        raise ValueError("position must be (row, col)")

    h_f, w_f = canvas_shape
    h_c, w_c = transmission.shape
    center_r, center_c = float(position_rc[0]), float(position_rc[1])

    # Top-left of paste in foodscan coordinates
    r0 = int(round(center_r - h_c / 2.0))
    c0 = int(round(center_c - w_c / 2.0))
    r1, c1 = r0 + h_c, c0 + w_c

    # Clip to foodscan bounds
    src_r0 = max(0, -r0)
    src_c0 = max(0, -c0)
    dst_r0 = max(0, r0)
    dst_c0 = max(0, c0)
    dst_r1 = min(h_f, r1)
    dst_c1 = min(w_f, c1)
    src_r1 = src_r0 + (dst_r1 - dst_r0)
    src_c1 = src_c0 + (dst_c1 - dst_c0)

    t_map = np.ones(canvas_shape, dtype=np.float64)
    if dst_r1 > dst_r0 and dst_c1 > dst_c0:
        t_map[dst_r0:dst_r1, dst_c0:dst_c1] *= transmission[
            src_r0:src_r1, src_c0:src_c1
        ]
    return t_map


def composite_contaminant_on_foodscan(
    foodscan_path: PathLike,
    contaminant_path: PathLike,
    position: Sequence[float],
    scale: float = 1.0,
    *,
    open_beam_contaminant: float | None = None,
    contaminant_strength: float = 1.0,
    return_uint16: bool = True,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """
    Place a gVXR contaminant projection onto a food X-ray scan.

    Parameters
    ----------
    foodscan_path :
        Path to the fillet / food radiograph (e.g. uint16 TIFF).
    contaminant_path :
        Path to ``projection.npy`` **or** a run folder that contains it.
    position :
        ``(row, col)`` centre of the contaminant on the foodscan pixel grid.
    scale :
        Spatial scale of the contaminant patch (1.0 = native npy size).
    open_beam_contaminant :
        I0 of the contaminant simulation (air). Default: ``DEFAULT_CONTAMINANT_I0``
        (80.0, matching the monochromatic gVXR demos).
    contaminant_strength :
        Exponent on transmission; 1.0 = physical extra path, larger = darker.
    return_uint16 :
        If True, clip/round to uint16 like the line cameras.

    Returns
    -------
    composite, meta
        Composite image and a small metadata dict (shapes, I0, etc.).

    Notes
    -----
    Fish attenuation is taken from the foodscan itself (already imaged).
    Soft-tissue mu at ~80 keV is roughly 0.18–0.22 /cm for reference only;
    it is not used in this composite because we lack a fish thickness map.
    """
    npy_path = _resolve_contaminant_npy(contaminant_path)
    fish = _load_foodscan(foodscan_path)
    contam = np.load(npy_path)

    if contam.ndim != 2:
        raise ValueError(f"Contaminant must be 2D, got shape {contam.shape}")

    i0 = (
        float(open_beam_contaminant)
        if open_beam_contaminant is not None
        else DEFAULT_CONTAMINANT_I0
    )

    t_c = _transmission_from_contaminant(contam, i0, contaminant_strength)
    t_c = _resize_transmission(t_c, scale)
    t_map = _paste_transmission(fish.shape, t_c, position)

    composite = fish * t_map

    meta: dict[str, Any] = {
        "foodscan_path": str(Path(foodscan_path).resolve()),
        "contaminant_npy": str(npy_path.resolve()),
        "position_row_col": (float(position[0]), float(position[1])),
        "scale": float(scale),
        "open_beam_contaminant": i0,
        "contaminant_strength": float(contaminant_strength),
        "fish_shape": tuple(int(x) for x in fish.shape),
        "contaminant_shape_native": tuple(int(x) for x in contam.shape),
        "contaminant_shape_scaled": tuple(int(x) for x in t_c.shape),
        "fish_min": float(fish.min()),
        "fish_max": float(fish.max()),
        "composite_min": float(composite.min()),
        "composite_max": float(composite.max()),
        # Reference only — not applied in the formula:
        "soft_tissue_mu_ref_per_cm_80keV": 0.20,
        "model": "I_out = I_fish * (I_c / I0) ** strength",
    }

    if return_uint16:
        composite = np.clip(np.rint(composite), 0, 65535).astype(np.uint16)

    return composite, meta
