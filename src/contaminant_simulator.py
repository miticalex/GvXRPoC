"""
Contaminant X-ray projection helper built from tutorials/first_xray_simulation.ipynb.

Immutable reference: tutorials/first_xray_simulation.ipynb
(Adapted cells: 8, 10, 14, 16, 18, 20, 22, 23, 25.)

Rotation convention
-------------------
``rotation`` is ``(rx_degrees, ry_degrees, rz_degrees)``.

Axes are Descartes / world XYZ, parallel to the imaging frustum axes, and they
pass through the **mesh bounding-box centre** (not the STL file origin).

gVXR's ``rotateNode`` rotates about the node's local origin. STL local origins
are often far from the geometric centre, so after ``moveToCentre`` we bake that
translation into the vertices with ``applyCurrentLocalTransformation``. After
that bake, the local origin coincides with the mesh centre and Rx/Ry/Rz spin
the object in place.

Fixed order after centering/bake:

    1. Rx — rotate about +X through mesh centre
    2. Ry — rotate about +Y through mesh centre
    3. Rz — rotate about +Z through mesh centre

Each call restores the unrotated (centred) mesh, then applies that call's
rotation only — rotations never accumulate.
Source, detector, spectrum, mesh position/scale match the notebook demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np
from gvxrPython3 import gvxr

_SAMPLE_ID = "Contaminant"

_SOURCE_POSITION_CM = (-40.0, 0.0, 0.0)
_DETECTOR_POSITION_CM = (10.0, 0.0, 0.0)
_DETECTOR_UP = (0, 0, -1)
_DETECTOR_PIXELS = (640, 320)
_DETECTOR_PIXEL_SIZE_MM = (0.5, 0.5)
_SPECTRUM_MEV = 0.08
_SPECTRUM_PHOTONS = 1000

_IDENTITY_4X4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

_context_ready = False
_loaded_mesh_path: Path | None = None
_mesh_centre_baked = False
_verbose = True


def _log(*args, **kwargs) -> None:
    if _verbose:
        print(*args, **kwargs)


def _ensure_opengl_context() -> None:
    global _context_ready
    if _context_ready:
        return
    gvxr.useLogFile()
    _log("Create an OpenGL context")
    gvxr.createOpenGLContext()
    _context_ready = True


def _ensure_unrotated_mesh(mesh_path: Path) -> None:
    """
    Load STL or reset rotation. When the path changes, swap meshes in the same
    OpenGL context (do not destroy the context — that breaks later computes).
    """
    global _loaded_mesh_path, _mesh_centre_baked
    resolved = mesh_path.resolve()

    if _loaded_mesh_path != resolved:
        if _loaded_mesh_path is not None:
            _log("Swap mesh — remove previous sample from renderer/scene")
            try:
                gvxr.removePolygonMeshesFromXRayRenderer()
            except Exception:
                pass
            try:
                gvxr.removeMeshAndChildren(_SAMPLE_ID)
            except Exception:
                pass
            _mesh_centre_baked = False
        _log("Load the mesh data from", mesh_path)
        gvxr.loadMeshFile(_SAMPLE_ID, str(mesh_path), "mm")
        _loaded_mesh_path = resolved
    else:
        _log("Reset mesh transform to identity (unrotated mesh state)")
        gvxr.setNodeTransformationMatrix(_SAMPLE_ID, _IDENTITY_4X4)

    _log("Object/mesh: id={!r}, path={}, unit=mm".format(_SAMPLE_ID, mesh_path))


def _centre_mesh_for_in_place_rotation() -> None:
    global _mesh_centre_baked

    if not _mesh_centre_baked:
        _log("Move", _SAMPLE_ID, "to the centre")
        gvxr.moveToCentre(_SAMPLE_ID)
        _log(
            "Bake centring into vertices so Rx/Ry/Rz pivot at mesh bbox centre "
            "(world-parallel axes)"
        )
        gvxr.applyCurrentLocalTransformation(_SAMPLE_ID)
        _mesh_centre_baked = True
    else:
        _log("Mesh vertices already centred at origin (pivot = bbox centre)")

    cx, cy, cz = gvxr.getNodeOnlyBoundingBoxCentre(_SAMPLE_ID, "cm")
    _log("Mesh bbox centre (cm): ({:.6g}, {:.6g}, {:.6g})".format(cx, cy, cz))


def _apply_rotation(rx_deg: float, ry_deg: float, rz_deg: float) -> None:
    gvxr.rotateNode(_SAMPLE_ID, float(rx_deg), 1.0, 0.0, 0.0)
    gvxr.rotateNode(_SAMPLE_ID, float(ry_deg), 0.0, 1.0, 0.0)
    gvxr.rotateNode(_SAMPLE_ID, float(rz_deg), 0.0, 0.0, 1.0)

    cx, cy, cz = gvxr.getNodeOnlyBoundingBoxCentre(_SAMPLE_ID, "cm")
    _log(
        "Mesh bbox centre after rotation (cm): ({:.6g}, {:.6g}, {:.6g}) "
        "(should stay ~0 if pivot is correct)".format(cx, cy, cz)
    )


def _setup_source_spectrum_detector() -> None:
    _log("Set up the beam")
    sx, sy, sz = _SOURCE_POSITION_CM
    gvxr.setSourcePosition(sx, sy, sz, "cm")
    gvxr.usePointSource()
    _log(
        f"Source: point source at ({sx}, {sy}, {sz}) cm; "
        f"spectrum monochromatic {_SPECTRUM_MEV} MeV "
        f"({_SPECTRUM_MEV * 1000:g} keV), {_SPECTRUM_PHOTONS} photons/pixel at SDD"
    )

    gvxr.setMonoChromaticPerPixelAtSDD(_SPECTRUM_MEV, "MeV", _SPECTRUM_PHOTONS)

    _log("Set up the detector")
    dx, dy, dz = _DETECTOR_POSITION_CM
    ux, uy, uz = _DETECTOR_UP
    nw, nh = _DETECTOR_PIXELS
    pw, ph = _DETECTOR_PIXEL_SIZE_MM
    gvxr.setDetectorPosition(dx, dy, dz, "cm")
    gvxr.setDetectorUpVector(ux, uy, uz)
    gvxr.setDetectorNumberOfPixels(nw, nh)
    gvxr.setDetectorPixelSize(pw, ph, "mm")
    _log(
        f"Detector: position ({dx}, {dy}, {dz}) cm; "
        f"up vector ({ux}, {uy}, {uz}); "
        f"{nw}x{nh} pixels; pixel size {pw}x{ph} mm"
    )


def generate_contaminant_projection(
    mesh_path: Union[str, Path],
    rotation: Sequence[float],
    material: str,
    *,
    verbose: bool = True,
) -> np.ndarray:
    """
    Simulate one raw 2D X-ray projection of a contaminant mesh.

    Returns float32 array shape (320, 640).
    """
    global _verbose
    _verbose = verbose

    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    if len(rotation) != 3:
        raise ValueError("rotation must be (rx_degrees, ry_degrees, rz_degrees)")

    rx, ry, rz = (float(rotation[0]), float(rotation[1]), float(rotation[2]))

    _ensure_opengl_context()
    _setup_source_spectrum_detector()
    _ensure_unrotated_mesh(mesh_path)
    _centre_mesh_for_in_place_rotation()

    _log("Set", _SAMPLE_ID, "'s material")
    _log("Material: element {!r} (gvxr.setElement)".format(material))
    gvxr.setElement(_SAMPLE_ID, material)

    _log(
        "Applied rotation (degrees), order Rx->Ry->Rz about mesh bbox centre "
        "(world-parallel / frustum-parallel axes): "
        "rx={:.6g}, ry={:.6g}, rz={:.6g}".format(rx, ry, rz)
    )
    _apply_rotation(rx, ry, rz)

    _log("Compute an X-ray image")
    projection = gvxr.computeXRayImage()
    projection_array = np.asarray(projection, dtype=np.float32)

    _log("Projection type (gVXR return):", type(projection))
    _log("Projection type (NumPy):", type(projection_array))
    _log("shape:", projection_array.shape)
    _log("dtype:", projection_array.dtype)
    _log("min:", np.nanmin(projection_array))
    _log("max:", np.nanmax(projection_array))
    _log("contains NaN:", bool(np.isnan(projection_array).any()))
    _log("contains infinity:", bool(np.isinf(projection_array).any()))

    return projection_array
