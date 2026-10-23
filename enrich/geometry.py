"""Pixel <-> pitch with the per-frame homography (normalised pitch 0-1 -> pixels, see DATA.md)."""
import numpy as np


def pixels_to_pitch(homography_row, px, py):
    """Image pixels -> normalised pitch coordinates for one frame. NaN if the matrix is missing.

    Assumes the point is on the ground plane: a lofted ball projects further from the camera than
    it is, which is one reason merge.py checks detections against the trajectory around them.
    """
    px, py = np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64)
    if not np.all(np.isfinite(homography_row)):
        return np.full(px.shape, np.nan), np.full(py.shape, np.nan)
    inverse = np.linalg.inv(np.asarray(homography_row, dtype=np.float64).reshape(3, 3))
    points = np.stack([px, py, np.ones_like(px)])
    x, y, w = inverse @ points.reshape(3, -1)
    return (x / w).reshape(px.shape), (y / w).reshape(py.shape)
