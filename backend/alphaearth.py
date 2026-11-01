"""Read the demo window from the public AlphaEarth GeoZarr and render a PCA RGB."""
from __future__ import annotations

import asyncio
import base64
import io
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy import ndimage
from PIL import Image
from obstore.store import S3Store
from zarr.api.asynchronous import open_group
from zarr.storage import ObjectStore

STORE = ObjectStore(
    S3Store(
        "us-west-2.opendata.source.coop",
        prefix="tge-labs/aef-mosaic/",
        region="us-west-2",
        skip_signature=True,
    ),
    read_only=True,
)
WEST, SOUTH, EAST, NORTH = 2.3815, 48.9335, 2.4285, 48.9465
MIN_COMPONENT_PIXELS = 50
_RAW_CACHE = {}
_DISPLACEMENT_CACHE = {}
_GEO_CACHE = None
_GEO_BY_BOUNDS = {}
CACHE_VERSION = 1
CACHE_PATH = Path(os.environ.get(
    "ALPHAEARTH_CACHE_DIR", Path(__file__).resolve().parent / "cache" / "alphaearth"
)) / f"window-v{CACHE_VERSION}-2017-2024.npz"


def _read_disk_cache():
    """Load the downloaded embedding window and its grid origin, if present."""
    global _GEO_CACHE
    try:
        with np.load(CACHE_PATH, allow_pickle=False) as cached:
            before = cached["before"]
            after = cached["after"]
            transform = tuple(cached["transform"].tolist())
            col0, row0 = (int(value) for value in cached["origin"])
        if before.ndim != 3 or after.shape != before.shape or len(transform) != 6:
            return None
        _GEO_CACHE = (transform, col0, row0)
        return before, after
    except (OSError, ValueError, KeyError):
        # Missing, truncated, or outdated cache files are fetched again.
        return None


def _write_disk_cache(before: np.ndarray, after: np.ndarray) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    transform, col0, row0 = _GEO_CACHE
    # Write then replace so an interrupted process never leaves a partial cache.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=CACHE_PATH.parent, suffix=".npz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            before=before,
            after=after,
            transform=np.asarray(transform, dtype=np.float64),
            origin=np.asarray([col0, row0], dtype=np.int64),
        )
        temporary_path.replace(CACHE_PATH)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


async def load_window(bounds=(WEST, SOUTH, EAST, NORTH)):
    global _GEO_CACHE
    west, south, east, north = bounds
    if not all(np.isfinite(value) for value in bounds) or west >= east or south >= north:
        raise ValueError("The selected zone must have valid west, south, east, and north bounds.")
    if bounds == (WEST, SOUTH, EAST, NORTH):
        cached = _read_disk_cache()
        if cached is not None:
            _GEO_BY_BOUNDS[tuple(bounds)] = _GEO_CACHE
            return cached
    root = await open_group(store=STORE, mode="r")
    embeddings = await root.get("embeddings")
    transform = root.metadata.attributes["spatial:transform"]
    # The mosaic uses an unrotated north-up EPSG:4326 transform.
    pixel_width, _, x_origin, _, pixel_height, y_origin = transform
    col0 = max(0, int(np.floor((west - x_origin) / pixel_width)))
    col1 = int(np.ceil((east - x_origin) / pixel_width))
    row0 = max(0, int(np.floor((north - y_origin) / pixel_height)))
    row1 = int(np.ceil((south - y_origin) / pixel_height))
    _GEO_CACHE = (transform, col0, row0)
    _GEO_BY_BOUNDS[tuple(bounds)] = _GEO_CACHE

    before, after = await asyncio.gather(
        embeddings.getitem((0, slice(None), slice(row0, row1), slice(col0, col1))),
        embeddings.getitem((7, slice(None), slice(row0, row1), slice(col0, col1))),
    )
    before, after = np.asarray(before), np.asarray(after)
    if bounds == (WEST, SOUTH, EAST, NORTH):
        _write_disk_cache(before, after)
    return before, after


def _png_data(image: np.ndarray, mode: str) -> str:
    output = io.BytesIO()
    Image.fromarray(image, mode).save(output, format="PNG", optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def pca_rgb_pair(before_raw: np.ndarray, after_raw: np.ndarray) -> tuple[str, str]:
    """Fit one PCA basis to both years, then render each year in shared RGB space."""
    valid = np.all(before_raw != -128, axis=0) & np.all(after_raw != -128, axis=0)
    before = np.sign(before_raw.astype(np.float32)) * (before_raw.astype(np.float32) / 127.5) ** 2
    after = np.sign(after_raw.astype(np.float32)) * (after_raw.astype(np.float32) / 127.5) ** 2
    h, w = valid.shape
    pixels = np.moveaxis(after, 0, -1).reshape(-1, 64)
    before_pixels = np.moveaxis(before, 0, -1).reshape(-1, 64)
    mask = valid.ravel()
    if mask.sum() < 3:
        raise ValueError("AlphaEarth returned too few valid pixels for PCA.")

    paired_samples = np.concatenate([before_pixels[mask], pixels[mask]], axis=0)
    mean = paired_samples.mean(axis=0)
    centered_pair = paired_samples - mean
    covariance = centered_pair.T @ centered_pair / max(len(paired_samples) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, np.argsort(eigenvalues)[-3:][::-1]]
    # Eigenvector sign is arbitrary; fix it for stable colours between requests.
    for index in range(3):
        pivot = np.argmax(np.abs(components[:, index]))
        if components[pivot, index] < 0:
            components[:, index] *= -1
    projected_before = (before_pixels - mean) @ components
    projected_after = (pixels - mean) @ components
    rgb_before = np.zeros((h * w, 4), dtype=np.uint8)
    rgb_after = np.zeros((h * w, 4), dtype=np.uint8)
    rgb_before[:, 3] = valid.ravel() * 255
    rgb_after[:, 3] = valid.ravel() * 255
    for channel in range(3):
        both = np.concatenate([projected_before[mask, channel], projected_after[mask, channel]])
        lo, hi = np.percentile(both, [2, 98])
        span = max(float(hi - lo), 1e-8)
        rgb_before[:, channel] = (np.clip((projected_before[:, channel] - lo) / span, 0, 1) * 255).astype(np.uint8)
        rgb_after[:, channel] = (np.clip((projected_after[:, channel] - lo) / span, 0, 1) * 255).astype(np.uint8)

    def resize_rgba(rgb: np.ndarray) -> str:
        image = Image.fromarray(rgb.reshape(h, w, 4), "RGBA")
        image.thumbnail((1200, 800), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="PNG", optimize=True)
        return base64.b64encode(out.getvalue()).decode("ascii")

    return resize_rgba(rgb_before), resize_rgba(rgb_after)


def displacement_layers(before_raw: np.ndarray, after_raw: np.ndarray):
    valid = np.all(before_raw != -128, axis=0) & np.all(after_raw != -128, axis=0)
    before = np.sign(before_raw.astype(np.float32)) * (before_raw.astype(np.float32) / 127.5) ** 2
    after = np.sign(after_raw.astype(np.float32)) * (after_raw.astype(np.float32) / 127.5) ** 2
    before /= np.maximum(np.linalg.norm(before, axis=0, keepdims=True), 1e-8)
    after /= np.maximum(np.linalg.norm(after, axis=0, keepdims=True), 1e-8)
    distance = np.linalg.norm(after - before, axis=0)
    distance[~valid] = 0

    # D is total displacement over seven years. The velocity legend reports D/year.
    velocity = distance / 7.0
    velocity_colors = np.array([[30, 67, 84], [49, 126, 126], [150, 179, 105], [247, 202, 95], [232, 116, 82]], dtype=np.float32)
    normalized = np.clip(velocity / (2.0 / 7.0), 0, 1)
    scaled = normalized * (len(velocity_colors) - 1)
    lower = np.floor(scaled).astype(np.int32)
    upper = np.minimum(lower + 1, len(velocity_colors) - 1)
    fraction = (scaled - lower)[..., None]
    rgb = velocity_colors[lower] * (1 - fraction) + velocity_colors[upper] * fraction
    velocity_rgba = np.zeros((*valid.shape, 4), dtype=np.uint8)
    velocity_rgba[..., :3] = rgb.astype(np.uint8)
    velocity_rgba[..., 3] = valid.astype(np.uint8) * 225

    return _png_data(velocity_rgba, "RGBA"), distance, valid


def detection_layer(distance: np.ndarray, valid: np.ndarray, threshold: float):
    threshold_mask = valid & (distance >= threshold)
    labels, component_count = ndimage.label(threshold_mask, structure=np.ones((3, 3), dtype=np.uint8))
    component_sizes = np.bincount(labels.ravel())
    keep_component = component_sizes > MIN_COMPONENT_PIXELS
    keep_component[0] = False
    detected = keep_component[labels]
    retained_components = int(keep_component.sum())
    detections_rgba = np.zeros((*valid.shape, 4), dtype=np.uint8)
    detections_rgba[detected] = [255, 122, 74, 220]
    return _png_data(detections_rgba, "RGBA"), detected, retained_components


async def _cached_arrays(bounds=(WEST, SOUTH, EAST, NORTH)):
    global _RAW_CACHE, _DISPLACEMENT_CACHE, _GEO_CACHE
    key = tuple(round(float(value), 8) for value in bounds)
    if key not in _RAW_CACHE:
        _RAW_CACHE[key] = await load_window(key)
    _GEO_CACHE = _GEO_BY_BOUNDS[key]
    if key not in _DISPLACEMENT_CACHE:
        _DISPLACEMENT_CACHE[key] = displacement_layers(*_RAW_CACHE[key])
    return _RAW_CACHE[key], _DISPLACEMENT_CACHE[key]


def _detection_summary(detected: np.ndarray, valid: np.ndarray, threshold: float, component_count: int) -> dict:
    pixel_width_m = 0.0000898311 * 111_320 * np.cos(np.deg2rad(48.94))
    pixel_height_m = 0.0000898311 * 111_132
    return {
        "detected_pixels": int(detected.sum()),
        "detected_components": component_count,
        "detected_area_ha": float(detected.sum() * pixel_width_m * pixel_height_m / 10_000),
        "valid_pixels": int(valid.sum()),
        "threshold": float(threshold),
    }


async def analyze(bounds=(WEST, SOUTH, EAST, NORTH)) -> dict:
    bounds = tuple(float(value) for value in bounds)
    (before, after), (velocity, distance, valid) = await _cached_arrays(bounds)
    alphaearth_2017, alphaearth_2024 = pca_rgb_pair(before, after)
    detections, detected, component_count = detection_layer(distance, valid, 0.65)
    return {
        "alphaearth_2017": alphaearth_2017,
        "alphaearth_2024": alphaearth_2024,
        "velocity": velocity,
        "detections": detections,
        "width": int(after.shape[2]),
        "height": int(after.shape[1]),
        "bounds": bounds,
        **_detection_summary(detected, valid, 0.65, component_count),
        "mean_displacement": float(distance[valid].mean()),
        "source": "Source Cooperative AlphaEarth GeoZarr",
        "years": [2017, 2024],
    }


async def update_threshold(threshold: float, bounds=(WEST, SOUTH, EAST, NORTH)) -> dict:
    _, (_, distance, valid) = await _cached_arrays(bounds)
    detections, detected, component_count = detection_layer(distance, valid, threshold)
    return {"detections": detections, **_detection_summary(detected, valid, threshold, component_count)}
