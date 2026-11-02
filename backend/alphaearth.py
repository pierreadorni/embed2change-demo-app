"""Read the demo window from the public AlphaEarth GeoZarr and render a PCA RGB."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
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
_GEO_CACHE: dict[tuple[float, ...], tuple[tuple[float, ...], int, int]] = {}
CACHE_VERSION = 3
CACHE_ROOT = Path(os.environ.get(
    "ALPHAEARTH_CACHE_DIR", Path(__file__).resolve().parent / "cache" / "alphaearth"
)) / f"tiles-v{CACHE_VERSION}"


def _cache_dir(bounds):
    key = ",".join(str(int(value)) for value in bounds)
    return CACHE_ROOT / hashlib.sha256(key.encode("ascii")).hexdigest()


def _write_disk_cache(bounds, before: np.ndarray, after: np.ndarray, geo) -> None:
    """Atomically persist one tile without making a compressed in-memory copy."""
    directory = _cache_dir(bounds)
    directory.parent.mkdir(parents=True, exist_ok=True)
    transform, col0, row0, col1, row1 = geo
    temporary_dir = Path(tempfile.mkdtemp(dir=directory.parent, prefix="tile-"))
    try:
        np.save(temporary_dir / "2017.npy", before, allow_pickle=False)
        np.save(temporary_dir / "2024.npy", after, allow_pickle=False)
        (temporary_dir / "metadata.json").write_text(json.dumps({
            "transform": list(transform), "origin": [int(col0), int(row0)],
            "extent": [int(col1), int(row1)],
        }))
        if directory.exists():
            shutil.rmtree(directory)
        temporary_dir.replace(directory)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)


async def load_window(bounds=(WEST, SOUTH, EAST, NORTH)):
    west, south, east, north = bounds
    if not all(np.isfinite(value) for value in bounds) or west >= east or south >= north:
        raise ValueError("The selected zone must have valid west, south, east, and north bounds.")
    root = await open_group(store=STORE, mode="r")
    embeddings = await root.get("embeddings")
    transform = root.metadata.attributes["spatial:transform"]
    # The mosaic uses an unrotated north-up EPSG:4326 transform.
    pixel_width, _, x_origin, _, pixel_height, y_origin = transform
    col0 = max(0, int(np.floor((west - x_origin) / pixel_width)))
    col1 = int(np.ceil((east - x_origin) / pixel_width))
    row0 = max(0, int(np.floor((north - y_origin) / pixel_height)))
    row1 = int(np.ceil((south - y_origin) / pixel_height))
    geo = (transform, col0, row0)
    request_key = tuple(round(float(value), 8) for value in bounds)
    _GEO_CACHE[request_key] = geo
    shape = (64, row1 - row0, col1 - col0)
    before = np.empty(shape, dtype=np.int8)
    after = np.empty(shape, dtype=np.int8)
    covered = np.zeros((row1 - row0, col1 - col0), dtype=bool)
    entries = []
    if CACHE_ROOT.exists():
        for directory in CACHE_ROOT.iterdir():
            item = _read_disk_cache_entry(directory)
            if item is not None:
                entries.append(item)
    for cached_before, cached_after, cached_geo in entries:
        _, cached_col0, cached_row0, cached_col1, cached_row1 = cached_geo
        left, top = max(col0, cached_col0), max(row0, cached_row0)
        right, bottom = min(col1, cached_col1), min(row1, cached_row1)
        if left >= right or top >= bottom:
            continue
        dst = (slice(top-row0, bottom-row0), slice(left-col0, right-col0))
        src = (slice(top-cached_row0, bottom-cached_row0), slice(left-cached_col0, right-cached_col0))
        before[:, dst[0], dst[1]] = cached_before[:, src[0], src[1]]
        after[:, dst[0], dst[1]] = cached_after[:, src[0], src[1]]
        covered[dst] = True
    # Fetch only uncovered horizontal runs, coalescing each run vertically while
    # its left/right edges remain identical.
    pending = {}
    for local_row in range(covered.shape[0] + 1):
        gaps = []
        if local_row < covered.shape[0]:
            row = ~covered[local_row]
            starts = np.flatnonzero(row & ~np.r_[False, row[:-1]])
            ends = np.flatnonzero(row & ~np.r_[row[1:], False]) + 1
            gaps = list(zip(starts.tolist(), ends.tolist()))
        current = set(gaps)
        for run, start_row in list(pending.items()):
            if run not in current:
                end_row = local_row
                await _fetch_and_cache(embeddings, before, after, run, start_row, end_row, col0, row0, transform)
                del pending[run]
        for run in gaps:
            pending.setdefault(run, local_row)
    return before, after


def _read_disk_cache_entry(directory):
    """Read a cache entry and its source pixel extent."""
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        before = np.load(directory / "2017.npy", mmap_mode="r", allow_pickle=False)
        after = np.load(directory / "2024.npy", mmap_mode="r", allow_pickle=False)
        transform = tuple(metadata["transform"])
        col0, row0 = metadata["origin"]
        col1, row1 = metadata["extent"]
        if before.ndim != 3 or after.shape != before.shape or before.shape[1:] != (row1-row0, col1-col0):
            return None
        return before, after, (transform, col0, row0, col1, row1)
    except (OSError, ValueError, KeyError, TypeError):
        return None


async def _fetch_and_cache(embeddings, before, after, run, start_row, end_row, col0, row0, transform):
    left, right = run
    global_row0, global_row1 = row0 + start_row, row0 + end_row
    global_col0, global_col1 = col0 + left, col0 + right
    fetched_before, fetched_after = await asyncio.gather(
        embeddings.getitem((0, slice(None), slice(global_row0, global_row1), slice(global_col0, global_col1))),
        embeddings.getitem((7, slice(None), slice(global_row0, global_row1), slice(global_col0, global_col1))),
    )
    fetched_before, fetched_after = np.asarray(fetched_before), np.asarray(fetched_after)
    before[:, start_row:end_row, left:right] = fetched_before
    after[:, start_row:end_row, left:right] = fetched_after
    _write_disk_cache((global_col0, global_row0, global_col1, global_row1), fetched_before, fetched_after,
                      (transform, global_col0, global_row0, global_col1, global_row1))


def _png_data(image: np.ndarray, mode: str) -> str:
    output = io.BytesIO()
    Image.fromarray(image, mode).save(output, format="PNG", optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _pixel_block(raw: np.ndarray, start_row: int, end_row: int):
    block = raw[:, start_row:end_row, :]
    valid = np.all(block != -128, axis=0)
    pixels = np.moveaxis(block, 0, -1).reshape(-1, 64).astype(np.float32)
    pixels /= 127.5
    pixels *= np.abs(pixels)
    return pixels, valid.ravel()


def pca_rgb_pair(before_raw: np.ndarray, after_raw: np.ndarray, basis=None):
    """Render both years with a shared, reusable PCA basis."""
    h, w = before_raw.shape[1:]
    row_block = 32
    valid = np.zeros((h, w), dtype=bool)
    if basis is None:
        sample_sum = np.zeros(64, dtype=np.float64)
        product_sum = np.zeros((64, 64), dtype=np.float64)
        valid_count = 0
        for start in range(0, h, row_block):
            end = min(start + row_block, h)
            before, before_valid = _pixel_block(before_raw, start, end)
            after, after_valid = _pixel_block(after_raw, start, end)
            mask = before_valid & after_valid
            valid[start:end] = mask.reshape(end - start, w)
            if not mask.any():
                continue
            before = before[mask]
            after = after[mask]
            sample_sum += before.sum(axis=0, dtype=np.float64) + after.sum(axis=0, dtype=np.float64)
            product_sum += before.T @ before + after.T @ after
            valid_count += int(mask.sum())
        sample_count = valid_count * 2
        if sample_count < 3:
            raise ValueError("AlphaEarth returned too few valid pixels for PCA.")
        mean = sample_sum / sample_count
        covariance = (product_sum - sample_count * np.outer(mean, mean)) / (sample_count - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        components = eigenvectors[:, np.argsort(eigenvalues)[-3:][::-1]]
        # Eigenvector sign is arbitrary; fix it for stable colours between requests.
        for index in range(3):
            pivot = np.argmax(np.abs(components[:, index]))
            if components[pivot, index] < 0:
                components[:, index] *= -1
        lows = highs = None
    else:
        mean = np.asarray(basis["mean"], dtype=np.float64)
        components = np.asarray(basis["components"], dtype=np.float64)
        lows = np.asarray(basis["lows"], dtype=np.float64)
        highs = np.asarray(basis["highs"], dtype=np.float64)
        if mean.shape != (64,) or components.shape != (64, 3) or lows.shape != (3,) or highs.shape != (3,):
            raise ValueError("The shared PCA basis is invalid.")
    projected_before = np.zeros((h, w, 3), dtype=np.float32)
    projected_after = np.zeros((h, w, 3), dtype=np.float32)
    for start in range(0, h, row_block):
        end = min(start + row_block, h)
        before, before_valid = _pixel_block(before_raw, start, end)
        after, after_valid = _pixel_block(after_raw, start, end)
        valid[start:end] = (before_valid & after_valid).reshape(end - start, w)
        projected_before[start:end] = ((before - mean) @ components).reshape(end - start, w, 3)
        projected_after[start:end] = ((after - mean) @ components).reshape(end - start, w, 3)
    rgb_before = np.zeros((h, w, 4), dtype=np.uint8)
    rgb_after = np.zeros((h, w, 4), dtype=np.uint8)
    rgb_before[..., 3] = valid * 255
    rgb_after[..., 3] = valid * 255
    if basis is None:
        lows = np.zeros(3, dtype=np.float64)
        highs = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        both = np.concatenate([projected_before[..., channel][valid], projected_after[..., channel][valid]])
        if basis is None:
            lows[channel], highs[channel] = np.percentile(both, [2, 98])
        lo, hi = lows[channel], highs[channel]
        span = max(float(hi - lo), 1e-8)
        rgb_before[..., channel] = (np.clip((projected_before[..., channel] - lo) / span, 0, 1) * 255).astype(np.uint8)
        rgb_after[..., channel] = (np.clip((projected_after[..., channel] - lo) / span, 0, 1) * 255).astype(np.uint8)

    def resize_rgba(rgb: np.ndarray) -> str:
        image = Image.fromarray(rgb, "RGBA")
        image.thumbnail((1200, 800), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="PNG", optimize=True)
        return base64.b64encode(out.getvalue()).decode("ascii")

    shared_basis = {
        "mean": mean.tolist(), "components": components.tolist(),
        "lows": lows.tolist(), "highs": highs.tolist(),
    }
    return resize_rgba(rgb_before), resize_rgba(rgb_after), shared_basis


def displacement_layers(before_raw: np.ndarray, after_raw: np.ndarray):
    h, w = before_raw.shape[1:]
    row_block = 32
    valid = np.zeros((h, w), dtype=bool)
    distance = np.zeros((h, w), dtype=np.float32)
    velocity_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    velocity_colors = np.array([[30, 67, 84], [49, 126, 126], [150, 179, 105], [247, 202, 95], [232, 116, 82]], dtype=np.float32)
    for start in range(0, h, row_block):
        end = min(start + row_block, h)
        before, before_valid = _pixel_block(before_raw, start, end)
        after, after_valid = _pixel_block(after_raw, start, end)
        mask = before_valid & after_valid
        valid[start:end] = mask.reshape(end - start, w)
        before /= np.maximum(np.linalg.norm(before, axis=1, keepdims=True), 1e-8)
        after /= np.maximum(np.linalg.norm(after, axis=1, keepdims=True), 1e-8)
        delta = after - before
        block_distance = np.linalg.norm(delta, axis=1).reshape(end - start, w)
        block_distance[~mask.reshape(end - start, w)] = 0
        distance[start:end] = block_distance
        scaled = np.clip(block_distance / 2.0, 0, 1) * (len(velocity_colors) - 1)
        lower = np.floor(scaled).astype(np.int8)
        upper = np.minimum(lower + 1, len(velocity_colors) - 1)
        fraction = (scaled - lower)[..., None]
        velocity_rgba[start:end, :, :3] = (velocity_colors[lower] * (1 - fraction) + velocity_colors[upper] * fraction).astype(np.uint8)
        velocity_rgba[start:end, :, 3] = mask.reshape(end - start, w).astype(np.uint8) * 225

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
    key = tuple(round(float(value), 8) for value in bounds)
    raw = await load_window(key)
    # Derived rasters are request scoped. Keeping them in a process-wide cache
    # caused memory to grow with every selected chunk and zone.
    derived = displacement_layers(*raw)
    return raw, derived


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


async def analyze(bounds=(WEST, SOUTH, EAST, NORTH), pca_basis=None) -> dict:
    bounds = tuple(float(value) for value in bounds)
    (before, after), (velocity, distance, valid) = await _cached_arrays(bounds)
    alphaearth_2017, alphaearth_2024, pca_basis = pca_rgb_pair(before, after, pca_basis)
    detections, detected, component_count = detection_layer(distance, valid, 0.45)
    return {
        "alphaearth_2017": alphaearth_2017,
        "alphaearth_2024": alphaearth_2024,
        "pca_basis": pca_basis,
        "velocity": velocity,
        "detections": detections,
        "width": int(after.shape[2]),
        "height": int(after.shape[1]),
        "bounds": bounds,
        **_detection_summary(detected, valid, 0.45, component_count),
        "mean_displacement": float(distance[valid].mean()),
        "source": "Source Cooperative AlphaEarth GeoZarr",
        "years": [2017, 2024],
    }


async def update_threshold(threshold: float, bounds=(WEST, SOUTH, EAST, NORTH)) -> dict:
    _, (_, distance, valid) = await _cached_arrays(bounds)
    detections, detected, component_count = detection_layer(distance, valid, threshold)
    return {"detections": detections, **_detection_summary(detected, valid, threshold, component_count)}
