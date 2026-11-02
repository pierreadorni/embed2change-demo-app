"""In-memory serial review queue for thresholded AlphaEarth regions."""
from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy import ndimage

from alphaearth import MIN_COMPONENT_PIXELS, _cached_arrays, detection_layer

ARTIFACTS = Path(__file__).resolve().parent / "cache" / "review-jobs"
JOBS: dict[str, dict] = {}
LOCK = threading.RLock()
# Keep jobs ordered, while reviewing several regions from the active job at once.
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-review")
REGION_WORKERS = 4
MAX_VISIBLE_CANDIDATES = 15
MAX_AI_ANALYSES = 15
IMAGE_SIZE = 448
CONTEXT_SCALE = 2.5
GOOGLE_AI_STUDIO = "google_ai_studio"
GOOGLE_MODEL = "gemini-3.5-flash-lite"

VISION_SYSTEM_PROMPT = (
    "You are a remote sensing expert verifying a potential land change. You will receive a pair of aligned aerial "
    "images: left is the earlier year and right is the later year, as labeled on the images. Nearby substitute "
    "years may be used when the requested year has no image coverage. Both images show the same zoomed-out area; "
    "a thick orange outline "
    "with a pale border marks the candidate in each image. Hatched purple regions labeled NO IMAGE DATA are "
    "missing orthophoto coverage and are not land features. First locate that outline, then compare only visible "
    "land inside it between years, using the surrounding area only for context. If the candidate overlaps missing "
    "coverage or cannot be compared reliably, return has_change true, category unclear_occluded, low confidence, "
    "and say coverage is incomplete. Describe only visible evidence.\n\n"
    "Return exactly one JSON object, never a JSON array, never markdown, and never explanatory text. Use exactly "
    "these keys: title, has_change, category, confidence, rationale. The title must be a short, "
    "sentence-case label of 2 to 6 words, such as \"New building complex\" or \"Cleared surface\". The rationale "
    "must be one short sentence of visible evidence, no more than 18 words. Do not mention the outline, outlined "
    "area, candidate, image, before/after, or the years/date range in either title or rationale; the interface "
    "already provides that context.\n\n"
    "Allowed category values when has_change is true: new_construction, demolition_clearing, surface_change, "
    "vegetation_gain, vegetation_loss, water_change, unclear_occluded. Set has_change to false only when no "
    "visible land change occurred; category is only used to describe change and must not be no_visible_change. "
    "confidence is a number from 0 to 1. has_change is a boolean.\n\n"
    "Example output (copy the structure, not the content): "
    "{\"title\":\"New building complex\",\"has_change\":true,\"category\":\"new_construction\","
    "\"confidence\":0.92,\"rationale\":\"New buildings and rectangular paved surfaces are visible.\"}"
)


def _load_env():
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env()


def provider_options():
    return [
        {"id": GOOGLE_AI_STUDIO, "label": "Google AI Studio · Gemini 2.5 Flash-Lite",
         "available": bool(os.getenv("GOOGLE_AI_STUDIO_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
         "detail": "Lowest-cost vision model · hosted"},
    ]


def _regions(mask: np.ndarray, distance: np.ndarray, geo):
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    candidates = [i for i in range(1, count + 1) if sizes[i] > MIN_COMPONENT_PIXELS]
    candidates.sort(key=lambda i: int(sizes[i]), reverse=True)
    transform, col0, row0 = geo
    pw, _, x0, _, ph, y0 = transform
    to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    found = []
    for rank, label in enumerate(candidates[:MAX_VISIBLE_CANDIDATES], 1):
        rows, cols = np.where(labels == label)
        top, bottom = int(rows.min()), int(rows.max()) + 1
        left, right = int(cols.min()), int(cols.max()) + 1
        lon0, lon1 = x0 + (col0 + left) * pw, x0 + (col0 + right) * pw
        lat1, lat0 = y0 + (row0 + top) * ph, y0 + (row0 + bottom) * ph
        xs, ys = to_l93.transform([lon0, lon1, lon1, lon0], [lat0, lat0, lat1, lat1])
        bounds = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))
        local = labels[top:bottom, left:right] == label
        affine = Affine(pw, 0, x0 + (col0 + left) * pw, 0, ph, y0 + (row0 + top) * ph)
        polygons = [shape for shape, value in shapes(local.astype(np.uint8), mask=local, transform=affine) if value == 1]
        geometry = {"type": "Polygon", "coordinates": polygons[0]["coordinates"]} if len(polygons) == 1 else {
            "type": "MultiPolygon", "coordinates": [polygon["coordinates"] for polygon in polygons]
        }
        edge = local & ~ndimage.binary_erosion(local, structure=np.ones((3, 3), dtype=bool), border_value=0)
        edge_rows, edge_cols = np.where(edge)
        found.append({
            "id": f"C-{rank:03d}", "rank": rank, "geometry": geometry,
            "centroid": [(lon0 + lon1) / 2, (lat0 + lat1) / 2], "bounds": bounds,
            "pixel_count": int(sizes[label]), "score": float(distance[labels == label].mean()),
            "analysis_eligible": rank <= MAX_AI_ANALYSES,
            "area_ha": float(sizes[label] * 0.0000898311 * 111320 * np.cos(np.deg2rad(48.94)) * 0.0000898311 * 111132 / 10000),
            "outline": list(zip(
                (x0 + (edge_cols + left + col0 + 0.5) * pw).astype(float),
                (y0 + (edge_rows + top + row0 + 0.5) * ph).astype(float),
            )),
        })
    return found, max(0, len(candidates) - len(found))


def _download(year: int, bounds: tuple[float, ...]) -> Image.Image:
    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": f"ORTHOIMAGERY.ORTHOPHOTOS{year}", "STYLES": "", "CRS": "EPSG:2154",
        "BBOX": ",".join(map(str, bounds)), "WIDTH": IMAGE_SIZE, "HEIGHT": IMAGE_SIZE, "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
    }
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get("https://data.geopf.fr/wms-r", params=params, timeout=(10, 45))
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGBA")
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError(f"IGN returned an unexpected {year} image size.")
            # Keep missing WMS coverage distinct from real pale ground/sky.
            # Draw a conspicuous pattern only where the service returned alpha=0.
            alpha = image.getchannel("A")
            missing = alpha.point(lambda value: 255 if value < 16 else 0)
            coverage = 1 - (missing.histogram()[255] / (IMAGE_SIZE * IMAGE_SIZE))
            if missing.getbbox():
                marker = Image.new("RGBA", image.size, (87, 52, 105, 255))
                draw = ImageDraw.Draw(marker)
                for offset in range(-IMAGE_SIZE, IMAGE_SIZE * 2, 18):
                    draw.line((offset, 0, offset - IMAGE_SIZE, IMAGE_SIZE), fill=(221, 180, 245, 255), width=3)
                ImageDraw.Draw(marker).text((12, IMAGE_SIZE - 30), "NO IMAGE DATA", fill="white", stroke_width=2, stroke_fill=(58, 35, 70, 255))
                image = Image.composite(marker, image, missing)
            image = image.convert("RGB")
            return image, coverage
        except (requests.RequestException, OSError, RuntimeError) as exc:
            last_error = exc
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code < 500 and exc.response.status_code != 429:
                break
            if attempt < 2:
                time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"Could not fetch IGN {year} orthophoto after 3 attempts: {last_error}") from last_error


def _download_best_coverage(year: int, bounds: tuple[float, ...]):
    """Use the requested year where possible, falling back to the nearest better-covered year."""
    candidates = [year]
    for offset in range(1, 4):
        candidates.extend((year - offset, year + offset))
    best = None
    errors = []
    for alternate_year in candidates:
        try:
            image, coverage = _download(alternate_year, bounds)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        if best is None or coverage > best[1]:
            best = (alternate_year, coverage, image)
        if coverage >= 0.999:
            break
    if best is None:
        raise RuntimeError(f"Could not fetch IGN imagery near {year}: {errors[-1] if errors else 'no imagery'}")
    return best


def _context_bounds(bounds: tuple[float, ...]) -> tuple[float, ...]:
    x0, y0, x1, y1 = bounds
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    half_size = max(x1 - x0, y1 - y0) * CONTEXT_SCALE / 2
    return (center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size)


def _outline_images(candidate: dict, before: Image.Image, after: Image.Image, bounds: tuple[float, ...]):
    x0, y0, x1, y1 = bounds
    polygons = candidate["geometry"]["coordinates"]
    if candidate["geometry"]["type"] == "Polygon":
        polygons = [polygons]
    to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    for image in (before, after):
        painter = ImageDraw.Draw(image)
        for polygon in polygons:
            for ring in polygon:
                xs, ys = to_l93.transform([point[0] for point in ring], [point[1] for point in ring])
                points = [((x - x0) / (x1 - x0) * IMAGE_SIZE, (y1 - y) / (y1 - y0) * IMAGE_SIZE) for x, y in zip(xs, ys)]
                if len(points) > 2:
                    painter.line(points, fill="#fff1c7", width=10, joint="curve")
                    painter.line(points, fill="#ff633f", width=6, joint="curve")
    return before, after


def _composite(before: Image.Image, after: Image.Image, before_year: int, after_year: int) -> Image.Image:
    result = Image.new("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE), "#f5f0df")
    result.paste(before, (0, 0)); result.paste(after, (IMAGE_SIZE, 0))
    ImageDraw.Draw(result).text((12, 12), f"{before_year} · BEFORE", fill="white", stroke_width=2, stroke_fill="#17231d")
    ImageDraw.Draw(result).text((IMAGE_SIZE + 12, 12), f"{after_year} · AFTER", fill="white", stroke_width=2, stroke_fill="#17231d")
    return result


# Local Gemma inference is temporarily disabled. Uncomment this block and the
# corresponding dispatch in _run/create_job to restore local inference.
# _MODEL = None
# _PROCESSOR = None
# _MODEL_ERROR = None
# _MODEL_LOCK = threading.Lock()
#
# def _ensure_local_model():
#     global _MODEL, _PROCESSOR, _MODEL_ERROR
#     with _MODEL_LOCK:
#         if _MODEL_ERROR:
#             raise RuntimeError(_MODEL_ERROR)
#         if _MODEL is None:
#             try:
#                 import torch
#                 from transformers import AutoModelForMultimodalLM, AutoProcessor
#                 if not torch.cuda.is_available():
#                     raise RuntimeError("Gemma 4 inference requires the CUDA-enabled environment used by the notebook.")
#                 model_id = "google/gemma-4-E2B-it"
#                 _PROCESSOR = AutoProcessor.from_pretrained(model_id)
#                 _PROCESSOR.image_processor.max_soft_tokens = 560
#                 _MODEL = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cuda")
#                 _MODEL.eval()
#             except ImportError as exc:
#                 _MODEL_ERROR = "Gemma inference dependencies are missing from the Python environment running the API."
#                 raise RuntimeError(_MODEL_ERROR) from exc
#             except Exception as exc:
#                 _MODEL_ERROR = str(exc)
#                 raise RuntimeError(f"Could not load local Gemma 4: {_MODEL_ERROR}") from exc
#
# def _ask_local_vlm(image: Image.Image) -> dict:
#     _ensure_local_model()
#     messages = [
#         {"role": "system", "content": VISION_SYSTEM_PROMPT},
#         {"role": "user", "content": [{"type": "image", "image": image}]},
#     ]
#     inputs = _PROCESSOR.apply_chat_template(messages, tokenize=True, return_dict=True,
#         return_tensors="pt", add_generation_prompt=True, enable_thinking=False).to(_MODEL.device)
#     input_len = inputs["input_ids"].shape[1]
#     outputs = _MODEL.generate(**inputs, max_new_tokens=256, do_sample=False)
#     text = _PROCESSOR.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
#     start, end = text.find("{"), text.rfind("}")
#     if start < 0 or end < start:
#         raise ValueError("Gemma returned text without a JSON review.")
#     result = json.loads(text[start:end + 1])
#     if isinstance(result, list) and len(result) == 1:
#         result = result[0]
#     if not isinstance(result, dict):
#         raise ValueError("Gemma returned a JSON array or scalar; expected one JSON object.")
#     if not isinstance(result.get("title"), str) or not result["title"].strip():
#         raise ValueError("Gemma returned a review without a short title.")
#     if not isinstance(result.get("has_change"), bool) or not isinstance(result.get("rationale"), str):
#         raise ValueError("Gemma returned an incomplete JSON review.")
#     result.setdefault("category", "unclear_occluded")
#     result.setdefault("confidence", None)
#     return result


def _ask_google_vlm(image: Image.Image) -> dict:
    api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Google AI Studio is not configured. Add GOOGLE_API_KEY to the project .env file.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Google AI Studio support is missing. Install google-genai from backend/requirements.txt.") from exc

    client = genai.Client(api_key=api_key)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    response = client.models.generate_content(
        model=GOOGLE_MODEL,
        contents=[types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg"), VISION_SYSTEM_PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
    )
    result = json.loads(response.text)
    if isinstance(result, list) and len(result) == 1:
        result = result[0]
    if not isinstance(result, dict):
        raise ValueError("Gemini returned a JSON array or scalar; expected one JSON object.")
    if not isinstance(result.get("title"), str) or not result["title"].strip():
        raise ValueError("Gemini returned a review without a short title.")
    if not isinstance(result.get("has_change"), bool) or not isinstance(result.get("rationale"), str):
        raise ValueError("Gemini returned an incomplete JSON review.")
    result.setdefault("category", "unclear_occluded")
    result.setdefault("confidence", None)
    return result


def _review_candidate(job_id: str, job: dict, candidate: dict):
    try:
        bounds = _context_bounds(candidate["bounds"])
        before_year, before_coverage, before = _download_best_coverage(2017, bounds)
        after_year, after_coverage, after = _download_best_coverage(2024, bounds)
        before, after = _outline_images(candidate, before, after, bounds)
        composite = _composite(before, after, before_year, after_year)
        path = ARTIFACTS / job_id / f"{candidate['id']}"
        path.parent.mkdir(parents=True, exist_ok=True)
        before.save(path.parent / f"{candidate['id']}-before.jpg", format="JPEG", quality=88, optimize=True)
        after.save(path.parent / f"{candidate['id']}-after.jpg", format="JPEG", quality=88, optimize=True)
        composite.save(path.with_suffix(".jpg"), format="JPEG", quality=88, optimize=True)
        candidate["composite_url"] = f"/api/jobs/{job_id}/images/{candidate['id']}"
        candidate["before_url"] = f"/api/jobs/{job_id}/images/{candidate['id']}-before"
        candidate["after_url"] = f"/api/jobs/{job_id}/images/{candidate['id']}-after"
        candidate["before_year"] = before_year
        candidate["after_year"] = after_year
        candidate["imagery_coverage"] = {"before": round(before_coverage, 3), "after": round(after_coverage, 3)}
        with LOCK:
            candidate["status"] = "analyzing"
        result = _ask_google_vlm(composite)
        if result["has_change"] is False:
            result["category"] = "no_visible_change"
        result["provider"] = job["provider"]
        with LOCK:
            candidate.update(result)
            candidate["status"] = "done"
    except Exception as exc:
        with LOCK:
            candidate["status"] = "error"
            candidate["error"] = str(exc)
    finally:
        candidate.pop("outline", None)
        candidate.pop("bounds", None)
        with LOCK:
            job["completed"] += 1
            job["active_ids"].discard(candidate["id"])
            job["active_id"] = next(iter(job["active_ids"]), None)


def _run(job_id: str):
    with LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["phase"] = "downloading_imagery_and_reviewing"
        job["active_ids"] = set()
    candidates = [candidate for candidate in job["candidates"] if candidate.get("analysis_eligible", True)]
    for candidate in job["candidates"]:
        if not candidate.get("analysis_eligible", True):
            candidate["status"] = "limited"
    with ThreadPoolExecutor(max_workers=REGION_WORKERS, thread_name_prefix="region-review") as workers:
        for start in range(0, len(candidates), REGION_WORKERS):
            if job.get("cancelled"):
                break
            batch = candidates[start:start + REGION_WORKERS]
            with LOCK:
                for candidate in batch:
                    candidate["status"] = "downloading"
                    job["active_ids"].add(candidate["id"])
                job["active_id"] = next(iter(job["active_ids"]), None)
            pending = [workers.submit(_review_candidate, job_id, job, candidate) for candidate in batch]
            for completed_future in as_completed(pending):
                completed_future.result()
    with LOCK:
        job["status"] = "cancelled" if job.get("cancelled") else "complete"
        job["phase"] = "complete"
        job["active_ids"] = set()
        job["active_id"] = None


def create_job(threshold: float, provider: str = GOOGLE_AI_STUDIO, bounds=(2.3815, 48.9335, 2.4285, 48.9465)) -> dict:
    # Reactivation: restore LOCAL_GEMMA to provider_options and this validation,
    # then restore the local model warm-up and dispatch in _run.
    if provider != GOOGLE_AI_STUDIO:
        raise ValueError("Only the hosted Google AI Studio provider is currently available.")
    if provider == GOOGLE_AI_STUDIO and not (os.getenv("GOOGLE_AI_STUDIO_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise ValueError("Google AI Studio is not configured. Add GOOGLE_API_KEY to the project .env file.")
    _, (__, distance, valid) = __import__("asyncio").run(_cached_arrays(bounds))
    _, detected, _ = detection_layer(distance, valid, threshold)
    geo = __import__("alphaearth")._GEO_CACHE[tuple(round(float(value), 8) for value in bounds)]
    candidates, omitted = _regions(detected, distance, geo)
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "status": "queued", "threshold": threshold, "provider": provider,
           "total": min(len(candidates), MAX_AI_ANALYSES), "completed": 0, "active_id": None, "active_ids": [],
           "omitted": omitted, "candidates": candidates}
    with LOCK:
        JOBS[job_id] = job
    EXECUTOR.submit(_run, job_id)
    return public_job(job)


def public_job(job: dict) -> dict:
    with LOCK:
        return {
            **{key: value for key, value in job.items() if key not in {"cancelled", "candidates", "active_ids"}},
            "active_ids": sorted(job.get("active_ids", ())),
            "candidates": [{key: value for key, value in candidate.items() if key not in {"outline", "bounds"}}
                           for candidate in job["candidates"]],
        }


def get_job(job_id: str) -> dict | None:
    with LOCK:
        job = JOBS.get(job_id)
        return public_job(job) if job else None


def get_image(job_id: str, candidate_id: str) -> bytes | None:
    if not job_id.isalnum() or not candidate_id.replace("-", "").isalnum():
        return None
    path = ARTIFACTS / job_id / f"{candidate_id}.jpg"
    return path.read_bytes() if path.is_file() else None
