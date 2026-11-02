"""Tiny local API used by the app demo."""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from alphaearth import analyze, update_threshold
from review_queue import create_job, get_image, get_job, provider_options

ANALYSIS_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parts = self.path.strip("/").split("/")
        if self.path == "/api/config":
            payload = json.dumps({"providers": provider_options()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return
        if len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "images":
            image = get_image(parts[2], parts[4])
            if image is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(image)
            return
        if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            job = get_job(parts[2])
            if job is None:
                self.send_error(404)
                return
            payload = json.dumps(job).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path not in {"/api/alphaearth", "/api/alphaearth/chunk", "/api/detection-threshold", "/api/vlm-analysis"}:
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 16384:
                self.send_error(413)
                return
            body = json.loads(self.rfile.read(size) or b"{}")
            if self.path in {"/api/alphaearth", "/api/alphaearth/chunk"}:
                bounds = body.get("bounds")
                if bounds is None:
                    bounds = [[2.3815, 48.9335], [2.4285, 48.9465]]
                if (not isinstance(bounds, list) or len(bounds) != 2
                        or any(not isinstance(point, list) or len(point) != 2 for point in bounds)):
                    self.send_error(400, "bounds must contain southwest and northeast coordinates")
                    return
                result = asyncio.run(analyze(
                    (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]),
                    body.get("pca_basis"),
                ))
            elif self.path == "/api/detection-threshold":
                threshold = body.get("threshold")
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.1 <= threshold <= 1.2:
                    self.send_error(400, "threshold must be between 0.1 and 1.2")
                    return
                bounds = body.get("bounds", [[2.3815, 48.9335], [2.4285, 48.9465]])
                with ANALYSIS_LOCK:
                    result = asyncio.run(update_threshold(float(threshold), (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1])))
            else:
                threshold = body.get("threshold")
                if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.1 <= threshold <= 1.2:
                    self.send_error(400, "threshold must be between 0.1 and 1.2")
                    return
                provider = body.get("provider", "google_ai_studio")
                bounds = body.get("bounds", [[2.3815, 48.9335], [2.4285, 48.9465]])
                with ANALYSIS_LOCK:
                    result = create_job(float(threshold), provider, (bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]))
            payload = json.dumps(result).encode()
            self.send_response(200)
        except Exception as exc:  # surfaced to the UI as a useful retryable error
            payload = json.dumps({"error": str(exc)}).encode()
            self.send_response(502)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print("[embed2change-api] " + fmt % args)


if __name__ == "__main__":
    print("AlphaEarth demo API listening on http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
