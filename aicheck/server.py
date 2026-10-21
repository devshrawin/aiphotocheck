"""Local web UI: `python -m aicheck serve`. Binds to 127.0.0.1 only; uploads are deleted right after each scan."""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .detectors import build_detectors
from .pipeline import analyze, load_calibration

MAX_BYTES = 50 * 1024 * 1024
WEB = Path(__file__).parent / "web"
SAFE_NAME = re.compile(r"^[\w.\-]+$")


class App:
    def __init__(self, cfg: dict, root: Path, calibration: Path | None, port: int):
        self.cfg, self.root, self.port = cfg, root, port
        self.calib = load_calibration(calibration)
        self.work = Path(tempfile.mkdtemp(prefix="aicheck_"))
        self.lock = threading.Lock()  # one scan at a time: the detectors are CPU/RAM heavy
        self._tiers: dict[str, list] = {}
        self._status: dict | None = None
        self._status_lock = threading.Lock()

    def tier(self, name: str) -> list:  # built once, so the models are loaded once
        if name not in self._tiers:
            self._tiers[name] = build_detectors(self.cfg, self.root, self.cfg["detectors"][name])
        return self._tiers[name]

    def detectors(self, deep: bool) -> list:
        return self.tier("fast") + (self.tier("deep") if deep else [])

    def status(self) -> dict:
        with self._status_lock:
            if self._status is None:
                dets = []
                for tier in ("fast", "deep"):
                    for d in self.tier(tier):
                        ok, why = d.available()
                        dets.append({"name": d.name, "tier": tier, "ready": ok, "why": why})
                self._status = {"detectors": dets, "calibrated": self.calib is not None,
                                "thresholds": {k: round(100 * v) for k, v in self.cfg["thresholds"].items()},
                                "calibration": {k: self.calib[k] for k in ("n_real", "n_ai")} if self.calib else None}
            return self._status

    def scan(self, data: bytes, name: str, deep: bool) -> dict:
        self.status()  # wait for the start-up model load to finish before scanning
        safe = re.sub(r"[^\w.\-]", "_", Path(name).name)[:80] or "upload"
        job = uuid.uuid4().hex[:8]
        path = self.work / "uploads" / f"{job}_{safe}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        try:
            with self.lock:
                rep = analyze(path, self.cfg, self.detectors(deep), self.work / "work", self.calib)
        finally:
            path.unlink(missing_ok=True)  # never keep a citizen's document
        rep["file"] = name
        rep["heatmaps"] = ["/h/" + Path(h).name for h in rep["heatmaps"]]
        rep["deep"] = deep
        return rep


def make_handler(app: App):
    allowed_hosts = {f"127.0.0.1:{app.port}", f"localhost:{app.port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the terminal quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def _host_ok(self) -> bool:  # blocks DNS-rebinding and cross-site requests to the local server
            if self.headers.get("Host", "") not in allowed_hosts:
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin.removeprefix("http://") in allowed_hosts

        def do_GET(self):
            if not self._host_ok():
                return self._json(403, {"error": "forbidden"})
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(200, app.status())
            if path.startswith("/h/"):
                name = path[3:]
                f = app.work / "work" / name
                if SAFE_NAME.match(name) and f.is_file():
                    return self._send(200, f.read_bytes(), "image/png")
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._host_ok():
                return self._json(403, {"error": "forbidden"})
            url = urlparse(self.path)
            if url.path != "/api/scan":
                return self._json(404, {"error": "not found"})
            q = parse_qs(url.query)
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_BYTES:
                return self._json(413, {"error": f"file must be between 1 byte and {MAX_BYTES // 2**20} MB"})
            data = self.rfile.read(n)
            try:
                rep = app.scan(data, q.get("name", ["upload"])[0], q.get("deep", ["0"])[0] == "1")
            except Exception as e:
                return self._json(500, {"error": f"scan failed: {str(e)[:200]}"})
            self._json(200, rep)

    return Handler


def serve(cfg: dict, root: Path, calibration: Path | None, port: int = 8765):
    app = App(cfg, root, calibration, port)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    threading.Thread(target=app.status, daemon=True).start()  # load models in the background so the first page view is quick
    print(f"aicheck is running at http://127.0.0.1:{port}  (Ctrl-C to stop)")
    print("Documents stay on this machine; uploads are deleted after each scan.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(app.work, ignore_errors=True)
