"""Local web UI:  uv run cavity-ui  [--port 8000] [--no-browser]"""
from __future__ import annotations

import argparse
import threading
import traceback
import webbrowser
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from . import api

WEB = Path(__file__).parent / "web"


def _wrap(fn):
    async def endpoint(request: Request):
        try:
            body = await request.json() if request.method == "POST" else dict(request.query_params)
        except Exception:
            return JSONResponse({"error": "Request body is not valid JSON."}, status_code=400)
        from starlette.concurrency import run_in_threadpool
        try:
            return JSONResponse(await run_in_threadpool(fn, body))
        except api.ApiError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:  # unexpected: show it, keep the server up
            traceback.print_exc()
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)
    return endpoint


def _index(request):
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


routes = [
    Route("/", _index),
    Route("/api/profile", _wrap(api.profile_preview), methods=["POST"]),
    Route("/api/solve", _wrap(api.solve), methods=["POST"]),
    Route("/api/figures", _wrap(lambda b: api.figures(b["solve_id"], float(b["beta"]))),
          methods=["POST"]),
    Route("/api/fields", _wrap(lambda b: api.mode_fields(b["solve_id"], int(b["index"]))),
          methods=["POST"]),
    Route("/api/track", _wrap(api.run_tracking), methods=["POST"]),
    Route("/api/beta", _wrap(lambda b: {"beta": api.beta_of(b["species"], float(b["Ek_MeV"]))}),
          methods=["POST"]),
    Mount("/static", StaticFiles(directory=WEB), name="static"),
]

app = Starlette(routes=routes)


def main():
    import uvicorn

    ap = argparse.ArgumentParser(description="Cavity modes web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://{a.host}:{a.port}/"
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    api.warm_up()
    print(f"Cavity modes UI at {url}  (Ctrl+C to stop)")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
