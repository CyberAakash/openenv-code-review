"""FastAPI application for the Code Review environment."""

import sys
import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app

try:
    from models import CodeReviewAction, CodeReviewObservation
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import CodeReviewAction, CodeReviewObservation

from server.code_review_environment import CodeReviewEnvironment

app = create_app(
    CodeReviewEnvironment,
    CodeReviewAction,
    CodeReviewObservation,
    env_name="code-review",
)

# ── Static files for the web UI ──────────────────────────────────
# Mounted at /ui — safe, does not conflict with any OpenEnv endpoints.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def _root_redirect(request: Request):
        """Redirect bare / to the UI (only when Gradio web interface is not mounted)."""
        return RedirectResponse(url="/ui/index.html")


def main(host: str = "0.0.0.0", port: int = 7860):
    """Run the server with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
