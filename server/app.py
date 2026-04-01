"""FastAPI application for the Code Review environment."""

import sys
import os

# Ensure the project root is on the path so models.py can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app

from models import CodeReviewAction, CodeReviewObservation
from server.code_review_environment import CodeReviewEnvironment

app = create_app(
    CodeReviewEnvironment,
    CodeReviewAction,
    CodeReviewObservation,
    env_name="code-review",
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """Run the server with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
