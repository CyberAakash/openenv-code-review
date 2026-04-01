"""Client wrapper for the Code Review environment."""

from openenv.core.env_client import EnvClient

from models import CodeReviewAction, CodeReviewObservation


class CodeReviewEnv(EnvClient[CodeReviewAction, CodeReviewObservation]):
    """Typed client for interacting with a remote Code Review environment."""

    def __init__(self, url: str = "http://localhost:7860") -> None:
        super().__init__(
            url=url,
            action_type=CodeReviewAction,
            observation_type=CodeReviewObservation,
        )
