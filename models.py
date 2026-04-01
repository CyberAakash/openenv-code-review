"""Pydantic models for the Code Review OpenEnv environment."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class CodeFinding(BaseModel):
    """A single issue found by the reviewing agent."""

    line_number: int = Field(
        ..., description="Line number where the issue occurs (1-indexed)"
    )
    issue_type: str = Field(
        ...,
        description="Category of the issue: 'style', 'bug', or 'security'",
    )
    description: str = Field(..., description="Human-readable explanation of the issue")
    severity: str = Field(
        default="medium",
        description="Severity level: 'low', 'medium', 'high', or 'critical'",
    )


class CodeReviewAction(Action):
    """Action submitted by the agent each step.

    The agent can either:
    - Submit one or more findings (findings list is non-empty, done=False)
    - Signal it is done reviewing (done=True, findings may be empty)
    """

    findings: List[CodeFinding] = Field(
        default_factory=list,
        description="List of code issues found in this step",
    )
    done: bool = Field(
        default=False,
        description="Set to True when the agent is finished reviewing",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class CodeReviewObservation(Observation):
    """Observation returned to the agent after each step / reset."""

    episode_id: str = Field(
        default="",
        description="Episode identifier for stateful multi-step interaction",
    )
    code_snippet: str = Field(
        default="",
        description="The code to review",
    )
    task_id: str = Field(
        default="",
        description="Identifier of the current task (easy / medium / hard)",
    )
    language: str = Field(
        default="python",
        description="Programming language of the snippet",
    )
    step_number: int = Field(
        default=0,
        description="Current step number in this episode",
    )
    max_steps: int = Field(
        default=10,
        description="Maximum number of steps allowed",
    )
    findings_so_far: int = Field(
        default=0,
        description="Total findings submitted so far",
    )
    feedback: str = Field(
        default="",
        description="Per-step feedback from the grader (e.g. 'correct', 'duplicate', 'false positive')",
    )


# ---------------------------------------------------------------------------
# State  (server-side, never sent to the agent)
# ---------------------------------------------------------------------------


class GroundTruthIssue(BaseModel):
    """A single ground-truth issue annotated in the task."""

    line_number: int
    issue_type: str  # style | bug | security
    severity: str  # low | medium | high | critical
    description: str


class EpisodeState(BaseModel):
    """Full server-side state for one episode."""

    episode_id: str = ""
    task_id: str = ""
    code_snippet: str = ""
    language: str = "python"
    ground_truth: List[GroundTruthIssue] = Field(default_factory=list)
    submitted_findings: List[CodeFinding] = Field(default_factory=list)
    matched_indices: List[int] = Field(
        default_factory=list
    )  # indices into ground_truth already matched
    step_number: int = 0
    max_steps: int = 10
    total_reward: float = 0.0
    done: bool = False
    last_feedback: str = ""
