"""Code Review Environment — core RL environment implementation.

Uses a module-level state store so that state persists across HTTP requests
(the OpenEnv HTTP server creates a new Environment instance per request).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment

import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    CodeFinding,
    CodeReviewAction,
    CodeReviewObservation,
    EpisodeState,
    GroundTruthIssue,
)
from server.tasks import get_task, list_task_ids, TASKS_BY_DIFFICULTY

# ---------------------------------------------------------------------------
# Module-level state store  (survives across Environment instances)
# ---------------------------------------------------------------------------
_STATE_STORE: Dict[str, EpisodeState] = {}
_LATEST_EPISODE_ID: str = ""  # Track latest episode for stateless HTTP

# ---------------------------------------------------------------------------
# Severity weights for reward calculation
# ---------------------------------------------------------------------------
SEVERITY_WEIGHTS: Dict[str, float] = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.5,
    "critical": 2.0,
}

LINE_TOLERANCE = 2  # +-2 lines for matching


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------


def _match_finding(
    finding: CodeFinding,
    ground_truth: list[GroundTruthIssue],
    already_matched: list[int],
) -> tuple[bool, int, str]:
    """Try to match a finding against unmatched ground truth issues.

    Returns (is_match, matched_gt_index, feedback_string).
    """
    for idx, gt in enumerate(ground_truth):
        if idx in already_matched:
            continue
        line_ok = abs(finding.line_number - gt.line_number) <= LINE_TOLERANCE
        type_ok = finding.issue_type.lower().strip() == gt.issue_type.lower().strip()
        if line_ok and type_ok:
            return True, idx, "correct"
    return False, -1, "false_positive"


def _is_duplicate(finding: CodeFinding, previous: list[CodeFinding]) -> bool:
    """Check if this finding duplicates a previous one (exact same line and same type)."""
    for prev in previous:
        if (
            finding.line_number == prev.line_number
            and finding.issue_type.lower().strip() == prev.issue_type.lower().strip()
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class CodeReviewEnvironment(
    Environment[CodeReviewAction, CodeReviewObservation, EpisodeState]
):
    """OpenEnv Environment for code review tasks."""

    def __init__(self) -> None:
        super().__init__()
        self._episode_id: str = ""

    # -- helpers to access the state store ----------------------------------

    def _get_state(self) -> EpisodeState:
        return _STATE_STORE[self._episode_id]

    def _set_state(self, state: EpisodeState) -> None:
        _STATE_STORE[self._episode_id] = state

    # -- Environment interface ----------------------------------------------

    def reset(self, **kwargs: Any) -> CodeReviewObservation:
        """Start a new episode.

        Accepted kwargs:
            task_id (str): which task to load (e.g. "easy_1", "medium_2", "hard_3").
                           Defaults to "easy_1".
        """
        global _LATEST_EPISODE_ID

        task_id: str = kwargs.get("task_id", "easy_1")
        task = get_task(task_id)

        episode_id = str(uuid.uuid4())
        self._episode_id = episode_id
        _LATEST_EPISODE_ID = episode_id

        ground_truth = [GroundTruthIssue(**issue) for issue in task["ground_truth"]]

        state = EpisodeState(
            episode_id=episode_id,
            task_id=task_id,
            code_snippet=task["code_snippet"],
            language=task.get("language", "python"),
            ground_truth=ground_truth,
            submitted_findings=[],
            matched_indices=[],
            step_number=0,
            max_steps=10,
            total_reward=0.0,
            done=False,
            last_feedback="",
        )
        self._set_state(state)

        return CodeReviewObservation(
            reward=0.0,
            done=False,
            metadata={"episode_id": episode_id, "available_tasks": list_task_ids()},
            episode_id=episode_id,
            code_snippet=state.code_snippet,
            task_id=state.task_id,
            language=state.language,
            step_number=0,
            max_steps=state.max_steps,
            findings_so_far=0,
            feedback="Episode started. Review the code and submit findings.",
        )

    def step(self, action: CodeReviewAction) -> CodeReviewObservation:
        """Process one agent action (findings or done signal)."""
        global _LATEST_EPISODE_ID

        # Recover episode_id: try action metadata, then instance, then latest
        episode_id = action.metadata.get("episode_id", "")
        if not episode_id or episode_id not in _STATE_STORE:
            episode_id = self._episode_id
        if not episode_id or episode_id not in _STATE_STORE:
            episode_id = _LATEST_EPISODE_ID
        if not episode_id or episode_id not in _STATE_STORE:
            # No valid state at all — force a reset
            return self.reset()

        self._episode_id = episode_id

        state = self._get_state()

        if state.done:
            return self._make_observation(
                state, reward=0.0, feedback="Episode already finished."
            )

        state.step_number += 1
        step_reward = 0.0
        feedbacks: list[str] = []

        # ── Process submitted findings FIRST (even if done=True) ──────────
        if action.findings:
            for finding in action.findings:
                # Duplicate check
                if _is_duplicate(finding, state.submitted_findings):
                    feedbacks.append(
                        f"Line {finding.line_number} ({finding.issue_type}): duplicate — ignored"
                    )
                    state.submitted_findings.append(finding)
                    continue

                is_match, gt_idx, fb = _match_finding(
                    finding, state.ground_truth, state.matched_indices
                )

                if is_match:
                    state.matched_indices.append(gt_idx)
                    gt_issue = state.ground_truth[gt_idx]
                    weight = SEVERITY_WEIGHTS.get(gt_issue.severity, 1.0)
                    reward_delta = 0.15 * weight
                    step_reward += reward_delta
                    feedbacks.append(
                        f"Line {finding.line_number} ({finding.issue_type}): CORRECT (+{reward_delta:.3f})"
                    )
                else:
                    step_reward -= 0.1
                    feedbacks.append(
                        f"Line {finding.line_number} ({finding.issue_type}): FALSE POSITIVE (-0.100)"
                    )

                state.submitted_findings.append(finding)

        state.total_reward += step_reward

        # ── Agent signals it is done ──────────────────────────────────────
        if action.done:
            state.done = True

            matched_count = len(state.matched_indices)
            total_issues = len(state.ground_truth)
            false_positives = len(state.submitted_findings) - matched_count

            # Recall bonus
            recall = matched_count / total_issues if total_issues > 0 else 0.0
            recall_bonus = 0.3 * recall

            # False-positive penalty
            fp_penalty = 0.05 * false_positives

            done_reward = recall_bonus - fp_penalty
            step_reward += done_reward
            state.total_reward += done_reward

            # Compute final F1
            precision = (
                matched_count / len(state.submitted_findings)
                if state.submitted_findings
                else 0.0
            )
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )

            feedbacks.append(
                f"Review complete. Matched {matched_count}/{total_issues} issues. "
                f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}. "
                f"Total reward={state.total_reward:.3f}"
            )
            state.last_feedback = " | ".join(feedbacks)
            self._set_state(state)
            return self._make_observation(
                state, reward=step_reward, feedback=state.last_feedback
            )

        # ── Not done yet — check for no findings ─────────────────────────
        if not action.findings:
            feedbacks.append("No findings submitted this step.")

        # Auto-terminate if max steps reached
        if state.step_number >= state.max_steps:
            state.done = True
            feedbacks.append("Max steps reached — episode auto-terminated.")
            # Apply end-of-episode bonuses as if agent said done
            matched_count = len(state.matched_indices)
            total_issues = len(state.ground_truth)
            recall = matched_count / total_issues if total_issues > 0 else 0.0
            false_positives = len(state.submitted_findings) - matched_count
            bonus = 0.3 * recall - 0.05 * false_positives
            step_reward += bonus
            state.total_reward += bonus

        state.last_feedback = " | ".join(feedbacks)
        self._set_state(state)
        return self._make_observation(
            state, reward=step_reward, feedback=state.last_feedback
        )

    @property
    def state(self) -> EpisodeState:
        """Return the full server-side state (not sent to the agent in production)."""
        if self._episode_id and self._episode_id in _STATE_STORE:
            return self._get_state()
        return EpisodeState()

    # -- internal -----------------------------------------------------------

    def _make_observation(
        self, state: EpisodeState, reward: float, feedback: str
    ) -> CodeReviewObservation:
        return CodeReviewObservation(
            reward=reward,
            done=state.done,
            metadata={"episode_id": state.episode_id},
            episode_id=state.episode_id,
            code_snippet=state.code_snippet if not state.done else "",
            task_id=state.task_id,
            language=state.language,
            step_number=state.step_number,
            max_steps=state.max_steps,
            findings_so_far=len(state.submitted_findings),
            feedback=feedback,
        )
