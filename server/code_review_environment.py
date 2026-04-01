"""Code Review Environment — core RL environment implementation.

Uses a module-level state store so that state persists across HTTP requests
(the OpenEnv HTTP server creates a new Environment instance per request).

Multi-step mechanics:
- "review": Submit findings for grading (default)
- "request_hint": Get a hint about unfound issue categories (costs -0.05, max 3 per episode)
- "request_analysis": Run simulated static analysis revealing obvious issues (costs -0.10, once per episode)

Grading enhancements:
- Description keyword matching: bonus for finding descriptions that mention key terms
- Severity accuracy: bonus when agent's severity matches ground-truth severity
- Step efficiency factor: bonus for finishing with fewer steps
"""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

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

SEVERITY_ORDER = ["low", "medium", "high", "critical"]

LINE_TOLERANCE = 2  # +-2 lines for matching

# Costs for information-gathering actions
HINT_COST = 0.05
ANALYSIS_COST = 0.10
MAX_HINTS = 3


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------


def _extract_keywords(description: str) -> set[str]:
    """Extract meaningful keywords from a description for fuzzy matching."""
    # Lowercase and split; filter short words
    words = set()
    for word in description.lower().replace("'", "").replace('"', "").split():
        # Strip punctuation
        cleaned = word.strip(".,;:!?()-_/")
        if len(cleaned) >= 4:  # Skip short words
            words.add(cleaned)
    return words


def _description_similarity(finding_desc: str, gt_desc: str) -> float:
    """Compute keyword overlap between finding and ground truth descriptions.

    Returns a score between 0.0 and 1.0.
    """
    finding_kw = _extract_keywords(finding_desc)
    gt_kw = _extract_keywords(gt_desc)
    if not gt_kw:
        return 0.0
    overlap = finding_kw & gt_kw
    return len(overlap) / len(gt_kw)


def _severity_bonus(finding_severity: str, gt_severity: str) -> float:
    """Return a bonus/penalty based on how close the severity match is.

    Exact match: +0.05
    Off by one level: 0.0
    Off by two+: -0.02
    """
    f_idx = (
        SEVERITY_ORDER.index(finding_severity)
        if finding_severity in SEVERITY_ORDER
        else 1
    )
    g_idx = SEVERITY_ORDER.index(gt_severity) if gt_severity in SEVERITY_ORDER else 1
    diff = abs(f_idx - g_idx)
    if diff == 0:
        return 0.05
    elif diff == 1:
        return 0.0
    else:
        return -0.02


def _match_finding(
    finding: CodeFinding,
    ground_truth: list[GroundTruthIssue],
    already_matched: list[int],
) -> tuple[bool, int, str, float]:
    """Try to match a finding against unmatched ground truth issues.

    Returns (is_match, matched_gt_index, feedback_string, extra_bonus).
    The extra_bonus includes description similarity and severity accuracy bonuses.
    """
    for idx, gt in enumerate(ground_truth):
        if idx in already_matched:
            continue
        line_ok = abs(finding.line_number - gt.line_number) <= LINE_TOLERANCE
        type_ok = finding.issue_type.lower().strip() == gt.issue_type.lower().strip()
        if line_ok and type_ok:
            # Compute extras
            desc_sim = _description_similarity(finding.description, gt.description)
            desc_bonus = 0.05 * desc_sim  # up to +0.05 for perfect description match
            sev_bonus = _severity_bonus(
                finding.severity.lower().strip(),
                gt.severity.lower().strip(),
            )
            total_extra = desc_bonus + sev_bonus
            return True, idx, "correct", total_extra
    return False, -1, "false_positive", 0.0


def _is_duplicate(finding: CodeFinding, previous: list[CodeFinding]) -> bool:
    """Check if this finding duplicates a previous one (exact same line and same type)."""
    for prev in previous:
        if (
            finding.line_number == prev.line_number
            and finding.issue_type.lower().strip() == prev.issue_type.lower().strip()
        ):
            return True
    return False


def _generate_hint(state: EpisodeState) -> str:
    """Generate a hint about unfound issue categories.

    Each hint reveals which *types* of issues remain unfound,
    without giving away exact locations.
    """
    unfound_types: Counter[str] = Counter()
    unfound_severities: Counter[str] = Counter()
    for idx, gt in enumerate(state.ground_truth):
        if idx not in state.matched_indices:
            unfound_types[gt.issue_type] += 1
            unfound_severities[gt.severity] += 1

    if not unfound_types:
        return "All issues have been found! You can submit 'done'."

    total_remaining = sum(unfound_types.values())
    parts = []
    for itype, count in unfound_types.most_common():
        parts.append(f"{count} {itype} issue(s)")

    severity_parts = []
    for sev in SEVERITY_ORDER:
        if sev in unfound_severities:
            severity_parts.append(f"{unfound_severities[sev]} {sev}")

    hint = (
        f"There are {total_remaining} unfound issue(s) remaining: "
        f"{', '.join(parts)}. "
        f"Severity breakdown: {', '.join(severity_parts)}."
    )

    # On second/third hint, reveal approximate line ranges
    if state.hints_used >= 1:
        unfound_lines = sorted(
            gt.line_number
            for idx, gt in enumerate(state.ground_truth)
            if idx not in state.matched_indices
        )
        if unfound_lines:
            # Group into ranges (within 5 lines)
            ranges = []
            start = unfound_lines[0]
            end = start
            for ln in unfound_lines[1:]:
                if ln - end <= 5:
                    end = ln
                else:
                    ranges.append((start, end))
                    start = ln
                    end = ln
            ranges.append((start, end))
            range_strs = [
                f"lines {s}-{e}" if s != e else f"line {s}" for s, e in ranges
            ]
            hint += f" Check around: {', '.join(range_strs)}."

    return hint


def _generate_analysis(state: EpisodeState) -> str:
    """Generate a static analysis output that reveals the most obvious issues.

    Reveals up to 2 of the easiest-to-detect issues (lowest severity first).
    Gives issue type and approximate line, but NOT the exact description.
    """
    unfound = [
        (idx, gt)
        for idx, gt in enumerate(state.ground_truth)
        if idx not in state.matched_indices
    ]
    if not unfound:
        return "Static analysis: No remaining issues detected."

    # Sort by severity (easiest first)
    sev_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    unfound.sort(key=lambda x: sev_rank.get(x[1].severity, 1))

    # Reveal up to 2
    revealed = unfound[:2]
    lines = []
    for _, gt in revealed:
        lines.append(
            f"  - Line {gt.line_number}: potential {gt.issue_type} issue "
            f"(severity: {gt.severity})"
        )

    return "Static analysis results:\n" + "\n".join(lines)


def _step_efficiency_bonus(state: EpisodeState) -> float:
    """Bonus for completing the review efficiently (fewer steps = better).

    Max bonus: +0.15 (finished in 1 step)
    Min bonus: 0.0 (used all steps)
    """
    if state.max_steps <= 1:
        return 0.15
    # Linear decay
    fraction_used = state.step_number / state.max_steps
    return max(0.0, 0.15 * (1.0 - fraction_used))


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

    def _available_actions(self, state: EpisodeState) -> list[str]:
        """Return the list of action types still available."""
        actions = ["review"]
        if state.hints_used < MAX_HINTS:
            actions.append("request_hint")
        if not state.analysis_used:
            actions.append("request_analysis")
        return actions

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
            hints_used=0,
            analysis_used=False,
            hint_categories_revealed=[],
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
            feedback="Episode started. Review the code and submit findings. You can also use 'request_hint' or 'request_analysis' actions to gather information (at a reward cost).",
            hint="",
            analysis_result="",
            available_actions=self._available_actions(state),
        )

    def step(self, action: CodeReviewAction) -> CodeReviewObservation:
        """Process one agent action."""
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
        action_type = (action.action_type or "review").lower().strip()

        # ── Handle request_hint ──────────────────────────────────────────
        if action_type == "request_hint":
            return self._handle_hint(state, action)

        # ── Handle request_analysis ──────────────────────────────────────
        if action_type == "request_analysis":
            return self._handle_analysis(state, action)

        # ── Handle review (default) ──────────────────────────────────────
        return self._handle_review(state, action)

    @property
    def state(self) -> EpisodeState:
        """Return the full server-side state (not sent to the agent in production)."""
        if self._episode_id and self._episode_id in _STATE_STORE:
            return self._get_state()
        return EpisodeState()

    # -- action handlers ----------------------------------------------------

    def _handle_hint(
        self, state: EpisodeState, action: CodeReviewAction
    ) -> CodeReviewObservation:
        """Handle a request_hint action."""
        if state.hints_used >= MAX_HINTS:
            feedback = f"Hint limit reached ({MAX_HINTS}/{MAX_HINTS}). Use 'review' to submit findings."
            self._set_state(state)
            return self._make_observation(state, reward=0.0, feedback=feedback)

        state.hints_used += 1
        hint_text = _generate_hint(state)
        cost = -HINT_COST
        state.total_reward += cost

        feedback = f"Hint {state.hints_used}/{MAX_HINTS} provided (cost: {cost:+.3f})."

        # Check for done signal
        if action.done:
            return self._finalize_episode(state, cost, feedback, hint=hint_text)

        # Auto-terminate if max steps reached
        if state.step_number >= state.max_steps:
            return self._auto_terminate(state, cost, feedback, hint=hint_text)

        state.last_feedback = feedback
        self._set_state(state)
        return self._make_observation(
            state, reward=cost, feedback=feedback, hint=hint_text
        )

    def _handle_analysis(
        self, state: EpisodeState, action: CodeReviewAction
    ) -> CodeReviewObservation:
        """Handle a request_analysis action."""
        if state.analysis_used:
            feedback = "Static analysis already used this episode. Use 'review' to submit findings."
            self._set_state(state)
            return self._make_observation(state, reward=0.0, feedback=feedback)

        state.analysis_used = True
        analysis_text = _generate_analysis(state)
        cost = -ANALYSIS_COST
        state.total_reward += cost

        feedback = f"Static analysis complete (cost: {cost:+.3f})."

        # Check for done signal
        if action.done:
            return self._finalize_episode(state, cost, feedback, analysis=analysis_text)

        # Auto-terminate if max steps reached
        if state.step_number >= state.max_steps:
            return self._auto_terminate(state, cost, feedback, analysis=analysis_text)

        state.last_feedback = feedback
        self._set_state(state)
        return self._make_observation(
            state, reward=cost, feedback=feedback, analysis=analysis_text
        )

    def _handle_review(
        self, state: EpisodeState, action: CodeReviewAction
    ) -> CodeReviewObservation:
        """Handle a review action with findings submission."""
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

                is_match, gt_idx, fb, extra_bonus = _match_finding(
                    finding, state.ground_truth, state.matched_indices
                )

                if is_match:
                    state.matched_indices.append(gt_idx)
                    gt_issue = state.ground_truth[gt_idx]
                    weight = SEVERITY_WEIGHTS.get(gt_issue.severity, 1.0)
                    base_reward = 0.15 * weight
                    reward_delta = base_reward + extra_bonus
                    step_reward += reward_delta
                    feedbacks.append(
                        f"Line {finding.line_number} ({finding.issue_type}): CORRECT "
                        f"(+{reward_delta:.3f}, base={base_reward:.3f}, bonus={extra_bonus:+.3f})"
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
            return self._finalize_episode(
                state, step_reward, " | ".join(feedbacks) if feedbacks else ""
            )

        # ── Not done yet — check for no findings ─────────────────────────
        if not action.findings:
            feedbacks.append("No findings submitted this step.")

        # Auto-terminate if max steps reached
        if state.step_number >= state.max_steps:
            return self._auto_terminate(state, step_reward, " | ".join(feedbacks))

        state.last_feedback = " | ".join(feedbacks)
        self._set_state(state)
        return self._make_observation(
            state, reward=step_reward, feedback=state.last_feedback
        )

    # -- episode finalization -----------------------------------------------

    def _finalize_episode(
        self,
        state: EpisodeState,
        step_reward: float,
        feedback_prefix: str,
        hint: str = "",
        analysis: str = "",
    ) -> CodeReviewObservation:
        """Compute final bonuses and terminate the episode."""
        state.done = True

        matched_count = len(state.matched_indices)
        total_issues = len(state.ground_truth)
        false_positives = len(state.submitted_findings) - matched_count

        # Recall bonus
        recall = matched_count / total_issues if total_issues > 0 else 0.0
        recall_bonus = 0.3 * recall

        # False-positive penalty
        fp_penalty = 0.05 * false_positives

        # Step efficiency bonus
        efficiency_bonus = _step_efficiency_bonus(state)

        done_reward = recall_bonus - fp_penalty + efficiency_bonus
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

        summary = (
            f"Review complete. Matched {matched_count}/{total_issues} issues. "
            f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}. "
            f"Efficiency bonus={efficiency_bonus:+.3f}. "
            f"Total reward={state.total_reward:.3f}"
        )

        final_feedback = (
            f"{feedback_prefix} | {summary}" if feedback_prefix else summary
        )
        state.last_feedback = final_feedback
        self._set_state(state)
        return self._make_observation(
            state,
            reward=step_reward,
            feedback=final_feedback,
            hint=hint,
            analysis=analysis,
        )

    def _auto_terminate(
        self,
        state: EpisodeState,
        step_reward: float,
        feedback_prefix: str,
        hint: str = "",
        analysis: str = "",
    ) -> CodeReviewObservation:
        """Auto-terminate when max steps is reached."""
        extra_feedback = "Max steps reached — episode auto-terminated."
        combined_prefix = (
            f"{feedback_prefix} | {extra_feedback}"
            if feedback_prefix
            else extra_feedback
        )
        return self._finalize_episode(
            state, step_reward, combined_prefix, hint=hint, analysis=analysis
        )

    # -- internal -----------------------------------------------------------

    def _make_observation(
        self,
        state: EpisodeState,
        reward: float,
        feedback: str,
        hint: str = "",
        analysis: str = "",
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
            hint=hint,
            analysis_result=analysis,
            available_actions=self._available_actions(state),
        )
