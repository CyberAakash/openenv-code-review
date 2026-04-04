"""Cross-episode memory and bandit strategy selection for the Code Review agent.

Implements three learning mechanisms:
  A) Per-task memory: remembers what worked/failed on each task across episodes
  B) Feedback parser: extracts structured results from step feedback strings
  E) Bandit strategy selection: epsilon-greedy multi-armed bandit over strategy choices

Memory is persisted to a JSON file so learning survives across inference runs.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEMORY_FILE = os.environ.get("MEMORY_FILE", "outputs/memory.json")

# Bandit parameters
EPSILON = 0.15  # Exploration rate (15% random, 85% exploit best)
UCB_C = 1.4  # UCB exploration constant

# Available strategies
STRATEGIES = [
    "ast_first",  # AST scan → review → hint → refine → fix
    "review_only",  # Review → done (minimal steps, max efficiency bonus)
    "review_hint",  # Review → hint → refine → done
    "full_pipeline",  # AST scan → review → hint → refine → fix → done
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TaskMemory:
    """Per-task learning history."""

    task_id: str
    attempts: int = 0
    best_score: float = -1.0
    last_score: float = 0.0
    avg_score: float = 0.0

    # What the agent got right/wrong
    matched_descriptions: List[str] = field(default_factory=list)
    missed_descriptions: List[str] = field(default_factory=list)
    false_positive_descriptions: List[str] = field(default_factory=list)

    # Successful fixes
    successful_fixes: List[Dict[str, str]] = field(default_factory=list)

    # Strategy that produced best score
    best_strategy: str = ""

    # Raw feedback from last episode (for prompt injection)
    last_feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "attempts": self.attempts,
            "best_score": self.best_score,
            "last_score": self.last_score,
            "avg_score": self.avg_score,
            "matched_descriptions": self.matched_descriptions[-10:],  # Keep last 10
            "missed_descriptions": self.missed_descriptions[-10:],
            "false_positive_descriptions": self.false_positive_descriptions[-10:],
            "successful_fixes": self.successful_fixes[-5:],
            "best_strategy": self.best_strategy,
            "last_feedback": self.last_feedback[:500],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskMemory":
        return cls(
            task_id=d.get("task_id", ""),
            attempts=d.get("attempts", 0),
            best_score=d.get("best_score", -1.0),
            last_score=d.get("last_score", 0.0),
            avg_score=d.get("avg_score", 0.0),
            matched_descriptions=d.get("matched_descriptions", []),
            missed_descriptions=d.get("missed_descriptions", []),
            false_positive_descriptions=d.get("false_positive_descriptions", []),
            successful_fixes=d.get("successful_fixes", []),
            best_strategy=d.get("best_strategy", ""),
            last_feedback=d.get("last_feedback", ""),
        )


@dataclass
class StrategyStats:
    """Reward statistics for a strategy (used by bandit)."""

    name: str
    total_reward: float = 0.0
    count: int = 0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.count if self.count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_reward": self.total_reward,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyStats":
        return cls(
            name=d.get("name", ""),
            total_reward=d.get("total_reward", 0.0),
            count=d.get("count", 0),
        )


# ---------------------------------------------------------------------------
# Feedback parser (Level B: within-episode adaptive)
# ---------------------------------------------------------------------------


@dataclass
class ParsedFeedback:
    """Structured result from parsing environment feedback strings."""

    correct_lines: List[int] = field(default_factory=list)
    false_positive_lines: List[int] = field(default_factory=list)
    correct_descriptions: List[str] = field(default_factory=list)
    false_positive_descriptions: List[str] = field(default_factory=list)
    duplicate_lines: List[int] = field(default_factory=list)
    total_correct: int = 0
    total_false_positives: int = 0
    total_duplicates: int = 0


def parse_step_feedback(feedback: str) -> ParsedFeedback:
    """Parse the environment's step feedback into structured data.

    Feedback format examples:
      "Line 12 (security): CORRECT (+0.300, base=0.300, bonus=+0.000)"
      "Line 3 (style): FALSE POSITIVE (-0.030)"
      "Line 5 (bug): duplicate — ignored"
    """
    result = ParsedFeedback()

    if not feedback:
        return result

    # Split on " | " (environment joins multiple feedbacks)
    parts = feedback.split(" | ")

    for part in parts:
        part = part.strip()

        # Extract line number
        line_num = _extract_line_number(part)

        if "CORRECT" in part:
            result.total_correct += 1
            if line_num:
                result.correct_lines.append(line_num)
            # Extract the type info for memory
            desc = _extract_type_from_feedback(part)
            if desc:
                result.correct_descriptions.append(desc)

        elif "FALSE POSITIVE" in part:
            result.total_false_positives += 1
            if line_num:
                result.false_positive_lines.append(line_num)
            desc = _extract_type_from_feedback(part)
            if desc:
                result.false_positive_descriptions.append(desc)

        elif "duplicate" in part.lower():
            result.total_duplicates += 1
            if line_num:
                result.duplicate_lines.append(line_num)

    return result


def parse_fix_feedback(fix_feedback: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Parse fix_feedback from the environment into a structured list.

    Returns list of dicts: {line, check_id, is_valid, score, feedback}
    """
    if not fix_feedback or "fixes" not in fix_feedback:
        return []
    return fix_feedback.get("fixes", [])


def _extract_line_number(text: str) -> Optional[int]:
    """Extract line number from feedback like 'Line 12 (security): ...'"""
    import re

    match = re.search(r"Line\s+(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _extract_type_from_feedback(text: str) -> str:
    """Extract 'Line N (type)' description from feedback."""
    import re

    match = re.match(r"(Line\s+\d+\s*\(\w+\))", text)
    if match:
        return match.group(1)
    return ""


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Manages cross-episode memory and bandit strategy selection."""

    def __init__(self, memory_file: str = MEMORY_FILE):
        self.memory_file = memory_file
        self.task_memories: Dict[str, TaskMemory] = {}
        self.strategy_stats: Dict[str, StrategyStats] = {
            name: StrategyStats(name=name) for name in STRATEGIES
        }
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self):
        """Load memory from JSON file."""
        if not os.path.exists(self.memory_file):
            return

        try:
            with open(self.memory_file, "r") as f:
                data = json.load(f)

            # Load task memories
            for task_id, task_data in data.get("tasks", {}).items():
                self.task_memories[task_id] = TaskMemory.from_dict(task_data)

            # Load strategy stats
            for strat_data in data.get("strategies", []):
                name = strat_data.get("name", "")
                if name in self.strategy_stats:
                    self.strategy_stats[name] = StrategyStats.from_dict(strat_data)

            print(
                f"  [Memory] Loaded: {len(self.task_memories)} tasks, "
                f"{sum(s.count for s in self.strategy_stats.values())} strategy trials"
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [Memory] Failed to load {self.memory_file}: {e}")

    def save(self):
        """Save memory to JSON file."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.memory_file) or ".", exist_ok=True)

        data = {
            "tasks": {tid: mem.to_dict() for tid, mem in self.task_memories.items()},
            "strategies": [stats.to_dict() for stats in self.strategy_stats.values()],
        }

        with open(self.memory_file, "w") as f:
            json.dump(data, f, indent=2)

    # ── Task memory ───────────────────────────────────────────────────

    def get_task_memory(self, task_id: str) -> Optional[TaskMemory]:
        """Get memory for a specific task (None if never seen)."""
        return self.task_memories.get(task_id)

    def update_task_memory(
        self,
        task_id: str,
        score: float,
        strategy: str,
        matched: List[str],
        missed: List[str],
        false_positives: List[str],
        fixes: List[Dict[str, str]],
        feedback: str,
    ):
        """Update memory after completing an episode."""
        mem = self.task_memories.get(task_id, TaskMemory(task_id=task_id))

        mem.attempts += 1
        mem.last_score = score

        # Running average
        if mem.attempts == 1:
            mem.avg_score = score
        else:
            mem.avg_score = mem.avg_score + (score - mem.avg_score) / mem.attempts

        if score > mem.best_score:
            mem.best_score = score
            mem.best_strategy = strategy

        # Accumulate unique descriptions (dedup)
        for desc in matched:
            if desc and desc not in mem.matched_descriptions:
                mem.matched_descriptions.append(desc)
        for desc in missed:
            if desc and desc not in mem.missed_descriptions:
                mem.missed_descriptions.append(desc)
        for desc in false_positives:
            if desc and desc not in mem.false_positive_descriptions:
                mem.false_positive_descriptions.append(desc)

        # Remove from "missed" anything that was matched this time
        mem.missed_descriptions = [
            d for d in mem.missed_descriptions if d not in matched
        ]

        # Successful fixes
        for fix in fixes:
            if fix not in mem.successful_fixes:
                mem.successful_fixes.append(fix)

        mem.last_feedback = feedback
        self.task_memories[task_id] = mem

    # ── Bandit strategy selection ─────────────────────────────────────

    def select_strategy(self, difficulty: str) -> str:
        """Select a strategy using epsilon-greedy with UCB1 tiebreaking.

        For easy tasks, prefer simpler strategies.
        For hard tasks, consider all strategies.
        """
        # Filter strategies by difficulty
        if difficulty == "easy":
            candidates = ["review_only", "review_hint", "ast_first"]
        elif difficulty == "medium":
            candidates = ["ast_first", "review_hint", "full_pipeline"]
        else:
            candidates = STRATEGIES  # All strategies for hard

        total_pulls = sum(self.strategy_stats[s].count for s in candidates)

        # If any candidate has never been tried, pick it (exploration)
        untried = [s for s in candidates if self.strategy_stats[s].count == 0]
        if untried:
            choice = random.choice(untried)
            print(f"  [Bandit] Exploring untried strategy: {choice}")
            return choice

        # Epsilon-greedy: explore with probability epsilon
        if random.random() < EPSILON:
            choice = random.choice(candidates)
            print(f"  [Bandit] Exploring randomly: {choice}")
            return choice

        # Exploit: pick best by UCB1
        best_strategy = ""
        best_ucb = -float("inf")

        for name in candidates:
            stats = self.strategy_stats[name]
            avg = stats.avg_reward
            exploration_bonus = UCB_C * math.sqrt(
                math.log(total_pulls + 1) / (stats.count + 1)
            )
            ucb_value = avg + exploration_bonus

            if ucb_value > best_ucb:
                best_ucb = ucb_value
                best_strategy = name

        print(
            f"  [Bandit] Exploiting best: {best_strategy} "
            f"(avg={self.strategy_stats[best_strategy].avg_reward:.3f}, "
            f"n={self.strategy_stats[best_strategy].count})"
        )
        return best_strategy

    def update_strategy(self, strategy: str, reward: float):
        """Update strategy stats after an episode."""
        if strategy in self.strategy_stats:
            stats = self.strategy_stats[strategy]
            stats.total_reward += reward
            stats.count += 1

    # ── Prompt enhancement ────────────────────────────────────────────

    def build_memory_prompt(self, task_id: str) -> str:
        """Build a prompt section with learning from previous attempts.

        Returns empty string if no memory exists for this task.
        """
        mem = self.get_task_memory(task_id)
        if not mem or mem.attempts == 0:
            return ""

        parts = [
            f"\n--- LEARNING FROM {mem.attempts} PREVIOUS ATTEMPT(S) ---",
            f"Your best score on this task: {mem.best_score:.3f} "
            f"(last: {mem.last_score:.3f}, avg: {mem.avg_score:.3f})",
        ]

        if mem.matched_descriptions:
            parts.append(
                f"\nIssues you correctly identified before:\n"
                + "\n".join(f"  + {d}" for d in mem.matched_descriptions[-8:])
            )

        if mem.missed_descriptions:
            parts.append(
                f"\nIssues you MISSED before (focus on finding these!):\n"
                + "\n".join(f"  ! {d}" for d in mem.missed_descriptions[-8:])
            )

        if mem.false_positive_descriptions:
            parts.append(
                f"\nFALSE POSITIVES you reported before (do NOT repeat these!):\n"
                + "\n".join(f"  x {d}" for d in mem.false_positive_descriptions[-8:])
            )

        if mem.successful_fixes:
            parts.append(
                f"\nFixes that worked before (reuse these patterns):\n"
                + "\n".join(
                    f"  * {f.get('check_id', '?')}: {f.get('description', '?')}"
                    for f in mem.successful_fixes[-5:]
                )
            )

        if mem.best_strategy:
            parts.append(f"\nBest strategy so far: {mem.best_strategy}")

        parts.append("--- END PREVIOUS LEARNING ---\n")
        return "\n".join(parts)

    def build_adaptive_prompt(
        self,
        parsed_feedback: ParsedFeedback,
        previous_findings: List[Dict[str, Any]],
    ) -> str:
        """Build a prompt section for within-episode adaptive refinement.

        Uses parsed step feedback to tell the LLM what worked and what didn't.
        """
        if (
            not parsed_feedback.correct_lines
            and not parsed_feedback.false_positive_lines
        ):
            return ""

        parts = ["\n--- STEP FEEDBACK ANALYSIS ---"]

        if parsed_feedback.correct_lines:
            correct_str = ", ".join(f"L{n}" for n in parsed_feedback.correct_lines)
            parts.append(f"Lines confirmed CORRECT: {correct_str}")

        if parsed_feedback.false_positive_lines:
            fp_str = ", ".join(f"L{n}" for n in parsed_feedback.false_positive_lines)
            parts.append(f"Lines that were FALSE POSITIVES (do NOT resubmit): {fp_str}")
            # Include what the FP findings were so the LLM avoids similar ones
            for line_num in parsed_feedback.false_positive_lines:
                for f in previous_findings:
                    if f.get("line_number") == line_num:
                        parts.append(
                            f"  -> L{line_num}: your '{f.get('issue_type')}' "
                            f'finding was WRONG: "{f.get("description", "")}"'
                        )

        if parsed_feedback.duplicate_lines:
            dup_str = ", ".join(f"L{n}" for n in parsed_feedback.duplicate_lines)
            parts.append(f"Duplicate submissions (already counted): {dup_str}")

        parts.append(
            f"Score: {parsed_feedback.total_correct} correct, "
            f"{parsed_feedback.total_false_positives} false positives"
        )
        parts.append("Focus on finding NEW issues on lines you haven't reported yet.")
        parts.append("--- END STEP FEEDBACK ---\n")
        return "\n".join(parts)

    def get_strategy_summary(self) -> str:
        """Return a human-readable summary of strategy performance."""
        lines = ["Strategy Performance:"]
        for name, stats in sorted(
            self.strategy_stats.items(),
            key=lambda x: x[1].avg_reward,
            reverse=True,
        ):
            if stats.count > 0:
                lines.append(f"  {name}: avg={stats.avg_reward:.3f}, n={stats.count}")
            else:
                lines.append(f"  {name}: (untried)")
        return "\n".join(lines)
