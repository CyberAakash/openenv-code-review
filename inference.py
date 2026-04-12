#!/usr/bin/env python3
"""Learning inference script for the Code Review environment.

Uses cross-episode memory, within-episode adaptive refinement, and
bandit-based strategy selection to improve over repeated runs.

Learning mechanisms:
  A) Cross-episode memory: remembers what worked/failed per task, injects
     into LLM prompts so it avoids past mistakes and focuses on missed issues.
  B) Within-episode adaptive: parses step feedback mid-episode, drops false
     positives, and refines the next submission based on what worked.
  E) Bandit strategy selection: epsilon-greedy UCB1 over 4 strategy variants,
     learns which approach works best for each difficulty level.

Strategy variants:
  - "review_only":   Review → Done  (fast, max efficiency bonus)
  - "review_hint":   Review → Hint → Refine → Done
  - "ast_first":     AST → Review → Hint → Refine → Done
  - "full_pipeline": AST → Review → Hint → Refine → Fix → Done

Environment variables (required by hackathon spec):
    API_BASE_URL  — The API endpoint for the LLM (default: https://api.openai.com/v1)
    MODEL_NAME    — The model identifier (default: gpt-4o-mini)
    HF_TOKEN      — Your API key for the LLM provider

Optional:
    ENV_URL       — Base URL of the environment server
    TASKS         — Comma-separated task IDs to run
    MEMORY_FILE   — Path to memory JSON file (default: outputs/memory.json)
    NUM_ROUNDS    — Number of rounds per task for learning (default: 1)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from memory import (
    MemoryManager,
    ParsedFeedback,
    parse_step_feedback,
    parse_fix_feedback,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hackathon-required variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Environment server URL
ENV_URL = os.environ.get("ENV_URL", "https://CyberAakash-code-review.hf.space")

# Tasks to run
DEFAULT_TASKS = ",".join(
    [
        "easy_1",
        "easy_2",
        "easy_3",
        "easy_4",
        "easy_5",
        "medium_1",
        "medium_2",
        "medium_3",
        "medium_4",
        "medium_5",
        "hard_1",
        "hard_2",
        "hard_3",
        "hard_4",
        "hard_5",
    ]
)
TASKS = os.environ.get("TASKS", DEFAULT_TASKS).split(",")

# Learning config
NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "1"))


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------


def env_reset(task_id: str) -> Dict[str, Any]:
    """Call /reset on the environment."""
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    """Call /step on the environment."""
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert code reviewer. You will be given a Python code snippet along with
structural metadata (AST summary) and optionally static analysis results.

Your job is to find ALL issues in the code. Issues fall into three categories:
- "style": PEP 8 violations, naming issues, unused imports, magic numbers, missing docstrings
- "bug": Logic errors, off-by-one errors, wrong operators, missing null checks, incorrect returns
- "security": SQL injection, command injection, hardcoded secrets, insecure deserialization, path traversal, weak crypto

For each issue you find, provide:
1. line_number: the exact line number (1-indexed)
2. issue_type: one of "style", "bug", or "security"
3. severity: one of "low", "medium", "high", or "critical"
4. description: a clear explanation of the issue

Return your findings as a JSON array. Example:
[
  {"line_number": 5, "issue_type": "style", "severity": "low", "description": "Unused import: os is imported but never used"},
  {"line_number": 12, "issue_type": "bug", "severity": "high", "description": "Off-by-one error in loop bound"}
]

IMPORTANT RULES:
- Be thorough but precise. Missing an issue is bad, but false positives are also penalized.
- Red herrings may be present — safe code patterns that look suspicious but are actually correct.
- Look carefully at whether imports are actually used before flagging them.
- subprocess.run() with a list (no shell=True) is SAFE — do not flag it.
- Return ONLY the JSON array, no other text.
"""

REFINEMENT_PROMPT = """\
You previously reviewed a code snippet and submitted some findings.
The environment has provided feedback and hints about remaining issues.

Your previous findings received this feedback:
{feedback}

{step_analysis}
{hint_section}
{analysis_section}
{memory_section}

Here is the original code again:
```python
{code_snippet}
```

Based on the feedback, hints, and learning from previous attempts, find any ADDITIONAL
issues you may have missed. Do NOT repeat findings for lines that were already correct.
Do NOT repeat false positive patterns.
Return ONLY a JSON array of new findings. If you found everything, return: []
"""

FIX_PROMPT = """\
You are an expert code reviewer. For each of the following security/bug issues found in
the code, provide a corrected version of the relevant code.

Original code:
```python
{code_snippet}
```

Issues to fix (provide fix_code for each):
{issues_json}

{fix_memory}

For each issue, return a JSON object with:
- line_number: the line of the issue
- issue_type: the type ("style", "bug", or "security")
- severity: the severity level
- description: brief description
- fix_code: the corrected code that fixes JUST this issue

CRITICAL RULES for fix_code:
- Must be valid Python with NO indentation errors (start at column 0)
- Must REMOVE the dangerous pattern entirely
- Must use the safe alternative (e.g., subprocess.run with list args instead of os.system)
- Keep fixes minimal — only the fixed line(s), not the whole function

Return ONLY a JSON array.

Example:
[
  {{
    "line_number": 17,
    "issue_type": "security",
    "severity": "critical",
    "description": "Command injection via os.system",
    "fix_code": "result = subprocess.run(['echo', user_input], capture_output=True, text=True, check=True)\\nreturn result.returncode"
  }}
]
"""


def _create_client() -> OpenAI:
    """Create an OpenAI client with the configured API settings."""
    return OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )


def _call_llm(client: OpenAI, system: str, user: str) -> str:
    """Make an LLM call and return the response content."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=3000,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines)

        return content
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        return "[]"


def _parse_findings(content: str) -> List[Dict[str, Any]]:
    """Parse JSON findings from LLM response."""
    try:
        findings = json.loads(content)
        if not isinstance(findings, list):
            findings = [findings]
        return findings
    except json.JSONDecodeError:
        # Try to extract JSON array from mixed content
        import re

        match = re.search(r"\[[\s\S]*\]", content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return []


def get_findings_from_llm(
    client: OpenAI,
    code_snippet: str,
    task_id: str,
    ast_summary: Optional[Dict[str, Any]] = None,
    analysis_result: str = "",
    feedback: str = "",
    hint: str = "",
    memory_prompt: str = "",
    adaptive_prompt: str = "",
    previous_findings: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Ask the LLM to review the code and return findings."""

    if feedback and (hint or analysis_result or adaptive_prompt):
        # ── Refinement pass (within-episode adaptive) ──
        hint_section = (
            f"Hint from environment:\n{hint}" if hint else "No hint available."
        )
        analysis_section = (
            f"Static analysis result:\n{analysis_result}"
            if analysis_result
            else "No analysis available."
        )
        step_analysis = adaptive_prompt if adaptive_prompt else ""
        memory_section = memory_prompt if memory_prompt else ""

        user_prompt = REFINEMENT_PROMPT.format(
            feedback=feedback,
            step_analysis=step_analysis,
            hint_section=hint_section,
            analysis_section=analysis_section,
            memory_section=memory_section,
            code_snippet=code_snippet,
        )
    else:
        # ── Initial pass — include AST summary, analysis, and memory ──
        context_parts = [f"Task: {task_id}"]

        # Inject cross-episode memory (Level A)
        if memory_prompt:
            context_parts.append(memory_prompt)

        if ast_summary:
            context_parts.append(
                f"\nAST Summary (structural metadata):\n"
                f"  Functions: {', '.join(ast_summary.get('functions', []))}\n"
                f"  Classes: {', '.join(ast_summary.get('classes', []))}\n"
                f"  Imports: {', '.join(ast_summary.get('imports', []))}\n"
                f"  Total lines: {ast_summary.get('total_lines', '?')}\n"
                f"  Function calls: {ast_summary.get('call_count', '?')}\n"
                f"  Dangerous imports: {ast_summary.get('dangerous_import_count', '?')}"
            )

        if analysis_result:
            context_parts.append(
                f"\nStatic Analysis Results (from AST analysis):\n{analysis_result}"
            )

        context_parts.append(
            f"\nReview this Python code and find ALL issues:\n\n```python\n{code_snippet}\n```"
        )

        user_prompt = "\n".join(context_parts)

    content = _call_llm(client, SYSTEM_PROMPT, user_prompt)
    findings = _parse_findings(content)

    # ── Filter out known false positives from memory ──
    if previous_findings:
        prev_lines = {f.get("line_number") for f in previous_findings}
        findings = [f for f in findings if f.get("line_number") not in prev_lines]

    return findings


def get_fixes_from_llm(
    client: OpenAI,
    code_snippet: str,
    issues: List[Dict[str, Any]],
    memory_manager: Optional[MemoryManager] = None,
    task_id: str = "",
) -> List[Dict[str, Any]]:
    """Ask the LLM to generate code fixes for detected issues."""
    # Only attempt fixes for security and high-severity bug issues
    fixable = [
        i
        for i in issues
        if i.get("issue_type") == "security"
        or (i.get("issue_type") == "bug" and i.get("severity") in ("high", "critical"))
    ]

    if not fixable:
        return []

    # Limit to top 3 fixes per step
    fixable = fixable[:3]

    # Build fix memory section
    fix_memory = ""
    if memory_manager and task_id:
        mem = memory_manager.get_task_memory(task_id)
        if mem and mem.successful_fixes:
            fix_memory = (
                "Previously successful fix patterns for this task:\n"
                + "\n".join(
                    f"  - {f.get('check_id', '?')}: {f.get('description', '?')}"
                    for f in mem.successful_fixes
                )
                + "\nReuse these patterns where applicable."
            )

    user_prompt = FIX_PROMPT.format(
        code_snippet=code_snippet,
        issues_json=json.dumps(fixable, indent=2),
        fix_memory=fix_memory,
    )

    content = _call_llm(client, SYSTEM_PROMPT, user_prompt)
    fixes = _parse_findings(content)

    # Ensure each fix has fix_code
    valid_fixes = [f for f in fixes if f.get("fix_code")]
    return valid_fixes


# ---------------------------------------------------------------------------
# Strategy execution functions
# ---------------------------------------------------------------------------


def _step_ast_analysis(
    episode_id: str,
) -> tuple[Dict[str, Any], float, str, bool]:
    """Execute AST analysis step. Returns (obs, reward, analysis_text, done)."""
    step_resp = env_step(
        {
            "action_type": "run_ast_analysis",
            "findings": [],
            "done": False,
            "metadata": {"episode_id": episode_id},
        }
    )
    obs = step_resp.get("observation", step_resp)
    reward = step_resp.get("reward", 0.0)
    done = step_resp.get("done", False)
    analysis_text = obs.get("analysis_result", "")
    return obs, reward, analysis_text, done


def _step_review(
    episode_id: str,
    findings: List[Dict[str, Any]],
    done: bool = False,
) -> tuple[Dict[str, Any], float, str, bool]:
    """Submit review findings. Returns (obs, reward, feedback, done)."""
    step_resp = env_step(
        {
            "action_type": "review",
            "findings": findings,
            "done": done,
            "metadata": {"episode_id": episode_id},
        }
    )
    obs = step_resp.get("observation", step_resp)
    reward = step_resp.get("reward", 0.0)
    is_done = step_resp.get("done", False)
    feedback = obs.get("feedback", "")
    return obs, reward, feedback, is_done


def _step_hint(episode_id: str) -> tuple[Dict[str, Any], float, str, bool]:
    """Request a hint. Returns (obs, reward, hint_text, done)."""
    step_resp = env_step(
        {
            "action_type": "request_hint",
            "findings": [],
            "done": False,
            "metadata": {"episode_id": episode_id},
        }
    )
    obs = step_resp.get("observation", step_resp)
    reward = step_resp.get("reward", 0.0)
    done = step_resp.get("done", False)
    hint_text = obs.get("hint", "")
    return obs, reward, hint_text, done


def _step_fix(
    episode_id: str,
    fixes: List[Dict[str, Any]],
    done: bool = True,
) -> tuple[Dict[str, Any], float, Dict[str, Any], bool]:
    """Submit fixes. Returns (obs, reward, fix_feedback, done)."""
    step_resp = env_step(
        {
            "action_type": "submit_fix",
            "findings": fixes,
            "done": done,
            "metadata": {"episode_id": episode_id},
        }
    )
    obs = step_resp.get("observation", step_resp)
    reward = step_resp.get("reward", 0.0)
    is_done = step_resp.get("done", False)
    fix_feedback = obs.get("fix_feedback", {})
    return obs, reward, fix_feedback, is_done


def _step_done(episode_id: str) -> tuple[Dict[str, Any], float]:
    """Send done signal. Returns (obs, reward)."""
    step_resp = env_step(
        {
            "action_type": "review",
            "findings": [],
            "done": True,
            "metadata": {"episode_id": episode_id},
        }
    )
    obs = step_resp.get("observation", step_resp)
    reward = step_resp.get("reward", 0.0)
    return obs, reward


# ---------------------------------------------------------------------------
# Episode runner with learning
# ---------------------------------------------------------------------------


def _emit_start(task_id: str) -> None:
    """Emit structured [START] block for the validator."""
    print(f"[START] task={task_id} env=code-review model={MODEL_NAME}", flush=True)


def _emit_step(
    step_num: int,
    action: str,
    reward: float,
    done: bool,
    error: str | None = None,
) -> None:
    """Emit structured [STEP] block for the validator.

    When done=True the reward is the final task score and must be strictly
    in (0, 1).  Clamp it here as a last-resort defence before printing.
    """
    err_str = error if error else "null"
    done_str = "true" if done else "false"
    if done:
        reward = max(0.01, min(0.99, reward))
    print(
        f"[STEP] step={step_num} action={action} reward={reward:.2f} done={done_str} error={err_str}",
        flush=True,
    )


def _emit_end(success: bool, steps: int, rewards: list[float]) -> None:
    """Emit structured [END] block for the validator.

    Clamps the total (sum of rewards) to strictly (0, 1) by adjusting
    the last reward element.  All values are rounded to 2 decimal places
    BEFORE the range check so that the formatted output is guaranteed to
    survive the :.2f formatting without drifting back to 0.00 or 1.00.
    A post-format verification re-parses the formatted strings and falls
    back to a safe ["0.50"] if floating-point drift still produces a
    boundary value.
    """
    # Ensure rewards list is non-empty
    if not rewards:
        rewards = [0.01]

    # Round to 2 dp first so we reason about what the validator will see
    rewards = [round(r, 2) for r in rewards]
    total = round(sum(rewards), 2)

    if total <= 0.0:
        rewards[-1] = round(rewards[-1] + (0.01 - total), 2)
    elif total >= 1.0:
        rewards[-1] = round(rewards[-1] + (0.99 - total), 2)

    # Post-format verification: re-parse the exact strings the validator
    # will read and confirm the sum is still in the open interval (0, 1).
    formatted = [f"{r:.2f}" for r in rewards]
    parsed_total = sum(float(x) for x in formatted)
    if parsed_total <= 0.0 or parsed_total >= 1.0:
        formatted = ["0.50"]  # safe fallback

    success_str = "true" if success else "false"
    rewards_str = ",".join(formatted)
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True
    )


def _clamp_score(val: float) -> float:
    """Clamp a task score to the strictly-open interval (0, 1).

    The OpenEnv validator requires each task score to be > 0.0 and < 1.0.
    Use 0.01 / 0.99 as the safe boundaries so the value also survives
    :.2f formatting without rounding back to 0.00 or 1.00.
    """
    return max(0.01, min(0.99, val))


def run_episode(
    task_id: str,
    memory: MemoryManager,
    round_num: int = 1,
) -> Dict[str, Any]:
    """Run a single episode using bandit-selected strategy with memory.

    Returns a result dict with score, strategy, matched/missed/fps for memory update.
    """
    difficulty = task_id.split("_")[0]  # easy, medium, hard

    print(f"\n{'=' * 60}")
    print(f"  Task: {task_id} (round {round_num})")
    print(f"{'=' * 60}")

    # ── Check memory for past attempts ────────────────────────────────
    task_mem = memory.get_task_memory(task_id)
    if task_mem and task_mem.attempts > 0:
        print(
            f"  [Memory] {task_mem.attempts} previous attempt(s), "
            f"best={task_mem.best_score:.3f}, last={task_mem.last_score:.3f}"
        )

    # ── Select strategy via bandit ────────────────────────────────────
    strategy = memory.select_strategy(difficulty)
    print(f"  [Strategy] Selected: {strategy}")

    # ── Build memory-enhanced prompt ──────────────────────────────────
    memory_prompt = memory.build_memory_prompt(task_id)
    if memory_prompt:
        print(f"  [Memory] Injecting learning from {task_mem.attempts} past attempt(s)")

    client = _create_client()

    # ── Emit structured [START] for validator ──────────────────────
    # Must be emitted before any step or exception so the validator
    # always sees a matching [START]/[END] pair for this task.
    _emit_start(task_id)

    # ── Step 0: Reset environment ─────────────────────────────────────
    reset_resp = env_reset(task_id)
    obs = reset_resp.get("observation", reset_resp)
    episode_id = obs.get("episode_id", "")
    code_snippet = obs.get("code_snippet", "")
    ast_summary = obs.get("ast_summary", {})

    print(f"  Episode ID: {episode_id}")
    print(f"  Code: {len(code_snippet)} chars, {code_snippet.count(chr(10))} lines")
    if ast_summary:
        print(
            f"  AST: {len(ast_summary.get('functions', []))} funcs, "
            f"{len(ast_summary.get('imports', []))} imports, "
            f"{ast_summary.get('dangerous_import_count', 0)} dangerous"
        )

    total_reward = 0.0
    step_num = 0
    last_feedback = ""
    analysis_text = ""
    hint_text = ""
    all_submitted_findings: List[Dict[str, Any]] = []
    accumulated_feedback = ParsedFeedback()
    episode_rewards: List[float] = []  # collect per-step rewards for [END] output

    # ══════════════════════════════════════════════════════════════════
    # Execute strategy
    # ══════════════════════════════════════════════════════════════════

    # ── Phase 1: Optional AST analysis ────────────────────────────────
    use_ast = strategy in ("ast_first", "full_pipeline")
    if use_ast:
        print(f"\n  [Step {step_num + 1}] Running AST analysis...")
        obs, reward, analysis_text, done = _step_ast_analysis(episode_id)
        step_num += 1
        if done:
            # done=True reward is the clamped total score, not a delta
            total_reward = _clamp_score(reward)
            episode_rewards = [total_reward]
        else:
            total_reward += reward
            episode_rewards.append(reward)
        _emit_step(step_num, "run_ast_analysis", reward, done)
        print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
        print(f"    Analysis: {analysis_text[:200]}")
        if done:
            _emit_end(total_reward > 0, step_num, episode_rewards)
            return _build_result(task_id, strategy, total_reward, [], step_num)

    # ── Phase 2: Initial LLM review ──────────────────────────────────
    print(f"\n  [Step {step_num + 1}] Querying {MODEL_NAME} for findings...")
    findings = get_findings_from_llm(
        client,
        code_snippet,
        task_id,
        ast_summary=ast_summary,
        analysis_result=analysis_text,
        memory_prompt=memory_prompt,
    )
    print(f"  LLM returned {len(findings)} findings")

    if not findings:
        print("  No findings — sending done signal")
        obs, reward = _step_done(episode_id)
        # done=True reward is the clamped total score, not a delta
        total_reward = _clamp_score(reward)
        step_num += 1
        episode_rewards = [total_reward]
        _emit_step(step_num, "review", reward, True)
        _emit_end(total_reward > 0, step_num, episode_rewards)
        return _build_result(task_id, strategy, total_reward, [], step_num)

    # Submit initial findings
    obs, reward, last_feedback, done = _step_review(episode_id, findings, done=False)
    step_num += 1
    all_submitted_findings.extend(findings)
    if done:
        # done=True reward is the clamped total score, not a delta
        total_reward = _clamp_score(reward)
        episode_rewards = [total_reward]
    else:
        total_reward += reward
        episode_rewards.append(reward)
    _emit_step(step_num, "review", reward, done)
    print(f"\n  Step {step_num}: submitted {len(findings)} findings")
    print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
    print(f"    Feedback: {last_feedback[:150]}")

    if done:
        _emit_end(total_reward > 0, step_num, episode_rewards)
        return _build_result(
            task_id,
            strategy,
            total_reward,
            all_submitted_findings,
            step_num,
            feedback=last_feedback,
        )

    # ── Parse step feedback for adaptive refinement (Level B) ─────────
    accumulated_feedback = parse_step_feedback(last_feedback)
    print(
        f"    [Adaptive] {accumulated_feedback.total_correct} correct, "
        f"{accumulated_feedback.total_false_positives} FPs"
    )

    # ── Phase 3: Optional hint ────────────────────────────────────────
    use_hint = strategy in ("review_hint", "ast_first", "full_pipeline")
    if use_hint and difficulty in ("medium", "hard"):
        print(f"\n  [Step {step_num + 1}] Requesting hint...")
        obs, reward, hint_text, done = _step_hint(episode_id)
        step_num += 1
        if done:
            # done=True reward is the clamped total score, not a delta
            total_reward = _clamp_score(reward)
            episode_rewards = [total_reward]
        else:
            total_reward += reward
            episode_rewards.append(reward)
        _emit_step(step_num, "request_hint", reward, done)
        print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
        print(f"    Hint: {hint_text[:150]}")
        if done:
            _emit_end(total_reward > 0, step_num, episode_rewards)
            return _build_result(
                task_id,
                strategy,
                total_reward,
                all_submitted_findings,
                step_num,
                feedback=last_feedback,
            )

    # ── Phase 4: Refinement with adaptive feedback ────────────────────
    use_refinement = strategy != "review_only"
    if use_refinement and (
        hint_text or analysis_text or accumulated_feedback.total_false_positives > 0
    ):
        # Build adaptive prompt from parsed feedback
        adaptive_prompt = memory.build_adaptive_prompt(
            accumulated_feedback, all_submitted_findings
        )

        print(f"\n  [Step {step_num + 1}] Adaptive refinement pass...")
        extra_findings = get_findings_from_llm(
            client,
            code_snippet,
            task_id,
            feedback=last_feedback,
            hint=hint_text,
            analysis_result=analysis_text,
            memory_prompt=memory_prompt,
            adaptive_prompt=adaptive_prompt,
            previous_findings=all_submitted_findings,
        )
        print(f"  LLM returned {len(extra_findings)} additional findings")

        if extra_findings:
            # Don't mark done if we still want to submit fixes
            want_fixes = strategy == "full_pipeline" and difficulty == "hard"
            obs, reward, last_feedback, done = _step_review(
                episode_id, extra_findings, done=not want_fixes
            )
            step_num += 1
            all_submitted_findings.extend(extra_findings)
            if done:
                # done=True reward is the clamped total score, not a delta
                total_reward = _clamp_score(reward)
                episode_rewards = [total_reward]
            else:
                total_reward += reward
                episode_rewards.append(reward)
            _emit_step(step_num, "review", reward, done)

            # Parse refinement feedback too
            ref_feedback = parse_step_feedback(last_feedback)
            print(
                f"\n  Step {step_num}: submitted {len(extra_findings)} additional"
                f" ({ref_feedback.total_correct} correct, {ref_feedback.total_false_positives} FPs)"
            )
            print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")

            if done:
                _emit_end(total_reward > 0, step_num, episode_rewards)
                return _build_result(
                    task_id,
                    strategy,
                    total_reward,
                    all_submitted_findings,
                    step_num,
                    feedback=last_feedback,
                )

    # ── Phase 5: Optional fix submission ──────────────────────────────
    use_fixes = strategy == "full_pipeline" and difficulty == "hard"
    fix_results = []
    if use_fixes:
        security_findings = [
            f for f in all_submitted_findings if f.get("issue_type") == "security"
        ]
        if security_findings:
            print(
                f"\n  [Step {step_num + 1}] Generating fixes for "
                f"{len(security_findings)} security issues..."
            )
            fixes = get_fixes_from_llm(
                client,
                code_snippet,
                security_findings,
                memory_manager=memory,
                task_id=task_id,
            )
            print(f"  LLM returned {len(fixes)} fixes")

            if fixes:
                obs, reward, fix_feedback, done = _step_fix(
                    episode_id, fixes, done=True
                )
                step_num += 1
                fix_results = parse_fix_feedback(fix_feedback)
                if done:
                    # done=True reward is the clamped total score, not a delta
                    total_reward = _clamp_score(reward)
                    episode_rewards = [total_reward]
                else:
                    total_reward += reward
                    episode_rewards.append(reward)
                _emit_step(step_num, "submit_fix", reward, done)

                valid_count = sum(1 for f in fix_results if f.get("is_valid"))
                print(
                    f"\n  Step {step_num}: submitted {len(fixes)} fixes "
                    f"({valid_count} valid)"
                )
                print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")

                _emit_end(total_reward > 0, step_num, episode_rewards)
                return _build_result(
                    task_id,
                    strategy,
                    total_reward,
                    all_submitted_findings,
                    step_num,
                    feedback=last_feedback,
                    fix_results=fix_results,
                )

    # ── Final done signal ─────────────────────────────────────────────
    print(f"\n  [Step {step_num + 1}] Sending done signal...")
    obs, reward = _step_done(episode_id)
    # done=True reward is the clamped total score, not a delta
    total_reward = _clamp_score(reward)
    step_num += 1
    final_feedback = obs.get("feedback", "")
    episode_rewards = [total_reward]
    _emit_step(step_num, "review", reward, True)
    print(f"    Final reward: {total_reward:.3f}")
    print(f"    Feedback: {final_feedback[:200]}")

    _emit_end(total_reward > 0, step_num, episode_rewards)
    return _build_result(
        task_id,
        strategy,
        total_reward,
        all_submitted_findings,
        step_num,
        feedback=final_feedback,
        fix_results=fix_results,
    )


def _build_result(
    task_id: str,
    strategy: str,
    total_reward: float,
    findings: List[Dict[str, Any]],
    steps: int,
    feedback: str = "",
    fix_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a structured result dict for memory updates."""
    # Parse final feedback to extract matched/missed/FP info
    parsed = parse_step_feedback(feedback)

    # Extract descriptions for memory
    matched_descs = []
    fp_descs = []
    for f in findings:
        line = f.get("line_number", 0)
        desc = f"{f.get('issue_type', '?')} L{line}: {f.get('description', '')[:80]}"
        if line in parsed.correct_lines:
            matched_descs.append(desc)
        elif line in parsed.false_positive_lines:
            fp_descs.append(desc)

    # Successful fixes
    successful_fixes = []
    if fix_results:
        for fr in fix_results:
            if fr.get("is_valid"):
                successful_fixes.append(
                    {
                        "check_id": fr.get("check_id", "?"),
                        "line": fr.get("line", 0),
                        "description": fr.get("feedback", ""),
                    }
                )

    return {
        "task_id": task_id,
        "strategy": strategy,
        "total_reward": total_reward,
        "steps": steps,
        "findings_count": len(findings),
        "matched": matched_descs,
        "missed": [],  # We can't know missed from step feedback alone
        "false_positives": fp_descs,
        "successful_fixes": successful_fixes,
        "feedback": feedback,
    }


# ---------------------------------------------------------------------------
# Main with learning loop
# ---------------------------------------------------------------------------


def main():
    """Run inference with cross-episode learning."""
    print("=" * 60)
    print("  Code Review Environment — Learning Inference")
    print(f"  Environment:  {ENV_URL}")
    print(f"  LLM API:      {API_BASE_URL}")
    print(f"  Model:        {MODEL_NAME}")
    print(f"  Tasks:        {len(TASKS)} tasks")
    print(f"  Rounds/task:  {NUM_ROUNDS}")
    print("=" * 60)

    # Health check
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10)
        print(f"\n  Health check: {health.status_code}")
    except Exception as e:
        print(f"\n  [WARN] Health check failed: {e}")

    # Initialize memory
    memory = MemoryManager()

    # Track scores for learning visualization
    all_results: List[Dict[str, Any]] = []

    for round_num in range(1, NUM_ROUNDS + 1):
        if NUM_ROUNDS > 1:
            print(f"\n{'#' * 60}")
            print(f"  ROUND {round_num} / {NUM_ROUNDS}")
            print(f"{'#' * 60}")

        round_results = {}
        for task_id in TASKS:
            task_id = task_id.strip()
            if not task_id:
                continue
            try:
                result = run_episode(task_id, memory, round_num)
                round_results[task_id] = result

                # ── Update memory after each episode ──────────────────
                memory.update_task_memory(
                    task_id=task_id,
                    score=result["total_reward"],
                    strategy=result["strategy"],
                    matched=result.get("matched", []),
                    missed=result.get("missed", []),
                    false_positives=result.get("false_positives", []),
                    fixes=result.get("successful_fixes", []),
                    feedback=result.get("feedback", ""),
                )
                memory.update_strategy(result["strategy"], result["total_reward"])
                memory.save()  # Persist after each episode

            except Exception as e:
                print(f"\n  [ERROR] Task {task_id} failed: {e}")
                # Spec requires [START]+[END] even on exception.
                # _emit_start is already called inside run_episode before any
                # work begins, so only emit [END] here to avoid a duplicate
                # [START] that could confuse the validator.
                _emit_end(False, 0, [])
                round_results[task_id] = {"error": str(e), "total_reward": 0.01}

        all_results.append(round_results)

    # ══════════════════════════════════════════════════════════════════
    # Final report
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")

    # Per-task scores across rounds
    if NUM_ROUNDS > 1:
        print("\n  Score progression (by round):")
        for task_id in TASKS:
            task_id = task_id.strip()
            scores = []
            for r in all_results:
                res = r.get(task_id, {})
                scores.append(res.get("total_reward", 0.0))
            trend = " -> ".join(f"{s:.3f}" for s in scores)
            improved = scores[-1] > scores[0] if len(scores) > 1 else False
            marker = " ^" if improved else ""
            print(f"    {task_id:12s}: {trend}{marker}")

    # Final round scores
    final_round = all_results[-1] if all_results else {}
    total_score = 0.0
    count = 0
    for task_id, result in final_round.items():
        score = result.get("total_reward", 0.0)
        strategy = result.get("strategy", "?")
        total_score += score
        count += 1
        print(f"  {task_id:12s}: {score:+.3f}  (strategy: {strategy})")

    if count > 0:
        print(f"\n  Average score: {total_score / count:.3f}")
        print(f"  Total score:   {total_score:.3f}")

    # Strategy performance
    print(f"\n  {memory.get_strategy_summary()}")

    # Memory stats
    print(f"\n  Memory: {len(memory.task_memories)} tasks learned")
    print(f"  Memory file: {memory.memory_file}")

    print(f"\n{'=' * 60}")
    print("  All episodes complete.")
    print(f"{'=' * 60}")

    return all_results


if __name__ == "__main__":
    main()
