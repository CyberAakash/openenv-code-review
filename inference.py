#!/usr/bin/env python3
"""Baseline inference script for the Code Review environment.

Uses an OpenAI-compatible LLM to review code snippets and submit findings.
Demonstrates multi-step interaction with hint and analysis actions.

Environment variables (required by hackathon spec):
    API_BASE_URL  — The API endpoint for the LLM (default: https://api.openai.com/v1)
    MODEL_NAME    — The model identifier to use for inference (default: gpt-4o-mini)
    HF_TOKEN      — Your Hugging Face / API key for the LLM provider

Optional:
    ENV_URL       — Base URL of the Code Review environment server
                    (default: https://CyberAakash-code-review.hf.space)
    TASKS         — Comma-separated list of task IDs to run
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hackathon-required variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Environment server URL (our deployed HF Space)
ENV_URL = os.environ.get("ENV_URL", "https://CyberAakash-code-review.hf.space")

# Tasks to run (all 15 by default)
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
You are an expert code reviewer. You will be given a Python code snippet.
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

Be thorough. Do not miss any issues. Do not report issues that don't exist.
Return ONLY the JSON array, no other text.
"""

REFINEMENT_PROMPT = """\
You previously reviewed a code snippet and submitted some findings.
The environment has provided feedback and hints about remaining issues.

Your previous findings received this feedback:
{feedback}

{hint_section}
{analysis_section}

Here is the original code again:
```python
{code_snippet}
```

Based on the feedback and hints, find any ADDITIONAL issues you may have missed.
Return ONLY a JSON array of new findings (do not repeat previous ones).
If you believe you have found all issues, return an empty array: []
"""


def get_findings_from_llm(
    code_snippet: str,
    task_id: str,
    feedback: str = "",
    hint: str = "",
    analysis: str = "",
) -> List[Dict[str, Any]]:
    """Ask the LLM to review the code and return findings."""
    client = OpenAI(
        api_key=HF_TOKEN or "dummy",
        base_url=API_BASE_URL,
    )

    if feedback and (hint or analysis):
        # Refinement pass: use feedback/hints to find more issues
        hint_section = (
            f"Hint from environment:\n{hint}" if hint else "No hint available."
        )
        analysis_section = (
            f"Static analysis result:\n{analysis}"
            if analysis
            else "No analysis available."
        )
        user_prompt = REFINEMENT_PROMPT.format(
            feedback=feedback,
            hint_section=hint_section,
            analysis_section=analysis_section,
            code_snippet=code_snippet,
        )
    else:
        # Initial pass
        user_prompt = f"Task: {task_id}\n\nReview this Python code and find ALL issues:\n\n```python\n{code_snippet}\n```"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
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

        findings = json.loads(content)
        if not isinstance(findings, list):
            findings = [findings]
        return findings
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Strategic episode runner
# ---------------------------------------------------------------------------


def run_episode(task_id: str) -> Dict[str, Any]:
    """Run a single episode using multi-step strategy.

    Strategy:
    1. Initial review: submit all findings from LLM
    2. If hard task and not all issues found: request_hint to learn what's missing
    3. If hint suggests remaining issues: request_analysis for specific line info
    4. Second review pass using feedback + hint + analysis
    5. Done
    """
    print(f"\n{'=' * 60}")
    print(f"  Task: {task_id}")
    print(f"{'=' * 60}")

    # Reset environment
    reset_resp = env_reset(task_id)
    obs = reset_resp.get("observation", reset_resp)
    episode_id = obs.get("episode_id", "")
    code_snippet = obs.get("code_snippet", "")
    available_actions = obs.get("available_actions", ["review"])

    print(f"  Episode ID: {episode_id}")
    print(
        f"  Code snippet ({len(code_snippet)} chars, {code_snippet.count(chr(10))} lines)"
    )
    print(f"  Available actions: {available_actions}")

    # Step 1: Initial LLM review
    print(f"\n  [Step 1] Querying {MODEL_NAME} for initial findings...")
    findings = get_findings_from_llm(code_snippet, task_id)
    print(f"  LLM returned {len(findings)} findings")

    if not findings:
        print("  No findings — sending done signal")
        step_resp = env_step(
            {
                "action_type": "review",
                "findings": [],
                "done": True,
                "metadata": {"episode_id": episode_id},
            }
        )
        return step_resp

    # Submit initial findings (first batch)
    batch_size = 5
    total_reward = 0.0
    step_num = 0
    last_feedback = ""
    last_obs = obs

    for i in range(0, len(findings), batch_size):
        batch = findings[i : i + batch_size]
        is_last_batch = (i + batch_size) >= len(findings)

        action = {
            "action_type": "review",
            "findings": batch,
            "done": False,  # Don't end yet — we might do hint/analysis
            "metadata": {"episode_id": episode_id},
        }

        step_resp = env_step(action)
        last_obs = step_resp.get("observation", step_resp)
        reward = step_resp.get("reward", last_obs.get("reward", 0.0))
        done = step_resp.get("done", last_obs.get("done", False))
        last_feedback = last_obs.get("feedback", "")

        total_reward += reward
        step_num += 1
        print(f"\n  Step {step_num}: submitted {len(batch)} findings")
        print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
        print(f"    Feedback: {last_feedback[:150]}")

        if done:
            return step_resp

    # Step 2: For medium/hard tasks, use hint to find missing issues
    difficulty = "easy"
    if task_id.startswith("medium"):
        difficulty = "medium"
    elif task_id.startswith("hard"):
        difficulty = "hard"

    hint_text = ""
    analysis_text = ""

    if difficulty in ("medium", "hard"):
        # Request a hint
        available_actions = last_obs.get("available_actions", [])
        if "request_hint" in available_actions:
            print(f"\n  [Step {step_num + 1}] Requesting hint...")
            step_resp = env_step(
                {
                    "action_type": "request_hint",
                    "findings": [],
                    "done": False,
                    "metadata": {"episode_id": episode_id},
                }
            )
            last_obs = step_resp.get("observation", step_resp)
            reward = step_resp.get("reward", last_obs.get("reward", 0.0))
            done = step_resp.get("done", last_obs.get("done", False))
            hint_text = last_obs.get("hint", "")
            total_reward += reward
            step_num += 1
            print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
            print(f"    Hint: {hint_text[:150]}")

            if done:
                return step_resp

    if difficulty == "hard":
        # For hard tasks, also use static analysis
        available_actions = last_obs.get("available_actions", [])
        if "request_analysis" in available_actions:
            print(f"\n  [Step {step_num + 1}] Requesting static analysis...")
            step_resp = env_step(
                {
                    "action_type": "request_analysis",
                    "findings": [],
                    "done": False,
                    "metadata": {"episode_id": episode_id},
                }
            )
            last_obs = step_resp.get("observation", step_resp)
            reward = step_resp.get("reward", last_obs.get("reward", 0.0))
            done = step_resp.get("done", last_obs.get("done", False))
            analysis_text = last_obs.get("analysis_result", "")
            total_reward += reward
            step_num += 1
            print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
            print(f"    Analysis: {analysis_text[:150]}")

            if done:
                return step_resp

    # Step 3: Refinement pass — use hint/analysis feedback to find more issues
    if hint_text or analysis_text:
        print(f"\n  [Step {step_num + 1}] Refinement pass with hints/analysis...")
        extra_findings = get_findings_from_llm(
            code_snippet,
            task_id,
            feedback=last_feedback,
            hint=hint_text,
            analysis=analysis_text,
        )
        print(f"  LLM returned {len(extra_findings)} additional findings")

        if extra_findings:
            action = {
                "action_type": "review",
                "findings": extra_findings,
                "done": True,
                "metadata": {"episode_id": episode_id},
            }
            step_resp = env_step(action)
            last_obs = step_resp.get("observation", step_resp)
            reward = step_resp.get("reward", last_obs.get("reward", 0.0))
            total_reward += reward
            step_num += 1
            print(
                f"\n  Step {step_num}: submitted {len(extra_findings)} additional + done"
            )
            print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
            print(f"    Feedback: {last_obs.get('feedback', '')[:150]}")
            return step_resp

    # Final done signal
    print(f"\n  [Step {step_num + 1}] Sending done signal...")
    step_resp = env_step(
        {
            "action_type": "review",
            "findings": [],
            "done": True,
            "metadata": {"episode_id": episode_id},
        }
    )
    last_obs = step_resp.get("observation", step_resp)
    reward = step_resp.get("reward", last_obs.get("reward", 0.0))
    total_reward += reward
    print(f"    Final reward: {total_reward:.3f}")
    print(f"    Feedback: {last_obs.get('feedback', '')[:150]}")

    return step_resp


def main():
    """Run inference across all configured tasks."""
    print("=" * 60)
    print("  Code Review Environment — Baseline Inference")
    print(f"  Environment: {ENV_URL}")
    print(f"  LLM API:    {API_BASE_URL}")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Tasks:      {len(TASKS)} tasks")
    print("=" * 60)

    if not HF_TOKEN:
        print("\n  [WARN] HF_TOKEN not set — LLM calls will likely fail.")
        print("         Set the HF_TOKEN environment variable to your API key.")

    # Health check
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=10)
        print(f"\n  Health check: {health.status_code}")
    except Exception as e:
        print(f"\n  [WARN] Health check failed: {e}")

    results = {}
    for task_id in TASKS:
        task_id = task_id.strip()
        if not task_id:
            continue
        try:
            result = run_episode(task_id)
            results[task_id] = result
        except Exception as e:
            print(f"\n  [ERROR] Task {task_id} failed: {e}")
            results[task_id] = {"error": str(e)}

    print(f"\n{'=' * 60}")
    print("  All episodes complete.")
    print(f"{'=' * 60}")
    return results


if __name__ == "__main__":
    main()
