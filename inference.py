#!/usr/bin/env python3
"""Baseline inference script for the Code Review environment.

Uses an OpenAI-compatible LLM to review code snippets and submit findings.

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

# Tasks to run (all 9 by default)
TASKS = os.environ.get(
    "TASKS",
    "easy_1,easy_2,easy_3,medium_1,medium_2,medium_3,hard_1,hard_2,hard_3",
).split(",")


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


def get_findings_from_llm(code_snippet: str, task_id: str) -> List[Dict[str, Any]]:
    """Ask the LLM to review the code and return findings."""
    client = OpenAI(
        api_key=HF_TOKEN or "dummy",
        base_url=API_BASE_URL,
    )

    user_prompt = f"Task: {task_id}\n\nReview this Python code and find ALL issues:\n\n```python\n{code_snippet}\n```"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        content = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last line (``` markers)
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
# Main loop
# ---------------------------------------------------------------------------


def run_episode(task_id: str) -> Dict[str, Any]:
    """Run a single episode on the given task."""
    print(f"\n{'=' * 60}")
    print(f"  Task: {task_id}")
    print(f"{'=' * 60}")

    # Reset environment
    reset_resp = env_reset(task_id)
    obs = reset_resp.get("observation", reset_resp)
    episode_id = obs.get("episode_id", "")
    code_snippet = obs.get("code_snippet", "")

    print(f"  Episode ID: {episode_id}")
    print(f"  Code snippet ({len(code_snippet)} chars):")
    for i, line in enumerate(code_snippet.split("\n")[:5], 1):
        print(f"    {i}: {line}")
    if code_snippet.count("\n") > 5:
        print(f"    ... ({code_snippet.count(chr(10))} lines total)")

    # Get findings from LLM
    print(f"\n  Querying {MODEL_NAME} for findings...")
    findings = get_findings_from_llm(code_snippet, task_id)
    print(f"  LLM returned {len(findings)} findings")

    if not findings:
        print("  No findings — sending done signal")
        step_resp = env_step(
            {
                "findings": [],
                "done": True,
                "metadata": {"episode_id": episode_id},
            }
        )
        return step_resp

    # Submit findings in batches of 3 (multi-step interaction)
    batch_size = 3
    total_reward = 0.0
    step_num = 0

    for i in range(0, len(findings), batch_size):
        batch = findings[i : i + batch_size]
        is_last_batch = (i + batch_size) >= len(findings)

        action = {
            "findings": batch,
            "done": is_last_batch,
            "metadata": {"episode_id": episode_id},
        }

        step_resp = env_step(action)
        obs = step_resp.get("observation", step_resp)
        reward = step_resp.get("reward", obs.get("reward", 0.0))
        done = step_resp.get("done", obs.get("done", False))
        feedback = obs.get("feedback", "")

        total_reward += reward
        step_num += 1
        print(f"\n  Step {step_num}: submitted {len(batch)} findings")
        print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
        print(f"    Feedback: {feedback[:120]}")

        if done:
            break

    # If we haven't sent done yet, send it now
    if not step_resp.get("done", step_resp.get("observation", {}).get("done", False)):
        step_resp = env_step(
            {
                "findings": [],
                "done": True,
                "metadata": {"episode_id": episode_id},
            }
        )
        obs = step_resp.get("observation", step_resp)
        reward = step_resp.get("reward", obs.get("reward", 0.0))
        total_reward += reward
        print(f"\n  Done signal sent. Final reward: {total_reward:.3f}")

    return step_resp


def main():
    """Run inference across all configured tasks."""
    print("=" * 60)
    print("  Code Review Environment — Baseline Inference")
    print(f"  Environment: {ENV_URL}")
    print(f"  LLM API:    {API_BASE_URL}")
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Tasks:      {TASKS}")
    print("=" * 60)

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
