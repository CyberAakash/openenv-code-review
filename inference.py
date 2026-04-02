#!/usr/bin/env python3
"""Baseline inference script for the Code Review environment.

Uses an OpenAI-compatible LLM to review code snippets and submit findings.
Demonstrates multi-step interaction with AST analysis, hints, review, and fix actions.

Strategy per episode:
1. Reset — observe ast_summary for structural clues
2. run_ast_analysis — get real static analysis findings (cost: -0.05)
3. Initial review — LLM reviews code with AST context, submits findings
4. (Medium/Hard) request_hint — learn what's still missing
5. Refinement review — submit additional findings using hints
6. submit_fix — attempt to fix high-confidence security issues for bonus reward
7. Done

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

Be thorough. Do not miss any issues. Do not report issues that don't exist.
Red herrings may be present — safe code patterns that look suspicious but are actually correct.
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

FIX_PROMPT = """\
You are an expert code reviewer. For each of the following security/bug issues found in
the code, provide a corrected version of the relevant code.

Original code:
```python
{code_snippet}
```

Issues to fix (provide fix_code for each):
{issues_json}

For each issue, return a JSON object with:
- line_number: the line of the issue
- issue_type: the type ("style", "bug", or "security")
- severity: the severity level
- description: brief description
- fix_code: the corrected code that fixes JUST this issue (a few lines around the fix)

Return ONLY a JSON array. Each fix_code should be valid Python that removes the
dangerous pattern and uses the safe alternative. Keep fixes minimal and focused.

Example:
[
  {{
    "line_number": 17,
    "issue_type": "security",
    "severity": "critical",
    "description": "Command injection via os.system",
    "fix_code": "def run_command(user_input):\\n    result = subprocess.run(['echo', user_input], capture_output=True)\\n    return result.returncode"
  }}
]
"""


def _create_client() -> OpenAI:
    """Create an OpenAI client with the configured API settings."""
    return OpenAI(
        api_key=HF_TOKEN or "dummy",
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
        return []


def get_findings_from_llm(
    client: OpenAI,
    code_snippet: str,
    task_id: str,
    ast_summary: Dict[str, Any] = None,
    analysis_result: str = "",
    feedback: str = "",
    hint: str = "",
) -> List[Dict[str, Any]]:
    """Ask the LLM to review the code and return findings."""
    if feedback and (hint or analysis_result):
        # Refinement pass
        hint_section = (
            f"Hint from environment:\n{hint}" if hint else "No hint available."
        )
        analysis_section = (
            f"Static analysis result:\n{analysis_result}"
            if analysis_result
            else "No analysis available."
        )
        user_prompt = REFINEMENT_PROMPT.format(
            feedback=feedback,
            hint_section=hint_section,
            analysis_section=analysis_section,
            code_snippet=code_snippet,
        )
    else:
        # Initial pass — include AST summary and analysis for context
        context_parts = [f"Task: {task_id}"]

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
    return _parse_findings(content)


def get_fixes_from_llm(
    client: OpenAI,
    code_snippet: str,
    issues: List[Dict[str, Any]],
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

    # Limit to top 3 fixes per step to stay within step budget
    fixable = fixable[:3]

    user_prompt = FIX_PROMPT.format(
        code_snippet=code_snippet,
        issues_json=json.dumps(fixable, indent=2),
    )

    content = _call_llm(client, SYSTEM_PROMPT, user_prompt)
    fixes = _parse_findings(content)

    # Ensure each fix has fix_code
    valid_fixes = [f for f in fixes if f.get("fix_code")]
    return valid_fixes


# ---------------------------------------------------------------------------
# Strategic episode runner
# ---------------------------------------------------------------------------


def run_episode(task_id: str) -> Dict[str, Any]:
    """Run a single episode using the full multi-step strategy.

    Strategy:
    1. Reset — get code + AST summary
    2. Run AST analysis — get real static analysis findings
    3. Initial review — submit findings informed by AST context
    4. (Medium/Hard) Request hint — learn what's missing
    5. (If hint reveals gaps) Refinement pass — find more issues
    6. (Hard) Submit fixes — attempt to fix high-severity issues
    7. Done
    """
    print(f"\n{'=' * 60}")
    print(f"  Task: {task_id}")
    print(f"{'=' * 60}")

    client = _create_client()

    # ── Step 0: Reset environment ─────────────────────────────────────
    reset_resp = env_reset(task_id)
    obs = reset_resp.get("observation", reset_resp)
    episode_id = obs.get("episode_id", "")
    code_snippet = obs.get("code_snippet", "")
    ast_summary = obs.get("ast_summary", {})
    available_actions = obs.get("available_actions", ["review"])

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

    # ── Step 1: Run AST analysis ──────────────────────────────────────
    if "run_ast_analysis" in available_actions:
        print(f"\n  [Step {step_num + 1}] Running AST analysis...")
        step_resp = env_step(
            {
                "action_type": "run_ast_analysis",
                "findings": [],
                "done": False,
                "metadata": {"episode_id": episode_id},
            }
        )
        obs = step_resp.get("observation", step_resp)
        reward = step_resp.get("reward", obs.get("reward", 0.0))
        done = step_resp.get("done", obs.get("done", False))
        analysis_text = obs.get("analysis_result", "")
        total_reward += reward
        step_num += 1
        print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
        print(f"    Analysis: {analysis_text[:200]}")
        if done:
            return step_resp

    # ── Step 2: Initial LLM review with AST context ──────────────────
    print(f"\n  [Step {step_num + 1}] Querying {MODEL_NAME} for initial findings...")
    findings = get_findings_from_llm(
        client,
        code_snippet,
        task_id,
        ast_summary=ast_summary,
        analysis_result=analysis_text,
    )
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

    # Submit initial findings in batches
    batch_size = 5
    for i in range(0, len(findings), batch_size):
        batch = findings[i : i + batch_size]

        action = {
            "action_type": "review",
            "findings": batch,
            "done": False,
            "metadata": {"episode_id": episode_id},
        }

        step_resp = env_step(action)
        obs = step_resp.get("observation", step_resp)
        reward = step_resp.get("reward", obs.get("reward", 0.0))
        done = step_resp.get("done", obs.get("done", False))
        last_feedback = obs.get("feedback", "")

        total_reward += reward
        step_num += 1
        print(f"\n  Step {step_num}: submitted {len(batch)} findings")
        print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
        print(f"    Feedback: {last_feedback[:150]}")

        if done:
            return step_resp

    # ── Step 3: For medium/hard, request hint ─────────────────────────
    difficulty = "easy"
    if task_id.startswith("medium"):
        difficulty = "medium"
    elif task_id.startswith("hard"):
        difficulty = "hard"

    hint_text = ""

    if difficulty in ("medium", "hard"):
        available_actions = obs.get("available_actions", [])
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
            obs = step_resp.get("observation", step_resp)
            reward = step_resp.get("reward", obs.get("reward", 0.0))
            done = step_resp.get("done", obs.get("done", False))
            hint_text = obs.get("hint", "")
            total_reward += reward
            step_num += 1
            print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
            print(f"    Hint: {hint_text[:150]}")

            if done:
                return step_resp

    # ── Step 4: Refinement pass with hints ────────────────────────────
    if hint_text or (analysis_text and difficulty != "easy"):
        print(f"\n  [Step {step_num + 1}] Refinement pass with hints/analysis...")
        extra_findings = get_findings_from_llm(
            client,
            code_snippet,
            task_id,
            feedback=last_feedback,
            hint=hint_text,
            analysis_result=analysis_text,
        )
        print(f"  LLM returned {len(extra_findings)} additional findings")

        if extra_findings:
            # For hard tasks, don't mark done yet — we want to try fixes
            is_done = difficulty != "hard"
            action = {
                "action_type": "review",
                "findings": extra_findings,
                "done": is_done,
                "metadata": {"episode_id": episode_id},
            }
            step_resp = env_step(action)
            obs = step_resp.get("observation", step_resp)
            reward = step_resp.get("reward", obs.get("reward", 0.0))
            done = step_resp.get("done", obs.get("done", False))
            total_reward += reward
            step_num += 1
            print(
                f"\n  Step {step_num}: submitted {len(extra_findings)} additional"
                + (" + done" if is_done else "")
            )
            print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
            print(f"    Feedback: {obs.get('feedback', '')[:150]}")

            if done:
                return step_resp

    # ── Step 5: Submit fixes for hard tasks ───────────────────────────
    if difficulty == "hard" and not obs.get("done", False):
        # Collect all security findings from our submissions
        all_security_findings = [
            f for f in findings if f.get("issue_type") == "security"
        ]

        if all_security_findings:
            print(
                f"\n  [Step {step_num + 1}] Generating fixes for {len(all_security_findings)} security issues..."
            )
            fixes = get_fixes_from_llm(client, code_snippet, all_security_findings)
            print(f"  LLM returned {len(fixes)} fixes")

            if fixes:
                fix_action = {
                    "action_type": "submit_fix",
                    "findings": fixes,
                    "done": True,
                    "metadata": {"episode_id": episode_id},
                }
                step_resp = env_step(fix_action)
                obs = step_resp.get("observation", step_resp)
                reward = step_resp.get("reward", obs.get("reward", 0.0))
                total_reward += reward
                step_num += 1
                print(f"\n  Step {step_num}: submitted {len(fixes)} fixes + done")
                print(f"    Reward: {reward:+.3f} (total: {total_reward:.3f})")
                print(f"    Feedback: {obs.get('feedback', '')[:150]}")
                return step_resp

    # ── Final done signal ─────────────────────────────────────────────
    if not obs.get("done", False):
        print(f"\n  [Step {step_num + 1}] Sending done signal...")
        step_resp = env_step(
            {
                "action_type": "review",
                "findings": [],
                "done": True,
                "metadata": {"episode_id": episode_id},
            }
        )
        obs = step_resp.get("observation", step_resp)
        reward = step_resp.get("reward", obs.get("reward", 0.0))
        total_reward += reward
        print(f"    Final reward: {total_reward:.3f}")
        print(f"    Feedback: {obs.get('feedback', '')[:150]}")
        return step_resp

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
