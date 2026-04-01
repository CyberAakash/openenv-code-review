---
title: Code Review OpenEnv
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Code Review OpenEnv Environment

An OpenEnv RL environment where an AI agent reviews Python code snippets and identifies issues across three difficulty levels. Features multi-step interaction with exploration vs. exploitation tradeoffs.

## Tasks

15 tasks across 3 difficulty levels:

| Difficulty | Task IDs | Issue Types | Issues per Snippet |
|-----------|----------|-------------|-------------------|
| Easy (5) | easy_1 — easy_5 | Style & syntax (unused imports, naming, magic numbers) | 4-13 |
| Medium (5) | medium_1 — medium_5 | Logic bugs (off-by-one, wrong operators, missing checks) | 4-8 |
| Hard (5) | hard_1 — hard_5 | Security vulnerabilities (SQL injection, command injection, hardcoded secrets) | 6-11 |

## Multi-Step Action Types

The agent can choose from three action types each step, creating an exploration vs. exploitation tradeoff:

| Action Type | Cost | Effect | Limit |
|------------|------|--------|-------|
| `review` | Free | Submit findings for grading | Unlimited |
| `request_hint` | -0.05 reward | Reveals unfound issue categories + approximate line ranges | 3 per episode |
| `request_analysis` | -0.10 reward | Simulated static analysis revealing 2 obvious issues | 1 per episode |

## API

Standard OpenEnv endpoints:

- `POST /reset` — Start a new episode. Body: `{"task_id": "easy_1"}`
- `POST /step` — Submit action. Body: `{"action": {"action_type": "review", "findings": [...], "done": false, "metadata": {"episode_id": "..."}}}`
- `GET /health` — Health check
- `GET /schema` — Action/Observation schemas

### Action Format

```json
{
  "action_type": "review",
  "findings": [
    {
      "line_number": 5,
      "issue_type": "style",
      "severity": "low",
      "description": "Unused import: os is imported but never used"
    }
  ],
  "done": false,
  "metadata": {"episode_id": "..."}
}
```

### Reward Design

- **Correct finding**: `+0.15 * severity_weight` (low=0.5, med=1.0, high=1.5, critical=2.0)
- **Description accuracy bonus**: up to `+0.05` for keyword overlap with ground truth
- **Severity accuracy bonus**: `+0.05` (exact match), `0.0` (off by one), `-0.02` (off by two+)
- **Duplicate**: `0.0`
- **False positive**: `-0.1`
- **Done bonus**: `+0.3 * recall - 0.05 * false_positive_count`
- **Step efficiency bonus**: up to `+0.15` (linear decay with steps used)
- **Hint cost**: `-0.05` per hint
- **Analysis cost**: `-0.10` per analysis

### Grading

Line-number matching within +/-2 lines tolerance, exact `issue_type` match, description keyword similarity, severity accuracy. Final score includes F1 of precision and recall.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline inference
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
python inference.py
```

## Docker

```bash
docker build -t code-review-env .
docker run -p 7860:7860 code-review-env
```

## Project Structure

```
├── models.py                          # Pydantic models (Action, Observation, State)
├── client.py                          # EnvClient subclass
├── inference.py                       # Baseline LLM inference with multi-step strategy
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Dependencies
├── Dockerfile                         # Container config
├── outputs/                           # Sample evaluation results
└── server/
    ├── app.py                         # FastAPI app
    ├── code_review_environment.py     # Environment implementation (3 action types, improved grading)
    └── tasks.py                       # Task definitions (15 tasks, 5 per difficulty)
```
