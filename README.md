---
title: Code Review OpenEnv
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Code Review OpenEnv Environment

An OpenEnv RL environment where an AI agent reviews Python code snippets and identifies issues across three difficulty levels.

## Tasks

| Task ID | Difficulty | Issue Types | Issues |
|---------|-----------|-------------|--------|
| easy_1, easy_2, easy_3 | Easy | Style & syntax (unused imports, naming, magic numbers) | 4-5 per snippet |
| medium_1, medium_2, medium_3 | Medium | Logic bugs (off-by-one, wrong operators, missing checks) | 4-5 per snippet |
| hard_1, hard_2, hard_3 | Hard | Security vulnerabilities (SQL injection, command injection, hardcoded secrets) | 6-7 per snippet |

## API

Standard OpenEnv endpoints:

- `POST /reset` — Start a new episode. Body: `{"task_id": "easy_1"}`
- `POST /step` — Submit findings. Body: `{"findings": [...], "done": false, "metadata": {"episode_id": "..."}}`
- `GET /health` — Health check
- `GET /schema` — Action/Observation schemas

### Action Format

```json
{
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

- Correct finding: `+0.15 * severity_weight` (low=0.5, med=1.0, high=1.5, critical=2.0)
- Duplicate: `0.0`
- False positive: `-0.1`
- Done bonus: `+0.3 * recall - 0.05 * false_positive_count`

### Grading

Line-number matching within +/-2 lines tolerance, exact `issue_type` match. Final score is F1 of precision and recall.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline inference
export OPENAI_API_KEY="your-key"
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
├── inference.py                       # Baseline LLM inference script
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Dependencies
├── Dockerfile                         # Container config
└── server/
    ├── app.py                         # FastAPI app
    ├── code_review_environment.py     # Environment implementation
    └── tasks.py                       # Task definitions (9 tasks, 3 per difficulty)
```
