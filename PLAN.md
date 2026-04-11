# TRIOZ Code Review Environment - Resubmission Fix Plan

**Last Updated**: April 12, 2026, 3:00 AM IST
**Deadline**: April 12, 2026, 11:59 PM IST
**Status**: Waitlisted, resubmitting

## Context

Our OpenEnv Code Review environment was **waitlisted** (not rejected) in the hackathon. We analyzed the troubleshooting guide and identified critical issues, particularly the **inference.py output format** being non-compliant with the spec.

## Key Discovery: inference.py Output Format is WRONG

The troubleshooting guide (https://docs.google.com/document/d/1nth7bAacQOQEpVk6oIHV917YuRcLOowcSS1Ed-uNQVQ) specifies exact output format:

```
# REQUIRED FORMAT
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

Our current format is wrong in ALL three line types:
- [START]: Missing `env=` and `model=`
- [STEP]: Missing `action=`, `done=`, `error=`; wrong decimal precision (4 vs 2)
- [END]: Completely wrong fields (we emit `task= score= steps=` instead of `success= steps= rewards=`)
- No [END] emitted on exception (spec says "always emitted, even on exception")

## Execution Order (by impact)

### P0: Fix inference.py Output Format (CRITICAL) - ~45 min
- File: `inference.py`
- Lines 497-509: Rewrite `_emit_start`, `_emit_step`, `_emit_end`
- Lines 512+: Thread action_type/done/error through call sites; collect `episode_rewards` list
- Lines 898-900: Add [END] in exception handler
- Line 58: HF_TOKEN should raise ValueError if None

### B1: Fix submit_fix Exploits - ~30 min
- File: `server/code_review_environment.py` lines 501-609
- Add duplicate fix checking (track `fixes_applied` set by line_number)
- Make `submit_fix` increment `review_steps`
- Track fix submissions in `submitted_findings` for FP calculation

### B2: Fix Easy Task Ground Truth - ~30 min
- File: `server/tasks.py`
- easy_1: Add unused variable `x` (line 13), `range(len())` anti-pattern (line 12)
- easy_4: Add unclosed file handle issue (if present in code)
- easy_5: Check for missing unused imports

### B3: Improve FP Penalties - ~15 min
- File: `server/code_review_environment.py`
- Per-step FP: -0.03 -> -0.08 (line 648)
- Finalization FP: -0.05 -> -0.08 (line 700)
- Consider reducing recall bonus from 0.3 to 0.15 (double-counting)

### B4: GroundTruthIssue Literal Types - ~10 min
- File: `models.py` lines 138-144
- Add Literal types for issue_type and severity
- Add Field descriptions

### C1: Add Genuinely Subtle Issues to Hard Tasks - ~1.5 hr
- File: `server/tasks.py`
- hard_1: Timing side-channel in password comparison
- hard_2: ReDoS vulnerability
- hard_3: TOCTOU race condition
- hard_4: JWT expiry not validated
- hard_5: Missing rate limiting on login

### A1: Fix README/Code Mismatch - ~15 min
- File: `README.md`
- `request_analysis` -> `run_ast_analysis`
- Analysis cost `-0.10` -> `-0.05`
- Add `submit_fix` documentation
- Fix available_actions list

### A2: Enrich pyproject.toml - ~5 min
- Add authors, license, readme, urls, keywords

### A3: Dockerfile Hardening - ~10 min
- Add LABEL, USER directive, HEALTHCHECK

### D1: Add Design Decisions to README - ~20 min
### D2: Clean up sys.path hack - ~5 min

### E1-E4: Deploy & Validate - ~30 min

## Files to Modify

| File | Priority | Changes |
|------|----------|---------|
| `inference.py` | P0 | Output format (emit functions + call sites + error handler) |
| `server/code_review_environment.py` | B1,B3 | submit_fix exploits, FP penalties |
| `server/tasks.py` | B2,C1 | Ground truth fixes, subtle hard task issues |
| `models.py` | B4 | Literal types + Field descriptions |
| `README.md` | A1,D1 | Action name fix, Design Decisions |
| `pyproject.toml` | A2 | Metadata enrichment |
| `Dockerfile` | A3 | Hardening |
| `server/app.py` | D2 | sys.path cleanup |

## DO NOT CHANGE

- **Web UI** (static/) - Complete, working, low impact
- **memory.py** - Only used by inference.py, working fine
- **client.py** - Standard, no issues
- **Task count** - Staying at 15
- **Core strategy logic in inference.py** - Only touch emit functions

## Progress Tracking

- [ ] PLAN.md written
- [ ] P0: inference.py output format fixed
- [ ] B1: submit_fix exploits fixed
- [ ] B2: Easy task ground truth fixed
- [ ] B3: FP penalties improved
- [ ] B4: GroundTruthIssue types added
- [ ] C1: Hard task subtle issues added
- [ ] A1: README fixed
- [ ] A2: pyproject.toml enriched
- [ ] A3: Dockerfile hardened
- [ ] D1: Design Decisions added
- [ ] D2: sys.path hack cleaned
- [ ] E1: Local validate passes
- [ ] E2: Docker build passes
- [ ] E3: HF Spaces deployed + remote validate
- [ ] E4: Smoke test inference.py

## Key Commands

```bash
# Start server locally
uv run python -m server.app

# Local validation
uv run openenv validate

# Deploy to HF
uv run openenv push -r CyberAakash/code-review

# Remote validation
uv run openenv validate --url https://CyberAakash-code-review.hf.space

# Test inference (single task)
NUM_ROUNDS=1 TASKS=easy_1 ENV_URL=http://localhost:7860 API_BASE_URL=http://localhost:11434/v1 MODEL_NAME=qwen2.5-coder:7b HF_TOKEN=dummy uv run python inference.py
```

## Hackathon Scoring Rubric

- Real-world utility (30%)
- Task & grader quality (25%)
- Environment design (20%)
- Code quality & spec compliance (15%)
- Creativity & novelty (10%)

## Team

- Santhosh (Team Lead)
- Aakash T
- Sriram
