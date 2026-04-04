# CREE — Causal Reverse Engineering Engine

> An OpenEnv-compliant SRE incident-response environment where an agent manages a
> production system with hidden internal state, discovering causal rules through interaction.

**Real-world task:** Site Reliability Engineering — the kind of on-call incident response that engineers do every day.
The agent acts as an operator who can see system metrics but cannot directly inspect internal state.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --port 8000

# Run the RL demo (separate terminal)
python demo.py

# Run the LLM baseline (requires API key)
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

---

## Environment Description

CREE simulates a **production microservices system** with two layers of state:

### Observation Space

| Variable | Type | Range | Description |
|---|---|---|---|
| `latency` | float | 5–600 ms | Response time |
| `error_rate` | float | 0.0–1.0 | Fraction of failed requests |
| `throughput` | float | 0–1000 rps | Successful requests per second |
| `cpu_load` | float | 0.0–1.0 | CPU utilization |
| `status` | string | normal/warning/critical/recovering | High-level health |

### Hidden State (agent cannot observe directly)

| Variable | Description |
|---|---|
| `risk_level` (0–10) | Internal pressure — feeds into all observable metrics |
| `memory_pressure` (0–10) | RAM saturation — can trigger cascade independently |
| `trigger_armed` (bool) | Hidden flag that makes `inject_load` catastrophic |
| `system_mode` | stable / stressed / cascade / failing / recovering |
| `cascade_counter` (0–5) | Countdown to failure once cascade starts |

### Action Space (10 actions)

| Action | Category | Effect |
|---|---|---|
| `probe_latency` | probe | Light probe, minor risk increase |
| `stress_cpu` | stress | CPU stress test, increases risk + memory |
| `inject_load` | stress | High traffic injection — **catastrophic if trigger is armed** |
| `wait` | neutral | Idle — allows risk and memory to decay |
| `reset_connections` | control | **Heals in stressed state, hurts in stable state** (Rule 3) |
| `force_gc` | repair | Reduces memory — but spikes latency if risk > 6 |
| `probe_memory` | probe | Strictly neutral signal |
| `toggle_debug` | control | Arms / disarms the hidden cascade trigger |
| `stabilize` | repair | Reduces risk by 2 — **only if risk < 6, silent fail otherwise** |
| `emergency_stop` | repair | Hard reset — clears all pressure, temporarily kills throughput |

### 7 Hidden Causal Rules

| # | Rule | Discovery method |
|---|---|---|
| 1 | `toggle_debug` → arms trigger; `inject_load` while armed → cascade | Sequence two actions, observe delayed catastrophic effect |
| 2 | `stress_cpu` ×3 consecutive → non-linear memory spike → cascade | Vary repetition count, note superlinear effect |
| 3 | `reset_connections` heals stressed system, damages stable system | Same action, compare outcomes across system states |
| 4 | `force_gc` backfires when risk > 6 | Cross-correlate with latency + risk-related observables |
| 5 | `stabilize` is a no-op when risk ≥ 6 | Compare before/after risk level across multiple attempts |
| 6 | Stress during recovery resets the recovery countdown | Act in recovering state, observe recovery duration |
| 7 | `stabilize` (risk < 6) can pause cascade countdown | Use stabilize after cascade starts, track countdown |

---

## Tasks

### Task 1 — Steady-State Maintenance `[easy]`
**Objective:** Keep system in `normal` status for as many of 25 steps as possible.

**Start state:** Clean system, risk=0, all normal.

**Scoring:** `(steps_in_normal / 25) × (0.5 ^ failures)`

**Expected difficulty:** An agent that learns to avoid `stress_cpu` and `inject_load` scores well quickly.

---

### Task 2 — Incident Recovery `[medium]`
**Objective:** System starts degraded (risk=6.5, status=warning). Restore it to `normal` within 20 steps.

**Start state:** `system_mode=stressed`, `risk_level=6.5`, `memory_pressure=3.0`

**Scoring:** `0.5 × speed_score + 0.5 × stability_score`, penalized per failure

**Expected difficulty:** Requires learning that `reset_connections` works in stressed state (Rule 3) and that `stabilize` is ineffective at high risk (Rule 5).

---

### Task 3 — Cascade Failure Prevention `[hard]`
**Objective:** Trigger is already armed and risk is rising. Prevent failure for 30 steps.

**Start state:** `trigger_armed=True`, `risk_level=5.0`, `memory_pressure=2.0`

**Scoring:** `min(1.0, (steps_survived/30 + 0.2×disarmed) × (0.4 ^ failures))`

**Expected difficulty:** Requires understanding that `toggle_debug` disarms the trigger (Rule 1) and that `inject_load` must be avoided while armed.

---

## Baseline Scores

Measured with `gpt-4o-mini` via `inference.py`:

| Task | Difficulty | Baseline Score |
|---|---|---|
| stability | easy | 0.60 |
| recovery | medium | 0.35 |
| cascade_prevention | hard | 0.18 |
| **Average** | | **0.38** |

---

## API Reference

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/reset` | POST | `{"task": "stability"}` | Reset env, optional task config |
| `/step` | POST | `{"action": "wait"}` | Take one step |
| `/state` | GET | — | Current observable state |
| `/actions` | GET | — | All valid actions |
| `/tasks` | GET | — | All task definitions |
| `/grade` | POST | — | Score current episode |
| `/health` | GET | — | Liveness probe |

---

## Project Structure

```
cree/
├── inference.py         # LLM baseline script (OpenAI client)
├── demo.py              # Rich terminal RL demo
├── openenv.yaml         # OpenEnv spec metadata
├── models.py            # Pydantic data structures
├── requirements.txt
├── Dockerfile
├── env/
│   └── environment.py   # Black-box simulator (7 hidden causal rules)
├── server/
│   └── app.py           # FastAPI server (OpenEnv-compliant)
├── client/
│   └── client.py        # HTTP client wrapper
├── agent/
│   └── agent.py         # CausalBeliefMap + CausalAgent
└── tasks/
    └── graders.py       # 3 task definitions + deterministic graders
```

---

## Docker

```bash
docker build -t cree .
docker run -p 8000:8000 cree
```

---

## Environment Variables for inference.py

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | API key |
| `API_BASE_URL` | Yes | OpenAI-compatible endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | No | Hugging Face token (for HF-hosted models) |
| `CREE_SERVER` | No | Server URL, default `http://localhost:8000` |
