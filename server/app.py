"""
CREE Environment Server — OpenEnv-compliant FastAPI

Endpoints
---------
POST /reset          → reset env (optional body: {"task": "task_id"})
POST /step           → take one action → StepResult
GET  /state          → current observable state
GET  /actions        → list all valid actions + descriptions
GET  /tasks          → list all task definitions
POST /grade          → score current episode for the active task
GET  /health         → liveness probe
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Optional

from env.environment import BlackBoxEnvironment, TASK_CONFIGS
from tasks.graders import TASKS, grade
from models import ACTIONS

app = FastAPI(
    title="CREE — Causal Reverse Engineering Engine",
    description=(
        "OpenEnv-compliant SRE incident-response environment. "
        "Agent manages a production system with hidden internal state."
    ),
    version="1.0.0",
)

env = BlackBoxEnvironment()


# ---------------------------------------------------------------------------
# Request schemas (Pydantic v2)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = None

    @field_validator('task')
    @classmethod
    def task_must_be_valid(cls, v):
        if v is not None and v not in TASK_CONFIGS:
            raise ValueError(f"Unknown task '{v}'. Valid: {list(TASK_CONFIGS)}")
        return v


class StepRequest(BaseModel):
    action: str

    @field_validator('action')
    @classmethod
    def action_must_be_valid(cls, v):
        from models import ACTION_NAMES
        if v not in ACTION_NAMES:
            raise ValueError(f"Unknown action '{v}'")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_dict(obs) -> dict:
    return {
        "latency":    round(obs.latency, 2),
        "error_rate": round(obs.error_rate, 4),
        "throughput": round(obs.throughput, 2),
        "cpu_load":   round(obs.cpu_load, 4),
        "status":     obs.status,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    obs = env.reset(task_id=request.task)
    return {
        "observation": _obs_dict(obs),
        "state":       _obs_dict(obs),   # alias for older clients
        "task":        request.task,
        "done":        False,
        "message":     f"Environment reset" + (f" for task '{request.task}'" if request.task else ""),
    }


@app.post("/step")
def step(request: StepRequest):
    try:
        result = env.step(request.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "observation": _obs_dict(result.state),
        "state":       _obs_dict(result.state),   # alias
        "reward":      result.reward,
        "done":        result.done,
        "info":        result.info,
    }


@app.get("/state")
def get_state():
    if env.state is None:
        raise HTTPException(status_code=409, detail="Environment not initialised — call /reset first")
    return {"observation": _obs_dict(env.state.observable)}


@app.get("/actions")
def list_actions():
    return {
        "actions": [
            {"name": a.name, "description": a.description, "category": a.category}
            for a in ACTIONS
        ]
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id":          t["id"],
                "name":        t["name"],
                "difficulty":  t["difficulty"],
                "max_steps":   t["max_steps"],
                "description": t["description"],
            }
            for t in TASKS.values()
        ]
    }


@app.post("/grade")
def grade_episode():
    if env.current_task is None:
        raise HTTPException(
            status_code=400,
            detail="No active task. Call /reset with a task id first."
        )
    result = grade(env.current_task, env.episode_metrics)
    return result


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
