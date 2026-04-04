"""
CREE Baseline Inference Script
================================
Runs an LLM agent against all 3 CREE tasks and reports scores.

Environment variables required:
    API_BASE_URL   — OpenAI-compatible API endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME     — Model identifier (e.g. gpt-4o-mini)
    OPENAI_API_KEY — API key
    HF_TOKEN       — (optional) Hugging Face token for HF-hosted models
    CREE_SERVER    — (optional) CREE server URL, default http://localhost:8000

Usage:
    python inference.py

Output:
    Prints score for each task and a final summary table.
"""

import os
import sys
import json
import time
import requests
from typing import Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL   = os.environ.get("API_BASE_URL",   "https://api.openai.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME",     "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN       = os.environ.get("HF_TOKEN",       "")
CREE_SERVER    = os.environ.get("CREE_SERVER",    "http://localhost:8000")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

# Initialise OpenAI client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL,
)


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class CREEEnv:
    def __init__(self, server_url: str):
        self.url = server_url.rstrip("/")

    def reset(self, task: Optional[str] = None) -> dict:
        body = {"task": task} if task else {}
        r = requests.post(f"{self.url}/reset", json=body, timeout=10)
        r.raise_for_status()
        return r.json()["observation"]

    def step(self, action: str) -> dict:
        r = requests.post(f"{self.url}/step", json={"action": action}, timeout=10)
        r.raise_for_status()
        d = r.json()
        return {
            "observation": d["observation"],
            "reward":      d["reward"],
            "done":        d["done"],
        }

    def grade(self) -> dict:
        r = requests.post(f"{self.url}/grade", timeout=10)
        r.raise_for_status()
        return r.json()

    def list_actions(self) -> list:
        r = requests.get(f"{self.url}/actions", timeout=10)
        r.raise_for_status()
        return r.json()["actions"]

    def get_task(self, task_id: str) -> dict:
        r = requests.get(f"{self.url}/tasks", timeout=10)
        r.raise_for_status()
        tasks = {t["id"]: t for t in r.json()["tasks"]}
        return tasks.get(task_id, {})


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) managing a production system.
At each step you observe the system's metrics and must choose exactly ONE action to take.

Your goal depends on the current task — read it carefully.

Respond with ONLY the action name — no explanation, no punctuation, just the action name.
If you are unsure, choose 'wait' or 'probe_latency'."""


def build_user_prompt(
    obs: dict,
    task_desc: str,
    actions: list,
    history: list,
    step: int,
    max_steps: int,
) -> str:
    action_list = "\n".join(
        f"  - {a['name']}: {a['description']}" for a in actions
    )
    history_str = ""
    if history:
        history_str = "\nLast 5 steps:\n" + "\n".join(
            f"  step {h['step']}: action={h['action']} → "
            f"status={h['obs']['status']} lat={h['obs']['latency']:.0f}ms "
            f"err={h['obs']['error_rate']:.3f} reward={h['reward']:+.2f}"
            for h in history[-5:]
        )

    return f"""TASK ({step}/{max_steps}): {task_desc}

Current system state:
  status:     {obs['status']}
  latency:    {obs['latency']:.1f} ms
  error_rate: {obs['error_rate']:.4f}
  throughput: {obs['throughput']:.1f} rps
  cpu_load:   {obs['cpu_load']:.4f}
{history_str}

Available actions:
{action_list}

Choose exactly one action name:"""


def choose_action(obs: dict, task_desc: str, actions: list, history: list,
                  step: int, max_steps: int, valid_names: list) -> str:
    prompt = build_user_prompt(obs, task_desc, actions, history, step, max_steps)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": prompt},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip().lower().replace("-", "_")
        # Find first valid action name that appears in the response
        for name in valid_names:
            if name in raw:
                return name
        # Fallback
        return "wait"
    except Exception as exc:
        print(f"  [LLM error: {exc}] → defaulting to 'wait'")
        return "wait"


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(env: CREEEnv, task_id: str) -> dict:
    task_meta = env.get_task(task_id)
    task_desc = task_meta.get("description", task_id)
    max_steps = task_meta.get("max_steps", 30)
    actions   = env.list_actions()
    valid_names = [a["name"] for a in actions]

    obs      = env.reset(task=task_id)
    history  = []
    total_r  = 0.0

    print(f"\n  Task: {task_meta.get('name', task_id)}  [{task_meta.get('difficulty','?')}]")
    print(f"  Max steps: {max_steps}")
    print(f"  Start state: status={obs['status']} lat={obs['latency']:.0f}ms")

    for step in range(1, max_steps + 1):
        action = choose_action(obs, task_desc, actions, history, step, max_steps, valid_names)
        result = env.step(action)
        total_r += result["reward"]

        history.append({
            "step":   step,
            "action": action,
            "obs":    result["observation"],
            "reward": result["reward"],
        })

        status = result["observation"]["status"]
        print(
            f"  step {step:2d}: {action:22s} → {status:10s} "
            f"lat={result['observation']['latency']:5.0f}ms "
            f"R={result['reward']:+.2f}",
            flush=True,
        )

        obs = result["observation"]
        if result["done"]:
            print("  ** FAILURE — episode ended early **")
            break

    grade_result = env.grade()
    score = grade_result.get("score", 0.0)
    print(f"  → Score: {score:.4f}  (total_reward={total_r:+.2f})")
    return grade_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CREE Baseline Inference")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Server: {CREE_SERVER}")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{CREE_SERVER}/health", timeout=5)
        r.raise_for_status()
        print(f"Server OK: {r.json()}")
    except Exception as exc:
        print(f"ERROR: Cannot reach CREE server at {CREE_SERVER}: {exc}")
        print("Start it with:  uvicorn server.app:app --port 8000")
        sys.exit(1)

    env    = CREEEnv(CREE_SERVER)
    scores = {}

    for task_id in ["stability", "recovery", "cascade_prevention"]:
        print(f"\n{'─'*60}")
        print(f"Running task: {task_id}")
        print('─' * 60)
        result        = run_task(env, task_id)
        scores[task_id] = result.get("score", 0.0)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SCORES")
    print('=' * 60)
    for task_id, score in scores.items():
        difficulty = {"stability": "easy", "recovery": "medium",
                      "cascade_prevention": "hard"}.get(task_id, "?")
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id:22s} [{difficulty:6s}]  {bar}  {score:.4f}")

    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average score: {avg:.4f}")
    print('=' * 60)

    # Machine-readable output for automated validators
    print("\nJSON_SCORES:" + json.dumps(scores))


if __name__ == "__main__":
    main()
