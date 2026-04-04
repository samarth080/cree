"""
CREE Task Graders — three tasks, each with a deterministic scorer (0.0–1.0).

Task 1  — stability          (easy)    Keep system healthy for 25 steps
Task 2  — recovery           (medium)  Recover from a degraded state within 20 steps
Task 3  — cascade_prevention (hard)    Prevent cascade failure for 30 steps

Each grader is a pure function:
    grade_*(metrics: dict, max_steps: int) -> float  in [0.0, 1.0]

metrics is the 'episode_metrics' dict from env.episode_metrics at episode end.
"""

from typing import Dict, Any


# ---------------------------------------------------------------------------
# Task metadata (mirrors TASK_CONFIGS in environment.py — single source of
# truth for grader logic is here; environment provides initial conditions)
# ---------------------------------------------------------------------------

TASKS = {
    "stability": {
        "id":          "stability",
        "name":        "Steady-State Maintenance",
        "difficulty":  "easy",
        "max_steps":   25,
        "description": (
            "You are an on-call SRE. Keep the production system in 'normal' status "
            "for as many of the 25 steps as possible. Any failure event halves your score."
        ),
    },
    "recovery": {
        "id":          "recovery",
        "name":        "Incident Recovery",
        "difficulty":  "medium",
        "max_steps":   20,
        "description": (
            "The system is already degraded (warning state, risk=6.5). "
            "Restore it to 'normal' status as quickly as possible and keep it stable. "
            "You have 20 steps."
        ),
    },
    "cascade_prevention": {
        "id":          "cascade_prevention",
        "name":        "Cascade Failure Prevention",
        "difficulty":  "hard",
        "max_steps":   30,
        "description": (
            "A hidden debug trigger is armed and risk is rising. "
            "Prevent any cascade failure for 30 steps. "
            "Disarming the trigger early earns a bonus."
        ),
    },
}


# ---------------------------------------------------------------------------
# Grader: Task 1 — Stability
# ---------------------------------------------------------------------------

def grade_stability(metrics: Dict[str, Any], max_steps: int = 25) -> float:
    """
    Score = fraction of steps spent in 'normal' status.
    Penalty: each failure event multiplies score by 0.5.

    Perfect score (1.0): 25/25 steps in normal, zero failures.
    """
    total   = max(1, metrics.get("total_steps", 1))
    normal  = metrics.get("steps_in_normal", 0)
    fails   = metrics.get("failures", 0)

    base    = min(1.0, normal / max_steps)
    penalty = 0.5 ** fails          # each failure halves the score
    return round(base * penalty, 4)


# ---------------------------------------------------------------------------
# Grader: Task 2 — Recovery
# ---------------------------------------------------------------------------

def grade_recovery(metrics: Dict[str, Any], max_steps: int = 20) -> float:
    """
    Score has two components (50/50):
      • Speed    = 1 - (recovery_step / max_steps)   [how fast you recovered]
      • Stability = (steps_in_normal / max_steps)     [how long you held it]

    If no recovery at all: score = 0.05 * (steps_in_normal / max_steps)  (partial credit)
    Failures additionally apply a 0.6 multiplier each.
    """
    total          = max(1, metrics.get("total_steps", 1))
    normal         = metrics.get("steps_in_normal", 0)
    fails          = metrics.get("failures", 0)
    recovery_step  = metrics.get("recovery_step", None)  # None if never recovered

    if recovery_step is not None:
        speed_score     = max(0.0, 1.0 - recovery_step / max_steps)
        stability_score = min(1.0, normal / max_steps)
        base = 0.5 * speed_score + 0.5 * stability_score
    else:
        # Partial credit: proportional to any improvement observed
        base = 0.05 * min(1.0, normal / max_steps)

    penalty = 0.6 ** fails
    return round(base * penalty, 4)


# ---------------------------------------------------------------------------
# Grader: Task 3 — Cascade Prevention
# ---------------------------------------------------------------------------

def grade_cascade_prevention(metrics: Dict[str, Any], max_steps: int = 30) -> float:
    """
    Score = steps_survived / max_steps
    Bonus:  +0.2 if trigger was cleanly disarmed (capped at 1.0)
    Penalty: each failure multiplies by 0.4 (hard — failure here is very bad)

    Perfect score (1.0): survive all 30 steps + disarm trigger early.
    """
    total    = max(1, metrics.get("total_steps", 1))
    fails    = metrics.get("failures", 0)
    disarmed = metrics.get("trigger_disarmed_while_armed", False)

    survival = min(1.0, total / max_steps)
    bonus    = 0.2 if disarmed else 0.0
    penalty  = 0.4 ** fails

    return round(min(1.0, (survival + bonus) * penalty), 4)


# ---------------------------------------------------------------------------
# Unified grader dispatcher
# ---------------------------------------------------------------------------

GRADER_MAP = {
    "stability":          grade_stability,
    "recovery":           grade_recovery,
    "cascade_prevention": grade_cascade_prevention,
}


def grade(task_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the grader for task_id against episode_metrics.
    Returns {"task_id", "score", "metrics"}.
    """
    if task_id not in GRADER_MAP:
        raise ValueError(f"Unknown task '{task_id}'. Valid: {list(GRADER_MAP)}")

    task     = TASKS[task_id]
    fn       = GRADER_MAP[task_id]
    score    = fn(metrics, task["max_steps"])

    return {
        "task_id":    task_id,
        "task_name":  task["name"],
        "difficulty": task["difficulty"],
        "score":      score,
        "metrics":    metrics,
    }
