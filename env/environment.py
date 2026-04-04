"""
BlackBoxEnvironment — the core simulation for CREE.

Observable state  : latency, error_rate, throughput, cpu_load, status
Hidden state      : system_mode, risk_level, memory_pressure, trigger_armed,
                    cascade_counter, recovery_steps, consecutive_stress

Hidden causal rules (what the agent must DISCOVER):

  Rule 1 — Armed Trigger
    toggle_debug arms/disarms trigger_armed.
    inject_load while trigger_armed=True starts the cascade countdown.
    inject_load while trigger_armed=False just raises risk normally.

  Rule 2 — Memory Bomb
    3+ consecutive stress_cpu actions cause a non-linear memory spike (+3 extra).
    memory_pressure >= 8 transitions system into 'cascade' mode.

  Rule 3 — Wrong Tool Trap
    reset_connections reduces risk heavily when mode is 'stressed'.
    reset_connections INCREASES risk when mode is 'stable' (paradox!).

  Rule 4 — GC Interference
    force_gc reduces memory_pressure by 3.
    But if risk_level > 6 at time of call, GC pause adds +1.5 extra risk.

  Rule 5 — Stabilize Window
    stabilize reduces risk by 2 ONLY if risk_level < 6.
    If risk_level >= 6, stabilize is silently ineffective.
    stabilize also decrements cascade_counter by 1 if risk < 6.

  Rule 6 — Recovery Protocol
    After failure, system enters 'recovering' for RECOVERY_STEPS steps.
    wait during recovery is optimal (completes recovery faster).
    Any stress action during recovery resets the recovery countdown.

  Rule 7 — Cascade Countdown
    Once cascade_counter > 0, it decrements by 1 each step.
    Reaching 0 causes system failure.
    Only stabilize (when risk < 6) can pause it.
"""

import random
from dataclasses import replace
from typing import Tuple, Dict, Any, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ObservableState, HiddenState, EnvironmentState,
    StepResult, ACTION_NAMES,
)

# ---------------------------------------------------------------------------
# Task definitions — initial hidden state overrides per task
# ---------------------------------------------------------------------------

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "stability": {
        "description": "Keep the production system in 'normal' status for as many steps as possible.",
        "difficulty": "easy",
        "max_steps": 25,
        "initial_hidden": dict(
            system_mode='stable', risk_level=0.0, memory_pressure=0.0,
            trigger_armed=False, cascade_counter=0, recovery_steps=0,
            consecutive_stress=0, last_actions=[],
        ),
    },
    "recovery": {
        "description": "System is degraded (warning). Return it to 'normal' within 20 steps.",
        "difficulty": "medium",
        "max_steps": 20,
        "initial_hidden": dict(
            system_mode='stressed', risk_level=6.5, memory_pressure=3.0,
            trigger_armed=False, cascade_counter=0, recovery_steps=0,
            consecutive_stress=0, last_actions=[],
        ),
    },
    "cascade_prevention": {
        "description": "Trigger is armed and risk is rising. Prevent cascade failure for 30 steps.",
        "difficulty": "hard",
        "max_steps": 30,
        "initial_hidden": dict(
            system_mode='stressed', risk_level=5.0, memory_pressure=2.0,
            trigger_armed=True, cascade_counter=0, recovery_steps=0,
            consecutive_stress=0, last_actions=[],
        ),
    },
}


class BlackBoxEnvironment:

    RISK_MAX       = 10.0
    MEMORY_MAX     = 10.0
    CASCADE_INIT   = 5    # steps before cascade causes failure
    RECOVERY_STEPS = 3    # steps spent in recovering mode

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)
        self._visited_signatures: set = set()
        self.state: EnvironmentState = None   # type: ignore
        self.current_task: Optional[str] = None
        # Episode-level metrics for grading
        self.episode_metrics: Dict[str, Any] = {}
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> ObservableState:
        self.current_task = task_id

        if task_id and task_id in TASK_CONFIGS:
            h_cfg = TASK_CONFIGS[task_id]["initial_hidden"]
            hidden = HiddenState(**h_cfg)
        else:
            hidden = HiddenState(
                system_mode='stable', risk_level=0.0, memory_pressure=0.0,
                trigger_armed=False, cascade_counter=0, recovery_steps=0,
                consecutive_stress=0, last_actions=[],
            )

        self.state = EnvironmentState(
            observable=ObservableState(
                latency=10.0, error_rate=0.0, throughput=500.0,
                cpu_load=0.1, status='normal',
            ),
            hidden=hidden,
            step_count=0,
        )
        # Project correct initial observable from (possibly non-default) hidden state
        self._project_observable()

        # Reset episode metrics
        self.episode_metrics = {
            "steps_in_normal": 0,
            "failures":        0,
            "recovery_step":   None,   # first step where status became 'normal' again
            "trigger_disarmed_while_armed": False,
            "total_steps":     0,
        }
        return self.state.observable

    def step(self, action_name: str) -> StepResult:
        if action_name not in ACTION_NAMES:
            raise ValueError(
                f"Unknown action '{action_name}'. Valid: {ACTION_NAMES}"
            )

        # Snapshot pre-step observable for reward logic
        prev_obs = self._snapshot_observable()

        h = self.state.hidden

        # Track consecutive stress
        if action_name in ('stress_cpu', 'inject_load'):
            h.consecutive_stress += 1
        else:
            h.consecutive_stress = 0

        # Rolling history of last 3 actions
        h.last_actions = (h.last_actions + [action_name])[-3:]

        # Apply action → hidden state
        self._apply_action(action_name)

        # Hidden dynamics (mode transitions, cascade countdown)
        self._update_dynamics()

        # Project hidden → observable (with measurement noise)
        self._project_observable()

        # Compute reward
        reward, components = self._compute_reward(action_name, prev_obs)

        self.state.step_count += 1

        sig = self._signature()
        is_new = sig not in self._visited_signatures
        if is_new:
            self._visited_signatures.add(sig)

        done = self.state.hidden.system_mode == 'failing'

        # Track episode-level grading metrics
        m = self.episode_metrics
        m["total_steps"] = self.state.step_count
        if self.state.observable.status == 'normal':
            m["steps_in_normal"] += 1
            if m["recovery_step"] is None and self.current_task == "recovery":
                m["recovery_step"] = self.state.step_count
        if done:
            m["failures"] += 1
        if (action_name == 'toggle_debug'
                and not self.state.hidden.trigger_armed
                and self.current_task == "cascade_prevention"):
            m["trigger_disarmed_while_armed"] = True

        info: Dict[str, Any] = {
            'step': self.state.step_count,
            'is_new_state': is_new,
            'reward_components': components,
            'episode_metrics': dict(m),
            # Hidden state exposed for demo / judge visualization only
            '_hidden_mode':     self.state.hidden.system_mode,
            '_hidden_risk':     round(self.state.hidden.risk_level, 1),
            '_hidden_memory':   round(self.state.hidden.memory_pressure, 1),
            '_hidden_trigger':  self.state.hidden.trigger_armed,
            '_hidden_cascade':  self.state.hidden.cascade_counter,
        }

        return StepResult(self.state.observable, round(reward, 3), done, info)

    # ------------------------------------------------------------------
    # Internal: action effects
    # ------------------------------------------------------------------

    def _apply_action(self, action: str):
        h = self.state.hidden

        if action == 'probe_latency':
            # Light probe — minor risk
            h.risk_level = min(self.RISK_MAX, h.risk_level + 0.5)

        elif action == 'stress_cpu':
            # CPU stress — risk+2, memory+1
            # RULE 2: 3+ consecutive → extra memory spike
            h.risk_level = min(self.RISK_MAX, h.risk_level + 2.0)
            h.memory_pressure = min(self.MEMORY_MAX, h.memory_pressure + 1.0)
            if h.consecutive_stress >= 3:
                h.memory_pressure = min(self.MEMORY_MAX, h.memory_pressure + 3.0)

        elif action == 'inject_load':
            # High load — risk+3
            # RULE 1: if trigger_armed, start cascade countdown
            h.risk_level = min(self.RISK_MAX, h.risk_level + 3.0)
            if h.trigger_armed and h.cascade_counter == 0:
                h.cascade_counter = self.CASCADE_INIT

        elif action == 'wait':
            h.risk_level = max(0.0, h.risk_level - 1.0)
            h.memory_pressure = max(0.0, h.memory_pressure - 0.5)

        elif action == 'reset_connections':
            # RULE 3: context-sensitive — good in stressed, bad in stable
            if h.system_mode in ('stressed', 'cascade'):
                h.risk_level = max(0.0, h.risk_level - 3.0)
            else:
                h.risk_level = min(self.RISK_MAX, h.risk_level + 2.0)

        elif action == 'force_gc':
            # RULE 4: memory relief, but GC pause if high risk
            h.memory_pressure = max(0.0, h.memory_pressure - 3.0)
            if h.risk_level > 6.0:
                h.risk_level = min(self.RISK_MAX, h.risk_level + 1.5)
            else:
                h.risk_level = min(self.RISK_MAX, h.risk_level + 0.3)

        elif action == 'probe_memory':
            pass  # strictly neutral

        elif action == 'toggle_debug':
            # RULE 1: arm/disarm the trigger
            h.trigger_armed = not h.trigger_armed
            h.risk_level = min(self.RISK_MAX, h.risk_level + 0.3)

        elif action == 'stabilize':
            # RULE 5: effective only below risk threshold
            if h.risk_level < 6.0:
                h.risk_level = max(0.0, h.risk_level - 2.0)
                h.cascade_counter = max(0, h.cascade_counter - 1)
            # else: silently does nothing

        elif action == 'emergency_stop':
            # Full reset of pressure — throughput penalty via mode
            h.risk_level = 0.0
            h.memory_pressure = 0.0
            h.trigger_armed = False
            h.cascade_counter = 0
            h.consecutive_stress = 0
            if h.system_mode != 'stable':
                h.system_mode = 'recovering'
                h.recovery_steps = self.RECOVERY_STEPS

    # ------------------------------------------------------------------
    # Internal: dynamics (mode transitions)
    # ------------------------------------------------------------------

    def _update_dynamics(self):
        h = self.state.hidden
        last = h.last_actions[-1] if h.last_actions else ''

        # Recovery state: count down and transition back to stable
        if h.system_mode == 'recovering':
            # RULE 6: stress during recovery resets countdown
            if last in ('stress_cpu', 'inject_load'):
                h.recovery_steps = self.RECOVERY_STEPS
            else:
                h.recovery_steps = max(0, h.recovery_steps - 1)
            if h.recovery_steps == 0:
                h.system_mode = 'stable'
            return  # no further transitions during recovery

        # Cascade countdown
        if h.cascade_counter > 0:
            h.cascade_counter -= 1
            if h.cascade_counter == 0:
                h.system_mode = 'failing'
                return

        # Failure auto-transitions to recovery
        if h.system_mode == 'failing':
            h.system_mode = 'recovering'
            h.recovery_steps = self.RECOVERY_STEPS
            h.risk_level = max(0.0, h.risk_level - 5.0)
            return

        # Memory bomb → cascade mode
        if h.memory_pressure >= 8.0 and h.system_mode not in ('cascade', 'failing'):
            h.system_mode = 'cascade'
            if h.cascade_counter == 0:
                h.cascade_counter = self.CASCADE_INIT
            return

        # Risk-based mode transitions
        if h.risk_level >= 7.0:
            h.system_mode = 'stressed'
        elif h.risk_level >= 4.0 and h.system_mode == 'stable':
            h.system_mode = 'stressed'
        elif h.risk_level < 3.5 and h.system_mode == 'stressed':
            h.system_mode = 'stable'

    # ------------------------------------------------------------------
    # Internal: project hidden → observable with measurement noise
    # ------------------------------------------------------------------

    def _project_observable(self):
        h = self.state.hidden
        o = self.state.observable
        n = self._rng.gauss

        # Per-mode constants
        mode_lat   = {'stable': 0,  'stressed': 50, 'cascade': 150, 'failing': 420, 'recovering': 25}
        mode_thr_f = {'stable': 1.0,'stressed': 0.7,'cascade': 0.4,  'failing': 0.04,'recovering': 0.55}
        mode_err   = {'stable': 0.0,'stressed': 0.04,'cascade': 0.18,'failing': 0.88,'recovering': 0.08}
        mode_cpu_b = {'stable': 0.0,'stressed': 0.2, 'cascade': 0.4,  'failing': 0.9, 'recovering': 0.1}

        m = h.system_mode

        o.latency = max(5.0,
            10.0
            + h.risk_level * 8.0
            + h.memory_pressure * 5.0
            + mode_lat.get(m, 0)
            + (20.0 if h.trigger_armed else 0.0)   # armed trigger: observable latency bleed
            + n(0, 5.0)
        )

        o.error_rate = min(1.0, max(0.0,
            h.risk_level * 0.025
            + h.memory_pressure * 0.02
            + mode_err.get(m, 0.0)
            + (0.12 if h.trigger_armed else 0.0)   # armed trigger: observable error bleed
            + n(0, 0.015)
        ))

        base_thr = 500.0 * mode_thr_f.get(m, 1.0)
        o.throughput = max(0.0,
            base_thr
            - h.risk_level * 12.0
            - h.memory_pressure * 8.0
            + n(0, 15.0)
        )

        o.cpu_load = min(1.0, max(0.0,
            0.1
            + h.risk_level * 0.065
            + h.memory_pressure * 0.045
            + mode_cpu_b.get(m, 0.0)
            + n(0, 0.025)
        ))

        status_map = {
            'stable':     'normal',
            'stressed':   'warning',
            'cascade':    'warning',
            'failing':    'critical',
            'recovering': 'recovering',
        }
        o.status = status_map.get(m, 'normal')

    # ------------------------------------------------------------------
    # Internal: reward calculation
    # ------------------------------------------------------------------

    def _compute_reward(
        self, action: str, prev_obs: ObservableState
    ) -> Tuple[float, Dict[str, float]]:
        h = self.state.hidden
        o = self.state.observable
        c: Dict[str, float] = {}

        # 1. Discovery bonus — first time we reach this discretized state
        sig = self._signature()
        c['discovery'] = 2.0 if sig not in self._visited_signatures else 0.0

        # 2. Novelty reward — significant observable change (agent learns new behavior)
        lat_delta  = abs(o.latency     - prev_obs.latency)
        err_delta  = abs(o.error_rate  - prev_obs.error_rate)
        thr_delta  = abs(o.throughput  - prev_obs.throughput)
        novelty = 0.0
        if lat_delta > 60 or err_delta > 0.15 or thr_delta > 100:
            novelty = 0.5
        c['novelty'] = novelty

        # 3. Stabilization reward — brought system back from warning/critical
        if prev_obs.status in ('warning', 'critical') and o.status == 'normal':
            c['stabilized'] = 1.5
        else:
            c['stabilized'] = 0.0

        # 4. Survival reward — per step in healthy state
        c['survival'] = 0.1 if h.system_mode == 'stable' else 0.0

        # 5. Failure penalty
        c['failure'] = -3.0 if h.system_mode == 'failing' else 0.0

        # 6. Repetition penalty — same action 3x in a row
        if len(h.last_actions) >= 3 and len(set(h.last_actions[-3:])) == 1:
            c['repetition'] = -0.3
        else:
            c['repetition'] = 0.0

        # 7. Emergency stop cost (throughput nuke)
        c['emergency_cost'] = -0.4 if action == 'emergency_stop' else 0.0

        # 8. Predictive probing reward — acting on trigger_armed signal correctly
        # Agent gets reward if it disarms trigger when cascade is pending
        if action == 'toggle_debug' and not h.trigger_armed and h.cascade_counter > 0:
            c['smart_disarm'] = 1.0
        else:
            c['smart_disarm'] = 0.0

        total = sum(c.values())
        return total, c

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _signature(self) -> tuple:
        o = self.state.observable
        return (
            o.status,
            int(o.latency     // 40),
            int(o.error_rate  *  5),
            int(o.throughput  // 120),
        )

    def _snapshot_observable(self) -> ObservableState:
        o = self.state.observable
        return ObservableState(
            latency=o.latency,
            error_rate=o.error_rate,
            throughput=o.throughput,
            cpu_load=o.cpu_load,
            status=o.status,
        )
