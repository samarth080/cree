"""
CausalAgent — the core learning component of CREE.

Architecture
------------
1. CausalBeliefMap
   Tracks, for every (action, observable_status) pair, the distribution of
   changes in each observable variable.  After MIN_OBS samples it begins
   forming "causal hypotheses" — strings that describe discovered rules.
   It can also PREDICT the next state for a given (state, action).

2. CausalAgent
   Uses UCB1 exploration over a Q-table.  Proceeds through four phases:

   Phase 1 — Blind Exploration   (steps   0–150)
     Nearly random, populating the belief map.

   Phase 2 — Pattern Detection   (steps 150–350)
     UCB-guided; causal map starts highlighting anomalies.

   Phase 3 — Hypothesis Testing  (steps 350–550)
     Deliberately probes under-explored (action, state) pairs to confirm
     or reject hypotheses.

   Phase 4 — Causal Mastery      (steps 550+)
     Strategic action selection based on the learned causal model.
     Prioritises stabilisation, avoidance of armed-trigger injection, etc.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.client import CREEClient
from models import ObservableState, ACTION_NAMES


# ---------------------------------------------------------------------------
# CausalBeliefMap
# ---------------------------------------------------------------------------

@dataclass
class EffectRecord:
    """Running statistics for one (action, status, variable) triplet."""
    deltas: List[float] = field(default_factory=list)

    def record(self, delta: float):
        self.deltas.append(delta)

    @property
    def n(self) -> int:
        return len(self.deltas)

    @property
    def mean(self) -> float:
        return sum(self.deltas) / self.n if self.n else 0.0

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean
        return math.sqrt(sum((x - m) ** 2 for x in self.deltas) / (self.n - 1))


class CausalBeliefMap:
    """
    Tracks cause-effect relationships derived from interaction.

    Internal structure:
        effects[action][status][variable] → EffectRecord
        transitions[action][from_status][to_status] → count

    After MIN_OBS samples for a (action, status) bucket the map:
      • Forms hypotheses (tentative rules)
      • Promotes stable hypotheses to verified_rules
      • Can predict next ObservableState
    """

    MIN_OBS        = 3    # samples before forming hypothesis
    PROMOTE_OBS    = 7    # samples to promote hypothesis → verified rule
    LAT_THRESHOLD  = 30.0 # ms change considered significant
    ERR_THRESHOLD  = 0.08
    THR_THRESHOLD  = 80.0
    HIGH_VAR_RATIO = 1.2  # std/|mean| above this → hidden variable suspected

    def __init__(self):
        # effects[action][status][variable] → EffectRecord
        self.effects: Dict[str, Dict[str, Dict[str, EffectRecord]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(EffectRecord))
        )
        # transitions[action][from_status][to_status] → count
        self.transitions: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self.total_obs: Dict[str, int] = defaultdict(int)
        self.hypotheses: List[str] = []
        self.verified_rules: List[str] = []
        # Track high-variance signatures (suspected hidden variable influence)
        self.high_variance_flags: List[str] = []

    def record(self, action: str, prev: ObservableState, nxt: ObservableState):
        st = prev.status

        for var, delta in [
            ('latency',    nxt.latency    - prev.latency),
            ('error_rate', nxt.error_rate - prev.error_rate),
            ('throughput', nxt.throughput - prev.throughput),
            ('cpu_load',   nxt.cpu_load   - prev.cpu_load),
        ]:
            self.effects[action][st][var].record(delta)

        self.transitions[action][st][nxt.status] += 1
        self.total_obs[action] += 1

        self._try_extract_insights(action, st)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def can_predict(self, action: str, status: str) -> bool:
        rec = self.effects[action][status].get('latency')
        return rec is not None and rec.n >= self.MIN_OBS

    def predict(self, state: ObservableState, action: str) -> Optional[ObservableState]:
        st = state.status
        if not self.can_predict(action, st):
            return None

        def adj(var, base, clamp_lo, clamp_hi) -> float:
            rec = self.effects[action][st].get(var)
            if rec is None or rec.n == 0:
                return base
            return max(clamp_lo, min(clamp_hi, base + rec.mean))

        # Most likely next status via mode of transition counts
        trans = self.transitions[action][st]
        pred_status = max(trans, key=trans.get) if trans else st

        return ObservableState(
            latency=adj('latency',    state.latency,    5.0,  600.0),
            error_rate=adj('error_rate', state.error_rate, 0.0, 1.0),
            throughput=adj('throughput', state.throughput, 0.0, 1000.0),
            cpu_load=adj('cpu_load',   state.cpu_load,   0.0, 1.0),
            status=pred_status,
        )

    def prediction_accuracy(self, predicted: ObservableState, actual: ObservableState) -> float:
        lat_err = abs(predicted.latency    - actual.latency)    / 200.0
        err_err = abs(predicted.error_rate - actual.error_rate)
        thr_err = abs(predicted.throughput - actual.throughput) / 500.0
        cpu_err = abs(predicted.cpu_load   - actual.cpu_load)
        status_ok = 1.0 if predicted.status == actual.status else 0.0

        num_acc = 1.0 - min(1.0, (lat_err + err_err + thr_err + cpu_err) / 4.0)
        return round((num_acc * 0.6 + status_ok * 0.4), 3)

    # ------------------------------------------------------------------
    # Rule extraction
    # ------------------------------------------------------------------

    def _try_extract_insights(self, action: str, status: str):
        eff = self.effects[action][status]
        lat = eff.get('latency')
        err = eff.get('error_rate')
        thr = eff.get('throughput')

        if lat is None or lat.n < self.MIN_OBS:
            return

        n = lat.n
        # candidate_keys: stable string keys for dedup; candidate_rules: display strings
        candidate_keys:  List[str] = []
        candidate_rules: List[str] = []

        # Significant latency effect
        if abs(lat.mean) > self.LAT_THRESHOLD:
            direction = "raises" if lat.mean > 0 else "lowers"
            key  = f"[LAT] '{action}' in '{status}' → {direction} latency"
            disp = f"{key} ~{abs(lat.mean):.0f}ms"
            candidate_keys.append(key)
            candidate_rules.append(disp)

        # Significant error_rate effect
        if err and abs(err.mean) > self.ERR_THRESHOLD:
            direction = "raises" if err.mean > 0 else "lowers"
            key  = f"[ERR] '{action}' in '{status}' → {direction} error_rate"
            disp = f"{key} ~{abs(err.mean):.3f}"
            candidate_keys.append(key)
            candidate_rules.append(disp)

        # Significant throughput effect
        if thr and abs(thr.mean) > self.THR_THRESHOLD:
            direction = "drops" if thr.mean < 0 else "boosts"
            key  = f"[THR] '{action}' in '{status}' → {direction} throughput"
            disp = f"{key} ~{abs(thr.mean):.0f} rps"
            candidate_keys.append(key)
            candidate_rules.append(disp)

        # High variance flag — evidence of hidden variable
        if lat.std > 0:
            ratio = lat.std / (abs(lat.mean) + 1e-6)
            flag_key = f"[HID] '{action}' in '{status}' high variance"
            if ratio > self.HIGH_VAR_RATIO and flag_key not in self.high_variance_flags:
                self.high_variance_flags.append(
                    f"[HID] '{action}' in '{status}' has HIGH latency variance "
                    f"(σ={lat.std:.0f}) — hidden variable suspected"
                )

        # Status transitions
        trans = self.transitions[action][status]
        total_trans = sum(trans.values())
        if total_trans >= self.MIN_OBS:
            for to_st, cnt in trans.items():
                rate = cnt / total_trans
                if to_st != status and rate >= 0.40:
                    key  = f"[MOD] '{action}' in '{status}' → '{to_st}'"
                    disp = f"{key} ({rate:.0%} of cases)"
                    candidate_keys.append(key)
                    candidate_rules.append(disp)

        # Promote hypothesis → verified, or add as new hypothesis
        for key, disp in zip(candidate_keys, candidate_rules):
            # Check if already verified (by key prefix)
            already = any(v.startswith(key) for v in self.verified_rules)
            if already:
                continue
            in_hyp = any(h.startswith(key) for h in self.hypotheses)
            if in_hyp:
                if n >= self.PROMOTE_OBS:
                    # Remove old hypothesis entry and add verified
                    self.hypotheses = [h for h in self.hypotheses if not h.startswith(key)]
                    self.verified_rules.append(disp)
            else:
                self.hypotheses.append(disp)

    def summary(self) -> str:
        lines = [f"Causal Belief Map  |  {len(self.verified_rules)} verified rules  |  {len(self.hypotheses)} hypotheses"]
        for r in self.verified_rules:
            lines.append(f"  ✓  {r}")
        if self.high_variance_flags:
            lines.append("Hidden variable suspects:")
            for f in self.high_variance_flags[-3:]:
                lines.append(f"  ?  {f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CausalAgent
# ---------------------------------------------------------------------------

PHASE_BOUNDARIES = [
    (0,   150, 1, "Blind Exploration"),
    (150, 350, 2, "Pattern Detection"),
    (350, 550, 3, "Hypothesis Testing"),
    (550, int(1e9), 4, "Causal Mastery"),
]


class CausalAgent:

    def __init__(self, client: CREEClient):
        self.client = client
        self.belief_map = CausalBeliefMap()

        # Q-table: state_key → action → value
        self.q: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {a: 0.0 for a in ACTION_NAMES}
        )
        # UCB visit counts: state_key → action → count
        self.visits: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {a: 0 for a in ACTION_NAMES}
        )

        self.total_steps    = 0
        self.episode_count  = 0
        self.exploration    = 1.0
        self.lr             = 0.15
        self.gamma          = 0.9

        # Metrics
        self.pred_accuracy_history: List[float] = []
        self.reward_history:        List[float] = []

    # ------------------------------------------------------------------
    # Phase logic
    # ------------------------------------------------------------------

    @property
    def phase(self) -> Tuple[int, str]:
        for lo, hi, num, name in PHASE_BOUNDARIES:
            if lo <= self.total_steps < hi:
                return num, name
        return 4, "Causal Mastery"

    # ------------------------------------------------------------------
    # State discretisation
    # ------------------------------------------------------------------

    def _sk(self, s: ObservableState) -> str:
        return (
            f"{s.status}|"
            f"{int(s.latency // 30)}|"
            f"{int(s.error_rate * 5)}|"
            f"{int(s.throughput // 120)}"
        )

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state: ObservableState) -> str:
        phase_num, _ = self.phase

        # Phase 4: strategic override based on causal model
        if phase_num >= 4:
            strategic = self._strategic_action(state)
            if strategic:
                return strategic

        # Phase 3: target under-sampled (action, status) pairs
        if phase_num == 3:
            least = self._least_sampled_action(state.status)
            if least:
                return least

        # ε-greedy with UCB exploration bonus
        if random.random() < self.exploration:
            return random.choice(ACTION_NAMES)

        sk = self._sk(state)
        total = sum(self.visits[sk].values()) + 1
        best, best_score = None, float('-inf')
        for a in ACTION_NAMES:
            q = self.q[sk][a]
            n = self.visits[sk][a] + 1
            score = q + 0.7 * math.sqrt(math.log(total) / n)
            if score > best_score:
                best_score = score
                best = a
        return best or random.choice(ACTION_NAMES)

    def _strategic_action(self, state: ObservableState) -> Optional[str]:
        """Phase-4 rule-based override derived from the causal belief map."""
        bm = self.belief_map

        # If in cascade/warning: check if reset_connections works here
        if state.status == 'warning':
            trans = bm.transitions.get('reset_connections', {}).get(state.status, {})
            if trans:
                normal_rate = trans.get('normal', 0) / (sum(trans.values()) + 1)
                if normal_rate >= 0.3:
                    return 'reset_connections'
            # Stabilize is likely effective — but only if risk isn't too high
            # We can infer risk < 6 from latency < ~110 (10 + 6*8 = 58 + mode offset)
            if state.latency < 110:
                return 'stabilize'
            return 'wait'

        if state.status == 'critical':
            return 'emergency_stop'

        if state.status == 'recovering':
            return 'wait'

        return None

    def _least_sampled_action(self, status: str) -> Optional[str]:
        """Return action with fewest samples in this status context."""
        min_n, min_action = float('inf'), None
        for a in ACTION_NAMES:
            rec = self.belief_map.effects[a][status].get('latency')
            n = rec.n if rec else 0
            if n < min_n:
                min_n = n
                min_action = a
        return min_action if min_n < self.belief_map.MIN_OBS else None

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(
        self,
        state: ObservableState,
        action: str,
        reward: float,
        next_state: ObservableState,
        done: bool,
    ):
        sk  = self._sk(state)
        nsk = self._sk(next_state)

        # Measure prediction accuracy before updating the belief map
        predicted = self.belief_map.predict(state, action)
        if predicted is not None:
            acc = self.belief_map.prediction_accuracy(predicted, next_state)
            self.pred_accuracy_history.append(acc)

        # Update causal belief map
        self.belief_map.record(action, state, next_state)

        # Q-learning
        cur_q     = self.q[sk][action]
        max_next  = max(self.q[nsk].values()) if not done else 0.0
        self.q[sk][action] = cur_q + self.lr * (reward + self.gamma * max_next - cur_q)

        self.visits[sk][action] += 1
        self.total_steps += 1
        self.exploration = max(0.05, self.exploration * 0.997)

    # ------------------------------------------------------------------
    # Episode runner
    # ------------------------------------------------------------------

    def run_episode(
        self,
        max_steps: int = 80,
        step_callback=None,
    ) -> dict:
        """
        Run one episode.

        step_callback(step_num, action, result, pred_acc) is called after
        each step if provided.
        """
        self.episode_count += 1
        state  = self.client.reset()
        total_reward = 0.0
        steps  = 0

        for step_num in range(max_steps):
            action     = self.choose_action(state)
            prediction = self.belief_map.predict(state, action)

            result    = self.client.step(action)
            pred_acc  = None
            if prediction is not None:
                pred_acc = self.belief_map.prediction_accuracy(prediction, result.state)

            self.learn(state, action, result.reward, result.state, result.done)

            if step_callback:
                step_callback(step_num, action, result, pred_acc)

            state        = result.state
            total_reward += result.reward
            steps        += 1

            if result.done:
                break

        self.reward_history.append(total_reward)

        recent_acc = self.pred_accuracy_history[-50:]
        avg_acc    = sum(recent_acc) / len(recent_acc) if recent_acc else 0.0

        return {
            'episode':         self.episode_count,
            'total_reward':    round(total_reward, 3),
            'steps':           steps,
            'phase':           self.phase,
            'exploration':     round(self.exploration, 3),
            'rules_verified':  len(self.belief_map.verified_rules),
            'hypotheses':      len(self.belief_map.hypotheses),
            'avg_pred_acc':    round(avg_acc, 3),
            'total_steps':     self.total_steps,
        }
