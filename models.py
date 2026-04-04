from dataclasses import dataclass, field
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Observable state — what the agent sees through the API
# ---------------------------------------------------------------------------

@dataclass
class ObservableState:
    latency: float      # response time in ms (10–500)
    error_rate: float   # fraction of failed requests (0–1)
    throughput: float   # requests per second (0–1000)
    cpu_load: float     # cpu utilization (0–1)
    status: str         # 'normal' | 'warning' | 'critical' | 'recovering'


# ---------------------------------------------------------------------------
# Hidden state — internal variables the agent cannot directly observe
# ---------------------------------------------------------------------------

@dataclass
class HiddenState:
    system_mode: str        # 'stable' | 'stressed' | 'cascade' | 'failing' | 'recovering'
    risk_level: float       # internal pressure 0–10
    memory_pressure: float  # memory saturation 0–10
    trigger_armed: bool     # armed by toggle_debug; makes inject_load catastrophic
    cascade_counter: int    # countdown to failure once cascade starts (0–5)
    recovery_steps: int     # steps remaining in post-failure recovery (0–3)
    consecutive_stress: int # number of consecutive stress actions (for memory bomb rule)
    last_actions: List[str] = field(default_factory=list)  # last 3 actions taken


@dataclass
class EnvironmentState:
    observable: ObservableState
    hidden: HiddenState
    step_count: int = 0


# ---------------------------------------------------------------------------
# Actions — the interface between agent and environment
# ---------------------------------------------------------------------------

@dataclass
class Action:
    name: str
    description: str
    category: str   # 'probe' | 'stress' | 'control' | 'repair' | 'neutral'


ACTIONS: List[Action] = [
    Action('probe_latency',    'Lightweight latency probe; slight risk increase',              'probe'),
    Action('stress_cpu',       'CPU stress test; increases risk and memory pressure',          'stress'),
    Action('inject_load',      'High traffic injection; severe risk; triggers cascade if armed','stress'),
    Action('wait',             'Idle period; allows risk and memory to decay',                 'neutral'),
    Action('reset_connections','Reset connection pool; helpful in stressed state, harmful in stable', 'control'),
    Action('force_gc',         'Force garbage collection; reduces memory but may cause GC pause','repair'),
    Action('probe_memory',     'Observe memory-related signals; neutral effect',               'probe'),
    Action('toggle_debug',     'Toggle debug mode; arms or disarms the cascade trigger',       'control'),
    Action('stabilize',        'Attempt risk reduction; only effective when risk < 6',         'repair'),
    Action('emergency_stop',   'Hard reset; zeroes risk/memory but kills throughput briefly',  'repair'),
]

ACTION_NAMES: List[str] = [a.name for a in ACTIONS]


# ---------------------------------------------------------------------------
# Step result — returned by /step
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    state: ObservableState
    reward: float
    done: bool
    info: Dict[str, Any]
