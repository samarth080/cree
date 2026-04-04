"""
CREE Demo — hackathon-grade terminal presentation.

Run order:
  1. Start server:  uvicorn server.app:app --port 8000
  2. Run demo:      python demo.py

What judges see:
  Phase 1 — Agent acts randomly, logs every step
  Phase 2 — Causal map starts filling, first rules emerge
  Phase 3 — Hypothesis testing: targeted exploration
  Phase 4 — Mastery: intentional failure + intentional prevention

Press Ctrl-C to abort at any time.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console    import Console
from rich.table      import Table
from rich.panel      import Panel
from rich.rule       import Rule
from rich            import box
from rich.text       import Text

from agent.agent  import CausalAgent
from client.client import CREEClient

console = Console()

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _sc(status: str) -> str:
    return {'normal': 'green', 'warning': 'yellow',
            'critical': 'red', 'recovering': 'cyan'}.get(status, 'white')

def _risk_color(risk) -> str:
    if isinstance(risk, (int, float)):
        return 'green' if risk < 4 else ('yellow' if risk < 7 else 'red')
    return 'white'

def _bar(value: float, width: int = 20, full='█', empty='░') -> str:
    filled = max(0, min(width, int(value * width)))
    return full * filled + empty * (width - filled)


def render_step(step_num: int, action: str, result, pred_acc, show_hidden: bool = False):
    s = result.state
    r = result.reward
    sc = _sc(s.status)
    rc = 'green' if r >= 0 else 'red'

    line = (
        f"  [{sc}]{s.status:10s}[/{sc}]  "
        f"step={step_num:3d}  "
        f"[bold cyan]{action:22s}[/bold cyan]  "
        f"R=[{rc}]{r:+.2f}[/{rc}]  "
        f"lat={s.latency:5.0f}ms  "
        f"err={s.error_rate:.3f}  "
        f"thr={s.throughput:5.0f}  "
        f"cpu={s.cpu_load:.2f}"
    )

    if pred_acc is not None:
        pac = 'green' if pred_acc > 0.7 else ('yellow' if pred_acc > 0.5 else 'red')
        line += f"  pred=[{pac}]{pred_acc:.0%}[/{pac}]"

    if result.info.get('is_new_state'):
        line += "  [bold yellow]★ NEW[/bold yellow]"

    console.print(line)

    if show_hidden:
        info = result.info
        risk = info.get('_hidden_risk', '?')
        mode = info.get('_hidden_mode', '?')
        trigger = info.get('_hidden_trigger', False)
        cascade = info.get('_hidden_cascade', 0)
        memory  = info.get('_hidden_memory', '?')
        rc2 = _risk_color(risk)
        tc  = 'red bold' if trigger else 'dim green'
        console.print(
            f"  [dim]  └─ hidden: mode=[italic]{mode}[/italic]  "
            f"risk=[{rc2}]{risk}[/{rc2}]  "
            f"mem={memory}  "
            f"trigger=[{tc}]{'ARMED' if trigger else 'safe'}[/{tc}]  "
            f"cascade_t={cascade}[/dim]"
        )


def render_episode_summary(result: dict):
    ep   = result['episode']
    ph_n, ph_name = result['phase']
    rew  = result['total_reward']
    stps = result['steps']
    rv   = result['rules_verified']
    hyp  = result['hypotheses']
    acc  = result['avg_pred_acc']
    expl = result['exploration']

    rc = 'green' if rew >= 0 else 'red'
    ph_colors = {1: 'dim', 2: 'yellow', 3: 'cyan', 4: 'bold green'}
    ph_c = ph_colors.get(ph_n, 'white')

    console.print(
        f"  Ep [bold]{ep:2d}[/bold]  [{ph_c}]Phase {ph_n}: {ph_name}[/{ph_c}]  "
        f"reward=[{rc}]{rew:+7.2f}[/{rc}]  "
        f"steps={stps:3d}  "
        f"rules=[cyan]{rv:2d}[/cyan]  "
        f"hyp=[dim]{hyp:2d}[/dim]  "
        f"pred_acc=[yellow]{acc:.0%}[/yellow]  "
        f"ε={expl:.3f}"
    )


def render_causal_map(agent: CausalAgent):
    bm = agent.belief_map

    if not bm.verified_rules and not bm.hypotheses and not bm.high_variance_flags:
        console.print("[dim]  No causal knowledge yet.[/dim]")
        return

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    table.add_column("Status", style="dim", width=8)
    table.add_column("Rule", no_wrap=False)

    for r in bm.verified_rules:
        tag = r[:5]
        body = r[7:]
        color_map = {'[LAT]': 'yellow', '[ERR]': 'red', '[THR]': 'blue',
                     '[MOD]': 'magenta', '[HID]': 'orange3'}
        col = color_map.get(tag, 'white')
        table.add_row(f"[{col}]{tag}[/{col}]", f"[green]✓[/green] {body}")

    for r in bm.hypotheses[-5:]:
        tag = r[:5]
        body = r[7:]
        table.add_row(f"[dim]{tag}[/dim]", f"[dim]? {body}[/dim]")

    for r in bm.high_variance_flags[-2:]:
        body = r[7:]
        table.add_row("[orange3][HID][/orange3]", f"[orange3]⚠ {body}[/orange3]")

    console.print(table)


def render_prediction_accuracy(agent: CausalAgent):
    hist = agent.pred_accuracy_history
    if not hist:
        return
    recent = hist[-50:]
    avg    = sum(recent) / len(recent)
    bar    = _bar(avg, 28)
    col    = 'green' if avg > 0.70 else ('yellow' if avg > 0.50 else 'red')
    console.print(
        f"  Prediction Accuracy: [{col}]{bar}[/{col}]  [{col}]{avg:.1%}[/{col}]"
        f"  (last {len(recent)} predictions)"
    )


# ---------------------------------------------------------------------------
# Phase banner
# ---------------------------------------------------------------------------

def phase_banner(phase_num: int, phase_name: str, episode: int):
    icons   = {1: '🔍', 2: '🧩', 3: '🧪', 4: '🎯'}
    colors  = {1: 'dim white', 2: 'yellow', 3: 'cyan', 4: 'bold green'}
    icon    = icons.get(phase_num, '•')
    color   = colors.get(phase_num, 'white')
    console.print()
    console.print(Rule(
        f"[{color}]{icon}  Phase {phase_num}: {phase_name}  — Episode {episode}[/{color}]",
        style=color
    ))


# ---------------------------------------------------------------------------
# Mastery demonstration
# ---------------------------------------------------------------------------

def mastery_demo(client: CREEClient, agent: CausalAgent):
    console.print()
    console.print(Rule("[bold green]MASTERY DEMONSTRATION[/bold green]", style="bold green"))

    # ── Demo A: Intentional cascade trigger ─────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold yellow]Demo A — Intentional Cascade Trigger[/bold yellow]\n"
        "[dim]Agent arms the hidden trigger, then injects load to force cascade failure.[/dim]",
        box=box.ROUNDED, expand=False
    ))

    client.reset()
    sequence = ['toggle_debug', 'stress_cpu', 'inject_load', 'inject_load', 'inject_load']
    for action in sequence:
        result = client.step(action)
        s = result.state
        sc = _sc(s.status)
        info = result.info
        risk = info.get('_hidden_risk', '?')
        trigger = info.get('_hidden_trigger', False)
        cascade = info.get('_hidden_cascade', 0)
        tc = 'red bold' if trigger else 'dim green'
        console.print(
            f"  [{sc}]{s.status:10s}[/{sc}]  "
            f"[cyan]{action:22s}[/cyan]  "
            f"lat={s.latency:5.0f}ms  err={s.error_rate:.3f}  "
            f"risk=[{_risk_color(risk)}]{risk}[/{_risk_color(risk)}]  "
            f"trigger=[{tc}]{'ARMED' if trigger else 'safe'}[/{tc}]  "
            f"cascade_t={cascade}"
        )
        time.sleep(0.15)
        if result.done:
            console.print("  [bold red]💥  FAILURE TRIGGERED — exactly as predicted![/bold red]")
            break

    time.sleep(0.5)

    # ── Demo B: Warning detected → failure averted ──────────────────────────
    console.print()
    console.print(Panel(
        "[bold cyan]Demo B — Failure Prevention[/bold cyan]\n"
        "[dim]Agent detects early warning signals and applies the learned stabilisation protocol.[/dim]",
        box=box.ROUNDED, expand=False
    ))

    client.reset()
    # Build up risk
    build_sequence = ['stress_cpu', 'probe_latency', 'stress_cpu', 'inject_load']
    for action in build_sequence:
        result = client.step(action)
        s = result.state
        sc = _sc(s.status)
        info = result.info
        risk = info.get('_hidden_risk', '?')
        console.print(
            f"  [{sc}]{s.status:10s}[/{sc}]  [dim]{action:22s}[/dim]  "
            f"lat={s.latency:5.0f}ms  risk={risk}"
        )
        time.sleep(0.1)

    if result.state.status in ('warning', 'critical'):
        console.print()
        console.print("  [bold yellow]⚠  WARNING DETECTED — engaging learned stabilisation[/bold yellow]")
        # Use agent's strategic knowledge
        state = result.state
        fix_sequence = []
        for _ in range(6):
            action = agent.choose_action(state)
            result = client.step(action)
            fix_sequence.append(action)
            s = result.state
            sc = _sc(s.status)
            info = result.info
            risk = info.get('_hidden_risk', '?')
            console.print(
                f"  [{sc}]{s.status:10s}[/{sc}]  [bold cyan]{action:22s}[/bold cyan]  "
                f"lat={s.latency:5.0f}ms  risk={risk}"
            )
            time.sleep(0.12)
            state = result.state
            if state.status == 'normal':
                console.print()
                console.print("  [bold green]✓  System stabilised — failure averted![/bold green]")
                break
    else:
        console.print("  [dim]System stayed stable — agent successfully navigated safely.[/dim]")

    time.sleep(0.5)

    # ── Demo C: Prediction showcase ──────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold magenta]Demo C — Live Prediction Accuracy[/bold magenta]\n"
        "[dim]Agent predicts next state before acting; we compare to reality.[/dim]",
        box=box.ROUNDED, expand=False
    ))

    client.reset()
    state = client.get_state()

    pred_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    pred_table.add_column("Action",      style="cyan",   width=22)
    pred_table.add_column("Pred Status", style="yellow", width=12)
    pred_table.add_column("Real Status", style="green",  width=12)
    pred_table.add_column("Pred Lat",    style="yellow", width=10)
    pred_table.add_column("Real Lat",    style="green",  width=10)
    pred_table.add_column("Accuracy",    style="bold",   width=10)

    probe_actions = ['probe_latency', 'stress_cpu', 'wait', 'stabilize',
                     'force_gc', 'probe_memory', 'reset_connections']

    for action in probe_actions:
        bm        = agent.belief_map
        predicted = bm.predict(state, action)
        result    = client.step(action)
        actual    = result.state

        if predicted:
            acc = bm.prediction_accuracy(predicted, actual)
            acc_col = 'green' if acc > 0.70 else ('yellow' if acc > 0.50 else 'red')
            pred_table.add_row(
                action,
                predicted.status,
                actual.status,
                f"{predicted.latency:.0f}ms",
                f"{actual.latency:.0f}ms",
                f"[{acc_col}]{acc:.0%}[/{acc_col}]",
            )
        else:
            pred_table.add_row(action, "[dim]?[/dim]", actual.status,
                               "[dim]?[/dim]", f"{actual.latency:.0f}ms", "[dim]no data[/dim]")

        state = result.state
        if result.done:
            break

    console.print(pred_table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    console.print()
    console.print(Panel.fit(
        Text.assemble(
            ("CREE — Causal Reverse Engineering Engine\n", "bold cyan"),
            ("Discovering hidden cause-effect relationships in a black-box system\n", "dim"),
            ("\nThe agent starts clueless. By the end it can predict, trigger,\n"
             "and prevent failures — without ever seeing hidden state.", "italic"),
        ),
        box=box.DOUBLE_EDGE,
        padding=(1, 4),
    ))
    console.print()

    client = CREEClient()

    # Server connection check
    try:
        client.health()
        console.print("[green]✓  Server reachable[/green]\n")
    except Exception as exc:
        console.print(f"[bold red]✗  Cannot reach server: {exc}[/bold red]")
        console.print("[yellow]Start it with:  uvicorn server.app:app --port 8000[/yellow]")
        sys.exit(1)

    agent = CausalAgent(client)

    N_EPISODES          = 14
    VERBOSE_EPISODES    = {1, 2, 7, 14}   # full step-by-step trace
    SHOW_HIDDEN_EPS     = {1, 14}          # also expose hidden state for these
    MAP_PRINT_EPISODES  = {3, 6, 9, 12, 14}

    prev_phase = 0

    for ep_num in range(1, N_EPISODES + 1):
        ph_num, ph_name = agent.phase

        # Print phase banner on transitions
        if ph_num != prev_phase:
            phase_banner(ph_num, ph_name, ep_num)
            prev_phase = ph_num

        verbose     = ep_num in VERBOSE_EPISODES
        show_hidden = ep_num in SHOW_HIDDEN_EPS

        if verbose:
            console.print(f"\n[bold]── Episode {ep_num} detail ──[/bold]")

        def _cb(step_num, action, result, pred_acc):
            if verbose:
                render_step(step_num, action, result, pred_acc, show_hidden)

        result = agent.run_episode(max_steps=70, step_callback=_cb)
        render_episode_summary(result)

        if ep_num in MAP_PRINT_EPISODES:
            console.print()
            console.print(f"[bold]Causal Belief Map after episode {ep_num}:[/bold]")
            render_causal_map(agent)
            render_prediction_accuracy(agent)
            console.print()

    # ── Mastery Demo ─────────────────────────────────────────────────────────
    mastery_demo(client, agent)

    # ── Final summary ────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]FINAL SUMMARY[/bold cyan]", style="cyan"))

    bm = agent.belief_map
    console.print(f"\n  Episodes run:         {agent.episode_count}")
    console.print(f"  Total steps taken:    {agent.total_steps}")
    console.print(f"  Verified causal rules:[bold cyan] {len(bm.verified_rules)}[/bold cyan]")
    console.print(f"  Hypotheses formed:    {len(bm.hypotheses)}")
    console.print(f"  Hidden var suspects:  {len(bm.high_variance_flags)}")

    if agent.pred_accuracy_history:
        last50 = agent.pred_accuracy_history[-50:]
        avg = sum(last50) / len(last50)
        console.print(f"  Final pred accuracy: [bold green]{avg:.1%}[/bold green]")

    reward_trend = "  →  ".join(f"{r:+.1f}" for r in agent.reward_history)
    console.print(f"\n  Reward trend:  {reward_trend}")

    console.print()
    console.print("[bold]Final Causal Map:[/bold]")
    render_causal_map(agent)
    console.print()


if __name__ == "__main__":
    main()
