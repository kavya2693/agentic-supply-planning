"""The agentic workflow: a supervisor plus four collaborating agents in LangGraph.

    supervisor ─► demand_agent ─► supervisor ─► inventory_agent ─► supervisor
               ─► po_agent (only if something needs ordering) ─► supervisor
               ─► workflow_agent ─► [human approval interrupt] ─► END

Agents share one state object. Each appends to an audit trail, so every decision can be
traced back to the forecast and policy that produced it."""

import operator
from datetime import UTC, datetime
from typing import Annotated, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from . import tools
from .config import AUTO_APPROVE_LIMIT, DEFAULT_DOMAIN
from .data import load
from .forecast import load_forecast
from .llm import get_llm


class State(TypedDict, total=False):
    domain: str
    refresh: bool
    interactive: bool
    done: Annotated[list[str], operator.add]
    audit: Annotated[list[dict], operator.add]
    metrics: dict
    health: list[dict]  # records, so the checkpointer can persist state
    pos: list[dict]
    budget: float
    kpis: dict
    notifications: list[str]
    briefing: str
    next: str


def _log(agent, action, **detail):
    return [
        {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "agent": agent, "action": action, **detail}
    ]


def _domain(state: State) -> str:
    return state.get("domain") or DEFAULT_DOMAIN


def _data(state: State):
    return load(_domain(state))


def _forecast(state: State, skus, sales, refresh=False):
    return load_forecast(skus, sales, _domain(state), refresh=refresh)


# ── Supervisor ───────────────────────────────────────────────────────────────
def supervisor(state: State) -> dict:
    done = set(state.get("done", []))
    if "demand" not in done:
        nxt = "demand_agent"
    elif "inventory" not in done:
        nxt = "inventory_agent"
    elif "po" not in done and any(h["status"] in ("critical", "reorder") for h in state["health"]):
        nxt = "po_agent"
    elif "workflow" not in done:
        nxt = "workflow_agent"
    else:
        nxt = END
    return {"next": nxt, "audit": _log("supervisor", "route", to=str(nxt))}


# ── 1. Demand agent ──────────────────────────────────────────────────────────
def demand_agent(state: State) -> dict:
    skus, sales, _ = _data(state)
    _, metrics = _forecast(state, skus, sales, refresh=state.get("refresh", False))
    beats = metrics["wape_model_p50"] < metrics["wape_seasonal_naive"]
    return {
        "done": ["demand"],
        "metrics": metrics,
        "audit": _log(
            "demand_agent",
            "forecast",
            skus=metrics["skus"],
            wape=metrics["wape_model_p50"],
            beats_baseline=bool(beats),
        ),
    }


# ── 2. Inventory monitoring agent ────────────────────────────────────────────
def inventory_agent(state: State) -> dict:
    skus, sales, inv = _data(state)
    fc, _ = _forecast(state, skus, sales)
    health = tools.inventory_health(skus, inv, fc)
    counts = health.status.value_counts().to_dict()
    return {
        "done": ["inventory"],
        "health": health.to_dict("records"),
        "audit": _log("inventory_agent", "classify_stock", **{k: int(v) for k, v in counts.items()}),
    }


# ── 3. Purchase-order agent ──────────────────────────────────────────────────
def po_agent(state: State) -> dict:
    skus, sales, _ = _data(state)
    fc, _ = _forecast(state, skus, sales)
    budget = tools.cycle_budget(skus, fc)
    pos = tools.propose_purchase_orders(skus, pd.DataFrame(state["health"]), budget)
    return {
        "done": ["po"],
        "pos": pos.to_dict("records"),
        "budget": budget,
        "audit": _log(
            "po_agent",
            "propose_pos",
            count=len(pos),
            value=round(float(pos.value.sum())),
            deferred_over_budget=int((~pos.within_budget).sum()),
        ),
    }


# ── 4. Workflow automation agent ─────────────────────────────────────────────
def workflow_agent(state: State) -> dict:
    skus, sales, inv = _data(state)
    fc, _ = _forecast(state, skus, sales)
    pos = state.get("pos")
    limit = AUTO_APPROVE_LIMIT[_domain(state)]
    pos = tools.route_approvals(pd.DataFrame(pos), limit) if pos else pd.DataFrame(columns=["decision"])
    audit = _log(
        "workflow_agent",
        "route_approvals",
        **{k: int(v) for k, v in pos.get("route", pd.Series(dtype=str)).value_counts().items()},
    )

    pending = pos[pos.decision == "pending"] if len(pos) else pos
    if len(pending) and state.get("interactive"):
        # Human in the loop: pause the graph and hand the planner the decisions it cannot make alone.
        answers = interrupt(
            {"pending": pending[["po_id", "sku", "status", "qty", "value", "route"]].to_dict("records")}
        )
        for po_id, ok in answers.items():
            pos.loc[pos.po_id == po_id, "decision"] = "approved" if ok else "rejected"
        audit += _log(
            "human_planner",
            "decisions",
            approved=sum(answers.values()),
            rejected=len(answers) - sum(answers.values()),
        )

    health = pd.DataFrame(state["health"])
    excess = health[health.status == "excess"]
    crit = pos[pos.get("status", pd.Series(dtype=str)) == "critical"] if len(pos) else pos
    notes = []
    if len(crit):
        notes.append(
            f"Procurement: {len(crit)} critical SKUs will stock out before a normal order "
            f"lands; expedite requests raised."
        )
    if (pos.decision == "pending").any():
        notes.append(f"Planner: {int((pos.decision == 'pending').sum())} POs await approval.")
    if (pos.decision == "deferred").any():
        notes.append(f"Finance: {int((pos.decision == 'deferred').sum())} POs deferred by the cycle budget.")
    if len(excess):
        notes.append(
            f"Sales & ops: {len(excess)} SKUs hold more than the target cover; "
            f"consider redistribution or promotion to bring DIO down."
        )
    kpis = tools.supply_chain_kpis(skus, inv, fc, health, pos if len(pos) else None, state.get("budget"))
    return {
        "done": ["workflow"],
        "pos": pos.to_dict("records"),
        "kpis": kpis,
        "notifications": notes,
        "briefing": _briefing(state, pos, kpis, notes),
        "audit": audit + _log("workflow_agent", "notify", messages=len(notes)),
    }


def _briefing(state, pos, kpis, notes) -> str:
    facts = {
        "forecast": state.get("metrics"),
        "stock_status": pd.DataFrame(state["health"]).status.value_counts().to_dict(),
        "po_decisions": pos.decision.value_counts().to_dict() if len(pos) else {},
        "kpis": kpis,
        "notifications": notes,
    }
    llm = get_llm()
    if llm is not None:
        try:
            msg = llm.invoke(
                "You are a supply-planning assistant. Write a 5-bullet briefing for the "
                "planning manager from these facts. Do not invent or change any number.\n"
                f"{facts}"
            )
            return msg.content
        except Exception as e:  # the workflow must never fail on the narrative
            return f"(LLM unavailable: {type(e).__name__}) " + "; ".join(notes)
    return "\n".join(f"- {n}" for n in notes)


def _plain(update: dict) -> dict:
    """Convert numpy/pandas scalars to plain Python so the checkpointer can store state."""
    import json

    return json.loads(json.dumps(update, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _node(fn):
    return lambda state: _plain(fn(state))


def build_graph():
    g = StateGraph(State)
    g.add_node("supervisor", _node(supervisor))
    for name, fn in (
        ("demand_agent", demand_agent),
        ("inventory_agent", inventory_agent),
        ("po_agent", po_agent),
        ("workflow_agent", workflow_agent),
    ):
        g.add_node(name, _node(fn))
        g.add_edge(name, "supervisor")
    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        ["demand_agent", "inventory_agent", "po_agent", "workflow_agent", END],
    )
    return g.compile(checkpointer=MemorySaver())
