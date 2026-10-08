import uuid

import pandas as pd
from langgraph.types import Command

from supply_planning import tools
from supply_planning.config import AUTO_APPROVE_LIMIT
from supply_planning.data import load
from supply_planning.forecast import load_forecast
from supply_planning.graph import build_graph


def _ctx():
    skus, sales, inv = load()
    fc, metrics = load_forecast(skus, sales)
    return skus, inv, fc, metrics


def test_forecast_beats_baselines():
    *_, m = _ctx()
    assert m["wape_model_p50"] < m["wape_seasonal_naive"]
    assert m["wape_model_p50"] < m["wape_moving_avg_4wk"]
    assert 0.8 <= m["p90_coverage"] <= 0.97


def test_po_quantities_are_moq_multiples_and_cover_the_gap():
    skus, inv, fc, _ = _ctx()
    health = tools.inventory_health(skus, inv, fc)
    pos = tools.propose_purchase_orders(skus, health).merge(health, on="sku")
    assert (pos.qty % pos.moq == 0).all()
    assert (pos.position + pos.qty >= pos.order_up_to - 1e-6).all()


def test_budget_is_never_exceeded():
    skus, inv, fc, _ = _ctx()
    pos = tools.propose_purchase_orders(skus, tools.inventory_health(skus, inv, fc), budget=500_000)
    assert pos[pos.within_budget].value.sum() <= 500_000
    assert (~pos.within_budget).any()


def test_critical_items_are_ordered_first():
    skus, inv, fc, _ = _ctx()
    pos = tools.propose_purchase_orders(skus, tools.inventory_health(skus, inv, fc), budget=200_000)
    funded = pos[pos.within_budget]
    assert (funded.status == "critical").all()


def test_routing_rules():
    df = pd.DataFrame(
        {
            "po_id": ["a", "b", "c", "d"],
            "status": ["reorder", "reorder", "critical", "reorder"],
            "value": [AUTO_APPROVE_LIMIT, AUTO_APPROVE_LIMIT + 1, 10, 10],
            "within_budget": [True, True, True, False],
        }
    )
    r = tools.route_approvals(df).set_index("po_id")
    assert r.loc["a", "route"] == "auto_approve" and r.loc["a", "decision"] == "approved"
    assert r.loc["b", "route"] == "human_approval" and r.loc["b", "decision"] == "pending"
    assert r.loc["c", "route"] == "human_expedite"
    assert r.loc["d", "decision"] == "deferred"


def test_graph_runs_without_llm_and_logs_every_agent(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("supply_planning.graph.get_llm", lambda: None)
    s = build_graph().invoke({}, {"configurable": {"thread_id": str(uuid.uuid4())}})
    agents = {a["agent"] for a in s["audit"]}
    assert {"supervisor", "demand_agent", "inventory_agent", "po_agent", "workflow_agent"} <= agents
    assert s["briefing"]


def test_human_in_the_loop_pauses_and_resumes(monkeypatch):
    monkeypatch.setattr("supply_planning.graph.get_llm", lambda: None)
    g, cfg = build_graph(), {"configurable": {"thread_id": str(uuid.uuid4())}}
    s = g.invoke({"interactive": True}, cfg)
    pending = s["__interrupt__"][0].value["pending"]
    assert pending, "large or expedited POs must wait for a human"
    first = pending[0]["po_id"]
    answers = {p["po_id"]: p["po_id"] == first for p in pending}  # approve one, reject the rest
    s = g.invoke(Command(resume=answers), cfg)
    pos = pd.DataFrame(s["pos"]).set_index("po_id")
    assert pos.loc[first, "decision"] == "approved"
    assert (pos.loc[[p["po_id"] for p in pending[1:]], "decision"] == "rejected").all()
    assert any(a["agent"] == "human_planner" for a in s["audit"])
