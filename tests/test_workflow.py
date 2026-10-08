import uuid

import pandas as pd
import pytest
from langgraph.types import Command

from supply_planning import tools
from supply_planning.config import AUTO_APPROVE_LIMIT, PERISHABLE_MAX_COVER_WEEKS
from supply_planning.data import load
from supply_planning.domains import DOMAINS, RETAIL
from supply_planning.forecast import load_forecast
from supply_planning.graph import build_graph

DOMAIN_NAMES = sorted(DOMAINS)


def _ctx(domain):
    skus, sales, inv = load(domain)
    fc, metrics = load_forecast(skus, sales, domain)
    return skus, inv, fc, metrics


@pytest.mark.parametrize("domain", DOMAIN_NAMES)
def test_forecast_beats_baselines(domain):
    *_, m = _ctx(domain)
    assert m["wape_model_p50"] < m["wape_seasonal_naive"]
    assert m["wape_model_p50"] < m["wape_moving_avg_4wk"]
    assert 0.8 <= m["p90_coverage"] <= 0.97
    assert abs(m["bias_p50"]) < 0.05


@pytest.mark.parametrize("domain", DOMAIN_NAMES)
def test_po_quantities_are_moq_multiples_and_cover_the_gap(domain):
    skus, inv, fc, _ = _ctx(domain)
    health = tools.inventory_health(skus, inv, fc)
    pos = tools.propose_purchase_orders(skus, health, tools.cycle_budget(skus, fc)).merge(health, on="sku")
    assert len(pos)
    assert (pos.qty % pos.moq == 0).all()
    assert (pos.position + pos.qty >= pos.order_up_to - 1e-6).all()


def test_budget_is_never_exceeded():
    skus, inv, fc, _ = _ctx("lubricants")
    pos = tools.propose_purchase_orders(skus, tools.inventory_health(skus, inv, fc), budget=500_000)
    assert pos[pos.within_budget].value.sum() <= 500_000
    assert (~pos.within_budget).any()


def test_critical_items_are_ordered_first():
    skus, inv, fc, _ = _ctx("lubricants")
    pos = tools.propose_purchase_orders(skus, tools.inventory_health(skus, inv, fc), budget=200_000)
    funded = pos[pos.within_budget]
    assert len(funded) and (funded.status == "critical").all()


def test_routing_rules():
    limit = 1_000
    df = pd.DataFrame(
        {
            "po_id": ["a", "b", "c", "d"],
            "status": ["reorder", "reorder", "critical", "reorder"],
            "value": [limit, limit + 1, 10, 10],
            "within_budget": [True, True, True, False],
        }
    )
    r = tools.route_approvals(df, limit).set_index("po_id")
    assert r.loc["a", "route"] == "auto_approve" and r.loc["a", "decision"] == "approved"
    assert r.loc["b", "route"] == "human_approval" and r.loc["b", "decision"] == "pending"
    assert r.loc["c", "route"] == "human_expedite"
    assert r.loc["d", "decision"] == "deferred"


def test_perishables_get_short_cover_and_fast_supply():
    skus, *_ = _ctx("retail")
    fresh = skus.family.isin([n for n, f in RETAIL.families.items() if f.perishable])
    assert fresh.any()
    assert (skus[fresh].max_cover_weeks == PERISHABLE_MAX_COVER_WEEKS).all()
    assert (skus[fresh].lead_time_weeks == 1).all()


def test_project_stock_counts_lost_sales():
    stockout, unmet = tools.project_stock(on_hand=10, on_order=0, eta=0, d50=[6, 6, 6])
    assert stockout == 2 and unmet == pytest.approx(8)
    stockout, unmet = tools.project_stock(10, 0, 0, [6, 6, 6], arrivals={2: 20})
    assert stockout is None and unmet == 0


@pytest.mark.parametrize("domain", DOMAIN_NAMES)
def test_graph_runs_without_llm_and_improves_fill_rate(domain, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("supply_planning.graph.get_llm", lambda: None)
    s = build_graph().invoke({"domain": domain}, {"configurable": {"thread_id": str(uuid.uuid4())}})
    agents = {a["agent"] for a in s["audit"]}
    assert {"supervisor", "demand_agent", "inventory_agent", "po_agent", "workflow_agent"} <= agents
    k = s["kpis"]
    assert k["fill_rate_no_action"] < k["fill_rate_with_approved"] <= k["fill_rate_if_pending_approved"]
    assert 0 < k["touchless_po_rate"] <= 1
    assert s["briefing"]


def test_human_in_the_loop_pauses_and_resumes(monkeypatch):
    monkeypatch.setattr("supply_planning.graph.get_llm", lambda: None)
    g, cfg = build_graph(), {"configurable": {"thread_id": str(uuid.uuid4())}}
    s = g.invoke({"domain": "retail", "interactive": True}, cfg)
    pending = s["__interrupt__"][0].value["pending"]
    assert pending, "large or expedited POs must wait for a human"
    assert all(p["value"] > AUTO_APPROVE_LIMIT["retail"] or p["status"] == "critical" for p in pending)
    first = pending[0]["po_id"]
    answers = {p["po_id"]: p["po_id"] == first for p in pending}  # approve one, reject the rest
    s = g.invoke(Command(resume=answers), cfg)
    pos = pd.DataFrame(s["pos"]).set_index("po_id")
    assert pos.loc[first, "decision"] == "approved"
    assert (pos.loc[[p["po_id"] for p in pending[1:]], "decision"] == "rejected").all()
    assert any(a["agent"] == "human_planner" for a in s["audit"])
