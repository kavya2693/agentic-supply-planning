"""Deterministic tools the agents call. Every quantity, value and approval route is
computed here, never by a language model, so each decision is reproducible and auditable."""

import math

import numpy as np
import pandas as pd

from .config import BUDGET_WEEKS, HORIZON, REVIEW_WEEKS


def project_stock(on_hand, on_order, eta, d50, arrivals=None):
    """Walk stock forward week by week against P50 demand. Returns the first week stock
    runs out (or None) and the units of demand that would go unserved."""
    stock, stockout, unmet = float(on_hand), None, 0.0
    arrivals = arrivals or {}
    for h in range(1, len(d50) + 1):
        if on_order and eta == h:
            stock += on_order
        stock += arrivals.get(h, 0)
        served = min(max(stock, 0.0), d50[h - 1])
        unmet += d50[h - 1] - served
        stock -= d50[h - 1]
        if stock < 0 and stockout is None:
            stockout = h
        stock = max(stock, 0.0)  # lost sales: unmet demand does not carry over
    return stockout, unmet


def inventory_health(skus: pd.DataFrame, inv: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Project stock week by week against the P50 forecast and classify each SKU."""
    p50 = fc.pivot(index="sku", columns="week_ahead", values="p50")
    p90 = fc.pivot(index="sku", columns="week_ahead", values="p90")
    df = skus.merge(inv, on="sku").set_index("sku").loc[p50.index]
    rows = []
    for sku, r in df.iterrows():
        d50, d90 = p50.loc[sku].to_numpy(), p90.loc[sku].to_numpy()
        stockout, _ = project_stock(r.on_hand, r.on_order, r.on_order_eta_weeks, d50)
        weekly = max(d50[:4].mean(), 1e-6)
        cover = r.on_hand / weekly
        cover_window = min(int(r.lead_time_weeks) + REVIEW_WEEKS, HORIZON)
        # This cycle's orders only protect lead time + review period; later weeks belong to later cycles.
        _, unmet = project_stock(r.on_hand, r.on_order, r.on_order_eta_weeks, d50[:cover_window])
        # Safety stock: weekly P90-P50 gaps are not all hit at once, so they add in quadrature
        # (summing weekly P90s would overstate demand and inflate stock).
        safety = np.sqrt(((d90[:cover_window] - d50[:cover_window]) ** 2).sum())
        order_up_to = d50[:cover_window].sum() + safety
        position = r.on_hand + r.on_order
        if stockout is not None and stockout <= r.lead_time_weeks:
            status = "critical"  # a normal order arrives too late
        elif position < order_up_to:
            status = "reorder"
        elif cover > r.max_cover_weeks:
            status = "excess"
        else:
            status = "healthy"
        rows.append(
            {
                "sku": sku,
                "status": status,
                "stockout_week": stockout,
                "weeks_of_cover": round(cover, 1),
                "order_up_to": round(order_up_to),
                "position": position,
                "weekly_p50": round(weekly, 1),
                "protection_weeks": cover_window,
                "demand_window": round(float(d50[:cover_window].sum()), 1),
                "unmet_no_action": round(unmet, 1),
            }
        )
    return pd.DataFrame(rows)


def cycle_budget(skus: pd.DataFrame, fc: pd.DataFrame) -> float:
    """Spend cap for one cycle: BUDGET_WEEKS of forecast demand at cost."""
    weekly = fc[fc.week_ahead <= 4].groupby("sku").p50.mean()
    cost = skus.set_index("sku").unit_cost.reindex(weekly.index)
    return float(round(BUDGET_WEEKS * (weekly * cost).sum(), -3))


def propose_purchase_orders(skus: pd.DataFrame, health: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Order-up-to policy rounded to MOQ, prioritised by urgency, trimmed to the budget."""
    need = health[health.status.isin(["critical", "reorder"])].merge(skus, on="sku")
    need["qty"] = [
        int(math.ceil(max(o - p, 0) / m) * m)
        for o, p, m in zip(need.order_up_to, need.position, need.moq, strict=True)
    ]
    need = need[need.qty > 0].copy()
    need["value"] = (need.qty * need.unit_cost).round(2)
    need["priority"] = np.where(need.status == "critical", 0, 1)
    need["stockout_sort"] = need.stockout_week.fillna(99)
    need = need.sort_values(["priority", "stockout_sort", "value"]).reset_index(drop=True)
    # Smaller orders may fill leftover budget within the same priority tier, but once a
    # critical order cannot be funded, no routine order is funded ahead of it.
    spent, within, blocked_below = 0.0, [], None
    for v, prio in zip(need.value, need.priority, strict=True):
        ok = spent + v <= budget and (blocked_below is None or prio <= blocked_below)
        if not ok and blocked_below is None:
            blocked_below = prio
        spent += v if ok else 0
        within.append(ok)
    need["within_budget"] = within
    need["po_id"] = [f"PO-{i + 1:04d}" for i in range(len(need))]
    cols = [
        "po_id",
        "sku",
        "family",
        "source",
        "status",
        "stockout_week",
        "qty",
        "moq",
        "unit_cost",
        "value",
        "lead_time_weeks",
        "within_budget",
    ]
    return need[cols]


def route_approvals(pos: pd.DataFrame, limit: float) -> pd.DataFrame:
    """Workflow rules: small routine POs go straight through; large or expedited ones
    need a planner's sign-off; anything over budget is deferred to the next cycle."""
    pos = pos.copy()
    pos["route"] = np.select(
        [~pos.within_budget, pos.status == "critical", pos.value > limit],
        ["deferred_budget", "human_expedite", "human_approval"],
        "auto_approve",
    )
    pos["decision"] = np.where(
        pos.route == "auto_approve",
        "approved",
        np.where(pos.route == "deferred_budget", "deferred", "pending"),
    )
    return pos


def supply_chain_kpis(skus, inv, fc, health, pos=None, budget=None) -> dict:
    """The numbers a supply-chain lead reviews each cycle."""
    m = skus.merge(inv, on="sku").merge(health, on="sku")
    weekly_cost = m.weekly_p50 * m.unit_cost
    stock_value = float((m.on_hand * m.unit_cost).sum())
    excess_units = (m.on_hand - m.max_cover_weeks * m.weekly_p50).clip(lower=0)
    demand = float(m.demand_window.sum())
    out = {
        "stock_value": round(stock_value),
        "dio_days": round(7 * stock_value / max(float(weekly_cost.sum()), 1e-9), 1),
        "excess_stock_value": round(float((excess_units * m.unit_cost).sum())),
        "skus_at_stockout_risk": int((m.unmet_no_action > 0).sum()),
        "fill_rate_no_action": round(1 - float(m.unmet_no_action.sum()) / demand, 3),
    }
    if pos is None or not len(pos):
        return out
    p50 = fc.pivot(index="sku", columns="week_ahead", values="p50")

    def fill_rate(decisions):
        orders = pos[pos.decision.isin(decisions)].set_index("sku")
        unmet = 0.0
        for r in m.itertuples():
            arrivals = {}
            if r.sku in orders.index:
                o = orders.loc[r.sku]
                arrivals = {min(int(o.lead_time_weeks), HORIZON): float(o.qty)}
            d50 = p50.loc[r.sku].to_numpy()[: int(r.protection_weeks)]
            unmet += project_stock(r.on_hand, r.on_order, r.on_order_eta_weeks, d50, arrivals)[1]
        return round(1 - unmet / demand, 3)

    out |= {
        "pos_total": len(pos),
        "touchless_po_rate": round(float((pos.route == "auto_approve").mean()), 3),
        "approved_po_value": round(float(pos[pos.decision == "approved"].value.sum())),
        "pending_po_value": round(float(pos[pos.decision == "pending"].value.sum())),
        "fill_rate_with_approved": fill_rate(["approved"]),
        "fill_rate_if_pending_approved": fill_rate(["approved", "pending"]),
    }
    if budget:
        out["budget"] = round(budget)
        out["budget_used"] = round(float(pos[pos.within_budget].value.sum()) / budget, 3)
    return out
