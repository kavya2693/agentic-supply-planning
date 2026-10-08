"""Deterministic tools the agents call. Every quantity, value and approval route is
computed here, never by a language model, so each decision is reproducible and auditable."""

import math

import numpy as np
import pandas as pd

from .config import AUTO_APPROVE_LIMIT, CYCLE_BUDGET, HORIZON, REVIEW_WEEKS, TARGET_COVER_WEEKS


def inventory_health(skus: pd.DataFrame, inv: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """Project stock week by week against the P50 forecast and classify each SKU."""
    p50 = fc.pivot(index="sku", columns="week_ahead", values="p50")
    p90 = fc.pivot(index="sku", columns="week_ahead", values="p90")
    df = skus.merge(inv, on="sku").set_index("sku").loc[p50.index]
    rows = []
    for sku, r in df.iterrows():
        d50, d90 = p50.loc[sku].to_numpy(), p90.loc[sku].to_numpy()
        stock, stockout = r.on_hand, None
        for h in range(HORIZON):
            if r.on_order and r.on_order_eta_weeks == h + 1:
                stock += r.on_order
            stock -= d50[h]
            if stock < 0 and stockout is None:
                stockout = h + 1
        weekly = max(d50[:4].mean(), 1e-6)
        cover = r.on_hand / weekly
        cover_window = min(int(r.lead_time_weeks) + REVIEW_WEEKS, HORIZON)
        # Safety stock: weekly P90-P50 gaps are not all hit at once, so they add in quadrature
        # (summing weekly P90s would overstate demand and inflate stock).
        safety = np.sqrt(((d90[:cover_window] - d50[:cover_window]) ** 2).sum())
        order_up_to = d50[:cover_window].sum() + safety
        position = r.on_hand + r.on_order
        if stockout is not None and stockout <= r.lead_time_weeks:
            status = "critical"  # a normal order arrives too late
        elif position < order_up_to:
            status = "reorder"
        elif cover > TARGET_COVER_WEEKS:
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
            }
        )
    return pd.DataFrame(rows)


def propose_purchase_orders(skus: pd.DataFrame, health: pd.DataFrame, budget=CYCLE_BUDGET) -> pd.DataFrame:
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


def route_approvals(pos: pd.DataFrame, limit=AUTO_APPROVE_LIMIT) -> pd.DataFrame:
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


def inventory_kpis(skus, inv, fc, pos=None) -> dict:
    """Days inventory outstanding at cost, before and after approved orders land."""
    m = skus.merge(inv, on="sku")
    weekly_cost = fc[fc.week_ahead <= 4].groupby("sku").p50.mean().reindex(m.sku).to_numpy() * m.unit_cost
    stock_value = (m.on_hand * m.unit_cost).sum()
    dio = 7 * stock_value / max(weekly_cost.sum(), 1e-9)
    out = {"stock_value": round(stock_value), "dio_days": round(dio, 1)}
    if pos is not None:
        approved = pos[pos.decision == "approved"].value.sum()
        out["approved_po_value"] = round(approved)
        out["pending_po_value"] = round(pos[pos.decision == "pending"].value.sum())
    return out
