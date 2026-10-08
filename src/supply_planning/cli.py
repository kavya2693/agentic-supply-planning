"""Run one planning cycle.

supply-planning                 # automatic: big POs wait in the approval queue
supply-planning --interactive   # pause for a planner to approve each big PO
supply-planning --refresh       # regenerate data and retrain the forecaster first
"""

import argparse
import json
import uuid

import pandas as pd
from langgraph.types import Command

from supply_planning.config import OUT
from supply_planning.data import generate
from supply_planning.graph import build_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    if args.refresh:
        generate()

    graph = build_graph()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke({"refresh": args.refresh, "interactive": args.interactive}, cfg)
    while "__interrupt__" in state:
        pending = state["__interrupt__"][0].value["pending"]
        print(f"\n{len(pending)} purchase orders need a planner decision:")
        answers = {}
        for p in pending:
            label = f"{p['po_id']} {p['sku']} {p['status']:8} qty {p['qty']:>6} value {p['value']:>10,.0f}"
            a = input(f"  {label}  approve? [y/N] ")
            answers[p["po_id"]] = a.strip().lower() == "y"
        state = graph.invoke(Command(resume=answers), cfg)

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(state["health"]).to_csv(OUT / "inventory_health.csv", index=False)
    pos = pd.DataFrame(state.get("pos") or [])
    if len(pos):
        pos.to_csv(OUT / "purchase_orders.csv", index=False)
        pos[pos.decision == "pending"].to_csv(OUT / "approval_queue.csv", index=False)
    with open(OUT / "audit_log.jsonl", "w") as f:
        for row in state["audit"]:
            f.write(json.dumps(row, default=str) + "\n")
    report = _report(state)
    (OUT / "run_report.md").write_text(report)
    print(report)


def _report(s) -> str:
    m, k, pos = s["metrics"], s["kpis"], pd.DataFrame(s.get("pos") or [])
    lines = [
        "# Planning cycle report",
        "",
        f"**Forecast:** {m['skus']} SKUs, 12-week backtest WAPE {m['wape_model_p50']:.1%} "
        f"(seasonal naive {m['wape_seasonal_naive']:.1%}, 4-week average {m['wape_moving_avg_4wk']:.1%}); "
        f"P90 covers {m['p90_coverage']:.0%} of actual weeks.",
        "",
        "**Stock status:** "
        + ", ".join(f"{k_}: {v}" for k_, v in pd.DataFrame(s["health"]).status.value_counts().items()),
        "",
    ]
    if len(pos):
        lines += [
            "**Purchase orders:** " + ", ".join(f"{k_}: {v}" for k_, v in pos.route.value_counts().items()),
            f"Approved value {k['approved_po_value']:,} · pending approval {k['pending_po_value']:,}",
            "",
        ]
    lines += [
        f"**Inventory:** stock value {k['stock_value']:,}, DIO {k['dio_days']} days",
        "",
        "## Briefing",
        s["briefing"],
        "",
        f"Audit trail: {len(s['audit'])} decisions logged to outputs/audit_log.jsonl",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
