# agentic-supply-planning

Demand forecasts turned into budgeted, approved purchase orders by a supervisor and four
LangGraph agents, for a lubricants supply chain.

## The problem

A lubricants supplier carries hundreds of SKUs, from 1 L motorcycle-oil bottles sold
through workshops to 209 L drums sold to fleets and factories. Some are blended locally
in two weeks; others are imported and take up to ten. Every two weeks a planner has to
decide, SKU by SKU, what to reorder, what to expedite and what is already overstocked,
within a purchasing budget, before the festive-season peak or a monsoon slowdown.

Done by hand in spreadsheets, this is where both failures come from: stock-outs on
fast-moving packs, and months of excess on slow ones (high days inventory outstanding).
A forecast alone does not fix it, because the forecast still has to be turned into
hundreds of order decisions, approvals and messages every cycle.

## Why it is hard

The modelling is not the hard part. The hard part is trust and control. Purchasing is
money leaving the business, so an automated system has to be reproducible, stay inside
budget and policy, show why it ordered what it ordered, and hand the expensive or urgent
decisions to a person. A language model choosing order quantities fails all four.

## The approach

An existing global **LightGBM** demand forecaster (P50 and P90 quantiles, 12 weeks ahead)
is extended into a **LangGraph** workflow where a supervisor coordinates four agents:

| Agent | Decides | How |
|---|---|---|
| **Demand agent** | Expected and high-side demand per SKU per week | Runs and backtests the forecaster |
| **Inventory monitoring agent** | Critical / reorder / healthy / excess per SKU | Projects stock week by week, incoming orders included |
| **Purchase-order agent** | What to order and how much | Order-up-to with quadrature safety stock, rounded to MOQ, critical first, trimmed to budget |
| **Workflow automation agent** | Who must approve, who gets told | Auto-approves small routine POs, pauses for a planner on large or expedited ones, defers over-budget POs, notifies procurement, finance and sales |

The key decision: **every quantity, value and route is computed by deterministic tools;
the language model only writes the planner briefing**, and is told not to change any
number. The workflow runs end to end with no LLM at all. Large or urgent POs trigger a
LangGraph `interrupt`, so a planner approves them before the cycle finishes, and every
agent step is written to an audit log.

```
supervisor ─► demand ─► inventory ─► purchase orders ─► workflow ─► [planner approval] ─► done
     ▲__________|____________|______________|_______________|   (shared state + audit log)
```

## Results (synthetic data, 800 SKUs, 3 years weekly)

| | |
|---|---|
| Forecast error, 12-week backtest (WAPE) | **17.8%** vs 27.4% seasonal naive, 26.6% 4-week average |
| P90 coverage | 90% of actual weeks fall at or below P90 (target 90%) |
| Stock status | 105 critical · 375 reorder · 146 healthy · 174 excess |
| Purchase orders | 476 proposed: 323 auto-approved, 133 sent to a planner (105 expedites, 28 large), 20 deferred by budget |
| Tests | 7 passing, including human-approval pause and resume |

The data is synthetic (no company data), so the operational numbers show the workflow
behaving sensibly, not business impact. The 174 excess SKUs are a reminder that the
workflow flags overstock but cannot fix it this cycle; that needs redistribution or promotion.

## Run it

```bash
uv sync
uv run supply-planning --refresh        # generate data, train, run one cycle
uv run supply-planning --interactive    # approve large / expedited POs yourself
make check                              # ruff, format, mypy, pytest
docker build -t supply-planning . && docker run --rm supply-planning
```

Outputs land in `outputs/`: `purchase_orders.csv`, `approval_queue.csv`,
`inventory_health.csv`, `audit_log.jsonl`, `run_report.md`. Set `OPENROUTER_API_KEY`
(see `.env.example`) or run Ollama locally for a model-written briefing; otherwise a
rule-based briefing is used. Policy (budget, approval limit, review period, target cover)
lives in `src/supply_planning/config.py`.

## Layout

```
src/supply_planning/
  data.py       seeded synthetic generator (SKUs, weekly sales, stock, open orders)
  forecast.py   global LightGBM, direct 12-week horizon, P50 and P90 heads, backtest
  tools.py      stock projection, order-up-to with safety stock, budget, approval routing
  graph.py      LangGraph supervisor + demand, inventory, purchase-order, workflow agents
  llm.py        optional briefing model (OpenRouter or Ollama)
  cli.py        one planning cycle, interactive approvals, report and audit log
docs/adr/       architecture decisions
tests/          policy, budget, routing, no-model run, human pause and resume
```

## Limitations

- Synthetic data only. The operational numbers show the workflow behaving sensibly, not business impact.
- Single echelon: one stocking point, no plant-to-depot-to-distributor network.
- No supplier capacity limits or price breaks; MOQ is the only order constraint.
- The model-written briefing path is not covered by tests (no key in CI); everything that decides an order is.
- State is checkpointed in memory; a production run would use a persistent checkpointer.

## License

MIT
