# agentic-supply-planning

Demand forecasts turned into budgeted, approved purchase orders by a supervisor and four
LangGraph agents. Ships with a retail range (groceries to electronics) and a lubricants
range; the agents run unchanged on both.

## The problem

A retailer carrying hundreds of SKUs, from fresh produce that spoils in days to imported
electronics accessories with a ten-week lead time, has to decide every two weeks, SKU by
SKU, what to reorder, what to expedite and what is already overstocked, within a
purchasing budget, ahead of Ramadan, back to school or the year-end sales season.

Done by hand in spreadsheets, this is where both failures come from: empty shelves on
fast movers, and cash tied up in slow or spoiling stock. A forecast alone does not fix it,
because the forecast still has to be turned into hundreds of order decisions, approvals
and messages every cycle.

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

## Results (synthetic data, 800 SKUs per range, 3 years weekly)

| | Retail | Lubricants |
|---|---|---|
| Forecast WAPE, 12-week backtest | **8.4%** (seasonal naive 19.6%, 4-week average 18.4%) | **18.2%** (28.2%, 27.4%) |
| Forecast bias / P90 coverage | −0.1% / 91% | −0.5% / 90% |
| Stock status | 23 critical · 202 reorder · 275 healthy · 300 excess | 93 critical · 376 reorder · 203 healthy · 128 excess |
| Purchase orders | 223: 197 auto-approved, 26 to a planner (23 expedites, 3 large) | 466: 306 auto-approved, 114 to a planner, 46 deferred by budget |
| Touchless PO rate | 88% | 66% |
| Fill rate over the protection window | 92.8% no action → 96.7% auto-approved → 99.1% queue approved | 91.0% → 94.9% → 98.6% |
| DIO / excess stock value | 49.8 days / 411,788 | 62.0 days / 1,608,509 |

The data is synthetic (no company data), so these numbers show the workflow behaving
sensibly, not business impact. Lubricants hits its budget cap (46 orders deferred) while
retail uses 27% of its cap; the excess counts are a reminder that the workflow flags
overstock but cannot fix it in one cycle. [docs/USE_CASE.md](docs/USE_CASE.md) walks
through one cycle and the KPIs in detail.

## Run it

```bash
uv sync
uv run supply-planning --refresh                       # retail: generate data, train, run one cycle
uv run supply-planning --domain lubricants --refresh   # same agents, lubricants range
uv run supply-planning --interactive    # approve large / expedited POs yourself
make check                              # ruff, format, mypy, pytest
docker build -t supply-planning . && docker run --rm supply-planning
```

Outputs land in `outputs/<domain>/`: `purchase_orders.csv`, `approval_queue.csv`,
`inventory_health.csv`, `audit_log.jsonl`, `run_report.md`. Set `OPENROUTER_API_KEY`
(see `.env.example`) or run Ollama locally for a model-written briefing; otherwise a
rule-based briefing is used. Policy (budget, approval limit, review period, target cover)
lives in `src/supply_planning/config.py`.

## Layout

```
src/supply_planning/
  domains.py    retail and lubricants profiles: families, channels, packs, suppliers, seasons
  data.py       seeded synthetic generator for any profile (SKUs, weekly sales, stock, open orders)
  forecast.py   global LightGBM, direct 12-week horizon, P50 and P90 heads, backtest
  tools.py      stock projection, order-up-to with safety stock, budget, routing, KPIs
  graph.py      LangGraph supervisor + demand, inventory, purchase-order, workflow agents
  llm.py        optional briefing model (OpenRouter or Ollama)
  cli.py        one planning cycle, interactive approvals, report and audit log
docs/           use case walkthrough, architecture decisions
tests/          both domains: policy, budget, routing, fill rate, no-model run, human pause and resume
```

## Limitations

- Synthetic data only. The operational numbers show the workflow behaving sensibly, not business impact.
- Single echelon: one stocking point per range, no store-level allocation or DC-to-store network.
- Perishables are modelled by a short cover limit, not by batch expiry dates.
- No supplier capacity limits or price breaks; MOQ is the only order constraint.
- The model-written briefing path is not covered by tests (no key in CI); everything that decides an order is.
- State is checkpointed in memory; a production run would use a persistent checkpointer.

## License

MIT
