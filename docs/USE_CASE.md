# From predicted volume to a placed order

This page walks through one planning cycle for the retail range (800 SKUs across nine
categories, from fresh produce to electronics accessories) and explains how the agents
hand work to each other.

## The cycle in one picture

```
            ┌──────────────────────── supervisor (orchestrator) ────────────────────────┐
            │  reads shared state, decides which agent runs next, skips steps not needed │
            └───────┬──────────────┬───────────────────┬────────────────────┬───────────┘
                    ▼              ▼                   ▼                    ▼
              Demand agent   Inventory agent    Purchase-order agent   Workflow agent
              forecast P50   stock projection   order-up-to, MOQ,      approval routing,
              and P90 per    status per SKU     budget, priority       human pause, notices
              SKU per week
                    │              │                   │                    │
                    └──────────────┴──── shared state + audit log ──────────┘
```

## How the agents communicate

The agents do not chat with each other. They communicate through one shared state object
that LangGraph checkpoints after every step:

| Agent | Reads from state | Writes to state |
|---|---|---|
| Supervisor | which agents are done, whether anything needs ordering | `next` (the agent to run) |
| Demand | domain, refresh flag | `metrics` (backtest WAPE, bias, P90 coverage); forecast table on disk |
| Inventory | forecast | `health`: status, weeks of cover, stock-out week, order-up-to level per SKU |
| Purchase order | `health` | `pos`: quantity, value, priority, within-budget flag; `budget` |
| Workflow | `pos`, `health`, `budget` | routes, decisions, `kpis`, `notifications`, `briefing` |

Every step also appends to `audit`, so each order can be traced back to the forecast,
the stock position and the policy that produced it.

This is a supervisor pattern with typed hand-offs. It was chosen over free-form agent
conversation because the hand-offs are data, not prose: the purchase-order agent needs a
stock status per SKU, not a paragraph about stock.

## Predicted volume to order, step by step

1. **Forecast.** The demand agent predicts P50 (expected) and P90 (high-side) units for each
   SKU for the next 12 weeks.
2. **Protection window.** An order placed now arrives after the lead time and must last
   until the next cycle, so each SKU is protected for *lead time + review period* weeks
   (3 weeks for a local supplier, up to 12 for imports).
3. **Order-up-to level.** P50 demand over the window, plus safety stock
   √Σ(P90 − P50)² over the same weeks.
4. **Quantity.** Order-up-to level − (on hand + on order), rounded up to the supplier MOQ.
5. **Status.** *Critical* if stock runs out before an order could arrive (needs an
   expedite); *reorder* if the position is below the order-up-to level; *excess* if cover
   is above the SKU's limit (2 weeks for fresh lines, longer for long-lead imports).
6. **Budget.** Critical orders are funded first, then the rest by how soon they run out,
   until the cycle budget (2.2 weeks of forecast demand at cost) is used.
7. **Approval.** Orders up to the auto-approval limit go straight through; larger ones and
   every expedite pause the graph for a planner; over-budget ones are deferred.
8. **Notify.** Procurement (expedites), planner (approval queue), finance (deferrals),
   sales and operations (excess to redistribute or promote).

## KPIs the cycle reports

| KPI | Why it matters | Retail run |
|---|---|---|
| Forecast WAPE | Accuracy of the volume the orders are built on | 8.4% (seasonal naive 19.6%) |
| Forecast bias | Systematic over- or under-ordering | −0.1% |
| P90 coverage | Whether safety stock is calibrated | 91% (target 90%) |
| Fill rate over the protection window | Share of demand served from stock | 92.8% → 96.7% with auto-approved orders → 99.1% with the queue approved |
| SKUs at stock-out risk | Where service will fail without action | 185 |
| Touchless PO rate | Share of orders needing no human | 88% |
| Days inventory outstanding | Cash tied up in stock | 49.8 days |
| Excess stock value | Stock above each SKU's cover limit | 411,788 |
| Budget used | Spend against the cycle cap | 27% of 742,000 |

The jump from 96.7% to 99.1% is the value of the planner's queue: 26 orders (23 expedites,
3 large) that the workflow deliberately does not approve on its own.

## Adding another business

The agents never see the domain. A new business is a profile in `domains.py`: product
families, channels and pack sizes, supply sources with lead times and MOQs, and seasonal
peaks. Retail and lubricants ship today; the same four agents run on both unchanged.
