# ADR-0001: Deterministic tools decide orders; the language model only writes the briefing

## Status
Accepted

## Context
The workflow spends money. Every purchase order must be reproducible from the forecast
and policy, stay inside the cycle budget, respect MOQ, and be explainable to a planner
and to finance afterwards. Agent frameworks make it easy to let a model choose quantities
through tool calls, but a model can invent or round a number, and the same input can give
a different answer on the next run.

## Decision
- A LangGraph `StateGraph` with a supervisor routes work to four agents in a fixed order:
  demand, inventory monitoring, purchase order, workflow automation. The supervisor skips
  the purchase-order agent when nothing needs ordering.
- Every quantity, value, priority and approval route is computed by plain Python in
  `tools.py`. Agents call these tools; no model output feeds a number.
- The only model call is the planner briefing, which receives the computed facts and is
  instructed not to change them. If no model is available, a rule-based briefing is used.
- Orders above the auto-approval limit, and all expedites, pause the graph with
  `interrupt()`. A planner's decisions resume it with `Command(resume=...)`.
- Order-up-to level = P50 demand over lead time plus review period, plus safety stock
  √Σ(P90 − P50)² over the same weeks. Summing weekly P90s was rejected because weekly
  peaks do not coincide, so it overstates lead-time demand and inflates stock.

## Alternatives rejected
- **Model-chosen quantities via tool calling.** Not reproducible; fails audit.
- **One script without agents.** Works, but every policy change touches one large function;
  separate agents let approval rules change without touching forecasting or ordering.
- **CrewAI / AutoGen role-play agents.** Conversation between agents adds tokens and
  non-determinism without adding a decision the tools cannot already make.

## Consequences
- The workflow runs end to end in CI with no API key.
- Adding a model-driven step later (for example, reading supplier emails) must not write
  to any field the order tools own.
