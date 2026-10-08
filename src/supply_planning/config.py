"""Shared constants. Every number the agents act on lives here, so a planner can change
policy without touching agent code."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SEED = 7
DEFAULT_DOMAIN = "retail"
N_SKUS = 800  # SKUs in the range
N_WEEKS = 156  # 3 years of weekly history
HORIZON = 12  # forecast weeks ahead
REVIEW_WEEKS = 2  # planning cycle: POs are raised every 2 weeks

# Purchase-order policy
SERVICE_QUANTILE = 0.9  # order-up-to uses the P90 forecast
BUDGET_WEEKS = 2.2  # cycle spend cap, in weeks of forecast demand at cost
TARGET_COVER_WEEKS = 8  # above this, stock counts as excess (drives DIO up)
PERISHABLE_MAX_COVER_WEEKS = 2  # fresh lines spoil, so excess starts much sooner

# Approval workflow: POs at or below this value are approved automatically
AUTO_APPROVE_LIMIT = {"retail": 5_000, "lubricants": 15_000}
