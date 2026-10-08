"""Shared constants. Every number the agents act on lives here, so a planner can change
policy without touching agent code."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

SEED = 7
N_SKUS = 800  # lubricants portfolio size
N_WEEKS = 156  # 3 years of weekly history
HORIZON = 12  # forecast weeks ahead
REVIEW_WEEKS = 2  # planning cycle: POs are raised every 2 weeks

# Purchase-order policy
SERVICE_QUANTILE = 0.9  # order-up-to uses the P90 forecast
CYCLE_BUDGET = 2_500_000  # spend cap per planning cycle, ~2 weeks of demand at cost
TARGET_COVER_WEEKS = 8  # above this, stock counts as excess (drives DIO up)

# Approval workflow
AUTO_APPROVE_LIMIT = 15_000  # POs at or below this value are approved automatically
