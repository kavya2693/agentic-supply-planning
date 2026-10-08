"""Synthetic lubricants supply-chain data, shaped like an Indian lubricants business:
B2C (retail, workshops) and B2B (fleets, industry), pack sizes from 1 L bottles to
209 L drums, festive-season peaks, monsoon dips, promotions and imported vs locally
blended supply. No real company data is used."""

import numpy as np
import pandas as pd

from .config import DATA, N_SKUS, N_WEEKS, SEED

FAMILIES = {  # family: (channel mix B2C share, base weekly units per pack litre)
    "Engine Oil - Passenger Car": (0.85, 60),
    "Engine Oil - Heavy Duty": (0.30, 40),
    "Motorcycle Oil": (0.95, 90),
    "Gear Oil": (0.50, 20),
    "Hydraulic Oil": (0.10, 25),
    "Industrial Grease": (0.15, 15),
}
PACKS = {"1L": 1, "3.5L": 3.5, "5L": 5, "20L": 20, "209L": 209}
SUPPLY = {  # source: (lead time weeks range, MOQ multiple)
    "Local blending plant": ((2, 3), 50),
    "Imported finished goods": ((6, 10), 200),
}


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    fams = list(FAMILIES)
    sku = pd.DataFrame({"sku": [f"LUB-{i:04d}" for i in range(N_SKUS)]})
    sku["family"] = rng.choice(fams, N_SKUS, p=[0.28, 0.18, 0.22, 0.12, 0.12, 0.08])
    sku["channel"] = [("B2C" if rng.random() < FAMILIES[f][0] else "B2B") for f in sku.family]
    sku["pack"] = [
        rng.choice(["209L", "20L"], p=[0.4, 0.6])
        if c == "B2B"
        else rng.choice(["1L", "3.5L", "5L"], p=[0.5, 0.25, 0.25])
        for c in sku.channel
    ]
    sku["source"] = rng.choice(list(SUPPLY), N_SKUS, p=[0.65, 0.35])
    sku["lead_time_weeks"] = [int(rng.integers(*SUPPLY[s][0], endpoint=True)) for s in sku.source]
    sku["moq"] = [
        SUPPLY[s][1] if p in ("1L", "3.5L", "5L") else max(4, SUPPLY[s][1] // 20)
        for s, p in zip(sku.source, sku.pack, strict=True)
    ]
    sku["unit_cost"] = (sku.pack.map(PACKS) * rng.uniform(3.0, 6.0, N_SKUS)).round(2)
    scale = sku.family.map(lambda f: FAMILIES[f][1]) / np.sqrt(sku.pack.map(PACKS))
    sku["base"] = (scale * rng.lognormal(0, 0.7, N_SKUS)).clip(0.3)
    sku["trend"] = rng.normal(0.0015, 0.002, N_SKUS)

    weeks = pd.date_range("2023-10-02", periods=N_WEEKS, freq="W-MON")
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    festive = np.isin(woy, range(40, 47)) * 0.35  # Diwali / festive service season
    monsoon = np.isin(woy, range(26, 36)) * -0.20  # monsoon slowdown
    season = 1 + festive + monsoon + 0.08 * np.sin(2 * np.pi * woy / 52)

    rows = []
    for r in sku.itertuples():
        promo = rng.random(N_WEEKS) < (0.08 if r.channel == "B2C" else 0.02)
        lam = r.base * season * (1 + r.trend) ** np.arange(N_WEEKS) * np.where(promo, 1.45, 1.0)
        if r.channel == "B2B":  # lumpy fleet / plant orders
            qty = rng.poisson(lam) * (rng.random(N_WEEKS) < 0.55) * 1.8
        else:
            qty = rng.poisson(lam)
        price = r.unit_cost * 1.35 * np.where(promo, 0.9, 1.0)
        rows.append(
            pd.DataFrame(
                {
                    "sku": r.sku,
                    "week": weeks,
                    "units": qty.round(),
                    "promo": promo.astype(int),
                    "price": price.round(2),
                }
            )
        )
    sales = pd.concat(rows, ignore_index=True)

    last = sales.week.max()
    recent = sales[sales.week > last - pd.Timedelta(weeks=8)].groupby("sku").units.mean()
    inv = sku[["sku"]].copy()
    # Stock is held relative to each SKU's replenishment time (lead time + review period),
    # with the usual spread: a few nearly empty, most adequate, some overstocked.
    factor = rng.choice([0.5, 1.0, 1.4, 2.0, 3.0], N_SKUS, p=[0.06, 0.20, 0.42, 0.20, 0.12])
    cover = factor * (sku.lead_time_weeks.to_numpy() + 2)
    inv["on_hand"] = (recent.reindex(inv.sku).fillna(0).to_numpy() * cover).round()
    has_po = rng.random(N_SKUS) < 0.3
    inv["on_order"] = np.where(has_po, (inv.on_hand * rng.uniform(0.3, 1.2, N_SKUS)).round(), 0)
    inv["on_order_eta_weeks"] = np.where(has_po, rng.integers(1, 6, N_SKUS), 0)

    sku = sku.drop(columns=["base", "trend"])
    for name, df in (("skus", sku), ("sales", sales), ("inventory", inv)):
        df.to_csv(DATA / f"{name}.csv", index=False)
    return sku, sales, inv


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (DATA / "sales.csv").exists():
        return generate()
    return (
        pd.read_csv(DATA / "skus.csv"),
        pd.read_csv(DATA / "sales.csv", parse_dates=["week"]),
        pd.read_csv(DATA / "inventory.csv"),
    )
