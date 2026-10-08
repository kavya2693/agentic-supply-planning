"""Seeded synthetic supply-chain data for any profile in `domains.py`: a SKU range with
families, channels, pack sizes and supply sources, three years of weekly sales with
seasonal peaks and promotions, and a current stock and open-order snapshot.
No real company data is used."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA, N_SKUS, N_WEEKS, PERISHABLE_MAX_COVER_WEEKS, SEED, TARGET_COVER_WEEKS
from .domains import DOMAINS


def data_dir(domain: str) -> Path:
    d = DATA / domain
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pick(rng, options: dict, weight) -> str:
    names = list(options)
    p = np.array([weight(options[n]) for n in names], float)
    return str(rng.choice(names, p=p / p.sum()))


def generate(domain: str = "retail", seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = DOMAINS[domain]
    rng = np.random.default_rng(seed)
    chan_names = list(spec.channels)
    rows = []
    for i in range(N_SKUS):
        fam_name = _pick(rng, spec.families, lambda f: f.share)
        fam = spec.families[fam_name]
        chan_name = chan_names[0] if rng.random() < fam.first_channel_share else chan_names[-1]
        chan = spec.channels[chan_name]
        pack = _pick(rng, chan.packs, lambda p: p[1])
        size = chan.packs[pack][0]
        src_name = _pick(rng, spec.sources, lambda s: s.share)
        src = spec.sources[src_name]
        lead = int(rng.integers(src.lead_weeks[0], src.lead_weeks[1], endpoint=True))
        if fam.perishable:  # fresh lines are sourced locally and quickly
            src_name, lead = "Local supplier" if "Local supplier" in spec.sources else src_name, 1
        rows.append(
            {
                "sku": f"{spec.prefix}-{i:04d}",
                "family": fam_name,
                "channel": chan_name,
                "pack": pack,
                "source": src_name,
                "lead_time_weeks": lead,
                "moq": max(4, round(src.moq_units / np.sqrt(size) / 4) * 4),
                "unit_cost": round(fam.unit_cost * size * rng.uniform(0.7, 1.4), 2),
                # Long-lead items legitimately carry more cover; fresh items spoil after a short window.
                "max_cover_weeks": PERISHABLE_MAX_COVER_WEEKS
                if fam.perishable
                else max(TARGET_COVER_WEEKS, round(1.5 * (lead + 2))),
                "gen_perishable": fam.perishable,
                "gen_base": max(0.3, fam.base_units / np.sqrt(size) * rng.lognormal(0, 0.7)),
                "gen_trend": rng.normal(0.0015, 0.002),
                "gen_promo": chan.promo_rate,
                "gen_lumpy": chan.lumpy,
            }
        )
    sku = pd.DataFrame(rows)

    weeks = pd.date_range(spec.start, periods=N_WEEKS, freq="W-MON")
    woy = weeks.isocalendar().week.to_numpy().astype(int)
    season = 1 + 0.08 * np.sin(2 * np.pi * woy / 52)
    for first, last, uplift in spec.seasons:
        season = season + np.isin(woy, range(first, last + 1)) * uplift

    sales = []
    for r in sku.itertuples():
        promo = rng.random(N_WEEKS) < r.gen_promo
        lam = r.gen_base * season * (1 + r.gen_trend) ** np.arange(N_WEEKS) * np.where(promo, 1.45, 1.0)
        qty = rng.poisson(lam)
        if r.gen_lumpy:  # large irregular account orders
            qty = qty * (rng.random(N_WEEKS) < 0.55) * 1.8
        price = r.unit_cost * 1.35 * np.where(promo, 0.9, 1.0)
        sales.append(
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
    sales_df = pd.concat(sales, ignore_index=True)

    # Stock is held relative to each SKU's replenishment time (lead time + review period),
    # with the usual spread: a few nearly empty, most adequate, some overstocked.
    last = sales_df.week.max()
    recent = sales_df[sales_df.week > last - pd.Timedelta(weeks=8)].groupby("sku").units.mean()
    inv = sku[["sku"]].copy()
    factor = rng.choice([0.5, 1.0, 1.4, 2.0, 3.0], N_SKUS, p=[0.06, 0.20, 0.42, 0.20, 0.12])
    cover = factor * np.where(sku.gen_perishable, 1.5, sku.lead_time_weeks.to_numpy() + 2)
    inv["on_hand"] = (recent.reindex(inv.sku).fillna(0).to_numpy() * cover).round()
    has_po = rng.random(N_SKUS) < 0.3
    inv["on_order"] = np.where(has_po, (inv.on_hand * rng.uniform(0.3, 1.2, N_SKUS)).round(), 0)
    inv["on_order_eta_weeks"] = np.where(has_po, rng.integers(1, 6, N_SKUS), 0)

    sku = sku.drop(columns=[c for c in sku.columns if c.startswith("gen_")])
    out = data_dir(domain)
    for name, df in (("skus", sku), ("sales", sales_df), ("inventory", inv)):
        df.to_csv(out / f"{name}.csv", index=False)
    return sku, sales_df, inv


def load(domain: str = "retail") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = data_dir(domain)
    if not (d / "sales.csv").exists():
        return generate(domain)
    return (
        pd.read_csv(d / "skus.csv"),
        pd.read_csv(d / "sales.csv", parse_dates=["week"]),
        pd.read_csv(d / "inventory.csv"),
    )
