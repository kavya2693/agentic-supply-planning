"""Product domains the synthetic generator can produce. The agents never see the domain:
they work on SKUs, stock, forecasts and purchase orders, so a new business is a new
profile here, not new agent code."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    share: float  # share of the SKU range
    base_units: float  # typical weekly units for a single-unit pack
    first_channel_share: float  # probability a SKU sells through the first channel
    unit_cost: float  # cost of one base unit
    perishable: bool = False  # perishables have a short maximum cover


@dataclass(frozen=True)
class Channel:
    promo_rate: float  # share of weeks on promotion
    lumpy: bool  # large irregular orders (B2B accounts) instead of steady sell-through
    packs: dict[str, tuple[float, float]]  # pack name -> (size multiple, share)


@dataclass(frozen=True)
class Source:
    lead_weeks: tuple[int, int]
    moq_units: int  # minimum order in base units; larger packs divide it down
    share: float


@dataclass(frozen=True)
class Domain:
    prefix: str
    start: str
    families: dict[str, Family]
    channels: dict[str, Channel]
    sources: dict[str, Source]
    seasons: tuple[tuple[int, int, float], ...]  # (first week, last week, demand uplift)


RETAIL = Domain(
    prefix="RTL",
    start="2023-10-02",
    families={
        "Fresh Produce": Family(0.11, 140, 0.92, 1.2, perishable=True),
        "Dairy & Bakery": Family(0.12, 150, 0.88, 1.6, perishable=True),
        "Beverages": Family(0.14, 170, 0.80, 0.9),
        "Snacks & Confectionery": Family(0.14, 130, 0.78, 1.1),
        "Personal Care": Family(0.12, 55, 0.70, 3.5),
        "Household Cleaning": Family(0.10, 50, 0.75, 2.8),
        "Baby & Kids": Family(0.08, 35, 0.68, 6.0),
        "Home & Kitchen": Family(0.10, 18, 0.60, 9.0),
        "Electronics Accessories": Family(0.09, 12, 0.50, 14.0),
    },
    channels={
        "Store": Channel(0.10, False, {"Each": (1, 0.6), "Multipack": (4, 0.3), "Case": (12, 0.1)}),
        "Online": Channel(0.14, False, {"Each": (1, 0.7), "Multipack": (4, 0.3)}),
    },
    sources={
        "Local supplier": Source((1, 2), 48, 0.45),
        "Regional distributor": Source((2, 4), 96, 0.35),
        "Imported": Source((6, 10), 240, 0.20),
    },
    # Ramadan and Eid, back to school, year-end sales season, summer slowdown (Gulf retail).
    seasons=((8, 13, 0.30), (34, 36, 0.15), (47, 52, 0.35), (27, 33, -0.15)),
)

LUBRICANTS = Domain(
    prefix="LUB",
    start="2023-10-02",
    families={
        "Engine Oil - Passenger Car": Family(0.28, 60, 0.85, 4.5),
        "Engine Oil - Heavy Duty": Family(0.18, 40, 0.30, 4.5),
        "Motorcycle Oil": Family(0.22, 90, 0.95, 4.0),
        "Gear Oil": Family(0.12, 20, 0.50, 5.0),
        "Hydraulic Oil": Family(0.12, 25, 0.10, 4.0),
        "Industrial Grease": Family(0.08, 15, 0.15, 6.0),
    },
    channels={
        "B2C": Channel(0.08, False, {"1L": (1, 0.5), "3.5L": (3.5, 0.25), "5L": (5, 0.25)}),
        "B2B": Channel(0.02, True, {"20L": (20, 0.6), "209L": (209, 0.4)}),
    },
    sources={
        "Local blending plant": Source((2, 3), 50, 0.65),
        "Imported finished goods": Source((6, 10), 200, 0.35),
    },
    # Festive service season, monsoon slowdown (India).
    seasons=((40, 46, 0.35), (26, 35, -0.20)),
)

DOMAINS = {"retail": RETAIL, "lubricants": LUBRICANTS}
