"""
Input data: model parameters (parameters.toml) and hourly time series (CSV).

Notation follows Section 3.1 of the paper. All hydrogen quantities are in
MWh based on the lower heating value (LHV); 1 t of hydrogen = 33.3 MWh.
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MWH_PER_TONNE_H2 = 33.3   # LHV of hydrogen, MWh per tonne
HOURS_PER_DAY = 24        # hydrogen is shipped in the last hour of every day

TIMESERIES_COLUMNS = ("wind_power_MWh", "price_EUR_per_MWh")


@dataclass(frozen=True)
class Parameters:
    """Model parameters (paper symbol in brackets)."""

    # Electricity
    grid_fee: float                  # [PG]  EUR/MWh_el
    # Hydrogen sales and purchase agreement
    h2_price: float                  # [PH]  EUR/MWh_H2
    shortfall_penalty: float         # [SP]  EUR/MWh_H2
    min_delivery_t: float            # minimum daily delivery, t/d (-> HSDL)
    initial_storage: float           # hs_0, MWh_H2
    final_storage: float | None      # storage level at the end of the year (strategic model), MWh_H2
    # Electrolyzer
    ely_capacity: float              # [EC]  MW_el input per unit
    ely_efficiency: float            # [EE]  MWh_H2 per MWh_el
    ely_hourly_cost: float           # [EP]  EUR/h per unit
    # Compressor
    comp_capacity: float             # [CC]  MWh_H2/h per unit
    comp_specific_energy: float      # [CE]  MWh_el per MWh_H2
    comp_hourly_cost: float          # [CP]  EUR/h per unit
    # Hydrogen storage
    stor_capacity: float             # [HSC] MWh_H2 per unit
    stor_hourly_cost: float          # [HSP] EUR/h per unit
    # Fuel cell
    fc_capacity: float               # [FC]  MWh_H2/h (hydrogen input) per unit
    fc_efficiency: float             # [FE]  MWh_el per MWh_H2
    fc_hourly_cost: float            # [FP]  EUR/h per unit
    # Scenarios and tactical model
    max_delivery_scenarios: tuple[float, ...]  # maximum daily delivery, t/d (inf = no limit)
    horizons: tuple[int, ...]                  # [L] foresight horizons (block lengths), h
    average_price: float | None                # [PE-bar] for the storage value, EUR/MWh_el
    # Solver
    mip_gap: float | None = None                # relative MIP gap of the strategic model (None = Gurobi default)

    def daily_bounds(self, max_delivery_t: float) -> tuple[float, float]:
        """Minimum and maximum daily delivery [HSDL, HSDH] in MWh_H2."""
        return (self.min_delivery_t * MWH_PER_TONNE_H2,
                max_delivery_t * MWH_PER_TONNE_H2)

    def storage_value(self, prices: np.ndarray) -> float:
        """Value of stored hydrogen [vs] = PE-bar / EE, Eq. (17), in EUR/MWh_H2."""
        avg = self.average_price if self.average_price is not None else float(np.mean(prices))
        return avg / self.ely_efficiency


def load_parameters(path: str | Path) -> Parameters:
    with open(path, "rb") as f:
        p = tomllib.load(f)
    h2, ely, comp = p["hydrogen"], p["electrolyzer"], p["compressor"]
    stor, fc, tac = p["storage"], p["fuel_cell"], p["tactical"]
    return Parameters(
        grid_fee=float(p["electricity"]["grid_fee"]),
        h2_price=float(h2["price"]),
        shortfall_penalty=float(h2["shortfall_penalty"]),
        min_delivery_t=float(h2["min_delivery"]),
        initial_storage=float(h2.get("initial_storage", 0.0)),
        final_storage=(float(h2["final_storage"]) if "final_storage" in h2 else None),
        ely_capacity=float(ely["capacity"]),
        ely_efficiency=float(ely["efficiency"]),
        ely_hourly_cost=float(ely["hourly_cost"]),
        comp_capacity=float(comp["capacity"]),
        comp_specific_energy=float(comp["specific_energy_kWh_per_MWh"]) / 1000.0,
        comp_hourly_cost=float(comp["hourly_cost"]),
        stor_capacity=float(stor["capacity"]),
        stor_hourly_cost=float(stor["hourly_cost"]),
        fc_capacity=float(fc["capacity"]),
        fc_efficiency=float(fc["efficiency"]),
        fc_hourly_cost=float(fc["hourly_cost"]),
        max_delivery_scenarios=tuple(float(x) for x in p["scenarios"]["max_delivery"]),
        horizons=tuple(int(x) for x in tac["horizons"]),
        average_price=(float(tac["average_price"]) if "average_price" in tac else None),
        mip_gap=(float(p["solver"]["mip_gap"]) if "mip_gap" in p.get("solver", {}) else None),
    )


@dataclass(frozen=True)
class TimeSeries:
    """Hourly available wind power [P_i] (MWh) and electricity price [PE_i] (EUR/MWh)."""

    wind: np.ndarray
    price: np.ndarray

    @property
    def hours(self) -> int:
        return len(self.wind)


def load_timeseries(path: str | Path, hours: int | None = None) -> TimeSeries:
    """Read a CSV with the columns wind_power_MWh and price_EUR_per_MWh (one row per hour).

    `hours` truncates the series, e.g. for quick tests. Use a multiple of 24 so that
    every day ends with a shipping hour.
    """
    df = pd.read_csv(path)
    missing = [c for c in TIMESERIES_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    if hours is not None:
        df = df.iloc[:hours]
    wind = df["wind_power_MWh"].to_numpy(dtype=float)
    price = df["price_EUR_per_MWh"].to_numpy(dtype=float)
    if np.isnan(wind).any() or np.isnan(price).any():
        raise ValueError(f"{path}: time series contains empty cells")
    if len(wind) % HOURS_PER_DAY:
        print(f"[warning] {len(wind)} hours is not a multiple of {HOURS_PER_DAY}; "
              "the last, incomplete day has no shipment.")
    return TimeSeries(wind=wind, price=price)


def scenario_name(max_delivery_t: float) -> str:
    """Label as in Table 2, e.g. 'H2 (4)' or 'H2 (inf)'."""
    return "H2 (inf)" if math.isinf(max_delivery_t) else f"H2 ({max_delivery_t:g})"
