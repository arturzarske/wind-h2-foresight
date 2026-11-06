"""
Physical representation of the wind-hydrogen system, Eqs. (5)-(16) of the paper.

The same function is used by the strategic model (installed units are integer
decision variables) and by the tactical model (installed units are fixed
numbers), so both models share exactly the same system representation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gurobipy as gp
import numpy as np
import pandas as pd

from inputs import HOURS_PER_DAY, MWH_PER_TONNE_H2, Parameters


@dataclass
class Units:
    """Installed units [elyins, compins, storins, fcins]: numbers or Gurobi variables."""

    electrolyzers: object
    compressors: object
    storage: object
    fuel_cells: object

    def as_ints(self) -> "Units":
        return Units(*(int(round(getattr(u, "X", u))) for u in
                       (self.electrolyzers, self.compressors, self.storage, self.fuel_cells)))


def hourly_fixed_cost(par: Parameters, u: Units):
    """EP*elyins + CP*compins + HSP*storins + FP*fcins (EUR/h), cf. Eq. (4)."""
    return (par.ely_hourly_cost * u.electrolyzers + par.comp_hourly_cost * u.compressors
            + par.stor_hourly_cost * u.storage + par.fc_hourly_cost * u.fuel_cells)


def shipping_hours(first_hour: int, n_hours: int) -> list[int]:
    """Local indices of the shipping hours (last hour of each day) within
    the global hours first_hour .. first_hour + n_hours - 1 (0-based)."""
    return [i for i in range(n_hours) if (first_hour + i + 1) % HOURS_PER_DAY == 0]


@dataclass
class Operation:
    """Variables and objective terms of one optimization problem."""

    vars: dict
    shipping: list[int]
    shortfall: gp.tupledict
    revenue: gp.LinExpr          # Eq. (3)
    penalty: gp.LinExpr          # SP * sum(hshort_k), Eq. (4)


def add_operation(m: gp.Model, wind: np.ndarray, price: np.ndarray, par: Parameters,
                  units: Units, hs_start: float, first_hour: int,
                  max_delivery_t: float) -> Operation:
    """Add the operation variables and constraints (5)-(16) for the given hours.

    wind, price : available wind power P_i (MWh) and price PE_i (EUR/MWh_el)
    units       : installed units (numbers or integer variables)
    hs_start    : storage level before the first hour, hs_0 (MWh_H2)
    first_hour  : global index of the first hour (to place the daily shipments)
    """
    T = range(len(wind))
    hsdl, hsdh = par.daily_bounds(max_delivery_t)

    ee = m.addVars(T, name="ee")              # electricity to electrolyzers (MWh)
    ecomp = m.addVars(T, name="ecomp")        # electricity to compressors (MWh)
    eg = m.addVars(T, name="eg")              # electricity sold to the grid (MWh)
    ec = m.addVars(T, name="ec")              # curtailed electricity (MWh)
    ef = m.addVars(T, name="ef")              # electricity from fuel cells (MWh)
    hcharge = m.addVars(T, name="hcharge")    # hydrogen produced (MWh_H2)
    hs = m.addVars(T, name="hs")              # storage level at end of hour (MWh_H2)
    hsold = m.addVars(T, name="hsold")        # hydrogen sold (MWh_H2)
    hf = m.addVars(T, name="hf")              # hydrogen to fuel cells (MWh_H2)

    # (5) Energy balance: wind power is sold, used, or curtailed (no grid purchase)
    m.addConstrs((wind[i] == ee[i] + ecomp[i] + eg[i] + ec[i] for i in T), "energy_balance")
    # (6)-(7) Electrolyzer capacity and hydrogen production
    m.addConstrs((ee[i] <= par.ely_capacity * units.electrolyzers for i in T), "ely_capacity")
    m.addConstrs((hcharge[i] == par.ely_efficiency * ee[i] for i in T), "h2_production")
    # (8)-(9) All hydrogen is compressed before storage
    m.addConstrs((hcharge[i] <= par.comp_capacity * units.compressors for i in T), "comp_capacity")
    m.addConstrs((ecomp[i] == par.comp_specific_energy * hcharge[i] for i in T), "comp_energy")
    # (10)-(11) Storage balance and capacity
    m.addConstrs((hs[i] == (hs[i - 1] if i > 0 else hs_start) + hcharge[i] - hsold[i] - hf[i]
                  for i in T), "storage_balance")
    m.addConstrs((hs[i] <= par.stor_capacity * units.storage for i in T), "storage_capacity")
    # (12)-(13) Fuel cells
    m.addConstrs((hf[i] <= par.fc_capacity * units.fuel_cells for i in T), "fc_capacity")
    m.addConstrs((ef[i] == par.fc_efficiency * hf[i] for i in T), "fc_production")

    # (14) Hydrogen is only sold in the shipping hours
    ship = shipping_hours(first_hour, len(wind))
    ship_set = set(ship)
    m.addConstrs((hsold[i] == 0 for i in T if i not in ship_set), "no_sale")
    # (15) Soft minimum delivery with shortfall, (16) hard maximum delivery
    K = range(len(ship))
    hshort = m.addVars(K, name="hshort")
    m.addConstrs((hsold[ship[k]] + hshort[k] >= hsdl for k in K), "min_delivery")
    if not math.isinf(hsdh):                       # no upper limit in H2 (inf)
        m.addConstrs((hsold[ship[k]] <= hsdh for k in K), "max_delivery")

    revenue = gp.quicksum((price[i] - par.grid_fee) * (eg[i] + ef[i]) + par.h2_price * hsold[i]
                          for i in T)
    penalty = par.shortfall_penalty * hshort.sum()
    return Operation(vars=dict(ee=ee, ecomp=ecomp, eg=eg, ec=ec, ef=ef, hcharge=hcharge,
                               hs=hs, hsold=hsold, hf=hf),
                     shipping=ship, shortfall=hshort, revenue=revenue, penalty=penalty)


def operation_values(m: gp.Model, op: Operation, wind, price, first_hour: int) -> dict:
    """Optimal hourly values as arrays (hshort is placed at the shipping hour)."""
    n = len(wind)
    data = {"hour": np.arange(first_hour, first_hour + n),
            "P": np.asarray(wind, dtype=float), "PE": np.asarray(price, dtype=float)}
    for name, v in op.vars.items():
        data[name] = np.fromiter(m.getAttr("X", v).values(), dtype=float, count=n)
    short = np.zeros(n)
    for k, i in enumerate(op.shipping):
        short[i] = op.shortfall[k].X
    data["hshort"] = short
    return data


def hourly_results(m: gp.Model, op: Operation, wind, price, first_hour: int) -> pd.DataFrame:
    """Optimal hourly values as a table (one row per hour, global hour index)."""
    return pd.DataFrame(operation_values(m, op, wind, price, first_hour))


def summarize(hourly: pd.DataFrame, par: Parameters, units: Units) -> dict:
    """Annual totals (EUR, MWh) and the allocation of the wind electricity (Fig. 2)."""
    n = len(hourly)
    net_price = hourly["PE"] - par.grid_fee
    rev_el = float((net_price * (hourly["eg"] + hourly["ef"])).sum())
    rev_h2 = float(par.h2_price * hourly["hsold"].sum())
    fixed = float(hourly_fixed_cost(par, units) * n)
    penalty = float(par.shortfall_penalty * hourly["hshort"].sum())
    total_wind = float(hourly["P"].sum())
    share = (lambda col: 100.0 * float(hourly[col].sum()) / total_wind if total_wind else 0.0)
    return {
        "revenue_hydrogen": rev_h2,
        "revenue_electricity": rev_el,
        "shortfall_penalty": penalty,
        "fixed_cost": fixed,
        "profit": rev_h2 + rev_el - fixed - penalty,
        "h2_delivered_t": float(hourly["hsold"].sum()) / MWH_PER_TONNE_H2,
        "h2_shortfall_t": float(hourly["hshort"].sum()) / MWH_PER_TONNE_H2,
        "share_grid_pct": share("eg"),
        "share_electrolyzer_pct": share("ee"),
        "share_compressor_pct": share("ecomp"),
        "share_curtailed_pct": share("ec"),
    }
