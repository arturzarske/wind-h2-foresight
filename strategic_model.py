"""
Strategic model (Section 3.2): sizing under perfect foresight.

Mixed-integer linear program that chooses the numbers of electrolyzer,
compressor, storage and fuel cell units together with their hourly operation
over the whole year, maximizing the annual profit, Eqs. (2)-(16).

Example:
    python strategic_model.py --max-delivery 4
    python strategic_model.py --max-delivery inf --hourly results/strategic_inf.csv
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from inputs import (MWH_PER_TONNE_H2, Parameters, TimeSeries, load_parameters,
                    load_timeseries, scenario_name)
from system import Units, add_operation, hourly_fixed_cost, hourly_results, summarize


@dataclass
class StrategicResult:
    units: Units                 # optimal installed units (integers)
    hourly: pd.DataFrame         # hourly operation
    summary: dict                # annual totals


def solve_strategic(ts: TimeSeries, par: Parameters, max_delivery_t: float,
                    env: gp.Env | None = None) -> StrategicResult:
    """Solve the strategic MILP for one purchase-agreement scenario.

    max_delivery_t : maximum daily delivery in t/d (float('inf') for no limit)
    """
    start = time.perf_counter()
    m = gp.Model("strategic", env=env)
    if par.mip_gap is not None:          # 0 = solve to proven optimality
        m.Params.MIPGap = par.mip_gap

    u = Units(*(m.addVar(vtype=GRB.INTEGER, name=n)
                for n in ("elyins", "compins", "storins", "fcins")))
    op = add_operation(m, ts.wind, ts.price, par, u, hs_start=par.initial_storage,
                       first_hour=0, max_delivery_t=max_delivery_t)
    if par.final_storage is not None:
        last = ts.hours - 1
        m.addConstr(op.vars["hs"][last] == par.final_storage, "final_storage")

    # Objective (2)-(4): revenue minus annualized fixed costs and shortfall penalty
    m.setObjective(op.revenue - ts.hours * hourly_fixed_cost(par, u) - op.penalty, GRB.MAXIMIZE)
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"Strategic model found no solution (Gurobi status {m.Status}).")

    units = u.as_ints()
    hourly = hourly_results(m, op, ts.wind, ts.price, first_hour=0)
    summary = {
        "scenario": scenario_name(max_delivery_t),
        "max_delivery_t_per_day": max_delivery_t,
        "electrolyzers": units.electrolyzers,
        "compressors": units.compressors,
        "storage_units": units.storage,
        "fuel_cells": units.fuel_cells,
        # Capacities as reported in Table 3
        "electrolyzer_MW": units.electrolyzers * par.ely_capacity,
        "compressor_kg_per_h": units.compressors * par.comp_capacity * 1000.0 / MWH_PER_TONNE_H2,
        "storage_t": units.storage * par.stor_capacity / MWH_PER_TONNE_H2,
        "fuel_cell_MW_el": units.fuel_cells * par.fc_capacity * par.fc_efficiency,
        **summarize(hourly, par, units),
        "objective": m.ObjVal,
        "mip_gap": m.MIPGap,
        "runtime_s": time.perf_counter() - start,
    }
    return StrategicResult(units=units, hourly=hourly, summary=summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="parameters.toml")
    ap.add_argument("--timeseries", default="data/timeseries.csv")
    ap.add_argument("--hours", type=int, default=None, help="use only the first N hours (testing)")
    ap.add_argument("--max-delivery", type=float, required=True,
                    help="maximum daily delivery in t/d, e.g. 2, 4, 6 or inf")
    ap.add_argument("--hourly", default=None, help="write the hourly operation to this CSV file")
    a = ap.parse_args()

    par = load_parameters(a.params)
    ts = load_timeseries(a.timeseries, a.hours)
    res = solve_strategic(ts, par, a.max_delivery)
    for k, v in res.summary.items():
        print(f"{k:28s} {v:,.4f}" if isinstance(v, float) else f"{k:28s} {v}")
    if a.hourly:
        res.hourly.to_csv(a.hourly, index=False)
        print(f"Hourly results written to {a.hourly}")


if __name__ == "__main__":
    main()
