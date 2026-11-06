"""
Tactical model (Section 3.3): operation under limited foresight.

The installed units are fixed (normally taken from the strategic model). The
year is divided into consecutive, non-overlapping blocks of length L (the
foresight horizon). Each block is solved as a linear program with perfect
information inside the block and none beyond it; the storage level at the end
of a block is the initial level of the next one. Hydrogen left in storage is
valued at vs = PE-bar / EE, Eqs. (17)-(19).

Example (units as found by the strategic model):
    python tactical_model.py --max-delivery 4 --horizon 24 \
        --electrolyzers 12 --compressors 8 --storage 249 --fuel-cells 0
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from inputs import Parameters, TimeSeries, load_parameters, load_timeseries, scenario_name
from system import Units, add_operation, hourly_fixed_cost, operation_values, summarize


@dataclass
class TacticalResult:
    hourly: pd.DataFrame
    summary: dict


def solve_block(wind, price, par: Parameters, units: Units, hs_start: float, first_hour: int,
                max_delivery_t: float, vs: float, env: gp.Env | None = None):
    """Solve one block b. Returns (hourly values, storage level at block end, objective)."""
    m = gp.Model("tactical_block", env=env)
    op = add_operation(m, wind, price, par, units, hs_start=hs_start,
                       first_hour=first_hour, max_delivery_t=max_delivery_t)
    T = range(len(wind))
    hs = op.vars["hs"]

    # (18) Hourly change in storage level, valued at vs in the objective
    dhs = m.addVars(T, lb=-GRB.INFINITY, name="dhs")
    m.addConstrs((dhs[i] == hs[i] - (hs[i - 1] if i > 0 else hs_start) for i in T), "dhs")

    # (19) Block profit incl. fixed costs of the |I_b| hours and the value of stored hydrogen.
    # The fixed costs are constant here; they only make the block objectives add up to
    # an annual profit that is comparable with the strategic model.
    fixed = len(wind) * hourly_fixed_cost(par, units)
    m.setObjective(op.revenue - fixed - op.penalty + vs * dhs.sum(), GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Block starting at hour {first_hour} not solved (status {m.Status}).")

    values = operation_values(m, op, wind, price, first_hour)
    values["dhs"] = np.fromiter(m.getAttr("X", dhs).values(), dtype=float, count=len(wind))
    return values, values["hs"][-1], m.ObjVal


def simulate(ts: TimeSeries, par: Parameters, units: Units, horizon: int,
             max_delivery_t: float, env: gp.Env | None = None) -> TacticalResult:
    """Sequential block-wise optimization over the whole year for one horizon L."""
    start = time.perf_counter()
    if env is None:
        env = gp.Env(params={"OutputFlag": 0})
    vs = par.storage_value(ts.price)
    n = ts.hours
    blocks, objective, hs = [], 0.0, par.initial_storage
    for b, t0 in enumerate(range(0, n, horizon)):
        t1 = min(t0 + horizon, n)          # the final block may be shorter
        values, hs, obj = solve_block(ts.wind[t0:t1], ts.price[t0:t1], par, units, hs, t0,
                                      max_delivery_t, vs, env)
        values["block"] = np.full(t1 - t0, b)
        blocks.append(values)
        objective += obj

    hourly = pd.DataFrame({k: np.concatenate([blk[k] for blk in blocks]) for k in blocks[0]})
    s = summarize(hourly, par, units)
    summary = {
        "scenario": scenario_name(max_delivery_t),
        "max_delivery_t_per_day": max_delivery_t,
        "horizon_h": horizon,
        "blocks": len(blocks),
        **s,
        "storage_value_EUR_per_MWh": vs,
        "storage_value_change": vs * (hourly["hs"].iloc[-1] - par.initial_storage),
        "objective_sum_of_blocks": objective,   # = profit + storage_value_change
        "runtime_s": time.perf_counter() - start,
    }
    return TacticalResult(hourly=hourly, summary=summary)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="parameters.toml")
    ap.add_argument("--timeseries", default="data/timeseries.csv")
    ap.add_argument("--hours", type=int, default=None, help="use only the first N hours (testing)")
    ap.add_argument("--max-delivery", type=float, required=True,
                    help="maximum daily delivery in t/d, e.g. 2, 4, 6 or inf")
    ap.add_argument("--horizon", type=int, required=True, help="foresight horizon L in hours")
    ap.add_argument("--electrolyzers", type=int, required=True)
    ap.add_argument("--compressors", type=int, required=True)
    ap.add_argument("--storage", type=int, required=True)
    ap.add_argument("--fuel-cells", type=int, default=0)
    ap.add_argument("--hourly", default=None, help="write the hourly operation to this CSV file")
    a = ap.parse_args()

    par = load_parameters(a.params)
    ts = load_timeseries(a.timeseries, a.hours)
    units = Units(a.electrolyzers, a.compressors, a.storage, a.fuel_cells)
    res = simulate(ts, par, units, a.horizon, a.max_delivery)
    for k, v in res.summary.items():
        print(f"{k:28s} {v:,.4f}" if isinstance(v, float) else f"{k:28s} {v}")
    if a.hourly:
        res.hourly.to_csv(a.hourly, index=False)
        print(f"Hourly results written to {a.hourly}")


if __name__ == "__main__":
    main()
