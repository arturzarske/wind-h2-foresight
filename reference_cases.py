"""
Reference cases without hydrogen infrastructure (Section 3.4), Eqs. (20)-(22).

Base    : all available wind power is sold, eg_i = P_i.
Curtail : wind power may be curtailed; maximizing the profit curtails the whole
          output in all hours with PE_i < PG. Both cases are solved in closed
          form, since every hour is independent.
"""
from __future__ import annotations

import numpy as np

from inputs import Parameters, TimeSeries


def reference_cases(ts: TimeSeries, par: Parameters) -> list[dict]:
    net = ts.price - par.grid_fee                  # PE_i - PG
    total = float(ts.wind.sum())
    curtail = net < 0
    rows = []
    for name, eg in (("Base", ts.wind), ("Curtail", np.where(curtail, 0.0, ts.wind))):
        profit = float((net * eg).sum())           # Eq. (20)
        rows.append({
            "scenario": name,
            "revenue_hydrogen": 0.0,
            "revenue_electricity": profit,
            "shortfall_penalty": 0.0,
            "fixed_cost": 0.0,
            "profit": profit,
            "share_grid_pct": 100.0 * float(eg.sum()) / total,
            "share_electrolyzer_pct": 0.0,
            "share_compressor_pct": 0.0,
            "share_curtailed_pct": 100.0 * (1.0 - float(eg.sum()) / total),
        })
    return rows
