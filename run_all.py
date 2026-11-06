"""
Reproduce all results of the paper.

1. Reference cases Base and Curtail (Section 3.4)
2. Strategic model for every purchase-agreement scenario (Section 3.2, Tables 3-4, Fig. 2)
3. Tactical model for every scenario and foresight horizon, using the units
   found in step 2 (Section 3.3, Figs. 3-5)

Results are written to results/strategic_results.csv and
results/tactical_results.csv (optionally also the hourly operation).

Examples:
    python run_all.py
    python run_all.py --timeseries data/example_timeseries.csv --hours 96   # quick test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gurobipy as gp
import pandas as pd

from inputs import load_parameters, load_timeseries
from reference_cases import reference_cases
from strategic_model import solve_strategic
from tactical_model import simulate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="parameters.toml")
    ap.add_argument("--timeseries", default="data/timeseries.csv")
    ap.add_argument("--hours", type=int, default=None, help="use only the first N hours (testing)")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--hourly", action="store_true", help="also write hourly results of every run")
    a = ap.parse_args()

    par = load_parameters(a.params)
    ts = load_timeseries(a.timeseries, a.hours)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = gp.Env(params={"OutputFlag": 0})

    # 1) Reference cases
    strategic_rows = reference_cases(ts, par)
    curtail_profit = strategic_rows[1]["profit"]
    for r in strategic_rows:
        r["change_vs_curtail_pct"] = 100.0 * (r["profit"] / curtail_profit - 1)

    tactical_rows = []
    for hmax in par.max_delivery_scenarios:
        # 2) Strategic model: sizing under perfect foresight
        st = solve_strategic(ts, par, hmax, env=env)
        st.summary["change_vs_curtail_pct"] = 100.0 * (st.summary["profit"] / curtail_profit - 1)
        strategic_rows.append(st.summary)
        name = st.summary["scenario"]
        u = st.units
        print(f"{name:9s} strategic: {u.electrolyzers} ely, {u.compressors} comp, "
              f"{u.storage} storage, {u.fuel_cells} FC units | "
              f"profit {st.summary['profit'] / 1e6:.3f} MEUR")
        if a.hourly:
            st.hourly.to_csv(out / f"hourly_strategic_{_slug(name)}.csv", index=False)

        # 3) Tactical model for all foresight horizons
        rows = []
        for L in par.horizons:
            tr = simulate(ts, par, u, L, hmax, env=env)
            rows.append(tr.summary)
            if a.hourly:
                tr.hourly.to_csv(out / f"hourly_tactical_{_slug(name)}_L{L}.csv", index=False)
        # Share of the perfect-foresight profit (L = N, i.e. one block for the whole year)
        pf = next((r["profit"] for r in rows if r["horizon_h"] >= ts.hours), st.summary["profit"])
        for r in rows:
            r["share_of_perfect_foresight_pct"] = 100.0 * r["profit"] / pf
            print(f"{'':9s} L = {r['horizon_h']:5d} h: profit {r['profit'] / 1e6:.3f} MEUR "
                  f"({r['share_of_perfect_foresight_pct']:5.1f} %), "
                  f"penalty {r['shortfall_penalty'] / 1e3:8.1f} kEUR")
        # Consistency check (Section 4.2): with L = N the tactical model must
        # reproduce the profit of the strategic model.
        if abs(pf - st.summary["profit"]) > 1e-4 * max(1.0, abs(pf)):
            print(f"[note] {name}: tactical profit at L = N ({pf:,.2f}) differs from the "
                  f"strategic profit ({st.summary['profit']:,.2f})")
        tactical_rows.extend(rows)

    pd.DataFrame(strategic_rows).to_csv(out / "strategic_results.csv", index=False)
    pd.DataFrame(tactical_rows).to_csv(out / "tactical_results.csv", index=False)
    print(f"\nResults written to {out.resolve()}")


def _slug(name: str) -> str:
    return name.replace(" ", "").replace("(", "_").replace(")", "")


if __name__ == "__main__":
    main()
