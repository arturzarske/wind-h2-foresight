# Wind–hydrogen foresight models

Optimization models used in the article

> A. Zarske, B. Samuelsson, *Assessment of the influence of wind foresight length on the
> profitability of wind–hydrogen systems*, International Journal of Hydrogen Energy
> (submitted). <!-- TODO: replace by the DOI of the article once published -->

The models size and operate a hydrogen system (PEM electrolyzers, compressors, pressurized
storage, fuel cells) attached to an existing, grid-connected wind farm. Hydrogen is sold
once per day under a purchase agreement with a minimum and a maximum daily delivery.

* **Strategic model** (MILP): optimal numbers of units under perfect foresight of one year.
* **Tactical model** (LP): operation of the sized system by sequential block-wise
  optimization, where the block length is the foresight horizon (1 h … 1 year).

## Files

| File | Content | Paper |
|---|---|---|
| `strategic_model.py` | Sizing MILP under perfect foresight | Sec. 3.2, Eqs. (2)–(16) |
| `tactical_model.py` | Sequential block-wise operation, value of stored hydrogen | Sec. 3.3, Eqs. (17)–(19) |
| `system.py` | System constraints shared by both models | Eqs. (5)–(16) |
| `reference_cases.py` | Wind farm without hydrogen (Base, Curtail) | Sec. 3.4, Eqs. (20)–(22) |
| `run_all.py` | Runs all scenarios and horizons (reproduces Tables 3–4, Figs. 2–5) | Sec. 4 |
| `compressor_energy.py` | Compression energy 30 → 250 bar with CoolProp (parameter CE) | Sec. 2.3 |
| `inputs.py` | Reading of parameters and time series | Sec. 3.1 |
| `parameters.toml` | All model parameters with units and paper symbols | Table 1, Sec. 2 |
| `data/example_timeseries.csv` | **Synthetic** example time series (see below) | – |

## Installation

Requires Python ≥ 3.11 and [Gurobi](https://www.gurobi.com) 13 with a licence
(free for academic use).

```bash
conda env create -f environment.yml
conda activate wind-h2-foresight
```

or `pip install -r requirements.txt`. The pip version of `gurobipy` includes a
size-limited licence, which is enough for the quick test below but not for a full year.

## Input data

`parameters.toml` contains all parameters used in the paper. The hourly time series are
read from a CSV file with one row per hour:

```
hour,wind_power_MWh,price_EUR_per_MWh
1,17.335,39.66
2,14.961,48.49
...
```

`wind_power_MWh` is the available wind power P_i, and `price_EUR_per_MWh` is the
day-ahead price PE_i (before the grid fee). A full year has 8760 rows.

**The measured production data of the wind farm used in the paper are confidential and
are not included.** Day-ahead prices for bidding zone SE3 are available from the
[ENTSO-E Transparency Platform](https://transparency.entsoe.eu).
`data/example_timeseries.csv` contains *synthetic* data for a fictitious 36 MW wind farm,
so that the code can be run without the original data; results obtained with it do not
correspond to those in the paper.

## Usage

Reproduce all results (expects the time series in `data/timeseries.csv`):

```bash
python run_all.py
```

Quick test with the synthetic data (runs with the size-limited pip licence):

```bash
python run_all.py --timeseries data/example_timeseries.csv --hours 96
```

The results are written to `results/strategic_results.csv` (installed units, revenues,
costs, profit and use of wind electricity per scenario, incl. Base and Curtail) and
`results/tactical_results.csv` (profit, shortfall penalty and share of the
perfect-foresight profit for every scenario and foresight horizon). Add `--hourly` to
also save the hourly operation of every run.

Single runs:

```bash
python strategic_model.py --max-delivery 4
python tactical_model.py --max-delivery 4 --horizon 24 \
    --electrolyzers 12 --compressors 8 --storage 249 --fuel-cells 0
python compressor_energy.py
```

On a laptop (Apple M3), each strategic run and each full-year tactical run takes a few
seconds.

## Modelling conventions

* Hourly time step; energy (MWh) and power (MW) are numerically equal.
* Hydrogen quantities in MWh based on the lower heating value, 1 t = 33.3 MWh.
* Hydrogen is shipped in the last hour of every day (hours 24, 48, …); deliveries below
  the minimum are allowed but penalized, the maximum is a hard limit.
* Storage is empty at the start of the year and, in the strategic model, at its end.
* The strategic model is solved to proven optimality (relative MIP gap 0, set in
  `[solver]` of `parameters.toml`); otherwise Gurobi default settings are used.

## Citation

If you use this code, please cite the article above and this software
(see `CITATION.cff`; the archived version has the DOI
[10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)). <!-- TODO -->

## Licence

MIT, see `LICENSE`.
