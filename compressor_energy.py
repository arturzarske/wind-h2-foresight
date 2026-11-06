"""
Specific energy demand of hydrogen compression (Section 2.3, "Compressor").

Multi-stage compression with intercooling and an equal pressure ratio per
stage. The isentropic work of each stage is computed from real-gas enthalpies,
dh_s = h(p2, s1) - h(p1, T1), using CoolProp [Bell et al., 2014], whose
default equation of state for hydrogen is that of Leachman et al. (2009).
The actual work follows from the isentropic efficiency, w = dh_s / eta_s.

Defaults are those of the paper: 30 -> 250 bar in two stages, eta_s = 0.80,
inlet and intercooling temperature 40 degC.

The script prints the energy demand per kg of hydrogen, the corresponding
model parameter CE (kWh_el per MWh_H2, LHV basis) and the electrical power of
one compressor module with a throughput of 1 MWh_H2/h (about 30 kg/h).

Example:
    python compressor_energy.py
    python compressor_energy.py --p-out 350 --stages 3
"""
from __future__ import annotations

import argparse

import CoolProp.CoolProp as CP

from inputs import MWH_PER_TONNE_H2

FLUID = "Hydrogen"
KWH_PER_KG_LHV = MWH_PER_TONNE_H2      # 33.3 MWh/t = 33.3 kWh/kg


def stage_work(p1_bar: float, p2_bar: float, T1_C: float, eta_s: float):
    """Isentropic and actual specific work of one stage (J/kg) and the outlet
    temperature before intercooling (degC)."""
    p1, p2, T1 = p1_bar * 1e5, p2_bar * 1e5, T1_C + 273.15
    h1 = CP.PropsSI("H", "P", p1, "T", T1, FLUID)
    s1 = CP.PropsSI("S", "P", p1, "T", T1, FLUID)
    h2s = CP.PropsSI("H", "P", p2, "S", s1, FLUID)   # isentropic end state
    w_s = h2s - h1
    w = w_s / eta_s
    T2_C = CP.PropsSI("T", "P", p2, "H", h1 + w, FLUID) - 273.15
    return w_s, w, T2_C


def compression_work(p_in_bar: float, p_out_bar: float, stages: int, eta_s: float,
                     T_in_C: float, T_intercool_C: float):
    """Total actual specific work (J/kg) and per-stage details."""
    r = (p_out_bar / p_in_bar) ** (1.0 / stages)      # equal pressure ratio per stage
    p, T, total, details = p_in_bar, T_in_C, 0.0, []
    for i in range(1, stages + 1):
        w_s, w, T_out = stage_work(p, p * r, T, eta_s)
        details.append((i, p, p * r, T, T_out, w_s, w))
        total += w
        p, T = p * r, T_intercool_C                  # intercooling before the next stage
    return total, details, r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p-in", type=float, default=30.0, help="inlet pressure, bar (electrolyzer outlet)")
    ap.add_argument("--p-out", type=float, default=250.0, help="storage pressure, bar")
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--eta", type=float, default=0.80, help="isentropic efficiency")
    ap.add_argument("--t-in", type=float, default=40.0, help="inlet temperature, degC")
    ap.add_argument("--t-intercool", type=float, default=40.0, help="intercooling temperature, degC")
    a = ap.parse_args()

    w, details, r = compression_work(a.p_in, a.p_out, a.stages, a.eta, a.t_in, a.t_intercool)
    print(f"{FLUID}: {a.p_in:.1f} -> {a.p_out:.1f} bar, {a.stages} stages "
          f"(pressure ratio {r:.3f} per stage), eta_s = {a.eta:.2f}")
    for i, p1, p2, T1, T2, w_s, w_i in details:
        print(f"  stage {i}: {p1:6.1f} -> {p2:6.1f} bar | T_in {T1:5.1f} degC, "
              f"T_out {T2:6.1f} degC | w_s {w_s / 1e3:7.1f} kJ/kg, w {w_i / 1e3:7.1f} kJ/kg")

    w_kwh_per_kg = w / 3.6e6
    ce = w_kwh_per_kg * 1000.0 / KWH_PER_KG_LHV        # kWh_el per MWh_H2
    module_kg_per_h = 1000.0 / KWH_PER_KG_LHV          # 1 MWh_H2/h in kg/h
    print(f"Specific work, isentropic : {sum(d[5] for d in details) / 1e3:8.1f} kJ/kg")
    print(f"Specific work, actual     : {w / 1e3:8.1f} kJ/kg = {w_kwh_per_kg:.3f} kWh/kg")
    print(f"Model parameter CE        : {ce:8.2f} kWh_el per MWh_H2 (LHV)")
    print(f"Power of one module       : {w_kwh_per_kg * module_kg_per_h:8.2f} kW "
          f"(throughput 1 MWh_H2/h = {module_kg_per_h:.2f} kg/h)")


if __name__ == "__main__":
    main()
