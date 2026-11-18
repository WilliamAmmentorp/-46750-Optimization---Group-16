"""Solve Q1 models over contract-term variations and tabulate community and DSO metrics."""
from __future__ import annotations
import pandas as pd
from .model import Model, Results
from .variations import with_discount, with_penalty, with_cap_scale

def _row(experiment: str, value: float, model: str, com, res: Results) -> dict:
    """One table row. Reference for 'gain' columns is added afterwards (cost in R, cost in I)."""
    m, h = res.meta["metrics"], res.hourly
    beta = 0.0 if model == "R" else com.contract.tariff_discount_ratio      # R pays full tariffs
    tariff_rev = (1 - beta) * (com.import_tariff * h["import"].sum() + com.export_tariff * h["export"].sum())
    return {
        "experiment": experiment, 
        "parameter_value": value, 
        "model": model,
        "total_cost_DKK": m["total_cost_DKK"],
        "cost_grid_DKK": m["cost_grid_DKK"], 
        "cost_pv_DKK": m["cost_pv_DKK"],
        "cost_disutility_DKK": m["cost_disutility_DKK"], 
        "cost_penalty_DKK": m["cost_penalty_DKK"],
        "peak_import_kW": m["peak_import_kW"], 
        "exceedance_kWh": m["exceedance_kWh"],
        "hours_above_cap": m["hours_above_cap"], 
        "energy_shifted_kWh": m["energy_shifted_kWh"],
        "dso_tariff_revenue_DKK": tariff_rev,
        "dso_penalty_revenue_DKK": m["cost_penalty_DKK"],

    }

def sensitivity_rows_Q1(com, experiments: dict, models=("I", "C"), verbose=False) -> pd.DataFrame:
    """experiments = {name: (label, base_value, {value: community_variant})}.
    R does not depend on the contract terms: solved once on the base case and repeated for plotting."""
    res_R = Model(com, "R").build().solve()
    rows = []
    for name, (_, _, cases) in experiments.items():
        for value, variant in cases.items():
            rows.append(_row(name, value, "R", com, res_R))
            for model in models:
                res = Model(variant, model, verbose=verbose).build().solve()
                rows.append(_row(name, value, model, variant, res))
    t = pd.DataFrame(rows)

    key = ["experiment", "parameter_value"]
    cost = t.pivot_table(index=key, columns="model", values="total_cost_DKK")
    rev = t.assign(r=t.dso_tariff_revenue_DKK + t.dso_penalty_revenue_DKK).pivot_table(index=key, columns="model", values="r")
    idx = pd.MultiIndex.from_frame(t[key])
    t["gain_vs_R_DKK"] = cost["R"].reindex(idx).to_numpy() - t["total_cost_DKK"]          # community point of view
    
    peak = t.pivot_table(index=key, columns="model", values="peak_import_kW")
    t["peak_reduction_vs_R_kW"] = peak["R"].reindex(idx).to_numpy() - t["peak_import_kW"]
    if "I" in cost:
        t["gain_vs_I_DKK"] = cost["I"].reindex(idx).to_numpy() - t["total_cost_DKK"]      # value of pooling behind the meter
    t["dso_revenue_change_vs_R_DKK"] = (t.dso_tariff_revenue_DKK + t.dso_penalty_revenue_DKK) - rev["R"].reindex(idx).to_numpy()
    return t

def contract_experiments_Q1(com) -> dict:
    """The sweeps of Q1: discount and penalty, one at a time around the base case."""
    c = com.contract
    return {
        "discount": ("Tariff discount beta", c.tariff_discount_ratio,
                     {b: with_discount(com, b) for b in [0.0, 0.25, 0.5, 0.75, 1.0]}),
        "penalty": ("Penalty alpha [DKK/kWh]", c.penalty_DKK_per_kWh,
                    {a: with_penalty(com, a) for a in [0.0, 1.0, 2.5, 5.0, 10.0, 20.0]}),
        "cap": ("Cap scale", 1.0, {s: with_cap_scale(com, s) for s in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]}),
    }


MODEL_VARIANTS = {
    "DP":       dict(price_per_member=False, bb="none"),
    "DP_dis":   dict(price_per_member=True,  bb="none"),
    "DP_bb":    dict(price_per_member=False, bb="eq"),
    "DP_disbb": dict(price_per_member=True,  bb="eq"),
}

def contract_experiments_Q2(com) -> dict:
    """Q2 sweeps: penalty, posted-price bound, and capacity scale, one at a time."""
    c = com.contract
    x_default = c.penalty_DKK_per_kWh + com.energy_price.max() + com.import_tariff    # the bound Model uses by default (13.0)
    return {
        "penalty": ("Penalty alpha [DKK/kWh]", c.penalty_DKK_per_kWh,
                    {a: with_penalty(com, a) for a in [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0]}),
        "price_bound": ("Price bound X [DKK/kWh]", x_default,
                        {X: com for X in [3.0, 5.0, 7.0, 10.0, x_default, 16.0, 20.0]}),
        "cap_scale": ("Capacity scale", 1.0,
                      {s: with_cap_scale(com, s) for s in [0.6, 0.8, 1.0, 1.2, 1.5]}),
    }

def sensitivity_rows_Q2(com, experiments: dict, models=("DP", "DP_dis", "DP_bb", "DP_disbb"), verbose=False) -> pd.DataFrame:
    rows = []
    for name, (_, _, cases) in experiments.items():
        for value, variant in cases.items():
            resC = Model(variant, "C").build().solve()                       # reference: central dispatch (cheap LP)
            rows.append({**_row(name, value, "C", variant, resC), "status": "OK"})
            for model in models:
                kw = dict(MODEL_VARIANTS[model])
                if name == "price_bound":
                    kw["price_bound"] = value
                print(f"[{name} = {value}] solving {model} ...")
                try:
                    res = Model(variant, "DP", verbose=verbose, **kw).build().solve()
                    row = _row(name, value, model, variant, res)
                    row["status"] = "OK"
                    row["budget_DKK"] = res.meta["metrics"]["manager_budget_DKK"]
                except RuntimeError as e:                                    # no feasible solution: record it, keep going
                    print("   no solution:", e)
                    row = dict(experiment=name, parameter_value=value, model=model, status="NO SOLUTION")
                rows.append(row)
    t = pd.DataFrame(rows)

    key = ["experiment", "parameter_value"]
    cost = t.pivot_table(index=key, columns="model", values="total_cost_DKK")
    idx = pd.MultiIndex.from_frame(t[key])
    t["loss_vs_C_DKK"] = t["total_cost_DKK"] - cost["C"].reindex(idx).to_numpy()      # efficiency cost of DP and its requirements
    return t