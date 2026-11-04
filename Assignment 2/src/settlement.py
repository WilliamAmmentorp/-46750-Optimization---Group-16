"""Settlement of the community contract (model C): internal prices, member payments, individual rationality, budget.

Under central dispatch the manager charges every member the uniform hourly price x_t = -(dual of the community balance),
so member i pays  pay_i = sum_t x_t * (import_i,t - export_i,t)  and bears its own PV and disutility cost.
Everything here is post-processing of solved models R, I and C: no optimisation is run.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TOL = 1e-6   # DKK tolerance for IR and budget tests


def _asset_class(archetype: str) -> str:
    """'L2+PV+BAT' -> 'PV+BAT', 'L2' -> 'none'."""
    parts = archetype.split("+")[1:]
    return "+".join(parts) if parts else "none"


def settle_Q1(res_C, res_R, res_I, com) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """Settle the central dispatch with the dual prices and test IR and budget balance.

    Inputs : Results of models C, R, I (solved on the same community ``com``).
    Returns: (members, hourly, summary, by_type)
      members : one row per member: payment, cost under C after settlement, cost in R and I, gains [DKK], IR flags.
      hourly  : posted price x_t, collected payments, manager's cost and surplus per hour [DKK].
      summary : community gain, budget (collected - manager cost), IR violations, shortfall [DKK].
      by_type : gains and IR violations summed by load class and by asset class.
    """
    h = res_C.hourly
    x = -h["internal_price_dual"].to_numpy()                       # internal price [DKK/kWh], same sign as model.py (DP)

    mC, mR, mI = (r.meta["members"] for r in (res_C, res_R, res_I))
    sched = res_C.meta["schedules"].copy()
    sched["net_import"] = sched["import"] - sched["export"]
    sched["pay"] = sched["hour"].map(dict(enumerate(x))) * sched["net_import"]
    pay = sched.groupby("member")["pay"].sum().reindex(mC.index)

    members = pd.DataFrame(index=mC.index)
    members["archetype"] = mC["archetype"]
    members["load_class"] = members["archetype"].str.split("+").str[0]
    members["assets"] = members["archetype"].map(_asset_class)
    members["net_import_kWh"] = sched.groupby("member")["net_import"].sum().reindex(mC.index)
    members["payment_DKK"] = pay
    members["cost_C_DKK"] = mC["pv"] + mC["disutility"] + pay      # own PV + disutility + settlement payment
    members["cost_R_DKK"] = mR["total"]
    members["cost_I_DKK"] = mI["total"]
    members["gain_vs_R_DKK"] = members["cost_R_DKK"] - members["cost_C_DKK"]
    members["gain_vs_I_DKK"] = members["cost_I_DKK"] - members["cost_C_DKK"]
    members["gain_vs_best_alt_DKK"] = np.minimum(members["cost_R_DKK"], members["cost_I_DKK"]) - members["cost_C_DKK"]
    members["IR_vs_R"] = members["gain_vs_R_DKK"] >= -TOL
    members["IR_vs_I"] = members["gain_vs_I_DKK"] >= -TOL

    # manager's side, hour by hour: pays the DSO for the metered flows and the penalty, collects x_t * (net import)
    d0, c = com.members[0], com.contract
    disc, alpha = 1.0 - c.tariff_discount_ratio, c.penalty_DKK_per_kWh
    imp, exp_, exc = h["import"].to_numpy(), h["export"].to_numpy(), h["excess"].to_numpy()
    mgr_cost = ((d0.energy_price + disc * d0.import_tariff) * imp - (d0.energy_price - disc * d0.export_tariff) * exp_
                + alpha * exc)
    collected = x * (imp - exp_)                                   # sum_i q_i,t = P_imp,t - P_exp,t
    hourly = pd.DataFrame({"hour": h.index if "hour" not in h else h["hour"], "price_x": x, "energy_price": d0.energy_price,
                           "cap": h["cap"], "import": imp, "export": exp_,
                           "collected_DKK": collected, "manager_cost_DKK": mgr_cost, "surplus_DKK": collected - mgr_cost})

    budget = float(collected.sum() - mgr_cost.sum())
    total_C = res_C.meta["metrics"]["total_cost_DKK"]
    # validation: members' settled costs = community cost + whatever the manager over-collects
    identity_gap = float(members["cost_C_DKK"].sum() - (total_C + budget))
    shortfall = lambda col: float(-members[col].clip(upper=0).sum())
    summary = dict(
        community_cost_C=total_C, community_cost_R=float(members["cost_R_DKK"].sum()), community_cost_I=float(members["cost_I_DKK"].sum()),
        community_gain_vs_R=float(members["cost_R_DKK"].sum() - total_C), community_gain_vs_I=float(members["cost_I_DKK"].sum() - total_C),
        collected_DKK=float(collected.sum()), manager_cost_DKK=float(mgr_cost.sum()), budget_DKK=budget,
        settled_cost_sum_DKK=float(members["cost_C_DKK"].sum()),
        n_members=len(members), IR_violations_vs_R=int((~members["IR_vs_R"]).sum()), IR_violations_vs_I=int((~members["IR_vs_I"]).sum()),
        shortfall_vs_R_DKK=shortfall("gain_vs_R_DKK"), shortfall_vs_I_DKK=shortfall("gain_vs_I_DKK"),
        identity_gap_DKK=identity_gap,
    )
    summary["surplus_covers_shortfall_vs_R"] = bool(budget + TOL >= summary["shortfall_vs_R_DKK"])
    summary["budget_balanced"] = bool(abs(budget) <= TOL)

    parts = []
    for key in ("load_class", "assets"):
        g = members.groupby(key).agg(n=("archetype", "size"), gain_vs_R_DKK=("gain_vs_R_DKK", "sum"), gain_vs_I_DKK=("gain_vs_I_DKK", "sum"),
                                     payment_DKK=("payment_DKK", "sum"), IR_viol_R=("IR_vs_R", lambda s: int((~s).sum())),
                                     IR_viol_I=("IR_vs_I", lambda s: int((~s).sum())))
        g.insert(0, "grouping", key)
        g.index.name = "type"
        parts.append(g.reset_index())
    return members, hourly, summary, pd.concat(parts, ignore_index=True)