"""Optimisation model of a single flexible consumer, implemented with gurobipy.

The class below separates the three steps you will repeat for every question:

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

``build()`` is the only method you need to complete for Question 1; the other questions
are variations of it (a different objective, an extra constraint). Copy this file or
subclass ``FlexibleConsumerModel`` and override ``build()`` to keep one model per question.

Two conventions make the dual variables easy to read out afterwards:

* Every constraint family is stored in ``self.con`` under a descriptive name, e.g.
  ``self.con["balance"] = self.m.addConstrs(...)``. ``solve()`` then returns the dual value
  (shadow price, Gurobi attribute ``Pi``) of every constraint in ``self.con`` automatically.
* Bounds that you want a dual for must be written as explicit constraints (``addConstr``),
  not as variable bounds (``lb=``/``ub=``). Gurobi reports the sensitivity of a variable
  bound in the reduced cost (``RC``), not in ``Pi``.
* Duals of quadratic constraints (``m.addQConstr``) are read from ``QCPi`` and require ``QCPDual = 1`` (set below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData
from collections import defaultdict



# From a notebook or your own script** (run from the repository root)

# DATA
# The hourly resolution makes kW and kWh/h numerically interchangeable, as in Assignment 1.
# `InputData` carries the *full* tariffs of `bus_params.json`; the discount of the contract is in
# `ContractData.tariff_discount_ratio` and applies to the metered flows of the community (model C) or of the
# member (model I) as the assignment defines them. The notation of the assignment follows Assignment 1
# (price p_t, tariffs tau_imp / tau_exp, member variables q_imp, q_exp, l, g, battery p_ch, p_dis, s) with a
# member index; the community's import and export at the connection point are P_imp_t and P_exp_t, the cap
# is P_DSO_t (`capacity_limit_kW`).





# from src.data_loader import load_community
# from src import scenario_generation as sg

# com = load_community("base")                       # day d1
# print(com.summary())
# for m in com.members:                              # Assignment 1 InputData objects
#     print(m.consumer_id, m.archetype, m.linear_disutility, m.min_daily_energy_kWh, m.pv_max_kW, m.battery_capacity_kWh)

# cap   = com.contract.capacity_limit_kW             # 24 values, kW, at the connection point
# beta  = com.contract.tariff_discount_ratio         # 0.75 in the base case
# alpha = com.contract.penalty_DKK_per_kWh           # 10 in the base case
# cap_i = com.contract.individual_capacity_limit_kW  # {consumer_id: 24 values}  (benchmark model I)

# com_d4 = load_community("base", day="d4")          # another day of the week (prices, PV, reference profiles and cap)
# com_nb = load_community("base")                    # a member-mix variation, built in code (Section 4.3): no batteries
# for m in com_nb.members:
#     for k in [k for k in vars(m) if k.startswith("battery_")]:
#         setattr(m, k, None)


def add_kkt(m, mm, x, tag, compl="bigM", M=None):
    """Embed member mm's KKT conditions in the joint single-level DP model.

    M contains the bounds used for the dual variables and for member primal
    variables/slacks. The complementarity pairs are linearized either with
    Gurobi SOS1 constraints or with binary Big-M constraints.

    Returns a dictionary containing the member's strong-duality cost.
    """
    if compl not in ("bigM", "sos1"):
        raise ValueError("compl must be 'bigM' or 'sos1'")

    M = M or dict(dual=20.0, primal=5.0)
    dual_bound = M["dual"]
    primal_bound = M["primal"]
    m.update()

    objective = mm.pv_cost + mm.disutility
    objective_coefficients = {
        objective.getVar(i): objective.getCoeff(i)
        for i in range(objective.size())
    }
    posted_price_coefficients = {}
    for t in mm.T:
        posted_price_coefficients[mm.var["import"][t]] = x[t]
        posted_price_coefficients[mm.var["export"][t]] = -x[t]

    # Read every member constraint as (linear row, sense, right-hand side).
    constraint_rows = []
    for c in mm.con.values():
        for cc in ([c] if isinstance(c, gp.Constr) else c.values()):
            constraint_rows.append((m.getRow(cc), cc.Sense, cc.RHS))

    complementarity_pairs = []
    equality_and_inequality_multipliers = []
    multiplier_contribution = defaultdict(gp.LinExpr)

    # Add equality multipliers and inequality multipliers together with their
    # primal slacks. Inequality multipliers are constrained to be non-negative.
    for constraint_index, (row, sense, rhs) in enumerate(constraint_rows):
        if sense == "=":
            multiplier = m.addVar(
                lb=-dual_bound,
                ub=dual_bound,
                name=f"pi_{tag}_{constraint_index}",
            )
        else:
            inequality_multiplier = m.addVar(
                lb=0,
                ub=dual_bound,
                name=f"lambda_{tag}_{constraint_index}",
            )
            primal_slack = m.addVar(
                lb=0,
                ub=primal_bound,
                name=f"slack_{tag}_{constraint_index}",
            )
            if sense == "<":
                m.addConstr(primal_slack == rhs - row)
                multiplier = inequality_multiplier
            else:
                m.addConstr(primal_slack == row - rhs)
                multiplier = -inequality_multiplier

            complementarity_pairs.append(
                (inequality_multiplier, primal_slack, dual_bound, primal_bound)
            )

        equality_and_inequality_multipliers.append(multiplier)
        for i in range(row.size()):
            member_variable = row.getVar(i)
            multiplier_contribution[member_variable] += (
                multiplier * row.getCoeff(i)
            )

    # Add a reduced cost and an explicit stationarity equation for every
    # member variable. The reduced-cost/primal-variable pair is complementary.
    for variable_family_name, variable_family in mm.var.items():
        for variable_index, member_variable in variable_family.items():
            member_variable.UB = primal_bound
            reduced_cost = m.addVar(
                lb=0,
                ub=dual_bound,
                name=f"reduced_cost_{tag}_{variable_family_name}_{variable_index}",
            )
            m.addConstr(
                reduced_cost
                == objective_coefficients.get(member_variable, 0.0)
                + posted_price_coefficients.get(member_variable, 0.0)
                + multiplier_contribution[member_variable]
            )
            complementarity_pairs.append(
                (reduced_cost, member_variable, dual_bound, primal_bound)
            )

    # Linearize each complementarity pair: either a <= 0 or b <= 0.
    for pair_index, (dual_value, primal_value, dual_bound, primal_bound) in enumerate(
        complementarity_pairs
    ):
        if compl == "sos1":
            m.addSOS(GRB.SOS_TYPE1, [dual_value, primal_value])
        else:
            active_dual = m.addVar(
                vtype=GRB.BINARY,
                name=f"complementarity_{tag}_{pair_index}",
            )
            m.addConstr(dual_value <= dual_bound * active_dual)
            m.addConstr(primal_value <= primal_bound * (1 - active_dual))

    strong_duality_cost = -gp.quicksum(
        multiplier * rhs
        for multiplier, (_, _, rhs) in zip(
            equality_and_inequality_multipliers,
            constraint_rows,
        )
    )
    return dict(dual_cost=strong_duality_cost)

    #return dict(dual_cost=-gp.quicksum(p * rhs for p, (_, _, rhs) in zip(pi, rows)))

@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write hourly profile, scalar summary, and any DataFrames stored in meta to CSV."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.meta.get("metrics", {}).items():
                f.write(f"{k:<22}: {v:.4f}\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")
        for key, val in self.meta.items():          # per-member tables etc.
            if isinstance(val, pd.DataFrame):
                val.to_csv(folder / f"{stem}_{key}.csv")



class MemberModel:
    
    def __init__(self, data, beta=0.0, alpha=0.0, cap=None, verbose=False, model=None, grid=True):
        self.data = data
        self.beta = beta
        self.alpha = alpha
        self.cap = cap
        self.grid = grid
        self.T = range(data.n_hours)
        self.own = model is None
        self.m = gp.Model(f"member_{data.consumer_id}") if self.own else model
        if self.own:
            self.m.Params.OutputFlag = 1 if verbose else 0          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)


    def build(self):
        # Inputs for Gurobi model
        d, m, T = self.data, self.m, self.T
        n = lambda s: f"{s}_{d.consumer_id}"

        # Initialize variables for import, export, load, pv, dev, charge, discharge, soc
        for k in ("load", "import", "export", "pv", "dev"):
            self.var[k] = m.addVars(T, lb=0, name=k)
        if d.has_battery:
            for k in ("charge", "discharge", "soc"):
                self.var[k] = m.addVars(T, lb=0, name=k)

        # Shorter variable names for easier access
        L = self.var["load"]
        imp = self.var["import"]
        exp = self.var["export"]
        PV = self.var["pv"]
        dev = self.var["dev"]
        charge = self.var["charge"] if d.has_battery else None
        discharge = self.var["discharge"] if d.has_battery else None
        soc = self.var["soc"] if d.has_battery else None

        # Constraints load
        self.con["load_max"] = m.addConstrs((L[t] <= d.load_upper_kWh[t] for t in T), name="load_upper")
        self.con["load_min"] = m.addConstrs((L[t] >= d.load_lower_kWh[t] for t in T), name="load_lower")

        # Constraints PV
        self.con["pv_max"] = m.addConstrs((PV[t] <= d.pv_available[t] for t in T), name="pv_max")
        self.con["balance"] = m.addConstrs((PV[t] + imp[t] + (discharge[t] if d.has_battery else 0) 
                                            == L[t] + exp[t] + (charge[t] if d.has_battery else 0) for t in T),
                                            name="balance")

        # Constraints disutility
        self.con["dev_above"] = m.addConstrs((dev[t] >= L[t] - d.reference_load[t] for t in T), name="dev_above")
        self.con["dev_below"] = m.addConstrs((dev[t] >= d.reference_load[t] - L[t] for t in T), name="dev_below")

        # Constraint minimum daily energy 
        self.con["min_energy"] = m.addConstr(gp.quicksum(L[t] for t in T) >= (d.min_daily_energy_kWh),name="min_energy")

        # Constraints battery related (only if the prosumer has a battery)
        if d.has_battery:
            self.con["charge_max"] = m.addConstrs((charge[t] <= d.battery_max_charge_kW for t in T), name="charge_max")
            self.con["discharge_max"] = m.addConstrs((discharge[t] <= d.battery_max_discharge_kW for t in T), name="discharge_max")


            self.con["soc_max"] = m.addConstrs((soc[t] <= d.battery_capacity_kWh for t in T), name="soc_max")

            self.con["soc_balance"] = m.addConstrs(
                (soc[t] == (d.battery_initial_soc_kWh if t == 0 else soc[t - 1])
                 + charge[t] * d.battery_charging_efficiency
                 - discharge[t] * (1 / d.battery_discharging_efficiency) 
                 for t in T), name="soc_balance")

            self.con["soc_final"] = m.addConstr(soc[T[-1]] == 0, name="soc_final") # battery_initial=0

        # Discount factor for tariffs
        discount = 1.0 - self.beta

        # Objective function: grid cost, PV cost, disutility, and penalty for exceeding cap
        self.pv_cost = gp.quicksum(d.pv_marginal_cost * PV[t] for t in T)
        self.disutility = gp.quicksum(d.linear_disutility * dev[t] for t in T)
        self.grid_cost, self.penalty = gp.LinExpr(), gp.LinExpr()

        # If the prosumer is connected to the grid
        if self.grid:
            self.grid_cost = gp.quicksum(
                (d.energy_price[t] + discount * d.import_tariff) * imp[t]
                - (d.energy_price[t] - discount * d.export_tariff) * exp[t] for t in T)
            if self.cap is not None:
                self.var["excess"] = m.addVars(T, lb=0, name=n("excess"))
                self.con["excess"] = m.addConstrs((self.var["excess"][t] >= imp[t] - self.cap[t] for t in T), name=n("excess_def"))
                self.penalty = self.alpha * gp.quicksum(self.var["excess"][t] for t in T)

        # Objective function with linear disutility
        # Minimize electricity procurement cost and disutility from deviating from reference load
        if self.own:
            m.setObjective(self.grid_cost + self.pv_cost + self.disutility + self.penalty, GRB.MINIMIZE)
        return self
    
    def extract(self, duals: bool = True) -> dict:
        T = self.T
        out = {k: np.array([x[t].X for t in T]) for k, x in self.var.items()}
        scalar = {}
        if duals:
            for name, c in self.con.items():
                if isinstance(c, gp.tupledict):
                    out[f"dual_{name}"] = np.array([c[t].Pi for t in T])
                else:
                    scalar[name] = c.Pi
        out["scalar_duals"] = scalar
        g, p, u, pen = (self.grid_cost.getValue(), self.pv_cost.getValue(),
                        self.disutility.getValue(), self.penalty.getValue())
        out["costs"] = dict(grid=g, pv=p, disutility=u, penalty=pen, total=g + p + u + pen)
        return out

    def solve(self) -> dict:
        self.m.optimize()
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{self.data.consumer_id}: status {_status_name(self.m.Status)}")
        return self.extract()

class Model():

    CONTRACTS = ("R", "I", "C", "DP")

    def __init__(self, com, contract="R", verbose=False, compl="bigM", price_bound=None, price_per_member=False, bb="none",
                 run_name=None):
        if contract not in self.CONTRACTS:
            raise ValueError(f"contract must be one of {self.CONTRACTS}")
        self.com  = com
        self.contract = contract
        self.verbose = verbose
        self.compl = compl
        self.price_bound = price_bound
        self.run_name = run_name
        self.members: dict[str, MemberModel] = {}
        self.m = None            # joint model (C only)
        self.manager = None      # manager's costs at the connection point (C only)
        self.price_per_member = price_per_member
        self.bb = bb

    """Build the model according to the specified contract type (R, I, C)."""
    def build(self):
        {"R": self._build_R, 
         "I": self._build_I, 
         "C": self._build_C,
         "DP": self._build_DP,
         }[self.contract]()

        return self
    
    """Model R: every member alone, full tariffs, no cap. 16 independent LPs."""
    def _build_R(self):
        for member in self.com.members:
            self.members[member.consumer_id] = MemberModel(member, verbose=self.verbose).build()

    """Model I: every member alone, full tariffs, individual caps. 16 independent LPs."""
    def _build_I(self):
        """
        Model I (Individual contracts): Each consumer schedules its flexible loads and assets independently
        and is charged by the DSO based on its own flows. It pays: the day-ahead energy price for its energy purchased
        (or sold), the discounted grid tariffs for its imports below their cap and its exports, and the penalty for its
        imports exceeding the hourly cap. Under a community contract
        """
        for member in self.com.members:
            self.members[member.consumer_id] = MemberModel(member, beta=self.com.contract.tariff_discount_ratio, 
                                                           alpha=self.com.contract.penalty_DKK_per_kWh,
                                                           cap=np.asarray(self.com.contract.individual_capacity_limit_kW[member.consumer_id]),
                                                           verbose=self.verbose).build()

    """Model C: members together, full tariffs, collective cap."""
    def _build_C(self):
        """
        Model C (Centralized dispatch): The manager knows every member's preferences
        (utilities and constraints) and dispatches their flexible loads and assets (e.g. PV, battery) in one
        optimization that maximizes the community's social welfare. This benchmark is an upper bound on the
        flexibility the community can provide and on its welfare under a given contract. All members' net hourly
        imports (or exports) are charged at a uniform hourly price, computed as the dual variable of the internal
        balance equation. Inside the community this price replaces every cost a member would face alone.
        """
        com, c = self.com, self.com.contract
        d0, T = com.members[0], range(com.members[0].n_hours)
        beta = c.tariff_discount_ratio
        alpha = c.penalty_DKK_per_kWh
        discount = 1.0 - beta
        cap = np.asarray(c.capacity_limit_kW)
        m = self.m = gp.Model("Q1_C")
        m.Params.OutputFlag = 1 if self.verbose else 0

        for member in com.members:                                       # members: no grid cost, no own cap
            self.members[member.consumer_id] = MemberModel(member, model=m, grid=False).build()

        P_imp = m.addVars(T, lb=0, name="P_imp")                    # metered import / export / excess
        P_exp = m.addVars(T, lb=0, name="P_exp")
        P_exc = m.addVars(T, lb=0, name="P_exc")
        self.P = dict(imp=P_imp, exp=P_exp, exc=P_exc)

        # hourly balance behind the meter: sum_i q_i,t = P_imp,t - P_exp,t   (its dual is the internal price)
        self.balance = m.addConstrs(
            (gp.quicksum(mm.var["import"][t] - mm.var["export"][t] for mm in self.members.values())
             == P_imp[t] - P_exp[t] for t in T), name="community_balance")
        m.addConstrs((P_exc[t] >= P_imp[t] - cap[t] for t in T), name="P_exc_def")

        grid = gp.quicksum((d0.energy_price[t] + discount * d0.import_tariff) * P_imp[t]
                           - (d0.energy_price[t] - discount * d0.export_tariff) * P_exp[t] for t in T)
        penalty = alpha * gp.quicksum(P_exc[t] for t in T)
        self.manager_expr = dict(
            grid=grid,
            penalty=penalty,
        )
        m.setObjective(gp.quicksum(mm.pv_cost + mm.disutility for mm in self.members.values())
                       + grid + penalty, GRB.MINIMIZE)

    def _build_DP(self):
        """Build the single-level MILP for the dynamic-pricing bilevel model.

        The manager chooses hourly posted prices. Each member remains an
        independent lower-level LP, whose KKT conditions are embedded with
        add_kkt(). Uniform and member-specific prices are selected by the
        price_per_member option. Budget balance is optional.
        """
        com = self.com
        contract = com.contract
        reference_member = com.members[0]
        hours = range(reference_member.n_hours)
        discount = 1.0 - contract.tariff_discount_ratio
        capacity_limit = np.asarray(contract.capacity_limit_kW)

        m = self.m = gp.Model("Q2_DP")
        m.Params.OutputFlag = 1 if self.verbose else 0
        m.Params.TimeLimit = 120
        m.Params.MIPGap = 1e-3

        # Bound the posted price. The default bound is also usable as a
        # dual-variable bound in the KKT reformulation.
        price_bound = self.price_bound or (
            contract.penalty_DKK_per_kWh
            + reference_member.energy_price.max()
            + reference_member.import_tariff
        )

        if self.price_per_member:
            # One independent 24-hour price vector for each member.
            self.x = {
                member_id: m.addVars(
                    hours,
                    lb=-price_bound,
                    ub=price_bound,
                    name=f"posted_price_{member_id}",
                )
                for member_id in com.member_ids
            }
        else:
            # One shared 24-hour price vector for the whole community.
            shared_prices = m.addVars(
                hours,
                lb=-price_bound,
                ub=price_bound,
                name="posted_price",
            )
            self.x = {
                member_id: shared_prices
                for member_id in com.member_ids
            }

        # Replace each member's LP by its primal, dual, stationarity,
        # and complementarity constraints.
        self.kkt = {}
        kkt_bounds = dict(dual=price_bound + 5, primal=5.0)
        for member in com.members:
            member_model = MemberModel(
                member,
                model=m,
                grid=False,
            ).build()
            member_id = member.consumer_id
            self.members[member_id] = member_model
            self.kkt[member_id] = add_kkt(
                m,
                member_model,
                self.x[member_id],
                member_id,
                self.compl,
                kkt_bounds,
            )

        # Community meter flows and capacity-exceedance variable.
        community_import = m.addVars(hours, lb=0, name="community_import")
        community_export = m.addVars(hours, lb=0, name="community_export")
        community_exceedance = m.addVars(hours, lb=0, name="community_exceedance")
        self.P = dict(
            imp=community_import,
            exp=community_export,
            exc=community_exceedance,
        )

        m.addConstrs(
            (
                gp.quicksum(
                    member_model.var["import"][t]
                    - member_model.var["export"][t]
                    for member_model in self.members.values()
                )
                == community_import[t] - community_export[t]
                for t in hours
            ),
            name="meter_balance",
        )
        m.addConstrs(
            (
                community_exceedance[t]
                >= community_import[t] - capacity_limit[t]
                for t in hours
            ),
            name="community_exceedance",
        )

        grid_cost = gp.quicksum(
            (
                reference_member.energy_price[t]
                + discount * reference_member.import_tariff
            ) * community_import[t]
            - (
                reference_member.energy_price[t]
                - discount * reference_member.export_tariff
            ) * community_export[t]
            for t in hours
        )
        exceedance_cost = contract.penalty_DKK_per_kWh * gp.quicksum(
            community_exceedance[t] for t in hours
        )
        self.manager_expr = dict(
            grid=grid_cost,
            penalty=exceedance_cost,
        )

        # Strong duality gives each member payment as:
        # payment = dual cost - PV cost - disutility.
        self.pay = {
            member_id: (
                self.kkt[member_id]["dual_cost"]
                - member_model.pv_cost
                - member_model.disutility
            )
            for member_id, member_model in self.members.items()
        }
        self.budget = (
            gp.quicksum(self.pay.values())
            - (grid_cost + exceedance_cost)
        )

        community_objective = (
            gp.quicksum(
                member_model.pv_cost + member_model.disutility
                for member_model in self.members.values()
            )
            + grid_cost
            + exceedance_cost
        )

        if self.bb == "eq":
            m.addConstr(
                self.budget == 0,
                name="budget_balance",
            )

        m.setObjective(community_objective, GRB.MINIMIZE)

        # Model C is a relaxation of DP. Its objective therefore provides a
        # valid lower bound and a useful feasible starting solution.
        centralized_solution = Model(self.com, "C").build().solve()
        m.addConstr(
            community_objective >= centralized_solution.objective - 1e-4,
            name="lower_bound_from_centralized_model",
        )

        central_posted_prices = np.clip(
            -centralized_solution.hourly["internal_price_dual"].to_numpy(),
            -price_bound,
            price_bound,
        )
        for t in hours:
            for member_id in com.member_ids:
                self.x[member_id][t].Start = central_posted_prices[t]
            community_import[t].Start = centralized_solution.hourly["import"].iloc[t]
            community_export[t].Start = centralized_solution.hourly["export"].iloc[t]

        for member_id, member_model in self.members.items():
            centralized_member = centralized_solution.meta["solutions"][member_id]
            for variable_name, variable_family in member_model.var.items():
                for t in hours:
                    variable_family[t].Start = centralized_member[variable_name][t]



    def solve(self) -> Results:
        if self.contract in ("C", "DP"):
            self.m.optimize()
            if self.m.SolCount == 0:
                raise RuntimeError(f"{self.contract}: status {_status_name(self.m.Status)}, no solution")
            if self.contract == "DP":
                print(f"DP status {_status_name(self.m.Status)}, MIP gap {self.m.MIPGap:.2%}")
            sol = {cid: mm.extract(duals=(self.contract == "C")) for cid, mm in self.members.items()}
            self.manager = {k: e.getValue() for k, e in self.manager_expr.items()}
        else:
            sol = {cid: mm.solve() for cid, mm in self.members.items()}
        return self._package(sol)

    def _package(self, sol: dict) -> Results:
        com = self.com
        cap = np.array(com.contract.capacity_limit_kW)
        T = range(len(cap))
        first = com.members[0]

        # ---- aggregate profile seen by the DSO (metered flows at the connection point)
        hourly = pd.DataFrame({"price": first.energy_price, "cap": cap})
        for k in ("load", "pv"):
            hourly[k] = sum(s[k] for s in sol.values())
        if self.contract in ("C", "DP"):
            hourly["import"] = [self.P["imp"][t].X for t in T]
            hourly["export"] = [self.P["exp"][t].X for t in T]
            hourly["gross_member_import"] = sum(s["import"] for s in sol.values())
        else:
            hourly["import"] = sum(s["import"] for s in sol.values())
            hourly["export"] = sum(s["export"] for s in sol.values())
        hourly["excess"] = np.maximum(hourly["import"] - cap, 0.0)
        hourly["pv_available"] = sum(d.pv_available for d in com.members)
        hourly["reference_load"] = sum(d.reference_load for d in com.members)
        hourly["shifted"] = 0.5 * sum(np.abs(s["load"] - d.reference_load)
                                      for s, d in zip(sol.values(), com.members))
        hourly["charge"] = sum(s.get("charge", 0.0) for s in sol.values())
        hourly["discharge"] = sum(s.get("discharge", 0.0) for s in sol.values())
        if self.contract == "C":
            hourly["internal_price_dual"] = [self.balance[t].Pi for t in T]
        elif self.contract == "DP":
            prices = pd.DataFrame({cid: [self.x[cid][t].X for t in T] for cid in com.member_ids})
            hourly["posted_price"] = prices.mean(axis=1)          # equals the price itself when it is uniform

        # ---- per-member costs (in C: pv + disutility only; grid and penalty sit with the manager)
        rows = {}
        for d in com.members:
            s = sol[d.consumer_id]
            rows[d.consumer_id] = dict(
                archetype=d.archetype, slope=d.linear_disutility, **s["costs"],
                import_kWh=s["import"].sum(), export_kWh=s["export"].sum(),
                shifted_kWh=0.5 * np.abs(s["load"] - d.reference_load).sum(),
                **{f"dual_{k}": v for k, v in s["scalar_duals"].items()})
        members = pd.DataFrame(rows).T
        members = members.astype({c: float for c in members.columns if c != "archetype"})
        if self.contract == "DP":
            members["payment"] = [self.pay[cid].getValue() for cid in members.index]
            members["settled_total"] = members["total"] + members["payment"]

        # ---- per-member hourly schedules (long format)
        frames = []
        for d in com.members:
            s = sol[d.consumer_id]
            df = pd.DataFrame({k: v for k, v in s.items() if k not in ("costs", "scalar_duals")})
            df.insert(0, "hour", list(T))
            df.insert(0, "member", d.consumer_id)
            frames.append(df)
        schedules = pd.concat(frames, ignore_index=True)

        # ---- community cost by component, and DSO metrics
        cost = {k: float(members[k].sum()) for k in ("grid", "pv", "disutility", "penalty")}
        if self.manager:
            cost["grid"] += self.manager["grid"]
            cost["penalty"] += self.manager["penalty"]
        metrics = dict(
            total_cost_DKK=sum(cost.values()),
            **{f"cost_{k}_DKK": v for k, v in cost.items()},
            daily_import_kWh=float(hourly["import"].sum()),
            daily_export_kWh=float(hourly["export"].sum()),
            peak_import_kW=float(hourly["import"].max()),
            exceedance_kWh=float(hourly["excess"].sum()),
            hours_above_cap=float((hourly["excess"] > 1e-6).sum()),
            max_exceedance_kW=float(hourly["excess"].max()),
            energy_shifted_kWh=float(hourly["shifted"].sum()),
        )
        if self.contract == "DP":
            metrics["manager_budget_DKK"] = self.budget.getValue()
        question_prefix = "Q1" if self.contract in ("R", "I", "C") else "Q2"
        question = self.run_name if self.run_name is not None else f"{question_prefix}_{self.contract}"
        return Results(question=question, status="OPTIMAL",
                       objective=metrics["total_cost_DKK"], hourly=hourly,
                       meta=dict(metrics=metrics, members=members, schedules=schedules, solutions=sol, prices=prices if self.contract == "DP" else None))



_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
