"""Entry point of the Assignment 2 starter package: load a community and summarize it.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)

No model is provided: build your Assignment 1 model class on the ``InputData`` of each member, then the community
models of the assignment (R, I, C, DP, ...) on top of it, one function per experiment, writing to ``results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from src.settlement import settle_Q1
from src import plotting as plot

from src.model import (
    Results,
    Model,
)

from src.data_loader import load_community, list_communities
from src.sensitivity import sensitivity_rows_Q1, contract_experiments_Q1, sensitivity_rows_Q2, contract_experiments_Q2

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def _plot_and_save_schedule(results, com, out: Path, model_type: str) -> None:
    # inputs of one member with PV (prices, tariffs, PV, reference load)
    plot.plot_inputs(com.member("C1_PV"), out / "inputs.png")

    # community level: what the DSO sees, and who shifts
    plot.plot_import_vs_cap(results, out / "aggregate_import_vs_cap.png")
    if model_type.startswith("DP"):
        plot.plot_schedule(results, com.members[0], out / "aggregate_schedule.png",
                   price_col="posted_price", price_label="DP: posted price x")

        # representative members: schedule and duals, as in Assignment 1
        for cid in ("C1", "C1_PVB", "C4", "C4_PVB"):
            d = com.member(cid)
            r = plot.member_results(results, d, price=results.hourly["posted_price"].to_numpy())
            plot.plot_schedule(r, d, out / f"schedule_{cid}.png", price_col="posted_price", price_label="DP: posted price x")
            plot.plot_duals(r, d, out / f"duals_{cid}.png")
    else:
        plot.plot_schedule(results, com.members[0], out / "aggregate_schedule.png")

        # representative members: schedule and duals, as in Assignment 1
        for cid in ("C1", "C1_PVB", "C4", "C4_PVB"):
            d = com.member(cid)
            r = plot.member_results(results, d)
            plot.plot_schedule(r, d, out / f"schedule_{cid}.png")
            plot.plot_duals(r, d, out / f"duals_{cid}.png")

    plt.close("all")

def run_Q1_sensitivity(com, out: Path,) -> None:
    """Sweep discount and penalty; solve R once, I and C at every value."""
    experiments = contract_experiments_Q1(com)
    table = sensitivity_rows_Q1(com, experiments)
    table.to_csv(out / "Q1_sensitivity_results.csv", index=False)
    plot.plot_contract_sweep(table, experiments, out, "Q1_sensitivity", first_model="R")
    print(table.round(2).to_string(index=False))

def run_Q1_settlement(com, out: Path) -> None:
    """Q1, part 2: settle model C with the dual prices; IR member by member vs R and I; manager's budget."""
    res = {k: Model(com, k).build().solve() for k in ("R", "I", "C")}
    members, hourly, summary, by_type = settle_Q1(res["C"], res["R"], res["I"], com)
    members.to_csv(out / "Q1_settlement_members.csv"); hourly.to_csv(out / "Q1_settlement_hourly.csv", index=False)
    by_type.to_csv(out / "Q1_settlement_by_type.csv", index=False)
    with open(out / "Q1_settlement_summary.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{k:<32}: {v}\n" for k, v in summary.items())
    plot.plot_settlement(members, hourly, summary, out / "Q1_settlement.png"); plt.close("all")
    print(members.round(2).to_string(), "\n"); print(by_type.round(2).to_string(index=False), "\n")
    print("\n".join(f"{k:<32}: {v}" for k, v in summary.items()))

def run_Q2_sensitivity(com, out: Path,) -> None:
    """Sweep different DP models; solve DP, DP_dis, DP_bb, and DP_disbb."""
    # Experiment: price_bound for Model. Penality for the new arrangement.
    experiments = contract_experiments_Q2(com)
    table = sensitivity_rows_Q2(com, experiments)
    table.to_csv(out / "Q2_sensitivity_results.csv", index=False)
    plot.plot_dp_sweep(table, experiments, out, "Q2_sensitivity")
    print(table.round(2).to_string(index=False))

def _build_and_run_model(com, out: Path, model_type: str) -> Results:
    if model_type == "DP_dis":
        print("Running DP with price per member")
        model = Model(com, "DP", price_per_member=True)
    elif model_type == "DP_bb":
        print("Running DP with budget balance")
        model = Model(com, "DP", bb="eq")
    elif model_type == "DP_disbb":
        print("Running DP with price per member and budget balance")
        model = Model(com, "DP", price_per_member=True, bb="eq")
    else:
        model = Model(com, model_type)
    model.build()
    print(model.m.getConstrByName("budget_balance"))
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None
    # print(results, "\n")
    results.save(out)

    # Plot the results
    _plot_and_save_schedule(results, com, out, model_type)

    return results

def main() -> None:
    MODEL = {"Q1_R": "R", 
             "Q1_I": "I", 
             "Q1_C": "C",
             "Q1_settle": run_Q1_settlement,
             "Q1_sens": run_Q1_sensitivity,
             "Q2_DP": "DP",
             "Q2_DP_dis": "DP_dis",
             "Q2_DP_bb": "DP_bb",
             "Q2_DP_disbb": "DP_disbb",
             "Q2_sens": run_Q2_sensitivity,
             }

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--model", choices=MODEL, help="run a specific model from a specific question (R, I, C, DP, ...)")
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)

    # Catch the case where the user specifies a model that is not in the MODEL dictionary
    if args.model not in MODEL:
        print(f"Available models: {MODEL}")
        return

    # If a specific model is requested, run it and save results to the appropriate directory
    if args.model is not None:
        out = RESULTS_DIR / args.model
        out.mkdir(parents=True, exist_ok=True)
        if args.model == "Q1_sens":
            run_Q1_sensitivity(com, out)
        elif args.model == "Q1_settle":
            run_Q1_settlement(com, out)
        elif args.model == "Q2_sens":
            run_Q2_sensitivity(com, out)
        else:
            _build_and_run_model(com, out, MODEL[args.model])
        return
    
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    (RESULTS_DIR / args.community).mkdir(parents=True, exist_ok=True)
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")


if __name__ == "__main__":
    main()



# INSERT IN README.md later

# Question 1
# \item 1 + 6: run ' bash
# python main.py --model Q1_R
# '
# \item 2: run ' bash
# python main.py --model Q1_I
# '
# \item 3 + 6: run ' bash
# python main.py --model Q1_C
# '
# \item 5: run ' bash
# python main.py --model Q1_sens
# '


# Question 2
# \item 1 - 3: run ' bash
# python main.py --model Q2_DP
# '
# \item 2: