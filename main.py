"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import json #Enables the use of the json files for Q2e

from src.data_loader import load_question, list_questions
from src.model import FlexibleConsumerModel, LinearDisutilityModel, QuadraticDisutilityModel,TemporalDisutilityModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs, set_linear_disutility, set_quadratic_disutility, set_prices

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    #If statements for the different questions, to run the correct model and save the results
    if question == "Q2_linear":
        model = LinearDisutilityModel(data).build()
    elif question == "Q2_quadratic":
        model = QuadraticDisutilityModel(data).build()
    else:        
        model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    scenarios = {
        "base": base,
        "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
        "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
        "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
        "no_pv": scale_pv(base, factor=0.0),
    }
    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def run_q2_linear_sweep(out: Path) -> dict[str, Results]:
    #Sweep over different linear disutility coefficients for Q2_linear.
    base = load_question("Q2_linear")
    
    scenarios = {
        "cL_0.5": set_linear_disutility(base, 0.5),
        "cL_1.0": set_linear_disutility(base, 1.0),
        "cL_1.43": set_linear_disutility(base, 1.43),
        "cL_2.0": set_linear_disutility(base, 2.0),
        "cL_2.1": set_linear_disutility(base, 2.1),
        "cL_3.5": set_linear_disutility(base, 3.5),
    }

    runs: dict[str, Results] = {}

    for name, data in scenarios.items():
        results = LinearDisutilityModel(data).build().solve()
        results.save(out, tag=name)
        plot_schedule(results, data, save_to=out / f"{name}_schedule.png")
        runs[name] = results

        print(f"{name}: objective {results.objective:.2f} DKK")

    return runs

def run_q2_quadratic_sweep(out: Path) -> dict[str, Results]:
    base = load_question("Q2_quadratic")
    
    scenarios = {
        "cQ_0.1": set_quadratic_disutility(base, 0.1),
        "cQ_0.5": set_quadratic_disutility(base, 0.5),
        "cQ_1.0": set_quadratic_disutility(base, 1.0),
        "cQ_2.0": set_quadratic_disutility(base, 2.0),
        "cQ_5.0": set_quadratic_disutility(base, 5.0),
        "cQ_10.0": set_quadratic_disutility(base, 10.0),
    }

    runs: dict[str, Results] = {}

    for name, data in scenarios.items():
        results = QuadraticDisutilityModel(data).build().solve()
        results.save(out, tag=name)
        plot_schedule(results, data, save_to=out / f"{name}_schedule.png")
        runs[name] = results

        print(f"{name}: objective {results.objective:.2f} DKK")

    return runs

def run_q2_temporal(out: Path) -> Results:
    data = load_question("Q2_quadratic")

    results = TemporalDisutilityModel(data).build().solve()
    results.question = "Q2_temporal"  # Set the question name for saving results

    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")

    print(f"Q2_temporal: objective {results.objective:.2f} DKK")

    return results

def run_q2_price_sweep(out: Path) -> dict[str, Results]:
    linear_base = load_question("Q2_linear")
    quadratic_base = load_question("Q2_quadratic")

    #Load the JSON files for the alternating and consecutive price scenarios
    with open("data/bus_params_alternating_prices.json", "r") as f:
        alternating_json = json.load(f)

    with open("data/bus_params_consecutive_prices.json", "r") as f:
        consecutive_json = json.load(f)

    alternating = np.array(alternating_json[0]["energy_price_DKK_per_kWh"])
    consecutive = np.array(consecutive_json[0]["energy_price_DKK_per_kWh"])

    price_scenarios = {
        "alternating_prices": alternating,
        "consecutive_prices": consecutive,
    }

    runs: dict[str, Results] = {}

    for price_name, prices in price_scenarios.items():

        #Linear model
        data = set_prices(linear_base, prices)
        results = LinearDisutilityModel(data).build().solve()
        name = f"linear_{price_name}"
        results.save(out, tag=name)
        plot_schedule(results, data, save_to=out / f"{name}_schedule.png")
        runs[name] = results

        #Quadratic model
        data = set_prices(quadratic_base, prices)
        results = QuadraticDisutilityModel(data).build().solve()
        name = f"quadratic_{price_name}"
        results.save(out, tag=name)
        plot_schedule(results, data, save_to=out / f"{name}_schedule.png")
        runs[name] = results

        #temporal model
        data = set_prices(quadratic_base, prices)
        results = TemporalDisutilityModel(data).build().solve()
        results.question = "Q2_temporal"  # Set the question name for saving results
        name = f"temporal_{price_name}"
        results.save(out, tag=name)
        plot_schedule(results, data, save_to=out / f"{name}_schedule.png")
        runs[name] = results

    for name, results in runs.items():
        deviation = results.hourly["abs_deviation"].sum()

        print(f"{name}: objective {results.objective:.2f} DKK, total deviation {deviation:.2f} kWh")

    return runs

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    special_runs = ["Q2_linear_sweep", "Q2_quadratic_sweep", "Q2_temporal", "Q2_price_sweep"]  # Added for the sweep runs and 2e
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions() + special_runs, help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")
    if args.question == "Q2_linear_sweep":
        run_q2_linear_sweep(out)
        print(f"\nOutputs written to {out}")
        return
    if args.question == "Q2_quadratic_sweep":
        run_q2_quadratic_sweep(out)
        print(f"\nOutputs written to {out}")
        return
    if args.question == "Q2_temporal":
        run_q2_temporal(out)
        print(f"\nOutputs written to {out}")
        return
    if args.question == "Q2_price_sweep":
        run_q2_price_sweep(out)
        print(f"\nOutputs written to {out}")
        return
    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
