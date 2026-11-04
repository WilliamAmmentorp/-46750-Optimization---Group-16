"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --all                    # run all data questions
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from src.data_loader import load_question, list_questions
from src.model import (
    FlexibleConsumerModel,
    LinearDisutilityModel,
    Q3Model,
    Q3Model_bat,
    QuadraticDisutilityModel,
    Results,
    TemporalDisutilityModel,
)
from src.plotting import (
    plot_duals,
    plot_inputs,
    plot_scenario_comparison,
    plot_schedule,
    plot_sensitivity_results,
)
from src.sensitivity import sensitivity_rows
import pandas as pd
from src.scenarios import (
    scale_prices,
    scale_quadratic_disutility,
    scale_pv,
    set_linear_disutility,
    set_load_preferences,
    set_prices,
    set_quadratic_disutility,
    set_tariffs,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_BY_QUESTION = {
    "Q1_caseA": FlexibleConsumerModel,
    "Q1_caseB": FlexibleConsumerModel,
    "Q2_linear": LinearDisutilityModel,
    "Q2_quadratic": QuadraticDisutilityModel,
    "Q2_temporal": TemporalDisutilityModel,
    "Q3": Q3Model,
    "Q3_battery": Q3Model_bat,
}


def model_for_question(question: str, data):
    """Create the model belonging to a loaded case."""
    try:
        model_class = MODEL_BY_QUESTION[question]
    except KeyError as exc:
        raise ValueError(
            f"No model class has been registered for {question!r}"
        ) from exc
    return model_class(data)


def run_cases(
    cases: dict[str, tuple[str, object]],
    out: Path,
    plot_schedules: bool = False,
) -> dict[str, Results]:
    """Solve, save, and optionally plot a collection of named cases."""
    runs: dict[str, Results] = {}
    for name, (question, data) in cases.items():
        results = model_for_question(question, data).build().solve()
        results.save(out, tag="" if name == "base" else name)
        if plot_schedules:
            plot_schedule(results, data, save_to=out / f"{name}_schedule.png")
        runs[name] = results
        print(f"{name}: objective {results.objective:.2f} DKK")
    return runs


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    model = model_for_question(question, data).build()
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


def run_all_questions(show: bool, scenarios: bool) -> None:
    """Run every data question using the standard base-case workflow."""
    for question in list_questions():
        out = RESULTS_DIR / question
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {question} ===")
        base = run_base_case(question, out, show)
        if scenarios and base is not None:
            run_scenarios(question, out)
        print(f"Outputs written to {out}")


def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g. -there is no 1.g in the assignemt, manly legacy code"""
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
        results = model_for_question(question, data).build().solve()
        results.save(out, tag=name)
        runs[name] = results
        print(f"{name:>14}: cost {results.objective:8.2f} DKK | import {results.hourly['import'].sum():5.1f} kWh"
              f" | export {results.hourly['export'].sum():5.1f} kWh")
    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


def run_q2_linear_sweep(out: Path) -> dict[str, Results]:
    """Compare Q2's linear model for several disutility coefficients."""
    base = load_question("Q2_linear")
    scenarios = {
        f"cL_{value}": ("Q2_linear", set_linear_disutility(base, value))
        for value in [0.5, 1.0, 1.43, 2.0, 2.1, 3.5]
    }
    return run_cases(scenarios, out, plot_schedules=True)


def run_q2_quadratic_sweep(out: Path) -> dict[str, Results]:
    """Compare Q2's quadratic model for several disutility coefficients."""
    base = load_question("Q2_quadratic")
    scenarios = {
        f"cQ_{value}": ("Q2_quadratic", set_quadratic_disutility(base, value))
        for value in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    }
    return run_cases(scenarios, out, plot_schedules=True)


def run_q2_temporal(out: Path) -> Results:
    """Run the temporal disutility model using the Q2 quadratic case data."""
    data = load_question("Q2_quadratic")
    results = model_for_question("Q2_temporal", data).build().solve()
    results.question = "Q2_temporal"
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    print(f"Q2_temporal: objective {results.objective:.2f} DKK")
    return results


def run_q2_price_sweep(out: Path) -> dict[str, Results]:
    """Compare Q2 models under alternating and consecutive price patterns."""
    data_dir = Path(__file__).resolve().parent / "data"
    with open(data_dir / "bus_params_alternating_prices.json", encoding="utf-8") as file:
        alternating = np.asarray(json.load(file)[0]["energy_price_DKK_per_kWh"])
    with open(data_dir / "bus_params_consecutive_prices.json", encoding="utf-8") as file:
        consecutive = np.asarray(json.load(file)[0]["energy_price_DKK_per_kWh"])

    price_scenarios = {
        "alternating_prices": alternating,
        "consecutive_prices": consecutive,
    }
    bases = {
        "linear": ("Q2_linear", load_question("Q2_linear")),
        "quadratic": ("Q2_quadratic", load_question("Q2_quadratic")),
        "temporal": ("Q2_temporal", load_question("Q2_quadratic")),
    }
    cases: dict[str, tuple[str, object]] = {}
    for price_name, prices in price_scenarios.items():
        for model_name, (question, base) in bases.items():
            data = set_prices(base, prices)
            name = f"{model_name}_{price_name}"
            cases[name] = (question, data)

    runs = run_cases(cases, out, plot_schedules=True)
    for name, results in runs.items():
        deviation = results.hourly["abs_deviation"].sum()
        print(
            f"{name}: objective {results.objective:.2f} DKK, "
            f"total deviation {deviation:.2f} kWh"
        )
    return runs


def run_q3_sensitivity(question: str, out: Path) -> None:
    """Run shared Q3 sensitivity tests, plus battery capacity when applicable."""
    if question not in ("Q3", "Q3_battery"):
        raise ValueError("Sensitivity analysis requires Q3 or Q3_battery")

    base = load_question(question)
    experiments = {
        "price_spread": (
            "Price spread factor",
            1.0,
            {
                factor: scale_prices(base, factor=factor, keep_mean=True)
                for factor in [0.0, 0.5, 1.0, 1.5, 2.0]
            },
        ),
    }

    if question == "Q3":
        experiments.update({
            "minimum_energy": (
                "Minimum daily energy",
                base.min_daily_energy_kWh,
                {
                    value: set_load_preferences(base, min_daily_energy_kWh=value)
                    for value in [20.0, 30.0, 40.0, 50.0, 60.0]
                },
            ),
            "disutility_factor": (
                "Quadratic disutility factor",
                1.0,
                {
                    factor: scale_quadratic_disutility(base, factor)
                    for factor in [0.25, 0.5, 1.0, 2.0, 4.0]
                },
            ),
        })
    else:
        experiments["battery_capacity"] = (
            "Battery capacity [kWh]",
            base.battery_capacity_kWh,
            {
                capacity: replace(
                    base,
                    battery_capacity_kWh=capacity,
                    battery_initial_soc_kWh=(
                        base.battery_initial_soc_kWh
                        / base.battery_capacity_kWh
                        * capacity
                    ),
                )
                for capacity in [1.0, 2.0, 4.0, 6.0, 8.0]
            },
        )

    rows = sensitivity_rows(
        question,
        experiments,
        ["import", "export", "load", "pv"],
        model_for_question,
    )
    table = pd.DataFrame(rows)
    table.to_csv(out / f"{question}_sensitivity_results.csv", index=False)
    plot_sensitivity_results(table, experiments, out, f"{question}_sensitivity")
    print(table.to_string(index=False))

def main() -> None:
    """Run a single question's base case, optionally with scenarios or sensitivity analysis. bit long to acount for difrent coding styles for difrent patisipants"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    special_runs = [
        "Q2_linear_sweep",
        "Q2_quadratic_sweep",
        "Q2_temporal",
        "Q2_price_sweep",
    ]
    question_args = parser.add_mutually_exclusive_group()
    question_args.add_argument(
        "--question",
        default="Q1_caseA",
        choices=list_questions() + special_runs,
        help="data case or analysis workflow to use",
    )
    question_args.add_argument(
        "--all",
        action="store_true",
        help="run every data question",
    )
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument("--q3-sensitivity", action="store_true", help="run sensitivity analysis for Q3 or Q3_battery")
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    if args.all:
        if args.q3_sensitivity:
            parser.error("--all cannot be combined with --q3-sensitivity")
        run_all_questions(args.show, args.scenarios)
        return

    out = RESULTS_DIR / (
        f"{args.question}_sensitivity" if args.q3_sensitivity else args.question
    )
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

 
    special_handlers = {
        "Q2_linear_sweep": run_q2_linear_sweep,
        "Q2_quadratic_sweep": run_q2_quadratic_sweep,
        "Q2_temporal": run_q2_temporal,
        "Q2_price_sweep": run_q2_price_sweep,
    }
    if args.question in special_handlers:
        special_handlers[args.question](out)
        print(f"\nOutputs written to {out}")
        return

    if args.q3_sensitivity:
        if args.question not in ("Q3", "Q3_battery"):
            parser.error("--q3-sensitivity requires --question Q3 or Q3_battery")
        run_q3_sensitivity(args.question, out)
        print(f"\nOutputs written to {out}")
        return

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
