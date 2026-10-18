"""Helpers for solving and tabulating sensitivity experiments."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .data_loader import InputData
from .model import Results


def sensitivity_rows(
    question: str,
    experiments: dict[str, tuple[str, Any, dict[float, InputData]]],
    hourly_totals: list[str],
    model_for_question: Callable[[str, InputData], Any],
) -> list[dict[str, Any]]:
    """Solve sensitivity cases and return rows ready for a DataFrame."""
    rows = []
    for experiment, (_, _, cases) in experiments.items():
        for value, data in cases.items():
            result: Results = model_for_question(question, data).build().solve()
            rows.append({
                "experiment": experiment,
                "parameter_value": value,
                "status": result.status,
                "objective_DKK": result.objective,
                **{
                    f"{column}_kWh": result.hourly[column].sum()
                    for column in hourly_totals
                },
                "minimum_energy_dual": result.duals.get(
                    "min_total_energy_per_day_kWh"
                ),
            })
    return rows
