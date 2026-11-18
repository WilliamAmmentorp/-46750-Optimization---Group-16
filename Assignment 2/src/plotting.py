"""Matplotlib figures for the input data and the optimisation results.

Every function returns the ``Figure`` and optionally saves it, so the same code works in a
script (``python main.py``) and in a notebook (``plot_schedule(results, data);``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data_loader import InputData
from .model import Results

_COLORS = {"R": "grey", "I": "C1", "C": "C0"}

def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_inputs(data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly prices (with tariffs) and available PV / load preferences, side by side."""
    h = data.hours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.step(h, data.energy_price, where="mid", label="energy price", color="k")
    ax1.step(h, data.energy_price + data.import_tariff, where="mid", ls="--", label="price + import tariff")
    ax1.step(h, data.energy_price - data.export_tariff, where="mid", ls=":", label="price - export tariff")
    ax1.set(xlabel="hour", ylabel="DKK/kWh", title="Electricity prices")
    ax1.legend(fontsize=8)

    ax2.fill_between(h, data.pv_available, step="mid", alpha=0.4, color="orange", label="PV available")
    ax2.axhline(data.load_max_kWh, color="C3", ls="--", label="max load")
    if data.load_min_kWh > 0:
        ax2.axhline(data.load_min_kWh, color="C3", ls=":", label="min load")
    if data.reference_load is not None:
        ax2.step(h, data.reference_load, where="mid", color="C0", label="reference load")
    ax2.set(xlabel="hour", ylabel="kWh/h", title="PV and load preferences")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Input data - {data.question}", fontsize=11)
    return _finish(fig, save_to)
def plot_schedule(results: Results, data: InputData, save_to: Path | str | None = None,
                  price_col: str = "price", price_label: str = "energy price") -> plt.Figure:
    """Optimal schedule: load, PV used, import/export, with a price column on a second axis."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.2))

    width = 0.8
    if "load" in hr:
        ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    if "pv" in hr:
        ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7, label="PV used (negative = generation)")
    if "pv_available" in hr and "pv" in hr:
        ax.step(h, -hr["pv_available"], where="mid", color="orange", ls="--", lw=1, label="PV available")
    if "import" in hr and "export" in hr:
        ax.plot(h, hr["import"] - hr["export"], "k.-", label="net import (+) / export (-)")
    if "reference_load" in hr:
        ax.step(h, hr["reference_load"], where="mid", color="C0", ls=":", label="reference load")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="hour", ylabel="kWh/h", title=f"Optimal schedule - {results.question} (cost {results.objective:.1f} DKK)")

    ax2 = ax.twinx()
    ax2.step(h, hr[price_col], where="mid", color="C3", lw=1.2, label=price_label)
    ax2.set_ylabel("DKK/kWh", color="C3")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)

# def plot_price_deviation(results: Results, com, save_to: Path | str | None = None) -> plt.Figure:
#     """Posted price of each member minus the community mean price, per hour (DKK/kWh).
#     Red = the member pays more than average in that hour, blue = less."""
#     p = results.meta["prices"]                                   # hour x member (one column per member)
#     dev = p.sub(p.mean(axis=1), axis=0).T                        # member x hour
#     spread = float((p.max(axis=1) - p.min(axis=1)).max())
#     lim = max(float(dev.abs().to_numpy().max()), 1e-6)
#     fig, ax = plt.subplots(figsize=(11, 5))
#     im = ax.imshow(dev.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
#     ax.set_yticks(range(len(dev)))
#     ax.set_yticklabels([f"{c} ({com.member(c).archetype})" for c in dev.index], fontsize=7)
#     ax.set(xlabel="hour")
#     if spread < 1e-6:
#         ax.set_title("Posted price minus the community mean: no difference between members (uniform prices)")
#     else:
#         ax.set_title(f"Posted price minus the community mean (largest spread {spread:.2f} DKK/kWh)")
#     fig.colorbar(im, ax=ax, label="DKK/kWh")
#     return _finish(fig, save_to)


def plot_duals(results: Results, data: InputData, save_to=None,
               keys=("load_max", "load_min", "pv_max", "balance")) -> plt.Figure:
    """Hourly duals of the chosen constraints against the price signals; daily minimum-energy multiplier as a line."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4))
    for k in keys:
        if f"dual_{k}" in hr:
            ax.step(h, hr[f"dual_{k}"], where="mid", label=k)
    mu = results.duals.get("min_energy")
    if mu is not None:
        ax.axhline(mu, color="C4", ls="-.", label=f"min_energy (daily): {mu:.2f}")
    ax.step(h, data.energy_price + data.import_tariff, where="mid", color="grey", ls="--", lw=1, label="price + import tariff")
    ax.step(h, data.energy_price - data.export_tariff, where="mid", color="grey", ls=":", lw=1, label="price - export tariff")
    ax.set(xlabel="hour", ylabel="DKK/kWh", title=f"Dual variables - {results.question}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)


def plot_scenario_comparison(
    runs: dict[str, Results], metric: str = "objective", save_to: Path | str | None = None
) -> plt.Figure:
    """Bar chart of one metric across scenarios. ``metric`` is ``"objective"`` or the name of an
    hourly column whose daily sum is compared (e.g. ``"import"``, ``"export"``, ``"load"``)."""
    names = list(runs)
    if metric == "objective":
        values = [r.objective for r in runs.values()]
        ylabel = "daily cost [DKK]"
    else:
        values = [r.hourly[metric].sum() for r in runs.values()]
        ylabel = f"daily {metric} [kWh]"
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.8))
    ax.bar(names, values, color="C0")
    ax.set(ylabel=ylabel, title=f"Scenario comparison - {metric}")
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, save_to)


def plot_sensitivity_results(
    table,
    experiments: dict,
    out: Path,
    filename_prefix: str,
) -> None:
    """Plot objective and import results for each sensitivity experiment."""
    figure, axes = plt.subplots(
        2,
        len(experiments),
        figsize=(5 * len(experiments), 7),
        squeeze=False,
        constrained_layout=True,
    )
    for column, (experiment, (title, base_value, _)) in enumerate(experiments.items()):
        data = table[table["experiment"] == experiment]
        axes[0, column].plot(data["parameter_value"], data["objective_DKK"], marker="o")
        axes[1, column].plot(data["parameter_value"], data["import_kWh"], marker="o")
        for row in range(2):
            axes[row, column].axvline(
                base_value,
                color="red",
                linestyle="--",
                label="base",
            )
            axes[row, column].set_xlabel(title)
            axes[row, column].set_title(experiment)
            axes[row, column].legend()
        axes[0, column].set_ylabel("objective [DKK]")
        axes[1, column].set_ylabel("import [kWh]")

    figure.savefig(out / f"{filename_prefix}.png", dpi=150)
    plt.close(figure)

_COLORS_Q2 = {"C": "k", "DP": "C0", "DP_dis": "C1", "DP_bb": "C2", "DP_disbb": "C3"}
_MARKERS_Q2 = {"C": "s", "DP": "o", "DP_dis": "x", "DP_bb": "^", "DP_disbb": "v"}   # different markers: identical lines stay visible

def plot_dp_sweep(table, experiments: dict, out: Path, prefix: str) -> None:
    """Compare total cost, efficiency loss, and manager budget across Q2 sweeps."""
    n = len(experiments)
    fig, axes = plt.subplots(3, n, figsize=(5.5 * n, 10), squeeze=False, constrained_layout=True)
    for col, (name, (label, base, _)) in enumerate(experiments.items()):
        t = table[(table["experiment"] == name) & (table["status"] == "OK")]
        for model, g in t.groupby("model"):
            g = g.sort_values("parameter_value")
            kw = dict(color=_COLORS_Q2[model], marker=_MARKERS_Q2[model], label=model)
            if model == "C":
                axes[0, col].plot(g["parameter_value"], g["total_cost_DKK"], **kw)
            else:
                axes[0, col].plot(g["parameter_value"], g["total_cost_DKK"], **kw)
                axes[1, col].plot(g["parameter_value"], g["loss_vs_C_DKK"], **kw)
                axes[2, col].plot(g["parameter_value"], g["budget_DKK"], **kw)
        axes[1, col].axhline(0, color="k", lw=0.8)
        axes[2, col].axhline(0, color="k", lw=0.8)
        for r in range(3):
            axes[r, col].axvline(base, color="red", ls="--", lw=1)
        axes[0, col].set(
            title=f"Total community cost: {label}",
            ylabel="total cost [DKK]",
        )
        axes[1, col].set(
            title=f"Efficiency loss relative to C: {label}",
            ylabel="DP cost - C cost [DKK]",
        )
        axes[2, col].set(
            xlabel=label,
            ylabel="payments - manager cost [DKK]",
            title="Budget imbalance (0 = balanced)",
        )
        axes[0, col].legend(fontsize=8)
    fig.savefig(out / f"{prefix}.png", dpi=150)
    plt.close(fig)

def member_results(results: Results, d: InputData, price=None) -> Results:
    """One member's Results (same layout as Assignment 1), including all dual_* columns."""
    sch = results.meta["schedules"]
    df = (sch[sch["member"] == d.consumer_id].drop(columns="member")
          .set_index("hour").dropna(axis=1, how="all"))
    df["price"] = d.energy_price
    if price is not None:
        df["posted_price"] = price
    df["pv_available"] = d.pv_available
    df["reference_load"] = d.reference_load

    row = results.meta["members"].loc[d.consumer_id]
    scalars = {c.removeprefix("dual_"): float(row[c])
               for c in row.index if c.startswith("dual_") and pd.notna(row[c])}
    objective = float(row["settled_total"]) if "settled_total" in row else float(row["total"])
    return Results(question=f"{results.question}_{d.consumer_id}", status=results.status,
                   objective=objective, hourly=df, duals=scalars)


def plot_import_vs_cap(results: Results, save_to: Path | str | None = None) -> plt.Figure:
    """Aggregate import against the cap, exceedance shaded, day-ahead price on a second axis."""
    hr = results.hourly
    h = hr.index.to_numpy()
    n_exc = int((hr["excess"] > 1e-6).sum())
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(h, hr["import"], where="mid", color="C0", label="aggregate import")
    ax.step(h, hr["cap"], where="mid", color="C3", ls="--", label="cap")
    ax.bar(h, hr["excess"], bottom=hr["cap"], width=1.0,
           color="C3", alpha=0.3, label="exceedance")
    ax.set(xlabel="hour", ylabel="kW",
           title=f"{results.question}: import exceeds the cap in {n_exc} hours "
                 f"({hr['excess'].sum():.1f} kWh, peak {hr['import'].max():.1f} kW)")
    ax2 = ax.twinx()
    ax2.step(h, hr["price"], where="mid", color="grey", lw=1, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="grey")
    l1, b1 = ax.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)



DSO_PANELS = [  # (column, ylabel, title)
    ("exceedance_kWh",        "exceedance above cap [kWh]", "DSO: total exceedance"),
    ("hours_above_cap",       "hours above cap",            "DSO: hours above cap"),
    ("peak_reduction_vs_R_kW","peak reduction vs R [kW]",   "DSO: peak reduction"),
]

def plot_contract_sweep(table, experiments: dict, out: Path, prefix: str, first_model: str) -> None:
    n, nrows = len(experiments), 1 + len(DSO_PANELS)
    fig, axes = plt.subplots(nrows, n, figsize=(5.2 * n, 3.3 * nrows), sharex="col", squeeze=False, constrained_layout=True)
    for col, (name, (label, base, _)) in enumerate(experiments.items()):
        t = table[table["experiment"] == name]
        for model, g in t.groupby("model"):
            g = g.sort_values("parameter_value")
            if model != first_model:
                axes[0, col].plot(g["parameter_value"], g[f"gain_vs_{first_model}_DKK"], marker="o", color=_COLORS[model], label=model)
            for r, (metric, _, _) in enumerate(DSO_PANELS, start=1):
                axes[r, col].plot(g["parameter_value"], g[metric], marker="o", color=_COLORS[model], label=model)
        axes[0, col].axhline(0, color=_COLORS[first_model], lw=1.2, label=f"{first_model} (reference)")
        axes[0, col].set(title=f"Community gain vs {first_model}: {label}", ylabel=f"{first_model} cost - cost [DKK]")
        for r, (_, ylabel, title) in enumerate(DSO_PANELS, start=1):
            axes[r, col].set(ylabel=ylabel, title=title)
        axes[-1, col].set(xlabel=label)
        for r in range(nrows):
            axes[r, col].axvline(base, color="red", ls="--", lw=1, label="base")
            axes[r, col].legend(fontsize=7)
    fig.savefig(out / f"{prefix}.png", dpi=150)
    plt.close(fig)


def plot_settlement(members: pd.DataFrame, hourly: pd.DataFrame, summary: dict, save_to: Path | str | None = None) -> plt.Figure:
    """Settlement of model C: member gains vs R and I (IR), internal price vs cost of import, manager's budget."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[2.2, 1.4, 1]))
    pos, w = range(len(members)), 0.4
    a1.bar([p - w / 2 for p in pos], members["gain_vs_R_DKK"], w, color=_COLORS["R"], label="gain vs R")
    a1.bar([p + w / 2 for p in pos], members["gain_vs_I_DKK"], w, color=_COLORS["I"], label="gain vs I")
    a1.axhline(0, color="k", lw=0.8)
    a1.set_xticks(list(pos)); a1.set_xticklabels(members.index, rotation=60, ha="right", fontsize=8)
    a1.set(ylabel="cost in reference - cost in C after settlement [DKK]", title="Individual rationality by member (< 0: worse off)")
    a1.legend(fontsize=8)
    a2.step(hourly["hour"], hourly["price_x"], where="mid", color=_COLORS["C"], label="internal price x")
    a2.step(hourly["hour"], hourly["energy_price"], where="mid", color="k", ls=":", label="day-ahead price")
    a2.set(xlabel="hour", ylabel="DKK/kWh", title="Internal price from the duals"); a2.legend(fontsize=8)
    bars = [summary["collected_DKK"], summary["manager_cost_DKK"], summary["budget_DKK"]]
    a3.bar(["collected", "manager cost\n(DSO)", "surplus"], bars, color=[_COLORS["C"], "grey", "tab:red"])
    for i, v in enumerate(bars):
        a3.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    a3.set(ylabel="DKK", title="Manager's budget")
    return _finish(fig, save_to)