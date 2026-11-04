"""Named variations of the base case (Table 11, dimension c: contract terms). Each returns a modified copy."""
from __future__ import annotations
from dataclasses import replace

def with_discount(com, beta: float):
    """Tariff discount beta in [0, 1]."""
    return replace(com, contract=replace(com.contract, tariff_discount_ratio=float(beta)))

def with_penalty(com, alpha: float):
    """Penalty alpha in DKK/kWh on every kWh imported above the cap."""
    return replace(com, contract=replace(com.contract, penalty_DKK_per_kWh=float(alpha)))


def with_cap_scale(com, s: float):
    """Scale the cap by s. Model I's individual caps are recomputed (kappa_i x cap), or I would keep the old ones."""
    c = com.contract
    cap = c.capacity_limit_kW * s
    return replace(com, contract=replace(
        c, capacity_limit_kW=cap,
        capacity_limit_daily_energy_kWh=float(cap.sum()),
        capacity_limit_daily_energy_factor=c.capacity_limit_daily_energy_factor * s,
        individual_capacity_limit_kW={cid: c.share[cid] * cap for cid in com.member_ids}))