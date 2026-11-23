import bisect

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

from .pcm_characterisics import DEFAULT_MATERIAL, PCM
from .soc import (
    enthalpy_from_state_of_charge,
    mean_state_of_charge,
    state_of_charge,
    state_of_charge_rate,
    temperature_from_enthalpy,
)


def discharge_1d(
    T_wall,
    *,
    f_init=1.0,
    f_target=0.0,
    t_max=48 * 3600.0,
    record_every=50,
    length=0.05,
    n=100,
    fourier_number=0.4,
    material: PCM = DEFAULT_MATERIAL,
):
    if T_wall >= material.T_m:
        raise ValueError("Discharging requires a wall temperature below the PCM melting temperature")
    if not 0.0 <= f_target <= f_init <= 1.0:
        raise ValueError("Discharging requires 0 <= f_target <= f_init <= 1")
    if n < 1 or record_every < 1 or t_max <= 0 or length <= 0:
        raise ValueError("n, record_every, t_max, and length must be positive")

    dx = length / n
    alpha_max = max(
        material.k_s / (material.rho * material.c_s),
        material.k_l / (material.rho * material.c_l),
    )
    dt = fourier_number * dx**2 / alpha_max
    n_max = int(t_max / dt)
    h = np.full(n, enthalpy_from_state_of_charge(f_init, material))
    t_hist, soc_hist = [0.0], [f_init]
    t_done = None

    for step in range(1, n_max + 1):
        temperature = temperature_from_enthalpy(h, material)
        fraction = state_of_charge(h, material)
        conductivity = material.k_s + (material.k_l - material.k_s) * fraction
        conductivity_face = (
            2 * conductivity[:-1] * conductivity[1:]
            / (conductivity[:-1] + conductivity[1:])
        )

        flux = np.zeros(n + 1)
        flux[1:-1] = -conductivity_face * (temperature[1:] - temperature[:-1]) / dx
        flux[0] = -conductivity[0] * (temperature[0] - T_wall) / (dx / 2)
        h += dt / (material.rho * dx) * (flux[:-1] - flux[1:])

        soc = mean_state_of_charge(h, material)
        if step % record_every == 0:
            t_hist.append(step * dt)
            soc_hist.append(soc)
        if soc <= f_target:
            t_done = step * dt
            t_hist.append(t_done)
            soc_hist.append(soc)
            break

    return np.array(t_hist), np.array(soc_hist), t_done


def discharge_2d(
    length,
    height,
    nx,
    ny,
    wall_mask,
    T_wall,
    *,
    f_init=None,
    t_max=48 * 3600.0,
    min_rate_pct_per_h=None,
    window=1800.0,
    snap_dt=None,
    record_every=10,
    material: PCM = DEFAULT_MATERIAL,
    target_soc=0.0,
):
    if T_wall >= material.T_m:
        raise ValueError("Discharging requires a wall temperature below the PCM melting temperature")
    if not 0.0 <= target_soc <= 1.0:
        raise ValueError("target_soc must be between 0 and 1")
    if nx < 1 or ny < 1 or length <= 0 or height <= 0 or t_max <= 0:
        raise ValueError("grid dimensions, geometry, and t_max must be positive")
    if record_every < 1:
        raise ValueError("record_every must be positive")
    wall_mask = np.asarray(wall_mask, dtype=bool)
    if wall_mask.shape != (ny,):
        raise ValueError("wall_mask must have one entry per y cell")

    dx, dy = length / nx, height / ny
    alpha_max = max(
        material.k_s / (material.rho * material.c_s),
        material.k_l / (material.rho * material.c_l),
    )
    dt = 0.8 / (2 * alpha_max * (1 / dx**2 + 1 / dy**2))
    n_max = int(t_max / dt)
    if f_init is None:
        f_init = 1.0
    if not 0.0 <= target_soc <= f_init <= 1.0:
        raise ValueError("Discharging requires 0 <= target_soc <= f_init <= 1")

    h = np.full((nx, ny), enthalpy_from_state_of_charge(f_init, material))
    flux_x = np.zeros((nx + 1, ny))
    flux_y = np.zeros((nx, ny + 1))
    energy_initial = material.rho * dx * dy * h.sum()
    energy_in = 0.0
    t_hist, soc_hist = [0.0], [f_init]
    snapshots = []
    next_snapshot = 0.0
    reason, t_done = "t_max reached", None

    for step in range(1, n_max + 1):
        temperature = temperature_from_enthalpy(h, material)
        conductivity = material.k_s + (material.k_l - material.k_s) * state_of_charge(h, material)
        conductivity_x = (
            2 * conductivity[:-1, :] * conductivity[1:, :]
            / (conductivity[:-1, :] + conductivity[1:, :])
        )
        conductivity_y = (
            2 * conductivity[:, :-1] * conductivity[:, 1:]
            / (conductivity[:, :-1] + conductivity[:, 1:])
        )
        flux_x[1:-1, :] = -conductivity_x * (temperature[1:, :] - temperature[:-1, :]) / dx
        flux_y[:, 1:-1] = -conductivity_y * (temperature[:, 1:] - temperature[:, :-1]) / dy
        flux_x[0, :] = np.where(
            wall_mask,
            -conductivity[0, :] * (temperature[0, :] - T_wall) / (dx / 2),
            0.0,
        )

        h += dt / material.rho * (
            (flux_x[:-1, :] - flux_x[1:, :]) / dx
            + (flux_y[:, :-1] - flux_y[:, 1:]) / dy
        )
        energy_in += dy * flux_x[0, :].sum() * dt

        if step % record_every:
            continue
        t_now = step * dt
        fraction = state_of_charge(h, material)
        soc = mean_state_of_charge(h, material)
        if snap_dt is not None and t_now >= next_snapshot:
            snapshots.append(
                (t_now, fraction.astype(np.float32),
                 temperature_from_enthalpy(h, material).astype(np.float32))
            )
            next_snapshot += snap_dt

        rate = None
        i0 = bisect.bisect_left(t_hist, t_now - window)
        if i0 < len(t_hist) and t_now - t_hist[i0] >= 0.9 * window:
            rate = state_of_charge_rate(soc, soc_hist[i0], t_now - t_hist[i0])

        t_hist.append(t_now)
        soc_hist.append(soc)
        if soc <= target_soc:
            reason, t_done = "target SoC reached", t_now
            break
        if min_rate_pct_per_h is not None and rate is not None and rate < min_rate_pct_per_h:
            reason, t_done = f"rate fell below {min_rate_pct_per_h} %/h", t_now
            break

    fraction = state_of_charge(h, material)
    if snap_dt is not None:
        snapshots.append(
            (t_hist[-1], fraction.astype(np.float32),
             temperature_from_enthalpy(h, material).astype(np.float32))
        )
    energy_end = material.rho * dx * dy * h.sum()
    return {
        "t": np.array(t_hist),
        "soc": np.array(soc_hist),
        "snaps": snapshots,
        "reason": reason,
        "t_done": t_done,
        "dt": dt,
        "f": fraction,
        "T": temperature_from_enthalpy(h, material),
        "dx": dx,
        "dy": dy,
        "energy_err": abs((energy_end - energy_initial) - energy_in) / max(abs(energy_in), 1e-30),
    }


def neumann_time(length, stefan_number, alpha):
    """Estimate the one-phase sharp-front time to cross the given length."""
    lam = brentq(
        lambda value: value * np.exp(value**2) * erf(value) - stefan_number / np.sqrt(np.pi),
        1e-6,
        5.0,
    )
    return length**2 / (4 * lam**2 * alpha)
