"""
2D enthalpy method (cell-centred finite volume, explicit), PCM slab of size length x height.

Array convention: field[i, j]  <->  (x_i, y_j).  Plots use field.T so x is horizontal.
Boundaries: heated/cooled wall on the left (x = 0) where wall_mask[j] is True,
            insulated (zero flux) everywhere else.

Stopping: the run ends when the first of these happens
  1. the PCM is completely converted (SoC reaches 1 when charging, 0 when discharging)
  2. the rate of SoC change falls below min_rate_pct_per_h (optional)
  3. t_max is reached (safety cap)
"""
import bisect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import brentq
from scipy.special import erf

# ---------------- material ----------------
rho = 800.0                  # kg/m^3
c_s, c_l = 2000.0, 2200.0    # J/(kg K)
k_s, k_l = 0.24, 0.20        # W/(m K)
L = 200e3                    # J/kg
T_m = 50.0                   # C

h_s = c_s * T_m              # enthalpy at melting onset
h_l = h_s + L                # enthalpy at melting end


def temperature(h):
    return np.where(h <= h_s, h / c_s,
                    np.where(h >= h_l, T_m + (h - h_l) / c_l, T_m))


def liquid_fraction(h):
    return np.clip((h - h_s) / L, 0.0, 1.0)


def stable_dt(dx, dy, safety=0.8):
    alpha_max = max(k_s / (rho * c_s), k_l / (rho * c_l))  #thermal diffusivity of the PCM
    return safety / (2 * alpha_max * (1 / dx**2 + 1 / dy**2))


def simulate(length, height, nx, ny, wall_mask, T_wall, f_init=None, t_max=48 * 3600.0,
             min_rate_pct_per_h=None, window=1800.0, snap_dt=None, record_every=10):
    dx, dy = length / nx, height / ny
    dt = stable_dt(dx, dy)
    n_max = int(t_max / dt)
    charging = T_wall > T_m
    if f_init is None:
        f_init = 0.0 if charging else 1.0
    f_goal = 1.0 if charging else 0.0

    h = np.full((nx, ny), h_s + f_init * L)
    Fx = np.zeros((nx + 1, ny))      # heat flux in +x on vertical faces
    Fy = np.zeros((nx, ny + 1))      # heat flux in +y on horizontal faces
    E0 = rho * dx * dy * h.sum()     # J per metre depth
    E_in = 0.0

    t_hist, soc_hist = [0.0], [f_init]
    snaps = []
    next_snap = 0.0
    reason, t_done = "t_max reached", None

    for n in range(1, n_max + 1):
        T = temperature(h)
        k = k_s + (k_l - k_s) * liquid_fraction(h)

        kx = 2 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])      # harmonic mean on faces
        ky = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
        Fx[1:-1, :] = -kx * (T[1:, :] - T[:-1, :]) / dx
        Fy[:, 1:-1] = -ky * (T[:, 1:] - T[:, :-1]) / dy
        Fx[0, :] = np.where(wall_mask, -k[0, :] * (T[0, :] - T_wall) / (dx / 2), 0.0)
        # Fx[-1, :], Fy[:, 0], Fy[:, -1] stay 0: insulated

        h += dt / rho * ((Fx[:-1, :] - Fx[1:, :]) / dx + (Fy[:, :-1] - Fy[:, 1:]) / dy)
        E_in += dy * Fx[0, :].sum() * dt

        if n % record_every:
            continue
        t_now = n * dt
        f = liquid_fraction(h)
        soc = f.mean()

        if snap_dt is not None and t_now >= next_snap:
            snaps.append((t_now, f.astype(np.float32), temperature(h).astype(np.float32)))
            next_snap += snap_dt

        # SoC rate over a moving window (single steps are too jumpy: cells switch phase one by one)
        rate = None
        i0 = bisect.bisect_left(t_hist, t_now - window)
        if i0 < len(t_hist) and t_now - t_hist[i0] >= 0.9 * window:
            rate = abs(soc - soc_hist[i0]) / (t_now - t_hist[i0]) * 3600 * 100   # %/h

        t_hist.append(t_now)
        soc_hist.append(soc)

        if abs(soc - f_goal) < 1e-12:
            reason, t_done = "PCM completely converted", t_now
            break
        if min_rate_pct_per_h is not None and rate is not None and rate < min_rate_pct_per_h:
            reason, t_done = f"rate fell below {min_rate_pct_per_h} %/h", t_now
            break

    f = liquid_fraction(h)
    if snap_dt is not None:
        snaps.append((t_hist[-1], f.astype(np.float32), temperature(h).astype(np.float32)))
    E_end = rho * dx * dy * h.sum()
    return dict(t=np.array(t_hist), soc=np.array(soc_hist), snaps=snaps, reason=reason,
                t_done=t_done, dt=dt, f=f, T=temperature(h), dx=dx, dy=dy,
                energy_err=abs((E_end - E0) - E_in) / max(abs(E_in), 1e-30))


def neumann_time(length, Ste, alpha):
    """Time for the 1D one-phase front to cross the full length."""
    lam = brentq(lambda x: x * np.exp(x**2) * erf(x) - Ste / np.sqrt(np.pi), 1e-6, 5.0)
    return length**2 / (4 * lam**2 * alpha)


# ---------------- plotting ----------------
def _draw_wall(ax, wall_mask, dy):
    j = np.where(wall_mask)[0]
    ax.vlines(np.zeros(len(j)), j * dy * 1e3, (j + 1) * dy * 1e3,
              colors="lime", linewidth=5, clip_on=False)


def plot_snapshots(res, length, height, wall_mask, n_panels=6, fname=None):
    snaps = res["snaps"]
    idx = np.linspace(0, len(snaps) - 1, n_panels).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    x = (np.arange(res["f"].shape[0]) + 0.5) * res["dx"] * 1e3
    y = (np.arange(res["f"].shape[1]) + 0.5) * res["dy"] * 1e3
    for ax, i in zip(axes.flat, idx):
        t, f, _ = snaps[i]
        im = ax.imshow(f.T, origin="lower", extent=[0, length * 1e3, 0, height * 1e3],
                       vmin=0, vmax=1, cmap="coolwarm", aspect="equal")
        if f.min() < 0.5 < f.max():
            ax.contour(x, y, f.T, levels=[0.5], colors="k", linewidths=0.8)
        _draw_wall(ax, wall_mask, res["dy"])
        ax.set_title(f"t = {t/3600:.1f} h,  SoC = {f.mean():.2f}")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    fig.colorbar(im, ax=axes, label="liquid fraction f  (0 = solid, 1 = liquid)")
    if fname:
        fig.savefig(fname, dpi=130)
    return fig


def animate(res, length, height, wall_mask, T_wall, fname="phase_evolution.gif", fps=10):
    snaps = res["snaps"]
    ext = [0, length * 1e3, 0, height * 1e3]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    im1 = a1.imshow(snaps[0][1].T, origin="lower", extent=ext, vmin=0, vmax=1, cmap="coolwarm")
    im2 = a2.imshow(snaps[0][2].T, origin="lower", extent=ext, cmap="inferno",
                    vmin=min(T_m, T_wall), vmax=max(T_m, T_wall))
    fig.colorbar(im1, ax=a1, label="liquid fraction f")
    fig.colorbar(im2, ax=a2, label="T [°C]")
    for a in (a1, a2):
        _draw_wall(a, wall_mask, res["dy"])
        a.set_xlabel("x [mm]")
        a.set_ylabel("y [mm]")

    def update(k):
        t, f, T = snaps[k]
        im1.set_data(f.T)
        im2.set_data(T.T)
        fig.suptitle(f"t = {t/3600:.1f} h,  SoC = {f.mean():.2f}")
        return im1, im2

    ani = FuncAnimation(fig, update, frames=len(snaps), interval=1000 / fps)
    ani.save(fname, writer="pillow", fps=fps)
    return ani


if __name__ == "__main__":
    length, height = 0.05, 0.04        # m
    dx_target = 1e-3                   # m: pick ny from height/dx so cells stay roughly square
    nx, ny = round(length / dx_target), round(height / dx_target)
    T_hot = 70.0

    # ---- 1) validation: wall covering the full height must reproduce the 1D result ----
    full = np.ones(ny, dtype=bool)
    res = simulate(length, height, nx, ny, full, T_hot)
    Ste = c_l * (T_hot - T_m) / L
    t_exact = neumann_time(length, Ste, k_l / (rho * c_l))
    print(f"[validation] 2D full-height wall: {res['t_done']/3600:.2f} h, "
          f"1D Neumann: {t_exact/3600:.2f} h, energy balance error {res['energy_err']:.1e}")

    # ---- 2) real 2D case: only the middle third of the left wall is a heat exchanger ----
    wall = np.zeros(ny, dtype=bool)
    wall[ny // 3: 2 * ny // 3] = True
    res = simulate(length, height, nx, ny, wall, T_hot, min_rate_pct_per_h=4.0, snap_dt=900.0)
    P_latent = rho * L * length * height * 4.0 / 100 / 3600      # W per m depth at the cutoff rate
    print(f"[2D partial wall] stopped after {res['t_done']/3600:.2f} h: {res['reason']} "
          f"(~{P_latent:.1f} W/m latent power); usable SoC = {res['soc'][-1]:.2f}; "
          f"energy balance error {res['energy_err']:.1e}")

    plot_snapshots(res, length, height, wall, fname="phase_snapshots.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(res["t"] / 3600, res["soc"] * 100)
    ax.set_xlabel("t [h]")
    ax.set_ylabel("SoC [%]")
    plt.tight_layout()

    # animate(res, length, height, wall, T_hot)   # writes phase_evolution.gif (slow for big grids)
    plt.show()