#Mass fraction capped 1D sim
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import erf

#PCM definition 
rho = 800.0                  # kg/m^3
c_s, c_l = 2000.0, 2200.0    # J/(kg K)
k_s, k_l = 0.24, 0.20        # W/(m K)
L = 200e3                    # J/kg
T_m = 50.0                   # C

h_s = c_s * T_m              # enthalpy at melting onset
h_l = h_s + L                # enthalpy at melting end

#Boundary and initial conditions
length = 0.05                # m
height = 0.01                # m
N = 100                      # number of cells
Fo = 0.4                     # Fourier number, explicit 1D limit is 0.5, Fo=alpha*t/L^2


def temperature(h):
    return np.where(h <= h_s, h / c_s,
                    np.where(h >= h_l, T_m + (h - h_l) / c_l, T_m))


def liquid_fraction(h):
    return np.clip((h - h_s) / L, 0.0, 1.0)


def simulate(T_wall, f_init, f_target, t_max, record_every=50):
    dx = length / N
    dy = height / N
    alpha_max = max(k_s / (rho * c_s), k_l / (rho * c_l))
    dt = Fo * dx**2 / alpha_max
    n_max = int(t_max / dt)
    charging = f_target > f_init

    h = np.full(N, h_s + f_init * L)         # uniform, at T_m for 0 < f_init <= 1
    t_hist, soc_hist = [0.0], [f_init]
    t_done = None

    for n in range(1, n_max + 1):
        T = temperature(h)
        f = liquid_fraction(h)
        k = k_s + (k_l - k_s) * f
        k_face = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])

        F = np.zeros(N + 1)
        F[1:-1] = -k_face * (T[1:] - T[:-1]) / dx
        F[0] = -k[0] * (T[0] - T_wall) / (dx / 2)    # wall at x = 0
        F[-1] = 0.0                                   # insulated end

        h += dt / (rho * dx) * (F[:-1] - F[1:])

        soc = liquid_fraction(h).mean()
        if n % record_every == 0:
            t_hist.append(n * dt)
            soc_hist.append(soc)

        if (charging and soc >= f_target) or (not charging and soc <= f_target):
            t_done = n * dt
            t_hist.append(t_done)
            soc_hist.append(soc)
            break

    return np.array(t_hist), np.array(soc_hist), t_done


def neumann_time(Ste, alpha):
    """Time for the front to cross the full length (valid for a sharp-front, one-phase problem)."""
    lam = brentq(lambda x: x * np.exp(x**2) * erf(x) - Ste / np.sqrt(np.pi), 1e-6, 5.0)
    return length**2 / (4 * lam**2 * alpha), lam


#Charing
T_hot, T_cold = 70.0, 30.0
t_ch, soc_ch, done_ch = simulate(T_hot, f_init=0.0, f_target=1.0, t_max=48 * 3600)
ex_ch, lam_ch = neumann_time(c_l * (T_hot - T_m) / L, k_l / (rho * c_l))

#Discharging
t_di, soc_di, done_di = simulate(T_cold, f_init=1.0, f_target=0.0, t_max=48 * 3600)
ex_di, lam_di = neumann_time(c_s * (T_m - T_cold) / L, k_s / (rho * c_s))


def report(name, done, exact):
    if done is None:
        print(f"{name}: target not reached within t_max")
    else:
        print(f"{name}: simulated {done/3600:.2f} h, Neumann {exact/3600:.2f} h "
              f"({100*(done-exact)/exact:+.2f} %)")


report("Charge   ", done_ch, ex_ch)
report("Discharge", done_di, ex_di)

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_ch / 3600, soc_ch, label=f"charge ({T_hot:.0f} °C wall)")
ax.plot(t_di / 3600, soc_di, label=f"discharge ({T_cold:.0f} °C wall)")
ax.set_xlabel("t [h]")
ax.set_ylabel("state of charge (mean liquid fraction)")
ax.legend()
plt.tight_layout()
plt.show()