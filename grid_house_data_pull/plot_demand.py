import pandas as pd
import matplotlib.pyplot as plt

# which PV columns to plot against demand
PV_FOR_ELEC = "pv_ac_kwh"
PV_FOR_HEAT = "solar_thermal_kwh"   # change to "pv_ac_kwh" if you want PV electricity vs heat

# ---- load + merge on local time ----
pv = pd.read_csv("karlsruhe_pv_hourly.csv", parse_dates=["timestamp_utc"])
pv["timestamp_local"] = pv["timestamp_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
elec = pd.read_csv("elec_demand_hourly.csv", parse_dates=["timestamp_local"])
heat = pd.read_csv("heat_demand_hourly.csv", parse_dates=["timestamp_local"])

df = pv.merge(elec, on="timestamp_local", how="left").merge(heat, on="timestamp_local", how="left")
print(len(df), "rows, NaNs in demand:", df[["elec_kwh", "heat_kwh"]].isna().sum().sum())

# ---- scatter: PV vs demand ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(df[PV_FOR_ELEC], df["elec_kwh"], s=3, alpha=0.4)
ax[0].set(xlabel=f"{PV_FOR_ELEC}", ylabel="Electricity demand (kWh)", title="PV vs electricity demand")
ax[1].scatter(df[PV_FOR_HEAT], df["heat_kwh"], s=3, alpha=0.4)
ax[1].set(xlabel=f"{PV_FOR_HEAT}", ylabel="Heat demand (kWh)", title="Solar vs heat demand")
plt.tight_layout()
plt.savefig("pv_vs_demand_scatter.png", dpi=150)

# ---- time series: one example week ----
week = df[(df["timestamp_local"] >= "2026-07-06") & (df["timestamp_local"] < "2026-07-13")]
fig2, ax2 = plt.subplots(2, 1, figsize=(12, 6))
ax2[0].plot(week["timestamp_local"], week[PV_FOR_ELEC], label="PV")
ax2[0].plot(week["timestamp_local"], week["elec_kwh"], label="Electricity demand")
ax2[0].set_ylabel("kWh"); ax2[0].legend()
winter = df[(df["timestamp_local"] >= "2026-01-12") & (df["timestamp_local"] < "2026-01-19")]
ax2[1].plot(winter["timestamp_local"], winter[PV_FOR_HEAT], label="Solar thermal")
ax2[1].plot(winter["timestamp_local"], winter["heat_kwh"], label="Heat demand")
ax2[1].set_ylabel("kWh"); ax2[1].legend()
ax2[0].set_title("Summer week (top) / winter week (bottom)")
plt.tight_layout()
plt.savefig("pv_vs_demand_timeseries.png", dpi=150)
plt.show()