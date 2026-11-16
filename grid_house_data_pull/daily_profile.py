import pandas as pd
import matplotlib.pyplot as plt

# ---- load + merge on local time (same as plot_demand.py) ----
pv = pd.read_csv("karlsruhe_pv_hourly.csv", parse_dates=["timestamp_utc"])
pv["timestamp_local"] = pv["timestamp_utc"].dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
elec = pd.read_csv("elec_demand_hourly.csv", parse_dates=["timestamp_local"])
heat = pd.read_csv("heat_demand_hourly.csv", parse_dates=["timestamp_local"])
df = pv.merge(elec, on="timestamp_local").merge(heat[["timestamp_local", "heat_kwh"]], on="timestamp_local")

df["hour"] = df["timestamp_local"].dt.hour
df["month"] = df["timestamp_local"].dt.month
cols = ["pv_ac_kwh", "elec_kwh", "solar_thermal_kwh", "heat_kwh"]

# ---- average value for each hour of the day (0-23) ----
profile = df.groupby("hour")[cols].mean()
profile.round(3).to_csv("daily_profile_avg.csv")
print("Hour of highest demand/production (yearly average):")
print(profile.idxmax().rename("hour"))

# ---- winter vs summer ----
winter = df[df["month"].isin([12, 1, 2])].groupby("hour")[cols].mean()
summer = df[df["month"].isin([6, 7, 8])].groupby("hour")[cols].mean()

fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

ax[0, 0].plot(profile.index, profile["pv_ac_kwh"], label="PV")
ax[0, 0].plot(profile.index, profile["elec_kwh"], label="Electricity demand")
ax[0, 0].set_title("Electricity: average day (whole year)")

ax[0, 1].plot(profile.index, profile["solar_thermal_kwh"], label="Solar thermal")
ax[0, 1].plot(profile.index, profile["heat_kwh"], label="Heat demand")
ax[0, 1].set_title("Heat: average day (whole year)")

ax[1, 0].plot(summer.index, summer["pv_ac_kwh"], label="PV summer")
ax[1, 0].plot(summer.index, summer["elec_kwh"], label="Demand summer")
ax[1, 0].plot(winter.index, winter["pv_ac_kwh"], "--", label="PV winter")
ax[1, 0].plot(winter.index, winter["elec_kwh"], "--", label="Demand winter")
ax[1, 0].set_title("Electricity: summer (Jun-Aug) vs winter (Dec-Feb)")

ax[1, 1].plot(summer.index, summer["solar_thermal_kwh"], label="Solar thermal summer")
ax[1, 1].plot(summer.index, summer["heat_kwh"], label="Heat demand summer")
ax[1, 1].plot(winter.index, winter["solar_thermal_kwh"], "--", label="Solar thermal winter")
ax[1, 1].plot(winter.index, winter["heat_kwh"], "--", label="Heat demand winter")
ax[1, 1].set_title("Heat: summer vs winter")

for a in ax.flat:
    a.set_ylabel("kWh per hour"); a.set_xticks(range(0, 24, 3)); a.legend(fontsize=8); a.grid(alpha=0.3)
for a in ax[1]:
    a.set_xlabel("Hour of day (local time)")
plt.tight_layout()
plt.savefig("daily_profile.png", dpi=150)
plt.show()