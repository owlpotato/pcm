import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("karlsruhe_pv_hourly.csv")

# parse UTC, then convert to Berlin time (handles the +01:00 / +02:00 DST switch)
df["time"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert("Europe/Berlin")
df = df.set_index("time")

pv = df["pv_ac_kwh"]
print(pv.describe())
print("Total PV yield [kWh]:", round(pv.sum()))

# 1) full hourly profile
pv.plot(figsize=(12, 4), ylabel="PV generation [kWh/h]", title="Hourly PV generation")
plt.tight_layout()
plt.show()

# 2) average day per month
profile = pv.groupby([pv.index.month, pv.index.hour]).mean().unstack(0)
profile.plot(figsize=(10, 5), xlabel="Hour of day", ylabel="kWh/h", title="Average day per month")
plt.tight_layout()
plt.show()

# 3) summer week vs winter week
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharey=True)
pv["2026-07-01":"2026-07-07"].plot(ax=ax[0], title="Summer week")
pv["2026-01-01":"2026-01-07"].plot(ax=ax[1], title="Winter week")
plt.tight_layout()
plt.show()  # expect 8760 (8784 in a leap year)