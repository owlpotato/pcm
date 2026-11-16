import pandas as pd
import holidays
from demandlib import bdew

# ---- inputs you can change ----
ANNUAL_ELEC_KWH = 3500     # household electricity per year
ANNUAL_HEAT_KWH = 15000    # space heat + hot water per year
HOUSE_TYPE = "EFH"         # EFH = single-family, MFH = multi-family
START, END = "2025-11-07 00:00", "2026-11-07 23:00"

# ---- air temp from your PV csv, as local wall-clock time ----
df = pd.read_csv("karlsruhe_pv_hourly.csv", parse_dates=["timestamp_utc"])
temp = df.set_index("timestamp_utc")["temp_air"]
temp.index = temp.index.tz_convert("Europe/Berlin").tz_localize(None)
temp = temp[~temp.index.duplicated()]                 # DST autumn: repeated hour
idx = pd.date_range(START, END, freq="h")             # continuous hourly index
temp = temp.reindex(idx).ffill()                      # DST spring: missing hour

# ---- holidays in Baden-Wuerttemberg ----
bw = holidays.Germany(subdiv="BW", years=[2025, 2026])

# ---- electricity demand (BDEW H0 household profile) ----
parts = []
for yr in (2025, 2026):
    slp = bdew.ElecSlp(yr, holidays=bw)
    parts.append(slp.get_scaled_power_profiles({"h0": ANNUAL_ELEC_KWH}))   # 15-min avg kW
elec_h = pd.concat(parts)["h0"].resample("h").mean()  # avg kW over 1 h = kWh per hour
elec_h = elec_h.reindex(idx)

# ---- thermal demand (BDEW heat profile driven by air temp) ----
heat = bdew.HeatBuilding(
    idx,
    holidays=bw,
    temperature=temp,
    shlp_type=HOUSE_TYPE,
    building_class=1,
    wind_class=0,
    annual_heat_demand=ANNUAL_HEAT_KWH,
    name=HOUSE_TYPE,
).get_bdew_profile()

out = pd.DataFrame({"temp_air": temp, "elec_kwh": elec_h, "heat_kwh": heat})
out.index.name = "timestamp_local"
out = out.round(4)
out[["elec_kwh"]].to_csv("elec_demand_hourly.csv")
out[["temp_air", "heat_kwh"]].to_csv("heat_demand_hourly.csv")

print(out.head())
print(out[["elec_kwh", "heat_kwh"]].sum().round(0))
print(len(out), "rows, NaNs:", out.isna().sum().sum())