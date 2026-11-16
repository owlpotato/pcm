<<<<<<< HEAD
import pandas as pd
from pathlib import Path

folder = Path(__file__).parent

prices = pd.read_csv(folder / "karlsruhe_prices_hourly.csv")
pv = pd.read_csv(folder / "karlsruhe_pv_hourly.csv")

# Parse UTC timestamps so both files match exactly
prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
pv["timestamp_utc"] = pd.to_datetime(pv["timestamp_utc"], utc=True)

# Both files have timestamp_local, so drop it from PV to avoid duplicate columns
pv = pv.drop(columns=["timestamp_local"])

# Merge on UTC time
merged = prices.merge(pv, on="timestamp_utc", how="outer",
                      validate="one_to_one", indicator=True)

# Sanity check: every row should say "both"
print(merged["_merge"].value_counts())
merged = merged.drop(columns="_merge").sort_values("timestamp_utc")

# Put local time first, then UTC, then the data columns
cols = ["timestamp_local", "timestamp_utc"] + [
    c for c in merged.columns if c not in ("timestamp_local", "timestamp_utc")
]
merged = merged[cols]

merged.to_csv(folder / "karlsruhe_merged_hourly.csv", index=False)
print(merged.shape)
=======
import pandas as pd
from pathlib import Path

folder = Path(__file__).parent

prices = pd.read_csv(folder / "karlsruhe_prices_hourly.csv")
pv = pd.read_csv(folder / "karlsruhe_pv_hourly.csv")

# Parse UTC timestamps so both files match exactly
prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True)
pv["timestamp_utc"] = pd.to_datetime(pv["timestamp_utc"], utc=True)

# Both files have timestamp_local, so drop it from PV to avoid duplicate columns
pv = pv.drop(columns=["timestamp_local"])

# Merge on UTC time
merged = prices.merge(pv, on="timestamp_utc", how="outer",
                      validate="one_to_one", indicator=True)

# Sanity check: every row should say "both"
print(merged["_merge"].value_counts())
merged = merged.drop(columns="_merge").sort_values("timestamp_utc")

# Put local time first, then UTC, then the data columns
cols = ["timestamp_local", "timestamp_utc"] + [
    c for c in merged.columns if c not in ("timestamp_local", "timestamp_utc")
]
merged = merged[cols]

merged.to_csv(folder / "karlsruhe_merged_hourly.csv", index=False)
print(merged.shape)
>>>>>>> fb3d7e9672a7f68ec7ba6bc4fcafbd491ac30ca2
print(merged.head())