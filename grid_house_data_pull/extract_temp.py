import pandas as pd

df = pd.read_csv("karlsruhe_pv_hourly.csv", parse_dates=["timestamp_local"])
temp = df[["timestamp_local", "temp_air"]]
temp["avg_temp_air"] = round(temp["temp_air"].mean(), 2)
temp.to_csv("air_temp.csv", index=False)
print(f"Average air temp: {temp['temp_air'].mean():.2f} °C")