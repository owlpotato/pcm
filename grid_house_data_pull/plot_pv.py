import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("karlsruhe_pv_hourly.csv")
print(df.head())
print(df.dtypes)
print(len(df))   # expect 8760 (8784 in a leap year)