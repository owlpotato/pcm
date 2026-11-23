"""
determining when to buy energy and when to sell

NOT FINISHED YET
"""
import sys

import pandas as pd

CSV_FILE = sys.argv[1] if len(sys.argv) > 1 else "karlsruhe_prices_hourly.csv"

df = pd.read_csv(CSV_FILE)


def run(power_from_PV, P_el):
    """
    power_from_PV, P_el: either a single number (same value every hour)
    or a list / array / pandas Series with one value per row of df (kW).
    """
    # a single number is used for every hour
    if isinstance(power_from_PV, (int, float)):
        power_from_PV = [power_from_PV] * len(df)
    if isinstance(P_el, (int, float)):
        P_el = [P_el] * len(df)

    if len(power_from_PV) != len(df) or len(P_el) != len(df):
        raise ValueError(f"power_from_PV and P_el need {len(df)} values (one per row)")

    power_sold_bought = 0   # kWh, positive = sold overall, negative = bought overall
    balance = 0             # ct, positive = earned, negative = spent

    for row, pv, load in zip(df.itertuples(), power_from_PV, P_el):
        local_time = row.timestamp_local                      # e.g. 2025-11-07T00:00+01:00
        price = row.electricity_day_ahead_ct_per_kwh          # ct/kWh
        power_from_grid = max(load - pv, 0.0)
        power_sold = max(pv - load, 0.0)

        if load - pv > 0:
            power_sold_bought -= power_from_grid
            balance -= power_from_grid * price
        else:
            # negative price rule: do not sell when the price is below zero
            # (the surplus is curtailed, so it is not counted as sold)
            if price < 0:
                power_sold = 0.0
            power_sold_bought += power_sold
            balance += power_sold * price

    return {
        "power_sold_bought": power_sold_bought, #kWh
        "balance": balance, #currently in ct
    }


if __name__ == "__main__":
    # same value every hour ...
    print(run(5.0, 3.0))

    # ... or one value per hour, e.g. from lists or columns of a second file
    # pv = [...]    # len(df) values in kW
    # load = [...]  # len(df) values in kW
    # print(run(pv, load))
