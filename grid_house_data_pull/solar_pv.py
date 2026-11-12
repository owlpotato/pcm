#!/usr/bin/env python3
"""
Hourly PV (and optional solar thermal) simulation for Karlsruhe with pvlib,
using real historical weather for an exact date range.

Weather: Open-Meteo historical archive (free, no API key, reanalysis data, available
up to about 2 days ago). Irradiance values are hourly means over the preceding hour.

Install:   pip install pvlib pandas numpy
Usage:     python karlsruhe_pv.py
           python karlsruhe_pv.py --start 2025-11-07 --end 2026-11-07 \
                  --kwp 8 --tilt 30 --azimuth 180 --collector-area 4

Output CSV (hourly, UTC and local timestamps, same format as karlsruhe_prices.py so
you can merge both files on timestamp_utc):
    pv_ac_kwh              AC energy of the PV system in that hour
    solar_thermal_kwh      heat output of the solar thermal collector in that hour
    plus the weather inputs (ghi, dni, dhi, temp_air, wind_speed, poa_global)

Model (simple, PVWatts-style, not as detailed as PVsyst):
    - Solar position + Hay-Davies transposition to the tilted plane
    - SAPM cell temperature (open rack, glass/polymer)
    - PVWatts DC model with a temperature coefficient, then a flat system loss factor
    - PVWatts inverter model
    - Solar thermal: flat-plate collector efficiency curve at a fixed mean fluid temperature.
      This is the heat the collector delivers, NOT the hot water actually used
      (no tank, no demand, no pipe/storage losses). Set --collector-area 0 to skip it.
Defaults for system size, losses and collector parameters are placeholders: replace
them with your real system data.
"""
import argparse
import json
import urllib.parse
import urllib.request
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib

BERLIN = ZoneInfo("Europe/Berlin")
LAT, LON, ALT = 49.0069, 8.4037, 115  # Karlsruhe


def fetch_weather(start, end):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation,"
                  "direct_normal_irradiance,diffuse_radiation",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    print("Downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "karlsruhe-pv-script"})
    with urllib.request.urlopen(req, timeout=120) as r:
        h = json.load(r)["hourly"]
    # Open-Meteo radiation values are means over the PRECEDING hour, labelled with the
    # END of that hour. We keep the label as hour_end and evaluate the sun at the
    # interval midpoint.
    df = pd.DataFrame({
        "hour_end": pd.to_datetime(h["time"], utc=True),
        "temp_air": h["temperature_2m"],
        "wind_speed": h["wind_speed_10m"],
        "ghi": h["shortwave_radiation"],
        "dni": h["direct_normal_irradiance"],
        "dhi": h["diffuse_radiation"],
    })
    return df


def simulate(df, args):
    # Interval midpoint for solar position; the hour label we output is the hour START.
    mid = df["hour_end"] - pd.Timedelta(minutes=30)
    times = pd.DatetimeIndex(mid)
    w = df.set_index(times).drop(columns="hour_end")

    loc = pvlib.location.Location(LAT, LON, tz="UTC", altitude=ALT)
    sp = loc.get_solarposition(times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=args.tilt,
        surface_azimuth=args.azimuth,  # 180 = south
        solar_zenith=sp["apparent_zenith"],
        solar_azimuth=sp["azimuth"],
        dni=w["dni"],
        ghi=w["ghi"],
        dhi=w["dhi"],
        dni_extra=dni_extra,
        model="haydavies",
        albedo=args.albedo,
    )
    poa_global = poa["poa_global"].fillna(0.0).clip(lower=0.0)

    # --- PV ---
    tp = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_polymer"]
    t_cell = pvlib.temperature.sapm_cell(poa_global, w["temp_air"], w["wind_speed"], **tp)
    pdc0 = args.kwp * 1000.0
    pdc = pvlib.pvsystem.pvwatts_dc(poa_global, t_cell, pdc0, gamma_pdc=args.gamma_pdc)
    pdc = pdc * (1.0 - args.losses)
    inv_pdc0 = pdc0 / args.dc_ac_ratio
    pac = pvlib.inverter.pvwatts(pdc, inv_pdc0, eta_inv_nom=args.inverter_eff)
    pv_kwh = (pac.clip(lower=0.0) / 1000.0).rename("pv_ac_kwh")  # W over 1 h -> kWh

    out = pd.DataFrame({
        "ghi": w["ghi"], "dni": w["dni"], "dhi": w["dhi"],
        "temp_air": w["temp_air"], "wind_speed": w["wind_speed"],
        "poa_global": poa_global, "t_cell": t_cell,
        "pv_ac_kwh": pv_kwh,
    })

    # --- Solar thermal (optional) ---
    if args.collector_area > 0:
        g = poa_global.where(poa_global >= args.min_irradiance)
        dT = args.collector_temp - w["temp_air"]
        eta = args.eta0 - args.a1 * dT / g - args.a2 * dT**2 / g
        eta = eta.clip(lower=0.0, upper=args.eta0).fillna(0.0)
        out["solar_thermal_kwh"] = eta * poa_global.where(g.notna(), 0.0) \
            * args.collector_area / 1000.0
    else:
        out["solar_thermal_kwh"] = np.nan

    # Re-label by hour START (UTC) to match the price script.
    out.index = out.index.floor("h")
    out.index.name = "timestamp_utc"
    return out


def main():
    today = date.today()
    default_end = today - timedelta(days=2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, default=default_end - timedelta(days=365))
    ap.add_argument("--end", type=date.fromisoformat, default=default_end)
    ap.add_argument("--kwp", type=float, default=8.0, help="PV DC size in kWp")
    ap.add_argument("--tilt", type=float, default=30.0)
    ap.add_argument("--azimuth", type=float, default=180.0, help="180 = south, 90 = east, 270 = west")
    ap.add_argument("--albedo", type=float, default=0.2)
    ap.add_argument("--losses", type=float, default=0.14, help="wiring, soiling, mismatch etc. (0-1)")
    ap.add_argument("--gamma-pdc", type=float, default=-0.004, help="power temp. coefficient per K")
    ap.add_argument("--dc-ac-ratio", type=float, default=1.15)
    ap.add_argument("--inverter-eff", type=float, default=0.96)
    ap.add_argument("--collector-area", type=float, default=4.0, help="solar thermal m2, 0 = skip")
    ap.add_argument("--collector-temp", type=float, default=50.0, help="mean fluid temp, deg C")
    ap.add_argument("--eta0", type=float, default=0.78)
    ap.add_argument("--a1", type=float, default=3.5, help="W/(m2 K)")
    ap.add_argument("--a2", type=float, default=0.015, help="W/(m2 K2)")
    ap.add_argument("--min-irradiance", type=float, default=100.0, help="W/m2 below which collector is off")
    ap.add_argument("--out", default="karlsruhe_pv_hourly.csv")
    args = ap.parse_args()

    df = fetch_weather(args.start, args.end)
    out = simulate(df, args)

    out.insert(0, "timestamp_local", [t.isoformat(timespec="minutes") for t in out.index.tz_convert(BERLIN)])
    out.index = out.index.strftime("%Y-%m-%dT%H:%M+00:00")
    out = out.round(4)
    out.to_csv(args.out)
    print(f"Wrote {len(out)} hourly rows to {args.out}")
    print(f"PV yield: {out['pv_ac_kwh'].sum():.0f} kWh "
          f"({out['pv_ac_kwh'].sum() / args.kwp:.0f} kWh/kWp)")
    if args.collector_area > 0:
        print(f"Solar thermal collector yield: {out['solar_thermal_kwh'].sum():.0f} kWh")


if __name__ == "__main__":
    main()
