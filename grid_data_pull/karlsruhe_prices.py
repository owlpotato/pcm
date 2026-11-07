#!/usr/bin/env python3
"""
Download the last 12 months of German-Luxembourg day-ahead electricity prices
(Karlsruhe is in this bidding zone) from Energy-Charts, convert them to hourly
averages, and add residential water price columns for Karlsruhe.

Usage:
    python karlsruhe_prices.py
    python karlsruhe_prices.py --water-price 2.10 --water-base-fee 60

Needs only Python 3.9+ (no extra packages).

Water: Karlsruhe residential water is a fixed price per m3 plus a base fee, not an
hourly price. Look up the current values on the Stadtwerke Karlsruhe website and pass
them with --water-price (EUR/m3) and --water-base-fee (EUR/year). The columns are
constant in the CSV and stay empty if you pass nothing.
"""
import argparse
import csv
import json
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

BERLIN = ZoneInfo("Europe/Berlin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--water-price", type=float, default=None, help="EUR per m3, residential")
    ap.add_argument("--water-base-fee", type=float, default=None, help="EUR per year, residential")
    ap.add_argument("--out", default="karlsruhe_prices_hourly.csv")
    args = ap.parse_args()

    end = datetime.now(BERLIN).date()
    start = end - timedelta(days=365)
    url = (
        "https://api.energy-charts.info/price"
        f"?bzn=DE-LU&start={start.isoformat()}&end={end.isoformat()}"
    )
    print("Downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.load(r)

    secs = data["unix_seconds"]
    prices = data["price"]  # EUR/MWh, hourly or 15-minute depending on date

    buckets = defaultdict(list)
    for s, p in zip(secs, prices):
        if p is None:
            continue
        hour_start = s - (s % 3600)
        buckets[hour_start].append(p)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp_local", "timestamp_utc",
            "electricity_day_ahead_eur_per_mwh", "electricity_day_ahead_ct_per_kwh",
            "water_price_eur_per_m3", "water_base_fee_eur_per_year",
        ])
        for hs in sorted(buckets):
            avg = sum(buckets[hs]) / len(buckets[hs])
            utc = datetime.fromtimestamp(hs, tz=timezone.utc)
            loc = utc.astimezone(BERLIN)
            w.writerow([
                loc.isoformat(timespec="minutes"),
                utc.isoformat(timespec="minutes"),
                round(avg, 2),
                round(avg / 10, 3),  # EUR/MWh -> ct/kWh
                "" if args.water_price is None else args.water_price,
                "" if args.water_base_fee is None else args.water_base_fee,
            ])
    print(f"Wrote {len(buckets)} hourly rows to {args.out}")


if __name__ == "__main__":
    main()
