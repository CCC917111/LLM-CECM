#!/usr/bin/env python3
"""Verify the bundled NSW1 reference against its original AEMO CSV, offline."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(source: Path) -> None:
    data = json.loads((ROOT / "src/data_integration/real_aemo_data_config.json").read_text())
    config = data["real_data_dynamics"]
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != config["source_sha256"]:
        raise ValueError("Source SHA-256 differs from the released source")
    start = datetime.fromisoformat(config["window_start"])
    end = datetime.fromisoformat(config["window_end"])
    groups, seen = defaultdict(list), set()
    with source.open(newline="", encoding="utf-8-sig") as stream:
        header = None
        for row in csv.reader(stream):
            if row and row[0] == "I" and "SETTLEMENTDATE" in row:
                header = {name: i for i, name in enumerate(row)}
                continue
            if not header or not row or row[0] != "D":
                continue
            if row[header["REGIONID"]] != config["region"]:
                continue
            stamp = datetime.strptime(row[header["SETTLEMENTDATE"]], "%Y/%m/%d %H:%M:%S")
            stamp = stamp.replace(tzinfo=start.tzinfo)
            if not start < stamp < end:
                continue
            if stamp in seen:
                raise ValueError(f"Duplicate retained timestamp: {stamp}")
            seen.add(stamp)
            value = float(row[header["RRP"]])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite source price: {stamp}")
            groups[stamp.replace(minute=0, second=0)].append(value)
    prices = config["ground_truth_rrp"]
    counts = config["hourly_observation_counts"]
    if not len(prices) == len(counts) == config["observation_count"] == len(groups) == 72:
        raise ValueError("Expected 72 consecutive hourly bins")
    with (ROOT / "src/data_integration/nsw1_hourly_rrp_20240715_17.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 72:
        raise ValueError("Expected 72 CSV rows")
    for i, (price, count, row) in enumerate(zip(prices, counts, rows)):
        hour = start + timedelta(hours=i)
        values = groups.get(hour, [])
        if count != len(values) or count != (11 if i == 0 else 12):
            raise ValueError(f"Unexpected interval count at {hour}")
        mean = math.fsum(values) / count
        if not math.isclose(price, mean, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"Reference differs from source at {hour}")
        if (int(row["round_index"]) != i or row["hour_start_aest"] != hour.isoformat()
                or int(row["n_observations"]) != count
                or float(row["rrp_aud_per_mwh"]) != price):
            raise ValueError(f"CSV and JSON disagree at {hour}")
    print("PASS: 72 hourly prices, timestamps, interval counts and source checksum match")
    print("Boundary: first hour uses 11 observations; remaining hours use 12 each")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    verify(parser.parse_args().source)
