from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
DERIVED = ROOT / "data/derived"
DATES = [8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21]
ORIGIN_ZONES = [61, 76, 42, 130]
HOURS = [8, 18, 22]


def haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def load_trips() -> pd.DataFrame:
    columns = [
        "request_datetime", "pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID",
        "trip_miles", "trip_time", "shared_request_flag", "shared_match_flag", "hvfhs_license_num",
    ]
    trips = pd.read_parquet(RAW / "fhvhv_tripdata_2025-01.parquet", columns=columns)
    trips = trips[
        (trips.request_datetime >= "2025-01-01") & (trips.request_datetime < "2025-02-01")
        & trips.PULocationID.between(1, 263) & trips.DOLocationID.between(1, 263)
        & trips.trip_miles.between(0.1, 60) & trips.trip_time.between(60, 10800)
        & trips.pickup_datetime.notna() & trips.dropoff_datetime.notna()
    ].copy()
    trips["hour_bucket"] = trips.request_datetime.dt.hour // 3
    trips["date"] = trips.request_datetime.dt.date
    return trips


def prepare_travel(trips: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    grouped = (
        trips.groupby(["PULocationID", "DOLocationID", "hour_bucket"])
        .agg(seconds=("trip_time", "median"), observations=("trip_time", "size"))
        .reset_index()
    )
    grouped["seconds"] = grouped.seconds.round().astype(int)
    grouped.to_parquet(DERIVED / "travel_time_od_hour.parquet", index=False)

    zones = gpd.read_file(RAW / "taxi_zones/taxi_zones.shp")[["LocationID", "zone", "borough", "geometry"]]
    zones["centroid"] = zones.geometry.centroid
    zones = zones.set_geometry("centroid").to_crs(4326)
    zones["lon"] = zones.geometry.x
    zones["lat"] = zones.geometry.y
    zones[["LocationID", "zone", "borough", "lon", "lat"]].to_csv(DERIVED / "zone_centroids.csv", index=False)

    centroids = zones.set_index("LocationID")[["lon", "lat"]]
    pair = (
        trips.groupby(["PULocationID", "DOLocationID"])
        .agg(seconds=("trip_time", "median"), observations=("trip_time", "size"))
        .reset_index()
    )
    pair = pair[pair.observations >= 20].join(centroids, on="PULocationID").join(
        centroids, on="DOLocationID", lsuffix="_pu", rsuffix="_do"
    )
    distance = haversine_km(pair.lon_pu.to_numpy(), pair.lat_pu.to_numpy(), pair.lon_do.to_numpy(), pair.lat_do.to_numpy())
    design = np.column_stack([np.ones(len(pair)), distance])
    weights = np.sqrt(pair.observations.to_numpy())
    intercept, slope = np.linalg.lstsq(design * weights[:, None], pair.seconds.to_numpy() * weights, rcond=None)[0]
    metadata = {
        "fallback_seconds": round(float(intercept), 3),
        "fallback_seconds_per_km": round(float(slope), 3),
        "fit_pairs": int(len(pair)),
        "travel_rows": int(len(grouped)),
    }
    (DERIVED / "travel_time_metadata.json").write_text(json.dumps(metadata, indent=2))
    return grouped, metadata


def prepare_scenarios(trips: pd.DataFrame) -> pd.DataFrame:
    shared = trips[trips.shared_request_flag == "Y"].copy()
    scenarios = []
    for day in DATES:
        for origin in ORIGIN_ZONES:
            for hour in HOURS:
                start = pd.Timestamp(2025, 1, day, hour)
                end = start + pd.Timedelta(hours=1)
                requests = shared[
                    (shared.PULocationID == origin) & (shared.request_datetime >= start) & (shared.request_datetime < end)
                ].sort_values("request_datetime")
                if len(requests) < 6:
                    continue
                requests = requests.head(30)
                scenario_id = f"rh-202501{day:02d}-z{origin}-h{hour:02d}"
                records = []
                for index, (_, row) in enumerate(requests.iterrows(), start=1):
                    records.append({
                        "id": f"R{index:02d}",
                        "pickup_zone": int(row.PULocationID),
                        "dropoff_zone": int(row.DOLocationID),
                        "request_s": int((row.request_datetime - start).total_seconds()),
                        "observed_pickup_s": int((row.pickup_datetime - start).total_seconds()),
                        "observed_trip_s": int(row.trip_time),
                        "observed_trip_miles": float(row.trip_miles),
                        "shared_matched": row.shared_match_flag == "Y",
                        "platform": row.hvfhs_license_num,
                    })
                scenarios.append({
                    "scenario_id": scenario_id,
                    "date": str(start.date()),
                    "day_type": "weekend" if start.dayofweek >= 5 else "weekday",
                    "origin_zone": int(origin),
                    "start": start.isoformat(),
                    "hour_bucket": int(hour // 3),
                    "split": "dev" if day <= 15 else "test",
                    "requests": records,
                    "fleet_size": max(2, math.ceil(len(records) / 4)),
                    "vehicle_capacity": 2,
                })
    with (DERIVED / "scenarios.jsonl").open("w") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(scenario) + "\n")
    return pd.DataFrame([{
        "scenario_id": s["scenario_id"], "split": s["split"], "day_type": s["day_type"],
        "origin_zone": s["origin_zone"], "requests": len(s["requests"]), "fleet_size": s["fleet_size"],
        "observed_shared_matches": sum(r["shared_matched"] for r in s["requests"]),
    } for s in scenarios])


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    trips = load_trips()
    _, metadata = prepare_travel(trips)
    summary = prepare_scenarios(trips)
    summary.to_csv(DERIVED / "scenario_summary.csv", index=False)
    report = {
        "clean_hvfh_v_trips": int(len(trips)),
        "scenarios": int(len(summary)),
        "dev_scenarios": int((summary.split == "dev").sum()),
        "test_scenarios": int((summary.split == "test").sum()),
        "requests": int(summary.requests.sum()),
        "requests_per_scenario": summary.requests.describe().round(2).to_dict(),
        "observed_match_rate": round(float(summary.observed_shared_matches.sum() / summary.requests.sum()), 4),
        **metadata,
    }
    (DERIVED / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
