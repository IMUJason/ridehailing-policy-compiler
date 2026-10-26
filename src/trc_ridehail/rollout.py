from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from trc_ridehail.simulator import Policy, TravelTime, simulate
from trc_ridehail.validation import validate_result

ROOT = Path(__file__).resolve().parents[2]


def policy_key(policy: Policy) -> str:
    return json.dumps(asdict(policy), sort_keys=True)


def score_record(record: dict) -> float:
    metrics = record["metrics"]
    waits = sum(detail.get("wait_s", 0) for detail in record["requests"] if detail["served"])
    detours = sum(detail.get("detour_s", 0) for detail in record["requests"] if detail["served"])
    return 7200 * metrics["unserved"] + waits + detours + 3600 * metrics["empty_travel_hours"]


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload))
    os.replace(temporary, path)


def run_cached(scenario: dict, policy: Policy, travel: TravelTime, time_limit_s: float, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"scenario": scenario, "policy": asdict(policy), "time_limit_s": time_limit_s}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"{digest}.json"
    if path.exists():
        try:
            result = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            result = None
        if result is not None:
            if result.get("status") == "ok":
                result["validation_errors"] = validate_result(scenario, policy, travel, result)
                result["score"] = score_record(result)
                write_json_atomic(path, result)
            return result
    result = simulate(scenario, policy, travel, time_limit_s)
    result["validation_errors"] = validate_result(scenario, policy, travel, result)
    result["score"] = score_record(result) if result["status"] == "ok" else None
    write_json_atomic(path, result)
    return result


def evaluate_portfolio(
    scenarios: list[dict],
    policies: dict[str, Policy],
    travel: TravelTime,
    time_limit_s: float = 5,
    cache_dir: Path | None = None,
    overrides: dict[str, Policy] | None = None,
) -> dict:
    cache_dir = cache_dir or ROOT / "results/cache"
    selected = dict(policies)
    if overrides:
        selected.update(overrides)
    records: dict[str, list[dict]] = {}
    for name, policy in selected.items():
        records[name] = [run_cached(scenario, policy, travel, time_limit_s, cache_dir) for scenario in scenarios]
        if any(record["status"] != "ok" or record["validation_errors"] for record in records[name]):
            return {"status": "invalid", "policy": name, "records": records}
    summary = {}
    for name, policy_records in records.items():
        requests = sum(record["metrics"]["requests"] for record in policy_records)
        served = sum(record["metrics"]["served"] for record in policy_records)
        pooled = sum(record["metrics"]["pooling_rate_served"] * record["metrics"]["served"] for record in policy_records)
        waits = [detail["wait_s"] for record in policy_records for detail in record["requests"] if detail["served"]]
        detours = [detail["detour_s"] for record in policy_records for detail in record["requests"] if detail["served"]]
        summary[name] = {
            "score": sum(record["score"] for record in policy_records),
            "requests": requests,
            "response_rate": served / requests,
            "pooling_rate_served": pooled / served if served else 0.0,
            "wait_mean_s": sum(waits) / len(waits) if waits else None,
            "detour_mean_s": sum(detours) / len(detours) if detours else None,
            "vehicle_hours": sum(record["metrics"]["vehicle_hours"] for record in policy_records),
            "empty_travel_hours": sum(record["metrics"]["empty_travel_hours"] for record in policy_records),
        }
    ranking = sorted(summary, key=lambda name: (summary[name]["score"], name))
    return {"status": "ok", "ranking": ranking, "best": ranking[0], "summary": summary, "records": records}


def decision_regret(reference: dict, selected_best: str) -> float:
    return reference["summary"][selected_best]["score"] - reference["summary"][reference["best"]]["score"]
