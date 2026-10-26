from __future__ import annotations

import json
from pathlib import Path

from trc_ridehail.simulator import Policy, TravelTime, simulate
from trc_ridehail.validation import validate_result

ROOT = Path(__file__).resolve().parents[1]
POLICIES = {
    "base": Policy(),
    "wait_300": Policy(max_wait_s=300),
    "detour_010": Policy(max_detour_ratio=0.10),
    "capacity_3": Policy(max_parties=3),
    "fleet_plus_2": None,
    "penalty_1800": Policy(unserved_penalty_s=1800),
}


def main() -> None:
    scenarios = [json.loads(line) for line in (ROOT / "data/derived/scenarios.jsonl").read_text().splitlines()][:12]
    travel = TravelTime()
    output = ROOT / "results/sanity"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for scenario in scenarios:
        for name, policy in POLICIES.items():
            if name == "fleet_plus_2":
                policy = Policy(fleet_size=scenario["fleet_size"] + 2)
            result = simulate(scenario, policy, travel, time_limit_s=5)
            result["policy_name"] = name
            result["validation_errors"] = validate_result(scenario, policy, travel, result)
            records.append(result)
            print(scenario["scenario_id"], name, result["status"], result.get("metrics", {}).get("served"), result["validation_errors"], flush=True)
    with (output / "records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    by_scenario: dict[str, dict[str, dict]] = {}
    for record in records:
        if record["status"] == "ok" and not record["validation_errors"]:
            by_scenario.setdefault(record["scenario_id"], {})[record["policy_name"]] = record["metrics"]
    by_scenario = {key: value for key, value in by_scenario.items() if len(value) == len(POLICIES)}
    checks = {key: [] for key in ["wait_lower", "detour_lower", "capacity_pooling_up", "fleet_response_up", "low_penalty_unserved_up"]}
    for metrics in by_scenario.values():
        base = metrics["base"]
        checks["wait_lower"].append(metrics["wait_300"]["wait_mean_s"] is None or base["wait_mean_s"] is None or metrics["wait_300"]["wait_mean_s"] <= base["wait_mean_s"] + 1e-6)
        checks["detour_lower"].append(metrics["detour_010"]["detour_mean_s"] is None or base["detour_mean_s"] is None or metrics["detour_010"]["detour_mean_s"] <= base["detour_mean_s"] + 1e-6)
        checks["capacity_pooling_up"].append(metrics["capacity_3"]["pooling_rate_served"] + 1e-9 >= base["pooling_rate_served"])
        checks["fleet_response_up"].append(metrics["fleet_plus_2"]["response_rate"] + 1e-9 >= base["response_rate"])
        checks["low_penalty_unserved_up"].append(metrics["penalty_1800"]["unserved"] >= base["unserved"])
    report = {
        "valid_records": sum(record["status"] == "ok" and not record["validation_errors"] for record in records),
        "total_records": len(records),
        "complete_scenarios": len(by_scenario),
        "directional": {name: {"passed": sum(values), "total": len(values), "share": sum(values) / len(values) if values else None} for name, values in checks.items()},
    }
    (output / "directional_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
