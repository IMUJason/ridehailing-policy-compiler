from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from trc_ridehail.rollout import evaluate_portfolio
from trc_ridehail.simulator import Policy, TravelTime

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--scenarios", type=int, default=4)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--time-limit-s", type=int, default=3)
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir
    corpus = {item["policy_id"]: item for item in map(json.loads, (ROOT / "data/policies/ambiguous_corpus.jsonl").read_text().splitlines())}
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    scenarios = [
        scenario for scenario in map(json.loads, (ROOT / "data/derived/scenarios.jsonl").read_text().splitlines())
        if scenario["split"] == args.split
    ][: args.scenarios]
    base_policies = {
        "eval-001-1": Policy(max_wait_s=300), "eval-004-1": Policy(max_detour_ratio=0.2),
        "eval-007-1": Policy(max_parties=2), "eval-009-1": Policy(fleet_size=6),
        "eval-011-1": Policy(unserved_penalty_s=3600), "eval-013-1": Policy(max_wait_s=300, max_detour_ratio=0.2),
        "eval-016-1": Policy(max_parties=2, max_detour_ratio=0.2),
    }
    travel = TravelTime()
    output = []
    for record in records:
        item = dict(record)
        policy_id = record["policy_id"]
        alternatives = corpus[policy_id]["alternatives"]
        gold = Policy(**record["gold"])
        if record["predicted"] is None:
            item["certificate"] = {"status": "review", "reason": "parse_failure"}
        elif not record.get("provenance", {}).get("accepted", False):
            item["certificate"] = {"status": "review", "reason": "source_provenance"}
        else:
            predicted = Policy(**record["predicted"])
            policies = dict(base_policies)
            policies[policy_id] = predicted
            candidates = [(policy_id, predicted)]
            for field, values in alternatives.items():
                for value in values:
                    candidates.append((f"{field}={value}", replace(predicted, **{field: value})))
            reference_policies = dict(base_policies)
            reference_policies[policy_id] = gold
            reference = evaluate_portfolio(scenarios, reference_policies, travel, time_limit_s=args.time_limit_s)
            if reference["status"] != "ok":
                raise RuntimeError(f"gold portfolio invalid at {policy_id}")
            results = []
            for label, candidate in candidates:
                result = evaluate_portfolio(scenarios, policies, travel, time_limit_s=args.time_limit_s, overrides={policy_id: candidate})
                if result["status"] != "ok":
                    results.append({"candidate": label, "best": None, "ranking": None})
                else:
                    results.append({"candidate": label, "best": result["best"], "ranking": result["ranking"]})
            best_values = {result["best"] for result in results}
            stable = None not in best_values and len(best_values) == 1
            selected_best = results[0]["best"]
            item["certificate"] = {
                "status": "auto_accept" if stable else "review",
                "reason": "rollout_stable" if stable else "rollout_unstable",
                "stable": stable,
                "selected_best": selected_best,
                "gold_best": reference["best"],
                "decision_correct": selected_best == reference["best"],
                "flipping_candidates": [result["candidate"] for result in results[1:] if result["best"] != selected_best],
                "candidates": results,
            }
        output.append(item)
        print(policy_id, item["evaluation"]["exact"], item["certificate"]["status"], item["certificate"].get("decision_correct"), flush=True)
    with (run_dir / "certificate_records.jsonl").open("w") as handle:
        for item in output:
            handle.write(json.dumps(item) + "\n")
    summary = {
        "records": len(output),
        "exact": sum(item["evaluation"]["exact"] for item in output),
        "auto_accept": sum(item["certificate"]["status"] == "auto_accept" for item in output),
        "review": sum(item["certificate"]["status"] == "review" for item in output),
        "wrong_auto_accept": sum(item["certificate"]["status"] == "auto_accept" and not item["certificate"].get("decision_correct", False) for item in output),
        "correct_auto_accept": sum(item["certificate"]["status"] == "auto_accept" and item["certificate"].get("decision_correct", False) for item in output),
    }
    (run_dir / "certificate_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
