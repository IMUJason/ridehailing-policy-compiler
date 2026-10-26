from __future__ import annotations

import argparse
import json
from pathlib import Path

from trc_ridehail.rollout import evaluate_portfolio
from trc_ridehail.simulator import Policy, TravelTime

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PORTFOLIO = [
    "eval-001-1", "eval-004-1", "eval-007-1", "eval-009-1", "eval-011-1",
    "eval-013-1", "eval-015-1", "eval-016-1", "eval-017-1", "eval-020-1",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--corpus", choices=["pilot", "evaluation"], default="pilot")
    parser.add_argument("--scenarios", type=int, default=4)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--time-limit-s", type=int, default=3)
    parser.add_argument("--family-filter", default="")
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir
    corpus_name = "pilot_corpus.jsonl" if args.corpus == "pilot" else "evaluation_corpus.jsonl"
    corpus = [json.loads(line) for line in (ROOT / f"data/policies/{corpus_name}").read_text().splitlines()]
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    if args.family_filter:
        terms = tuple(args.family_filter.split(","))
        records = [record for record in records if any(term in record["family"] for term in terms)]
    scenarios = [
        scenario for scenario in map(json.loads, (ROOT / "data/derived/scenarios.jsonl").read_text().splitlines())
        if scenario["split"] == args.split
    ][: args.scenarios]
    all_gold = {item["policy_id"]: Policy(**item["gold"]) for item in corpus}
    portfolio_ids = list(all_gold) if args.corpus == "pilot" else EVALUATION_PORTFOLIO
    base_policies = {policy_id: all_gold[policy_id] for policy_id in portfolio_ids}
    travel = TravelTime()
    output = []
    for record in records:
        item = dict(record)
        gold_policy = Policy(**record["gold"])
        reference_policies = dict(base_policies)
        reference_policies[record["policy_id"]] = gold_policy
        reference = evaluate_portfolio(scenarios, reference_policies, travel, time_limit_s=args.time_limit_s)
        if reference["status"] != "ok":
            raise RuntimeError(f"gold portfolio invalid at {record['policy_id']}")
        if record["predicted"] is None:
            item["rollout"] = {"valid": False, "best_match": False, "decision_regret": None}
        else:
            result = evaluate_portfolio(
                scenarios, base_policies, travel, time_limit_s=args.time_limit_s,
                overrides={record["policy_id"]: Policy(**record["predicted"])},
            )
            if result["status"] != "ok":
                item["rollout"] = {
                    "valid": False,
                    "simulation_failed": True,
                    "best": None,
                    "gold_best": reference["best"],
                    "best_match": False,
                    "decision_regret": None,
                    "gold_ranking": reference["ranking"],
                }
            else:
                regret = result["summary"][result["best"]]["score"] - reference["summary"][reference["best"]]["score"]
                item["rollout"] = {
                    "valid": True,
                    "simulation_failed": False,
                    "best": result["best"],
                    "gold_best": reference["best"],
                    "best_match": result["best"] == reference["best"],
                    "decision_regret": regret,
                    "ranking": result["ranking"],
                    "gold_ranking": reference["ranking"],
                }
        output.append(item)
        print(item["policy_id"], item["evaluation"]["exact"], item["rollout"], flush=True)
    with (run_dir / "rollout_records.jsonl").open("w") as handle:
        for item in output:
            handle.write(json.dumps(item) + "\n")
    regrets = sorted(item["rollout"]["decision_regret"] for item in output if item["rollout"]["decision_regret"] is not None)
    summary = {
        "records": len(output),
        "exact": sum(item["evaluation"]["exact"] for item in output),
        "rollout_best_match": sum(item["rollout"]["best_match"] for item in output),
        "simulation_failures": sum(item["rollout"].get("simulation_failed", False) for item in output),
        "median_decision_regret": regrets[len(regrets) // 2] if regrets else None,
    }
    (run_dir / "rollout_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
