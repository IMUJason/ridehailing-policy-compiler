from __future__ import annotations

import argparse
import json
from pathlib import Path

from lm_client import LMStudioClient
from trc_ridehail.policy_io import build_policy_schema, parse_policy, policy_errors
from trc_ridehail.provenance import provenance_errors

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = """Example input: Limit pickup waiting to 600 seconds and additional onboard time to 25 percent of a direct trip.
Example output: {\"max_wait_s\":600,\"max_detour_ratio\":0.25}
Example input: Use eight vehicles and allow no more than two passenger parties per vehicle.
Example output: {\"fleet_size\":8,\"max_parties\":2}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen/qwen3-coder-30b")
    parser.add_argument("--schema-mode", choices=["static", "rich", "instance", "activated", "unit", "subset"], required=True)
    parser.add_argument("--shots", type=int, choices=[0, 2], default=0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--corpus", choices=["pilot", "evaluation", "ambiguous"], default="pilot")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()

    scenarios = [json.loads(line) for line in (ROOT / "data/derived/scenarios.jsonl").read_text().splitlines()]
    scenario = scenarios[0]
    corpus_files = {"pilot": "pilot_corpus.jsonl", "evaluation": "evaluation_corpus.jsonl", "ambiguous": "ambiguous_corpus.jsonl"}
    corpus = [json.loads(line) for line in (ROOT / f"data/policies/{corpus_files[args.corpus]}").read_text().splitlines()][: args.limit]
    client = LMStudioClient(args.model, temperature=0.0, timeout_s=30.0, max_retries=1)
    prefix = f"{args.run_name}-" if args.run_name else ""
    output = ROOT / f"results/policy-{args.corpus}/{prefix}{args.schema_mode}-shots{args.shots}"
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    completed = {json.loads(line)["policy_id"] for line in records_path.read_text().splitlines()} if records_path.exists() else set()
    records = []
    for item in corpus:
        if item["policy_id"] in completed:
            continue
        context = f"Operating context: origin zone {scenario['origin_zone']}, hour bucket {scenario['hour_bucket']}, {len(scenario['requests'])} requests."
        instruction = (
            "Translate the shared ride-hailing operating policy into a JSON policy patch. "
            "The schema field names already encode the units used by this policy. Copy the numeric value exactly as stated; do not convert units.\n"
            if args.schema_mode == "unit"
            else "Translate the shared ride-hailing operating policy into a JSON policy patch. "
            "Use seconds for waits and penalties, a fraction for detour ratios, and integers for parties and vehicles. "
            "Only change fields explicitly requested by the policy.\n"
        )
        prompt = (
            instruction
            + (EXAMPLES if args.shots else "")
            + context
            + "\nPolicy text: "
            + item["text"]
        )
        messages = [
            {"role": "system", "content": "You are a precise policy configuration compiler for shared ride-hailing simulation."},
            {"role": "user", "content": prompt},
        ]
        llm = client.complete(messages, response_format=build_policy_schema(args.schema_mode, scenario, item["text"]))
        record = {"policy_id": item["policy_id"], "family": item["family"], "language": item["language"], "gold": item["gold"], "llm": llm}
        try:
            _, predicted = parse_policy(llm["content"] or "", scenario)
            record["predicted"] = predicted
            record["evaluation"] = policy_errors(item["gold"], predicted)
            record["provenance"] = provenance_errors(item["text"], predicted)
            record["parse_error"] = None
        except Exception as exc:
            record["predicted"] = None
            record["evaluation"] = {"exact": False, "wrong_fields": {}}
            record["provenance"] = {"accepted": False, "rejected_fields": {}}
            record["parse_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        with (output / "records.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(item["policy_id"], args.schema_mode, record["evaluation"]["exact"], record["parse_error"], flush=True)
    all_records = [json.loads(line) for line in records_path.read_text().splitlines()] if records_path.exists() else []
    exact = sum(record["evaluation"]["exact"] for record in all_records)
    print(json.dumps({"schema_mode": args.schema_mode, "shots": args.shots, "exact": exact, "total": len(all_records)}))


if __name__ == "__main__":
    main()
