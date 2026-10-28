from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", default=["static-shots0", "activated-shots0", "unit-shots0"])
    args = parser.parse_args()
    for run in args.runs:
        path = ROOT / "results/policy-evaluation" / run / "rollout_records.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        unparsed = [row for row in records if row["predicted"] is None]
        valid = [row for row in records if row["rollout"].get("valid")]
        regrets = sorted(row["rollout"]["decision_regret"] for row in valid)
        print(
            run,
            "statements", len(records),
            "unparsed", len(unparsed),
            "valid", len(valid),
            "best_match", sum(1 for row in records if row["rollout"].get("best_match")),
            "regrets", regrets,
            "median_regret_valid", statistics.median(regrets) if regrets else None,
        )


if __name__ == "__main__":
    main()
