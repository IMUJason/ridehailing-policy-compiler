from __future__ import annotations

import argparse
import json
from pathlib import Path

from trc_ridehail.provenance import provenance_errors

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--corpus", choices=["pilot", "evaluation", "ambiguous"], default="evaluation")
    parser.add_argument("--strip-failed-timeouts", action="store_true")
    args = parser.parse_args()
    run_dir = ROOT / args.run_dir
    corpus_files = {"pilot": "pilot_corpus.jsonl", "evaluation": "evaluation_corpus.jsonl", "ambiguous": "ambiguous_corpus.jsonl"}
    texts = {item["policy_id"]: item["text"] for item in map(json.loads, (ROOT / f"data/policies/{corpus_files[args.corpus]}").read_text().splitlines())}
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    annotated = []
    for record in records:
        if args.strip_failed_timeouts and record["parse_error"] and any(
            (attempt.get("error") or {}).get("stage") == "timeout" for attempt in record["llm"].get("attempts", [])
        ):
            continue
        if record["predicted"] is None:
            record["provenance"] = {"accepted": False, "rejected_fields": {}, "parse_failure": record["parse_error"]}
        else:
            record["provenance"] = provenance_errors(texts[record["policy_id"]], record["predicted"])
        annotated.append(record)
    with (run_dir / "records.jsonl").open("w") as handle:
        for record in annotated:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "records": len(annotated),
        "exact": sum(record["evaluation"]["exact"] for record in annotated),
        "provenance_accepted": sum(record["provenance"]["accepted"] for record in annotated),
        "exact_and_provenance": sum(record["evaluation"]["exact"] and record["provenance"]["accepted"] for record in annotated),
        "provenance_rejections": sum(not record["provenance"]["accepted"] for record in annotated),
    }
    (run_dir / "provenance_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
