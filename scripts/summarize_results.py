from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from trc_ridehail.provenance import provenance_errors

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "30B static": "results/policy-evaluation/static-shots0",
    "30B rich": "results/policy-evaluation/rich-shots0",
    "30B activated": "results/policy-evaluation/activated-shots0",
    "30B unit": "results/policy-evaluation/unit-shots0",
    "30B subset": "results/policy-evaluation/subset-shots0",
    "30B instance": "results/policy-evaluation/instance-shots0",
    "8B static": "results/policy-evaluation/deepseek8b-static-shots0",
    "8B activated": "results/policy-evaluation/deepseek8b-activated-shots0",
    "8B unit": "results/policy-evaluation/deepseek8b-unit-shots0",
}
TEXTS = {item["policy_id"]: item["text"] for item in map(json.loads, (ROOT / "data/policies/evaluation_corpus.jsonl").read_text().splitlines())}


def main() -> None:
    rows = []
    for label, relative in RUNS.items():
        path = ROOT / relative / "records.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        model = Counter(record["llm"].get("model") for record in records).most_common(1)[0][0]
        records = [record for record in records if record["llm"].get("model") == model]
        seen = set()
        unique = []
        for record in records:
            if record["policy_id"] not in seen:
                seen.add(record["policy_id"])
                unique.append(record)
        for record in unique:
            if "provenance" not in record:
                record["provenance"] = provenance_errors(TEXTS[record["policy_id"]], record["predicted"] or {}) if record["predicted"] else {"accepted": False}
        exact = sum(record["evaluation"]["exact"] for record in unique)
        accepted = sum(record["provenance"]["accepted"] for record in unique)
        false_accept = sum(record["provenance"]["accepted"] and not record["evaluation"]["exact"] for record in unique)
        latencies = [record["llm"]["latency_s"] for record in unique if record["llm"].get("latency_s") is not None]
        family = defaultdict(lambda: [0, 0])
        for record in unique:
            family[record["family"]][1] += 1
            family[record["family"]][0] += bool(record["evaluation"]["exact"])
        rows.append({
            "run": label,
            "model": model,
            "n": len(unique),
            "exact": exact,
            "exact_share": exact / len(unique),
            "parse_failures": sum(record["predicted"] is None for record in unique),
            "accepted": accepted,
            "review": len(unique) - accepted,
            "review_share": (len(unique) - accepted) / len(unique),
            "false_accept": false_accept,
            "latency_median_s": median(latencies) if latencies else None,
            "family_exact": {name: f"{value[0]}/{value[1]}" for name, value in sorted(family.items())},
        })
    output = ROOT / "results/summary"
    output.mkdir(parents=True, exist_ok=True)
    (output / "main_table.json").write_text(json.dumps(rows, indent=2))
    header = "run,n,exact,share,parse,accepted,review,review_share,false_accept,latency_s"
    lines = [header]
    for row in rows:
        lines.append(f"{row['run']},{row['n']},{row['exact']},{row['exact_share']:.3f},{row['parse_failures']},{row['accepted']},{row['review']},{row['review_share']:.3f},{row['false_accept']},{row['latency_median_s']}")
    (output / "main_table.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({row["run"]: row["family_exact"] for row in rows}, indent=2))


if __name__ == "__main__":
    main()
