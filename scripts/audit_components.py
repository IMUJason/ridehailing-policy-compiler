from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from trc_ridehail.policy_io import active_fields
from trc_ridehail.provenance import canonical_candidates

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    "unit-shots0", "activated-shots0", "static-shots0",
    "deepseek8b-unit-shots0", "deepseek8b-activated-shots0", "deepseek8b-static-shots0",
]
GOLD_FIELDS = {
    "paraphrase-01": {"max_wait_s"},
    "paraphrase-02": {"max_parties"},
    "paraphrase-03": {"max_detour_ratio"},
    "paraphrase-04": {"fleet_size"},
    "paraphrase-05": {"unserved_penalty_s"},
    "paraphrase-06": {"max_wait_s", "max_parties", "max_detour_ratio"},
    "paraphrase-07": {"max_wait_s", "fleet_size"},
    "paraphrase-08": {"max_detour_ratio", "unserved_penalty_s"},
    "paraphrase-09": {"fleet_size", "max_wait_s"},
    "paraphrase-10": {"max_parties", "max_wait_s"},
    "paraphrase-11": {"max_detour_ratio", "max_parties"},
    "paraphrase-12": {"fleet_size", "max_detour_ratio"},
}
PROBES = [
    ("keep the pickup wait at 5 minutes", "max_wait_s", 300),
    ("keep the pickup wait at 8 minutes", "max_wait_s", 480),
    ("keep the pickup wait at 12 min", "max_wait_s", 720),
    ("接驾等待不超过八分钟", "max_wait_s", 480),
    ("keep the pickup wait at about 10 minutes", "max_wait_s", 600),
    ("keep the pickup wait between 10 and 12 minutes", "max_wait_s", 600),
    ("charge unserved requests 1.5 hours", "unserved_penalty_s", 5400),
    ("charge unserved requests half an hour", "unserved_penalty_s", 1800),
    ("charge unserved requests 90 minutes", "unserved_penalty_s", 5400),
    ("limit the detour to 30 percent", "max_detour_ratio", 0.30),
    ("limit the detour to forty percent", "max_detour_ratio", 0.40),
    ("put 6 vehicles into service", "fleet_size", 6),
]


def prf(pairs: list[tuple[set[str], set[str]]]) -> dict:
    tp = sum(len(g & p) for g, p in pairs)
    fp = sum(len(p - g) for g, p in pairs)
    fn = sum(len(g - p) for g, p in pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def rule_config(text: str) -> dict:
    config = {}
    for field in active_fields(text):
        candidates = canonical_candidates(text, field)
        if len(candidates) == 1:
            config[field] = candidates[0]
    return config


def strict_accept(text: str, config: dict) -> bool:
    if set(config) != set(active_fields(text)):
        return False
    for field, value in config.items():
        candidates = canonical_candidates(text, field)
        if not candidates or not any(abs(value - candidate) < 1e-9 for candidate in candidates):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey-db", default="")
    parser.add_argument("--out", default="results/components.json")
    args = parser.parse_args()

    corpus = [json.loads(line) for line in (ROOT / "data/policies/evaluation_corpus.jsonl").read_text().splitlines()]
    texts = {row["policy_id"]: row["text"] for row in corpus}
    result: dict = {}

    result["activation_corpus"] = prf([(set(row["gold"]), set(active_fields(row["text"]))) for row in corpus])
    no_change = [row for row in corpus if not row["gold"]]
    result["activation_corpus"]["no_change_items"] = len(no_change)
    result["activation_corpus"]["false_activations_on_no_change"] = sum(
        1 for row in no_change if active_fields(row["text"])
    )

    if args.survey_db:
        connection = sqlite3.connect(args.survey_db)
        rows = [dict(zip(("intent_id", "response"), row)) for row in connection.execute(
            "SELECT intent_id, response FROM paraphrases"
        )]
        connection.close()
        pairs = [(GOLD_FIELDS[row["intent_id"]], set(active_fields(row["response"]))) for row in rows]
        result["activation_human"] = prf(pairs)
        result["activation_human"]["statements"] = len(rows)

    rule_rows = [(row, rule_config(row["text"])) for row in corpus]
    failures = [row for row, config in rule_rows if config != row["gold"]]
    result["rule_baseline"] = {
        "exact": len(corpus) - len(failures),
        "total": len(corpus),
        "families": dict(Counter(row["family"] for row in failures)),
        "single_field_failures": sum(1 for row in failures if len(row["gold"]) == 1),
    }

    strict = {}
    for run in RUNS:
        path = ROOT / "results/policy-evaluation" / run / "records.jsonl"
        if not path.exists():
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        parsed = [row for row in records if row["predicted"] is not None]
        admitted = [row for row in parsed if strict_accept(texts[row["policy_id"]], row["predicted"])]
        strict[run] = {
            "parsed": len(parsed),
            "admitted": len(admitted),
            "wrong_admitted": sum(1 for row in admitted if not row["evaluation"]["exact"]),
        }
    result["strict_guard"] = strict

    probes = []
    for text, field, value in PROBES:
        candidates = canonical_candidates(text, field)
        probes.append({
            "text": text, "field": field, "value": value, "candidates": candidates,
            "handled": any(abs(value - candidate) < 1e-9 for candidate in candidates),
        })
    result["numeric_probes"] = {"handled": sum(1 for row in probes if row["handled"]), "total": len(probes), "cases": probes}

    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "numeric_probes"}, indent=2))
    print("numeric probes handled:", result["numeric_probes"]["handled"], "/", result["numeric_probes"]["total"])


if __name__ == "__main__":
    main()
