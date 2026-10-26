from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITEMS = {item["item_id"]: item for item in json.loads((HERE / "data" / "items.json").read_text())}
INTENTS = {item["intent_id"]: item for item in json.loads((HERE / "data" / "intents.json").read_text())}
FLAG = {"reject", "clarify"}


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def condition(item_id: str) -> str:
    for prefix in ("correct-", "wrong-unit-", "wrong-static-", "wrong-activated-"):
        if item_id.startswith(prefix):
            return prefix.rstrip("-")
    return "other"


def fleiss(rows: list[list[int]]) -> float:
    rows = [r for r in rows if sum(r) >= 2]
    if not rows:
        return float("nan")
    total = sum(sum(r) for r in rows)
    p_j = [sum(r[j] for r in rows) / total for j in range(2)]
    p_bar = sum(p * p for p in p_j)
    scores = []
    for r in rows:
        n = sum(r)
        scores.append((sum(v * v for v in r) - n) / (n * (n - 1)))
    agree = sum(scores) / len(scores)
    if abs(1 - p_bar) < 1e-12:
        return float("nan")
    return (agree - p_bar) / (1 - p_bar)


def cohen(a: list[int], b: list[int]) -> float:
    if not a:
        return float("nan")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)


def parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(HERE / "survey.db"))
    ap.add_argument("--out", default=str(HERE / "analysis"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    parts = [dict(r) for r in conn.execute("SELECT * FROM participants")]
    reviews = [dict(r) for r in conn.execute("SELECT * FROM reviews")]
    paras = [dict(r) for r in conn.execute("SELECT * FROM paraphrases")]

    report: dict[str, object] = {}
    report["participants"] = len(parts)
    report["completed"] = sum(1 for p in parts if p["completed_at"])
    spans = [
        (parse(p["completed_at"]) - parse(p["started_at"])).total_seconds() / 60
        for p in parts
        if p["completed_at"] and p["started_at"]
    ]
    report["minutes_median"] = round(statistics.median(spans), 1) if spans else None
    report["minutes_range"] = [round(min(spans), 1), round(max(spans), 1)] if spans else None

    per_code = Counter(r["code"] for r in reviews)
    report["reviews_total"] = len(reviews)
    report["reviews_per_participant"] = dict(sorted(per_code.items()))
    report["review_minutes_median"] = (
        round(statistics.median(r["duration_ms"] for r in reviews) / 60000, 1) if reviews else None
    )

    by_cond: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "accept": 0, "reject": 0, "clarify": 0})
    by_cond_hi: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "accept": 0, "reject": 0, "clarify": 0})
    by_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "accept": 0, "reject": 0, "clarify": 0})
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "accept": 0, "reject": 0, "clarify": 0})
    by_cross: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "accept": 0, "reject": 0, "clarify": 0})
    loc_hit = loc_tot = 0
    guard_confusion = Counter()

    for r in reviews:
        item = ITEMS.get(r["item_id"])
        if item is None:
            continue
        cond = condition(r["item_id"])
        d = r["decision"]
        for bucket, key in [(by_cond, cond), (by_lang, item["language"]), (by_family, item["family"]),
                            (by_cross, f"{cond}|{item['language']}")]:
            bucket[key]["n"] += 1
            bucket[key][d] += 1
        if r["confidence"] >= 3:
            by_cond_hi[cond]["n"] += 1
            by_cond_hi[cond][d] += 1
        if cond != "correct" and d == "reject":
            loc_tot += 1
            flagged = set(json.loads(r["suspicious"]))
            if flagged & set(item["error_fields"]):
                loc_hit += 1
        guard_confusion[(bool(r["guard_shown"]), d in FLAG)] += 1

    def summarise(table: dict[str, dict[str, int]]) -> dict[str, object]:
        out_rows = {}
        for key, c in sorted(table.items()):
            lo, hi = wilson(c["reject"], c["n"])
            alo, ahi = wilson(c["accept"], c["n"])
            out_rows[key] = {
                "n": c["n"], "accept": c["accept"], "reject": c["reject"], "clarify": c["clarify"],
                "accept_rate": round(c["accept"] / c["n"], 3) if c["n"] else None,
                "accept_ci95": [round(alo, 3), round(ahi, 3)],
                "reject_rate": round(c["reject"] / c["n"], 3) if c["n"] else None,
                "reject_ci95": [round(lo, 3), round(hi, 3)],
            }
        return out_rows

    report["by_condition"] = summarise(by_cond)
    report["by_condition_confident"] = summarise(by_cond_hi)
    report["by_language"] = summarise(by_lang)
    report["by_family"] = summarise(by_family)
    report["by_condition_language"] = summarise(by_cross)
    wrong_n = sum(c["n"] for k, c in by_cond.items() if k.startswith("wrong"))
    wrong_accept = sum(c["accept"] for k, c in by_cond.items() if k.startswith("wrong"))
    report["headline"] = {
        "human_false_accept_on_wrong_configs": round(wrong_accept / wrong_n, 3) if wrong_n else None,
        "n_wrong_decisions": wrong_n,
        "human_accept_on_correct_configs": (
            round(by_cond["correct"]["accept"] / by_cond["correct"]["n"], 3) if by_cond["correct"]["n"] else None
        ),
        "n_correct_decisions": by_cond["correct"]["n"],
    }
    report["field_localisation"] = {
        "n_reject_wrong": loc_tot,
        "correct_field_flagged": loc_hit,
        "rate": round(loc_hit / loc_tot, 3) if loc_tot else None,
    }
    report["human_vs_guard"] = {
        f"guard_{'shown' if g else 'hidden'}_{'flag' if f else 'accept'}": v
        for (g, f), v in sorted(guard_confusion.items())
    }

    votes: dict[str, dict[str, int]] = defaultdict(lambda: {"flag": 0, "accept": 0})
    for r in reviews:
        votes[r["item_id"]]["flag" if r["decision"] in FLAG else "accept"] += 1
    report["fleiss_kappa"] = round(fleiss([[v["flag"], v["accept"]] for v in votes.values()]), 3)

    raters: dict[str, dict[str, int]] = defaultdict(dict)
    for r in reviews:
        raters[r["code"]][r["item_id"]] = 1 if r["decision"] in FLAG else 0
    codes = sorted(raters)
    pairs = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            shared = sorted(set(raters[codes[i]]) & set(raters[codes[j]]))
            if len(shared) >= 5:
                pairs.append(cohen([raters[codes[i]][s] for s in shared], [raters[codes[j]][s] for s in shared]))
    report["pairwise_kappa_mean"] = round(statistics.mean(pairs), 3) if pairs else None
    report["pairwise_kappa_n"] = len(pairs)

    (out / "survey_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    with (out / "blind_review.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "item_id", "condition", "language", "family", "decision", "confidence",
                    "flags_correct_field", "guard_shown", "duration_ms", "note"])
        for r in reviews:
            item = ITEMS.get(r["item_id"], {})
            flagged = set(json.loads(r["suspicious"]))
            w.writerow([r["code"], r["item_id"], condition(r["item_id"]), item.get("language", ""),
                        item.get("family", ""), r["decision"], r["confidence"],
                        int(bool(flagged & set(item.get("error_fields", [])))), r["guard_shown"],
                        r["duration_ms"], r["note"].replace("\n", " ")])

    with (out / "paraphrases.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["code", "intent_id", "slot", "reference", "response", "chars"])
        for r in paras:
            meta = INTENTS.get(r["intent_id"], {})
            text = r["response"].replace("\n", " ").strip()
            w.writerow([r["code"], r["intent_id"], meta.get("slot", ""),
                        meta.get("prompt", ""), text, len(text)])

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
