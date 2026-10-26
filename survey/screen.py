from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ITEMS = {item["item_id"]: item for item in json.loads((HERE / "data" / "items.json").read_text())}

MIN_REVIEWS = 30
MIN_PARAPHRASES = 6
FAST_MEDIAN_S = 4.0
FAST_SHARE_S = 3.0
FAST_SHARE_FRAC = 0.5
STRAIGHT_MEDIAN_S = 8.0
PERFECT_MEDIAN_S = 6.0
ZERO_MAX = 5


def gold(item_id: str) -> str:
    return "accept" if ITEMS[item_id]["correct"] else "reject"


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
    conn.close()

    by_code: dict[str, list[dict]] = {}
    for r in reviews:
        by_code.setdefault(r["code"], []).append(r)
    para_count = Counter(p["code"] for p in paras)

    rows, excluded = [], []
    for p in sorted(parts, key=lambda x: x["name"]):
        code, name = p["code"], p["name"]
        rs = by_code.get(code, [])
        n = len(rs)
        times = [r["duration_ms"] / 1000 for r in rs]
        median_s = round(statistics.median(times), 1) if times else None
        share_fast = round(sum(1 for t in times if t < FAST_SHARE_S) / n, 2) if n else None
        decisions = Counter(r["decision"] for r in rs)
        agreements = sum(1 for r in rs if r["decision"] == gold(r["item_id"])) if n else 0
        zero = sum(1 for r in rs if r["duration_ms"] <= 0)
        inconsistent = sum(
            1 for r in rs
            if (r["decision"] == "accept" and json.loads(r["suspicious"]))
            or (r["decision"] == "reject" and not json.loads(r["suspicious"]))
        )

        reason = None
        if n < MIN_REVIEWS or para_count[code] < MIN_PARAPHRASES:
            reason = "E1_incomplete"
        elif median_s is not None and (median_s < FAST_MEDIAN_S or share_fast >= FAST_SHARE_FRAC):
            reason = "E2_mechanical"
        elif n and len(decisions) == 1 and median_s is not None and median_s < STRAIGHT_MEDIAN_S:
            reason = "E3_straight_line"
        elif n == MIN_REVIEWS and agreements == MIN_REVIEWS and median_s is not None and median_s < PERFECT_MEDIAN_S:
            reason = "E4_perfect_fast"
        elif zero > ZERO_MAX:
            reason = "E5_timer"

        row = {
            "name": name, "code": code, "reviews": n, "paraphrases": para_count[code],
            "median_s": median_s, "share_fast": share_fast,
            "gold_agreement": round(agreements / n, 3) if n else None,
            "accept": decisions.get("accept", 0), "reject": decisions.get("reject", 0),
            "clarify": decisions.get("clarify", 0), "zero_time": zero,
            "contradictions": inconsistent,
            "completed": bool(p["completed_at"]),
            "verdict": "excluded" if reason else "included",
            "reason": reason or "",
        }
        rows.append(row)
        if reason:
            excluded.append(row)

    cols = ["name", "reviews", "paraphrases", "median_s", "share_fast", "gold_agreement",
            "accept", "reject", "clarify", "zero_time", "contradictions", "completed", "verdict", "reason"]
    with (out / "quality_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})

    clean_db = out / "survey_clean.db"
    shutil.copyfile(args.db, clean_db)
    cc = sqlite3.connect(clean_db)
    for code in [r["code"] for r in excluded]:
        cc.execute("DELETE FROM reviews WHERE code=?", (code,))
        cc.execute("DELETE FROM paraphrases WHERE code=?", (code,))
        cc.execute("DELETE FROM participants WHERE code=?", (code,))
    cc.commit()
    cc.close()

    summary = {
        "participants_total": len(parts),
        "completed_total": sum(1 for p in parts if p["completed_at"]),
        "included": len(rows) - len(excluded),
        "excluded": len(excluded),
        "excluded_detail": [{"name": r["name"], "reason": r["reason"]} for r in excluded],
        "reviews_total": len(reviews),
        "reviews_included": sum(r["reviews"] for r in rows if r["verdict"] == "included"),
        "median_time_range_included": [
            min(r["median_s"] for r in rows if r["verdict"] == "included"),
            max(r["median_s"] for r in rows if r["verdict"] == "included"),
        ] if any(r["verdict"] == "included" for r in rows) else None,
    }
    (out / "quality_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for r in rows:
        print(f"{r['name']:<8} n={r['reviews']:>2} med={r['median_s']} fast={r['share_fast']} "
              f"agree={r['gold_agreement']} {r['verdict']} {r['reason']}")


if __name__ == "__main__":
    main()
