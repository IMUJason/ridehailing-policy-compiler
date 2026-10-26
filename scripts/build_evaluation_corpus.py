from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATES = [
    ("wait-minutes", "en", "Do not keep passengers waiting more than {wait_min} minutes for pickup.", lambda v: {"max_wait_s": v["wait_min"] * 60}),
    ("wait-seconds", "en", "Set the maximum pickup wait to {wait_s} seconds.", lambda v: {"max_wait_s": v["wait_s"]}),
    ("wait-paraphrase", "en", "Passengers should not wait longer than {wait_min} minutes before being picked up.", lambda v: {"max_wait_s": v["wait_min"] * 60}),
    ("detour-percent", "en", "Additional onboard time may not exceed {detour_percent} percent of the direct ride.", lambda v: {"max_detour_ratio": v["detour_percent"] / 100}),
    ("detour-fraction", "en", "Cap the onboard detour at {detour_fraction} of direct travel time.", lambda v: {"max_detour_ratio": v["detour_fraction"]}),
    ("detour-delay", "en", "A shared passenger's extra time in the vehicle is limited to {detour_percent} percent of a direct trip.", lambda v: {"max_detour_ratio": v["detour_percent"] / 100}),
    ("parties-groups", "en", "No vehicle may carry more than {parties} passenger groups at once.", lambda v: {"max_parties": v["parties"]}),
    ("parties-per-vehicle", "en", "Per vehicle, allow at most {parties} separate passenger parties.", lambda v: {"max_parties": v["parties"]}),
    ("fleet-deploy", "en", "Deploy {fleet} vehicles for this operating window.", lambda v: {"fleet_size": v["fleet"]}),
    ("fleet-service", "en", "Use {fleet} vehicles into service during the peak.", lambda v: {"fleet_size": v["fleet"]}),
    ("penalty-hours", "en", "Charge an unserved shared request as {penalty_hours} hours of passenger delay.", lambda v: {"unserved_penalty_s": v["penalty_hours"] * 3600}),
    ("penalty-minutes", "en", "Treat each unmet shared request as a penalty of {penalty_min} minutes.", lambda v: {"unserved_penalty_s": v["penalty_min"] * 60}),
    ("combo-wait-detour", "en", "Cap pickup wait at {wait_min} minutes and additional onboard time at {detour_percent} percent.", lambda v: {"max_wait_s": v["wait_min"] * 60, "max_detour_ratio": v["detour_percent"] / 100}),
    ("combo-wait-parties", "en", "Limit pickup wait to {wait_min} minutes and vehicles to {parties} parties.", lambda v: {"max_wait_s": v["wait_min"] * 60, "max_parties": v["parties"]}),
    ("combo-fleet-wait-detour", "en", "Put {fleet} vehicles into service, cap pickup wait at {wait_min} minutes, and keep detours below {detour_percent} percent.", lambda v: {"fleet_size": v["fleet"], "max_wait_s": v["wait_min"] * 60, "max_detour_ratio": v["detour_percent"] / 100}),
    ("combo-detour-parties", "en", "Keep each vehicle to {parties} parties and limit detour to {detour_percent} percent of the direct ride.", lambda v: {"max_parties": v["parties"], "max_detour_ratio": v["detour_percent"] / 100}),
    ("zh-wait", "zh", "接驾等待时间不得超过{wait_min}分钟。", lambda v: {"max_wait_s": v["wait_min"] * 60}),
    ("zh-detour", "zh", "拼车额外乘坐时间不得超过直达时间的{detour_percent}%。", lambda v: {"max_detour_ratio": v["detour_percent"] / 100}),
    ("zh-parties", "zh", "每辆车同时最多服务{parties}组乘客。", lambda v: {"max_parties": v["parties"]}),
    ("zh-fleet-wait", "zh", "本时段投入{fleet}辆车，接驾等待上限为{wait_min}分钟。", lambda v: {"fleet_size": v["fleet"], "max_wait_s": v["wait_min"] * 60}),
    ("no-change-en", "en", "Keep the current pooling policy unchanged.", lambda v: {}),
    ("no-change-zh", "zh", "保持当前拼车策略不变。", lambda v: {}),
    ("distractor-fleet-wait", "en", "Fleet utilization should be monitored, but the only policy change is a {wait_min}-minute pickup wait cap.", lambda v: {"max_wait_s": v["wait_min"] * 60}),
    ("distractor-idle-detour", "en", "Vehicles should avoid unnecessary idling; cap additional onboard time to {detour_percent} percent of a direct ride.", lambda v: {"max_detour_ratio": v["detour_percent"] / 100}),
]

VALUE_SETS = [
    {"wait_min": 5, "wait_s": 300, "detour_percent": 20, "detour_fraction": 0.2, "parties": 2, "fleet": 6, "penalty_hours": 1, "penalty_min": 60},
    {"wait_min": 8, "wait_s": 480, "detour_percent": 25, "detour_fraction": 0.25, "parties": 3, "fleet": 8, "penalty_hours": 2, "penalty_min": 90},
    {"wait_min": 10, "wait_s": 600, "detour_percent": 30, "detour_fraction": 0.3, "parties": 2, "fleet": 10, "penalty_hours": 1, "penalty_min": 45},
    {"wait_min": 12, "wait_s": 720, "detour_percent": 40, "detour_fraction": 0.4, "parties": 3, "fleet": 12, "penalty_hours": 2, "penalty_min": 75},
    {"wait_min": 15, "wait_s": 900, "detour_percent": 50, "detour_fraction": 0.5, "parties": 4, "fleet": 15, "penalty_hours": 3, "penalty_min": 120},
]


def main() -> None:
    records = []
    for template_index, (family, language, template, gold_fn) in enumerate(TEMPLATES, start=1):
        for value_index, values in enumerate(VALUE_SETS, start=1):
            text = template.format(**values)
            gold = gold_fn(values)
            records.append({
                "policy_id": f"eval-{template_index:03d}-{value_index}",
                "family": family,
                "language": language,
                "text": text,
                "gold": gold,
            })
    output = ROOT / "data/policies/evaluation_corpus.jsonl"
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"records": len(records), "templates": len(TEMPLATES), "output": str(output)}))


if __name__ == "__main__":
    main()
