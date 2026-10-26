from __future__ import annotations

import json
import re
from dataclasses import asdict
from itertools import combinations

from trc_ridehail.simulator import Policy

FIELDS = {"max_wait_s", "max_detour_ratio", "max_parties", "fleet_size", "unserved_penalty_s"}
BOUNDS = {
    "max_wait_s": (0, 7200),
    "max_detour_ratio": (0.0, 3.0),
    "max_parties": (1, 4),
    "fleet_size": (1, 50),
    "unserved_penalty_s": (0, 100000),
}
DESCRIPTIONS = {
    "max_wait_s": "Maximum seconds between a request and its pickup.",
    "max_detour_ratio": "Maximum additional onboard time as a fraction of direct ride time; 0.5 means 50 percent.",
    "max_parties": "Maximum simultaneous passenger parties per vehicle.",
    "fleet_size": "Number of vehicles available in the operating window.",
    "unserved_penalty_s": "Penalty in seconds charged for each unserved shared request.",
}
UNIT_FIELDS = {
    "max_wait_minutes": ("max_wait_s", 60, "Maximum pickup wait in minutes exactly as stated in the policy."),
    "max_wait_seconds": ("max_wait_s", 1, "Maximum pickup wait in seconds exactly as stated in the policy."),
    "max_detour_percent": ("max_detour_ratio", 0.01, "Maximum additional onboard time in percent exactly as stated in the policy."),
    "unserved_penalty_hours": ("unserved_penalty_s", 3600, "Unserved-request penalty in hours exactly as stated in the policy."),
    "unserved_penalty_minutes": ("unserved_penalty_s", 60, "Unserved-request penalty in minutes exactly as stated in the policy."),
}
OUTPUT_FIELDS = FIELDS | set(UNIT_FIELDS)


KEYWORDS = {
    "max_wait_s": ("wait", "waiting", "pickup", "等待", "接驾"),
    "max_detour_ratio": ("detour", "additional onboard", "onboard delay", "extra time", "direct ride", "绕行", "额外", "直达"),
    "max_parties": ("parties", "passenger parties", "groups", "乘客组", "组乘客", "两组", "每组乘客"),
}
FLEET_PATTERNS = (
    r"\b(?:use|put|deploy|set)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+vehicles?\b",
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+vehicles?\s+into\s+service\b",
    r"\bfleet\s+(?:size|level)\s+(?:is|to|=)?\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    r"投入\s*[0-9一二三四五六七八九十]+\s*辆车",
    r"\d+\s*辆车",
)
PENALTY_KEYWORDS = ("unserved", "unmet", "penalty", "equivalent", "未服务", "罚")
NO_CHANGE_PATTERNS = (r"keep\b.{0,40}\bunchanged", r"do\s+not\s+change", r"maintain\s+the\s+current", r"维持", r"保持.{0,20}不变")


def requests_no_change(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in NO_CHANGE_PATTERNS)


def active_fields(text: str) -> set[str]:
    lowered = text.lower()
    fields = {field for field, terms in KEYWORDS.items() if any(term in lowered for term in terms)}
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in FLEET_PATTERNS):
        fields.add("fleet_size")
    if any(term in lowered for term in PENALTY_KEYWORDS):
        fields.add("unserved_penalty_s")
    return fields


def source_unit_fields(text: str) -> set[str]:
    lowered = text.lower()
    fields = set()
    for canonical in active_fields(text):
        if canonical == "max_wait_s":
            if "minute" in lowered or "分钟" in text:
                fields.add("max_wait_minutes")
            elif "second" in lowered or "秒" in text:
                fields.add("max_wait_seconds")
            else:
                fields.add(canonical)
        elif canonical == "max_detour_ratio":
            if "percent" in lowered or "%" in text or "percentage" in lowered:
                fields.add("max_detour_percent")
            else:
                fields.add(canonical)
        elif canonical == "unserved_penalty_s":
            if "hour" in lowered or "小时" in text:
                fields.add("unserved_penalty_hours")
            elif "minute" in lowered or "分钟" in text:
                fields.add("unserved_penalty_minutes")
            else:
                fields.add(canonical)
        else:
            fields.add(canonical)
    return fields


UNIT_BOUNDS = {
    "max_wait_minutes": (0, 120),
    "max_wait_seconds": (0, 7200),
    "max_detour_percent": (0, 300),
    "unserved_penalty_hours": (0, 30),
    "unserved_penalty_minutes": (0, 1800),
}


def field_property(field: str) -> dict:
    if field in UNIT_FIELDS:
        lower, upper = UNIT_BOUNDS[field]
        return {"type": "number", "minimum": lower, "maximum": upper, "description": UNIT_FIELDS[field][2]}
    lower, upper = BOUNDS[field]
    return {
        "type": "integer" if field in {"max_wait_s", "max_parties", "fleet_size", "unserved_penalty_s"} else "number",
        "minimum": lower,
        "maximum": upper,
        "description": DESCRIPTIONS[field],
    }


def subset_schema(fields: set[str]) -> dict:
    ordered = sorted(fields)
    branches = []
    for size in range(1, len(ordered) + 1):
        for combination in combinations(ordered, size):
            branches.append({
                "type": "object",
                "additionalProperties": False,
                "required": list(combination),
                "properties": {field: field_property(field) for field in combination},
            })
    branches.append({"type": "object", "additionalProperties": False, "properties": {}})
    return {"oneOf": branches}


def build_policy_schema(mode: str, scenario: dict, text: str | None = None) -> dict:
    if mode == "subset":
        fields = active_fields(text) if text else FIELDS
        return {
            "type": "json_schema",
            "json_schema": {"name": "ridehail_policy_subset", "strict": True, "schema": subset_schema(fields)},
        }
    if mode in {"activated", "unit"} and text and requests_no_change(text):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "ridehail_policy_no_change",
                "strict": True,
                "schema": {"type": "object", "additionalProperties": False, "properties": {}},
            },
        }
    fields = source_unit_fields(text) if mode == "unit" and text else active_fields(text) if mode == "activated" and text else FIELDS
    if mode in {"activated", "unit"} and not fields:
        fields = FIELDS
    properties = {}
    for field in sorted(fields):
        if mode == "static":
            properties[field] = {"type": "number"}
        else:
            properties[field] = field_property(field)
    schema = {"type": "object", "additionalProperties": False, "properties": properties}
    if mode in {"activated", "unit"}:
        schema["required"] = sorted(fields)
    elif mode == "instance":
        schema["required"] = ["context"]
        schema["properties"]["context"] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["origin_zone", "hour_bucket", "request_count"],
            "properties": {
                "origin_zone": {"const": scenario["origin_zone"]},
                "hour_bucket": {"const": scenario["hour_bucket"]},
                "request_count": {"const": len(scenario["requests"])},
            },
        }
    return {"type": "json_schema", "json_schema": {"name": f"ridehail_policy_{mode}", "strict": True, "schema": schema}}


def parse_policy(content: str, scenario: dict) -> tuple[Policy, dict]:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("policy output must be a JSON object")
    unknown = set(payload) - OUTPUT_FIELDS - {"context"}
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")
    if "context" in payload:
        expected = {"origin_zone": scenario["origin_zone"], "hour_bucket": scenario["hour_bucket"], "request_count": len(scenario["requests"])}
        if payload["context"] != expected:
            raise ValueError("instance context mismatch")
    values = {}
    for field, raw_value in payload.items():
        if field == "context":
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"non-numeric value for {field}")
        if field in UNIT_FIELDS:
            canonical, multiplier, _ = UNIT_FIELDS[field]
            lower, upper = UNIT_BOUNDS[field]
            if not lower <= raw_value <= upper:
                raise ValueError(f"out-of-domain value for {field}")
            value = round(float(raw_value) * multiplier, 10)
            target = canonical
        else:
            lower, upper = BOUNDS[field]
            if not lower <= raw_value <= upper:
                raise ValueError(f"out-of-domain value for {field}")
            value = raw_value
            target = field
        if target in {"max_wait_s", "max_parties", "fleet_size", "unserved_penalty_s"} and int(value) != value:
            raise ValueError(f"non-integer canonical value for {target}")
        if target in values and values[target] != value:
            raise ValueError(f"conflicting values for {target}")
        values[target] = int(value) if target in {"max_wait_s", "max_parties", "fleet_size", "unserved_penalty_s"} else float(value)
    return Policy(**values), values


def policy_errors(gold: dict, predicted: dict) -> dict:
    default = asdict(Policy())
    effective_gold = {**default, **gold}
    effective_predicted = {**default, **predicted}
    wrong = {field: {"gold": effective_gold[field], "predicted": effective_predicted[field]} for field in default if effective_gold[field] != effective_predicted[field]}
    return {"exact": not wrong, "wrong_fields": wrong}
