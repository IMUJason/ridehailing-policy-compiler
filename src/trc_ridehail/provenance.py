from __future__ import annotations

import re

from trc_ridehail.policy_io import UNIT_FIELDS, source_unit_fields

NUMBER_WORD = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?|" + "|".join(NUMBER_WORD), flags=re.IGNORECASE)


def source_numbers(text: str) -> list[float]:
    values = []
    for match in NUMBER_PATTERN.findall(text):
        values.append(float(match) if match.replace(".", "").isdigit() else float(NUMBER_WORD[match.lower()]))
    return values


def canonical_candidates(text: str, canonical_field: str) -> list[float]:
    unit_fields = source_unit_fields(text)
    multiplier = 1.0
    if canonical_field == "max_wait_s" and "max_wait_minutes" in unit_fields:
        multiplier = 60.0
    elif canonical_field == "max_detour_ratio" and "max_detour_percent" in unit_fields:
        multiplier = 0.01
    elif canonical_field == "unserved_penalty_s":
        if "unserved_penalty_hours" in unit_fields:
            multiplier = 3600.0
        elif "unserved_penalty_minutes" in unit_fields:
            multiplier = 60.0
    values = [round(value * multiplier, 10) for value in source_numbers(text)]
    if canonical_field in {"max_wait_s", "max_parties", "fleet_size", "unserved_penalty_s"}:
        values = [int(value) for value in values if int(value) == value]
    return sorted(set(values))


def provenance_errors(text: str, predicted: dict) -> dict:
    rejected = {}
    for field, value in predicted.items():
        candidates = canonical_candidates(text, field)
        if candidates and not any(abs(value - candidate) < 1e-9 for candidate in candidates):
            rejected[field] = {"predicted": value, "source_candidates": candidates}
    return {"accepted": not rejected, "rejected_fields": rejected}


def ambiguous_fields(text: str, predicted: dict) -> dict[str, list[float]]:
    result = {}
    for field in predicted:
        candidates = [value for value in canonical_candidates(text, field) if value != predicted[field]]
        if candidates:
            result[field] = candidates
    return result
