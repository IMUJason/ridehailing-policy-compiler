from __future__ import annotations

from dataclasses import replace

from trc_ridehail.rollout import evaluate_portfolio
from trc_ridehail.simulator import Policy, TravelTime


def single_field_interpretations(policy: Policy, alternatives: dict[str, list], include_base: bool = True) -> list[tuple[str, Policy]]:
    candidates = [("base", policy)] if include_base else []
    for field, values in alternatives.items():
        for value in values:
            if value != getattr(policy, field):
                candidates.append((f"{field}={value}", replace(policy, **{field: value})))
    return candidates


def certify_rollout(
    scenarios: list[dict],
    policies: dict[str, Policy],
    target: str,
    candidates: list[tuple[str, Policy]],
    travel: TravelTime,
    time_limit_s: float = 3,
) -> dict:
    outcomes = []
    for label, candidate in candidates:
        result = evaluate_portfolio(
            scenarios, policies, travel, time_limit_s=time_limit_s,
            overrides={target: candidate},
        )
        if result["status"] != "ok":
            return {"status": "invalid", "candidate": label, "details": result}
        outcomes.append({"candidate": label, "best": result["best"], "ranking": result["ranking"], "summary": result["summary"]})
    baseline_best = outcomes[0]["best"]
    flips = [outcome["candidate"] for outcome in outcomes[1:] if outcome["best"] != baseline_best]
    return {
        "status": "ok",
        "stable": not flips,
        "certified_best": baseline_best,
        "flipping_interpretations": flips,
        "coverage_note": "Stability holds only for the evaluated demand scenarios and supplied single-field interpretations.",
        "outcomes": outcomes,
    }
