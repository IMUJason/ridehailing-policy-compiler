from __future__ import annotations

from trc_ridehail.simulator import Policy, TravelTime


def validate_result(scenario: dict, policy: Policy, travel: TravelTime, result: dict) -> list[str]:
    if result.get("status") != "ok":
        return [f"status:{result.get('status')}"]
    errors = []
    requests = {request["id"]: request for request in scenario["requests"]}
    seen = set()
    intervals = []
    for detail in result["requests"]:
        request_id = detail["id"]
        if request_id in seen:
            errors.append(f"duplicate:{request_id}")
        seen.add(request_id)
        if not detail["served"]:
            continue
        request = requests[request_id]
        direct = travel.seconds(request["pickup_zone"], request["dropoff_zone"], scenario["hour_bucket"])
        if detail["wait_s"] < -1e-6 or detail["wait_s"] > policy.max_wait_s + 1e-6:
            errors.append(f"wait:{request_id}")
        if detail["onboard_s"] < 0:
            errors.append(f"precedence:{request_id}")
        if detail["detour_s"] > direct * policy.max_detour_ratio + 1:
            errors.append(f"detour:{request_id}")
        intervals.append((detail["vehicle"], detail["pickup_time_s"], detail["dropoff_time_s"], request_id))
    if seen != set(requests):
        errors.append("request_coverage")
    vehicles = sorted({interval[0] for interval in intervals})
    for vehicle in vehicles:
        events = []
        for other_vehicle, start, end, _ in intervals:
            if other_vehicle == vehicle:
                events.append((start, 1))
                events.append((end, -1))
        events.sort(key=lambda event: (event[0], event[1]))
        occupancy = 0
        peak = 0
        for _, change in events:
            occupancy += change
            peak = max(peak, occupancy)
        if peak > policy.max_parties:
            errors.append(f"capacity:vehicle{vehicle}")
    metrics = result["metrics"]
    served = sum(bool(detail["served"]) for detail in result["requests"])
    if metrics["served"] != served or metrics["unserved"] != len(requests) - served:
        errors.append("metric_count")
    return errors
