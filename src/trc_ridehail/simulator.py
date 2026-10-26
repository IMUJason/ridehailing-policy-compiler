from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data/derived"


@dataclass(frozen=True)
class Policy:
    max_wait_s: int = 900
    max_detour_ratio: float = 0.5
    max_parties: int = 2
    fleet_size: int | None = None
    unserved_penalty_s: int = 7200
    priority_weight: float = 0.0


class TravelTime:
    def __init__(self) -> None:
        frame = pd.read_parquet(DERIVED / "travel_time_od_hour.parquet")
        self.lookup = {
            (int(r.PULocationID), int(r.DOLocationID), int(r.hour_bucket)): int(r.seconds)
            for r in frame.itertuples()
        }
        metadata = json.loads((DERIVED / "travel_time_metadata.json").read_text())
        self.intercept = metadata["fallback_seconds"]
        self.slope = metadata["fallback_seconds_per_km"]
        centroids = pd.read_csv(DERIVED / "zone_centroids.csv").set_index("LocationID")
        self.centroids = centroids[["lon", "lat"]].to_dict("index")

    def seconds(self, origin: int, destination: int, bucket: int) -> int:
        if origin == destination:
            return 0
        key = (origin, destination, bucket)
        if key in self.lookup:
            return self.lookup[key]
        a, b = self.centroids[origin], self.centroids[destination]
        lon1, lat1, lon2, lat2 = map(np.radians, [a["lon"], a["lat"], b["lon"], b["lat"]])
        h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        km = 6371.0 * 2 * np.arcsin(np.sqrt(h))
        return max(60, round(self.intercept + self.slope * km))


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def simulate(scenario: dict, policy: Policy, travel: TravelTime, time_limit_s: float = 5.0) -> dict:
    requests = scenario["requests"]
    n = len(requests)
    fleet = policy.fleet_size or scenario["fleet_size"]
    bucket = scenario["hour_bucket"]
    direct = [travel.seconds(r["pickup_zone"], r["dropoff_zone"], bucket) for r in requests]
    horizon = 4 * 3600

    manager = pywrapcp.RoutingIndexManager(2 * n + 1, fleet, 0)
    routing = pywrapcp.RoutingModel(manager)
    zone = [scenario["origin_zone"]] + [r["pickup_zone"] for r in requests] + [r["dropoff_zone"] for r in requests]

    def transit(from_index: int, to_index: int) -> int:
        return travel.seconds(zone[manager.IndexToNode(from_index)], zone[manager.IndexToNode(to_index)], bucket)

    transit_index = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    def demand(index: int) -> int:
        node = manager.IndexToNode(index)
        if 1 <= node <= n:
            return 1
        if n + 1 <= node <= 2 * n:
            return -1
        return 0

    demand_index = routing.RegisterUnaryTransitCallback(demand)
    routing.AddDimensionWithVehicleCapacity(demand_index, 0, [policy.max_parties] * fleet, True, "Parties")
    parties = routing.GetDimensionOrDie("Parties")

    def time_callback(from_index: int, to_index: int) -> int:
        return transit(from_index, to_index)

    time_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_index, horizon, horizon, True, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for i, request in enumerate(requests, start=1):
        pickup = manager.NodeToIndex(i)
        delivery = manager.NodeToIndex(i + n)
        routing.AddPickupAndDelivery(pickup, delivery)
        routing.solver().Add(routing.VehicleVar(pickup) == routing.VehicleVar(delivery))
        routing.solver().Add(
            time_dimension.CumulVar(delivery) - time_dimension.CumulVar(pickup)
            <= round(direct[i - 1] * (1 + policy.max_detour_ratio))
        )
        routing.AddDisjunction([pickup, delivery], policy.unserved_penalty_s, 2)
        time_dimension.CumulVar(pickup).SetRange(request["request_s"], request["request_s"] + policy.max_wait_s)
        time_dimension.CumulVar(delivery).SetRange(0, horizon)

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.FromSeconds(max(1, round(time_limit_s)))
    assignment = routing.SolveWithParameters(parameters)
    if assignment is None:
        empty = routing.ReadAssignmentFromRoutes([[] for _ in range(fleet)], True)
        if empty is None:
            return {"status": "no_solution", "scenario_id": scenario["scenario_id"], "policy": asdict(policy)}
        assignment = routing.SolveFromAssignmentWithParameters(empty, parameters)
    if assignment is None:
        return {"status": "no_solution", "scenario_id": scenario["scenario_id"], "policy": asdict(policy)}

    served: dict[int, dict] = {}
    vehicle_travel = []
    empty_travel = []
    route_request_counts = []
    for vehicle in range(fleet):
        index = routing.Start(vehicle)
        route_nodes = []
        load = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            if 1 <= node <= n:
                load += 1
            elif n + 1 <= node <= 2 * n:
                load -= 1
            next_index = assignment.Value(routing.NextVar(index))
            leg = travel.seconds(zone[node], zone[manager.IndexToNode(assignment.Value(routing.NextVar(index)))], bucket)
            empty_travel.append(leg if load == 0 else 0)
            index = next_index
        route_nodes.append(manager.IndexToNode(index))
        end_time = assignment.Min(time_dimension.CumulVar(index))
        vehicle_travel.append(end_time)
        route_request_counts.append(sum(1 for node in route_nodes if 1 <= node <= n))
        for node in route_nodes:
            if 1 <= node <= n:
                request_index = node - 1
                delivery_index = node + n
                served[request_index] = {
                    "pickup_time_s": assignment.Min(time_dimension.CumulVar(manager.NodeToIndex(node))),
                    "dropoff_time_s": assignment.Min(time_dimension.CumulVar(manager.NodeToIndex(delivery_index))),
                    "vehicle": vehicle,
                }

    waits, onboard, detours = [], [], []
    details = []
    for i, request in enumerate(requests):
        if i not in served:
            details.append({"id": request["id"], "served": False})
            continue
        result = served[i]
        wait = result["pickup_time_s"] - request["request_s"]
        ride = result["dropoff_time_s"] - result["pickup_time_s"]
        detour = ride - direct[i]
        waits.append(wait)
        onboard.append(ride)
        detours.append(detour)
        details.append({
            "id": request["id"], "served": True, "vehicle": result["vehicle"],
            "wait_s": wait, "onboard_s": ride, "detour_s": detour,
            "pickup_time_s": result["pickup_time_s"], "dropoff_time_s": result["dropoff_time_s"],
        })

    pooled_ids = set()
    intervals = [(i, value["pickup_time_s"], value["dropoff_time_s"], value["vehicle"]) for i, value in served.items()]
    for left in range(len(intervals)):
        for right in range(left + 1, len(intervals)):
            a, b = intervals[left], intervals[right]
            if a[3] == b[3] and max(a[1], b[1]) < min(a[2], b[2]):
                pooled_ids.update([a[0], b[0]])
    pooled = len(pooled_ids)
    metrics = {
        "requests": n,
        "served": len(served),
        "unserved": n - len(served),
        "response_rate": len(served) / n,
        "pooling_rate_served": pooled / len(served) if served else 0.0,
        "wait_mean_s": mean(waits) if waits else None,
        "wait_p90_s": percentile(waits, 90),
        "detour_mean_s": mean(detours) if detours else None,
        "detour_p90_s": percentile(detours, 90),
        "onboard_mean_s": mean(onboard) if onboard else None,
        "vehicle_hours": sum(vehicle_travel) / 3600,
        "empty_travel_hours": sum(empty_travel) / 3600,
        "active_vehicles": sum(count > 0 for count in route_request_counts),
    }
    return {
        "status": "ok",
        "scenario_id": scenario["scenario_id"],
        "split": scenario["split"],
        "day_type": scenario["day_type"],
        "origin_zone": scenario["origin_zone"],
        "policy": asdict(policy),
        "metrics": metrics,
        "requests": details,
    }
