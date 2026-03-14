import os
import glob
import math
import random
import time
from typing import List, Tuple, Set, Optional, Dict

import numpy as np
import vrplib


class Instance:
    """CVRP problem instance."""

    def __init__(self, name: str, capacity: float, n_customers: int,
                 demands: List[float], dist: np.ndarray,
                 coords: Dict[int, Tuple[float, float]]) -> None:
        self.name = name
        self.capacity = capacity
        self.n = n_customers
        self.demands = demands
        self.dist = dist
        self.coords = coords

    def demand(self, customer: int) -> float:
        return self.demands[customer]

    def distance(self, i: int, j: int) -> float:
        return self.dist[i, j]

    @property
    def n_nodes(self) -> int:
        return self.n + 1


class Route:
    """Single vehicle route (list of customers)."""

    def __init__(self, customers: Optional[List[int]] = None) -> None:
        self.customers = customers if customers is not None else []

    def copy(self) -> 'Route':
        return Route(self.customers.copy())

    def cost(self, inst: Instance) -> float:
        if not self.customers:
            return 0.0
        total = inst.distance(0, self.customers[0])
        for i in range(len(self.customers) - 1):
            total += inst.distance(self.customers[i], self.customers[i + 1])
        total += inst.distance(self.customers[-1], 0)
        return total

    def load(self, inst: Instance) -> float:
        return sum(inst.demand(c) for c in self.customers)

    def feasible(self, inst: Instance) -> bool:
        return self.load(inst) <= inst.capacity

    def insert(self, pos: int, customer: int) -> None:
        self.customers.insert(pos, customer)

    def remove(self, customer: int) -> None:
        self.customers.remove(customer)

    def to_list(self) -> List[int]:
        return self.customers.copy()


class Solution:
    """Solution consisting of several routes."""

    def __init__(self, routes: Optional[List[Route]] = None) -> None:
        self.routes = routes if routes is not None else []

    def copy(self) -> 'Solution':
        return Solution([r.copy() for r in self.routes])

    def total_cost(self, inst: Instance) -> float:
        return sum(r.cost(inst) for r in self.routes)

    def add_route(self, route: Route) -> None:
        self.routes.append(route)

    def remove_route(self, idx: int) -> Route:
        return self.routes.pop(idx)

    def get_routes(self) -> List[Route]:
        return [r.copy() for r in self.routes]

    def feasible(self, inst: Instance) -> bool:
        return all(r.feasible(inst) for r in self.routes)

    def all_customers(self) -> Set[int]:
        cust = set()
        for r in self.routes:
            cust.update(r.customers)
        return cust


class VRPParser:
    """Parser for CVRPLIB files (EUC_2D and EXPLICIT)."""

    @staticmethod
    def parse_vrp(path: str) -> Instance:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        name = ""
        capacity = 0.0
        dim = 0
        edge_weight_type = ""
        edge_weight_format = ""

        coords: Dict[int, Tuple[float, float]] = {}
        demands_dict: Dict[int, float] = {}
        depot_id = 1
        edge_weights: List[float] = []

        section: Optional[str] = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if ":" in line and section is None:
                key, value = line.split(":", 1)
                key = key.strip().upper()
                value = value.strip()

                if key == "NAME":
                    name = value
                elif key == "DIMENSION":
                    dim = int(value)
                elif key == "CAPACITY":
                    capacity = float(value)
                elif key == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = value.upper()
                elif key == "EDGE_WEIGHT_FORMAT":
                    edge_weight_format = value.upper()
                continue

            if line.upper().startswith("NODE_COORD_SECTION"):
                section = "NODE_COORD"
                continue
            if line.upper().startswith("DEMAND_SECTION"):
                section = "DEMAND"
                continue
            if line.upper().startswith("DEPOT_SECTION"):
                section = "DEPOT"
                continue
            if line.upper().startswith("EDGE_WEIGHT_SECTION"):
                section = "EDGE_WEIGHT"
                continue
            if line.upper().startswith("EOF"):
                break
            if line.upper().startswith("DISPLAY_DATA_SECTION"):
                section = "SKIP"
                continue

            if section == "NODE_COORD":
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        nid = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        coords[nid] = (x, y)
                    except ValueError:
                        pass

            elif section == "DEMAND":
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        nid = int(parts[0])
                        dmd = float(parts[1])
                        demands_dict[nid] = dmd
                    except ValueError:
                        pass

            elif section == "DEPOT":
                if line != "-1":
                    try:
                        depot_id = int(line)
                    except ValueError:
                        pass

            elif section == "EDGE_WEIGHT":
                parts = line.split()
                for p in parts:
                    try:
                        edge_weights.append(float(p))
                    except ValueError:
                        pass

        if dim == 0:
            if coords:
                dim = len(coords)
            else:
                nw = len(edge_weights)
                dim = int((1 + math.sqrt(1 + 8 * nw)) / 2)

        n_cust = dim - 1

        if coords:
            dist = VRPParser._build_dist_from_coords(coords, depot_id, dim)
        elif edge_weights:
            dist = VRPParser._build_dist_from_weights(edge_weights, edge_weight_format, dim)
        else:
            raise ValueError(f"Cannot recognise file format: {path}")

        demands = VRPParser._build_demands_list(demands_dict, depot_id, dim)

        if not name:
            name = os.path.basename(path).split('.')[0]

        return Instance(name, capacity, n_cust, demands, dist, coords)

    @staticmethod
    def _build_dist_from_coords(coords: Dict[int, Tuple[float, float]],
                                depot: int, dim: int) -> np.ndarray:
        all_ids = sorted(coords.keys())
        id_to_idx = {depot: 0}
        nxt = 1
        for orig in all_ids:
            if orig != depot:
                id_to_idx[orig] = nxt
                nxt += 1

        mat = np.zeros((dim, dim))
        for i_id in all_ids:
            for j_id in all_ids:
                if i_id != j_id:
                    i = id_to_idx[i_id]
                    j = id_to_idx[j_id]
                    dx = coords[i_id][0] - coords[j_id][0]
                    dy = coords[i_id][1] - coords[j_id][1]
                    mat[i, j] = int(round(math.hypot(dx, dy)))
        return mat

    @staticmethod
    def _build_dist_from_weights(weights: List[float], fmt: str, dim: int) -> np.ndarray:
        mat = np.zeros((dim, dim))
        fmt = fmt.upper()

        if fmt in ("LOWER_ROW", "LOWER_DIAG_ROW", ""):
            idx = 0
            for i in range(1, dim):
                for j in range(i):
                    if idx < len(weights):
                        val = weights[idx]
                        mat[i, j] = val
                        mat[j, i] = val
                        idx += 1

        elif fmt == "UPPER_ROW":
            idx = 0
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    if idx < len(weights):
                        val = weights[idx]
                        mat[i, j] = val
                        mat[j, i] = val
                        idx += 1

        elif fmt == "FULL_MATRIX":
            idx = 0
            for i in range(dim):
                for j in range(dim):
                    if idx < len(weights):
                        mat[i, j] = weights[idx]
                        idx += 1

        else:
            exp_low = dim * (dim - 1) // 2
            if len(weights) == exp_low:
                idx = 0
                for i in range(1, dim):
                    for j in range(i):
                        if idx < len(weights):
                            val = weights[idx]
                            mat[i, j] = val
                            mat[j, i] = val
                            idx += 1
            elif len(weights) == dim * dim:
                idx = 0
                for i in range(dim):
                    for j in range(dim):
                        if idx < len(weights):
                            mat[i, j] = weights[idx]
                            idx += 1

        return mat

    @staticmethod
    def _build_demands_list(dem_dict: Dict[int, float], depot: int, dim: int) -> List[float]:
        all_ids = sorted(dem_dict.keys())
        id_to_idx = {depot: 0}
        nxt = 1
        for orig in all_ids:
            if orig != depot:
                id_to_idx[orig] = nxt
                nxt += 1

        demands = [0.0] * dim
        for orig, d in dem_dict.items():
            if orig in id_to_idx:
                demands[id_to_idx[orig]] = d
        return demands

    @staticmethod
    def parse_sol(path: str) -> Optional[float]:
        return vrplib.read_solution(path)


class DestroyOperator:
    """Base class for destroy operators."""

    def destroy(self, sol: Solution, k: int) -> Tuple[Solution, Set[int]]:
        raise NotImplementedError


class RandomDestroy(DestroyOperator):
    """Random removal of customers."""

    def destroy(self, sol: Solution, k: int) -> Tuple[Solution, Set[int]]:
        routes = [r.copy() for r in sol.routes]
        all_cust = [c for r in routes for c in r.customers]
        if not all_cust or k <= 0:
            return sol.copy(), set()

        k = min(k, len(all_cust))
        removed = set(random.sample(all_cust, k))

        new_routes = []
        for r in routes:
            new_cust = [c for c in r.customers if c not in removed]
            if new_cust:
                new_routes.append(Route(new_cust))

        return Solution(new_routes), removed


class RepairOperator:
    """Base class for repair operators."""

    def repair(self, partial: Solution, removed: Set[int], inst: Instance) -> Solution:
        raise NotImplementedError


class GreedyRepair(RepairOperator):
    """Greedy insertion: best position for each removed customer."""

    def repair(self, partial: Solution, removed: Set[int], inst: Instance) -> Solution:
        routes = [r.copy() for r in partial.routes]

        for cust in removed:
            best_r = -1
            best_pos = -1
            best_inc = float('inf')

            for r_idx, r in enumerate(routes):
                if r.load(inst) + inst.demand(cust) <= inst.capacity:
                    for pos in range(len(r.customers) + 1):
                        prev = r.customers[pos - 1] if pos > 0 else 0
                        nxt = r.customers[pos] if pos < len(r.customers) else 0
                        inc = (inst.distance(prev, cust) +
                               inst.distance(cust, nxt) -
                               inst.distance(prev, nxt))
                        if inc < best_inc:
                            best_inc = inc
                            best_r = r_idx
                            best_pos = pos

            if best_r != -1:
                routes[best_r].insert(best_pos, cust)
            else:
                routes.append(Route([cust]))

        return Solution(routes)


class LocalSearch:
    """2‑opt local search for a single route."""

    @staticmethod
    def optimize(route: Route, inst: Instance) -> Route:
        if len(route.customers) < 2:
            return route.copy()

        improved = True
        best = route.copy()
        best_cost = best.cost(inst)

        while improved:
            improved = False
            for i in range(len(best.customers) - 1):
                for j in range(i + 1, len(best.customers)):
                    new_cust = (best.customers[:i] +
                                best.customers[i:j + 1][::-1] +
                                best.customers[j + 1:])
                    new_route = Route(new_cust)
                    new_cost = new_route.cost(inst)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = new_route
                        improved = True
        return best


class ALNSSolver:
    """Adaptive Large Neighbourhood Search for CVRP."""

    def __init__(self,
                 max_iter: int = 2500,
                 start_temp: float = 100.0,
                 cooling: float = 0.995,
                 destroy: Optional[DestroyOperator] = None,
                 repair: Optional[RepairOperator] = None) -> None:
        self.max_iter = max_iter
        self.start_temp = start_temp
        self.cooling = cooling
        self.destroy = destroy if destroy else RandomDestroy()
        self.repair = repair if repair else GreedyRepair()
        self.ls = LocalSearch()

    def _greedy_initial(self, inst: Instance) -> Solution:
        unvisited = set(range(1, inst.n_nodes))
        routes = []

        while unvisited:
            cur_route, cur_load, cur_node = [], 0.0, 0
            while unvisited:
                best_node = None
                best_dist = float('inf')
                for node in unvisited:
                    if cur_load + inst.demand(node) <= inst.capacity:
                        d = inst.distance(cur_node, node)
                        if d < best_dist:
                            best_dist = d
                            best_node = node
                if best_node is None:
                    break
                cur_route.append(best_node)
                cur_load += inst.demand(best_node)
                unvisited.remove(best_node)
                cur_node = best_node
            if cur_route:
                routes.append(Route(cur_route))

        return Solution(routes)

    def solve(self, inst: Instance) -> Tuple[Solution, float, float]:
        current = self._greedy_initial(inst)
        current.routes = [self.ls.optimize(r, inst) for r in current.routes]
        cur_cost = current.total_cost(inst)

        best = current.copy()
        best_cost = cur_cost

        T = self.start_temp
        k = max(2, int(inst.n * 0.15))

        for _ in range(self.max_iter):
            partial, removed = self.destroy.destroy(current, k)
            new_sol = self.repair.repair(partial, removed, inst)
            new_sol.routes = [self.ls.optimize(r, inst) for r in new_sol.routes]
            new_cost = new_sol.total_cost(inst)

            delta = new_cost - cur_cost
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current = new_sol
                cur_cost = new_cost
                if new_cost < best_cost:
                    best = new_sol.copy()
                    best_cost = new_cost

            T *= self.cooling

        return best, best_cost