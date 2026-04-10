"""
Route optimization: greedy insertion, simulated annealing, route helpers.
"""
import numpy as np
import math
from .pathfinding import rcost, rcost_fatigue, rpts


def greedy_route(cm, pts, bud_eff):
    """Greedy insertion heuristic: add best ratio node iteratively."""
    n = len(pts)
    route = []; visited = {0}
    while True:
        best_s = -1; best_n = -1
        for j in range(1, n):
            if j in visited:
                continue
            trial = route + [j]
            tc = rcost(cm, trial)
            if tc > bud_eff:
                continue
            ac = max(tc - rcost(cm, route), 0.1)
            s = pts[j] / ac
            if s > best_s:
                best_s = s; best_n = j
        if best_n < 0:
            break
        route.append(best_n); visited.add(best_n)
    return route


def optimize_route(cm, pts, bud_eff, bud_raw, fatigue_rate):
    """Greedy + 2-opt improvement."""
    n = len(pts)
    route = greedy_route(cm, pts, bud_eff)
    print(f"    Greedy: {len(route)}ctrl {rpts(route, pts):.0f}pts "
          f"base={rcost(cm, route):.0f}")

    # 2-opt improvement
    imp = True; it = 0
    while imp and it < 50:
        imp = False; it += 1
        for i in range(len(route)):
            for j in range(i + 2, len(route)):
                nr = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                if rcost(cm, nr) < rcost(cm, route):
                    route = nr; imp = True

    # Try inserting unvisited nodes
    visited = set(route) | {0}
    for j in range(1, n):
        if j in visited:
            continue
        best_pos = -1; best_cost = rcost(cm, route)
        for pos in range(len(route) + 1):
            trial = route[:pos] + [j] + route[pos:]
            tc = rcost(cm, trial)
            if tc < bud_eff and tc < best_cost + pts[j] * 0.5:
                best_cost = tc; best_pos = pos
        if best_pos >= 0:
            route = route[:best_pos] + [j] + route[best_pos:]
            visited.add(j)

    print(f"    Insert: {len(route)}ctrl {rpts(route, pts):.0f}pts")

    # Final SA-style improvement
    import random
    rng = random.Random(42)
    best_route = route[:]
    best_score = rpts(route, pts)
    for _ in range(5000):
        if len(route) < 2:
            break
        mv = rng.random()
        new_route = route[:]
        if mv < 0.3 and new_route:
            idx = rng.randint(0, len(new_route) - 1)
            new_route.pop(idx)
        elif mv < 0.6:
            unv = [j for j in range(1, n) if j not in set(new_route) | {0}]
            if not unv:
                continue
            j = rng.choice(unv)
            pos = rng.randint(0, len(new_route))
            new_route.insert(pos, j)
        elif mv < 0.8 and len(new_route) >= 2:
            i, j = sorted(rng.sample(range(len(new_route)), 2))
            new_route[i:j + 1] = new_route[i:j + 1][::-1]
        else:
            unv = [j for j in range(1, n) if j not in set(new_route) | {0}]
            if not unv or not new_route:
                continue
            idx = rng.randint(0, len(new_route) - 1)
            old = new_route[idx]
            new_route[idx] = rng.choice(unv)

        nc = rcost(cm, new_route)
        if nc > bud_eff:
            continue
        if rcost_fatigue(cm, new_route, bud_raw, fatigue_rate) > bud_raw:
            continue
        ns = rpts(new_route, pts)
        if ns > best_score or (ns == best_score and nc < rcost(cm, best_route)):
            best_route = new_route[:]
            best_score = ns
            route = new_route[:]

    print(f"    Final:  {len(best_route)}ctrl {best_score:.0f}pts "
          f"base={rcost(cm, best_route):.0f} "
          f"fatigue={rcost_fatigue(cm, best_route, bud_raw, fatigue_rate):.0f}")
    return best_route
