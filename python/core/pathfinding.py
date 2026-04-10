"""
Pathfinding: anisotropic Dijkstra, A*, cost matrix construction, route cost helpers.
"""
import numpy as np
import math
import os
from .cost_functions import minetti_factor

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def compute_slope_dir(elev, cell_m):
    """Compute directional slope components (dx, dy) from DEM."""
    e = elev.astype(np.float64)
    dx = np.zeros_like(e); dy = np.zeros_like(e)
    dx[:, 1:-1] = (e[:, 2:] - e[:, :-2]) / (2 * cell_m)
    dx[:, 0] = (e[:, 1] - e[:, 0]) / cell_m
    dx[:, -1] = (e[:, -1] - e[:, -2]) / cell_m
    dy[1:-1, :] = (e[2:, :] - e[:-2, :]) / (2 * cell_m)
    dy[0, :] = (e[1, :] - e[0, :]) / cell_m
    dy[-1, :] = (e[-1, :] - e[-2, :]) / cell_m
    return np.nan_to_num(dx), np.nan_to_num(dy)


def ds_grid(grid, factor):
    """Downsample a grid by averaging factor×factor blocks."""
    h, w = grid.shape
    h_ds, w_ds = h // factor, w // factor
    return grid[:h_ds * factor, :w_ds * factor].reshape(
        h_ds, factor, w_ds, factor).mean(axis=(1, 3))


def rcost(cm, route):
    """Base route cost (no fatigue)."""
    if not route:
        return 0.0
    c = cm[0][route[0]]
    for i in range(len(route) - 1):
        c += cm[route[i]][route[i + 1]]
    return c + cm[route[-1]][0]


def rcost_fatigue(cm, route, budget, fatigue_rate):
    """Route cost with cumulative fatigue."""
    if not route:
        return 0.0
    total = 0.0; elapsed = 0.0
    seq = [0] + list(route) + [0]
    for i in range(len(seq) - 1):
        leg = cm[seq[i]][seq[i + 1]]
        total += leg * (1.0 + fatigue_rate * (elapsed / max(budget, 1.0)))
        elapsed += leg
    return total


def rcost_fatigue_detail(cm, route, budget, fatigue_rate):
    """Detailed per-leg fatigue breakdown."""
    if not route:
        return []
    details = []
    elapsed = 0.0
    seq = [0] + list(route) + [0]
    for i in range(len(seq) - 1):
        leg = cm[seq[i]][seq[i + 1]]
        fm = 1.0 + fatigue_rate * (elapsed / max(budget, 1.0))
        actual = leg * fm
        details.append({
            'src': seq[i], 'dst': seq[i + 1],
            'base': leg, 'fatigue': fm, 'actual': actual
        })
        elapsed += leg
    return details


def rpts(route, pts):
    """Total points collected on a route."""
    return sum(pts[v] for v in route)


def compute_cost_matrix(base_cost, slope_x, slope_y, nodes, ds=8,
                        cache_enabled=True):
    """Build asymmetric cost matrix via anisotropic Dijkstra from each node."""
    h, w = base_cost.shape
    h_ds, w_ds = h // ds, w // ds

    ds_base = ds_grid(base_cost, ds) * ds
    ds_sx = ds_grid(slope_x, ds)
    ds_sy = ds_grid(slope_y, ds)

    n = len(nodes)
    dn = [(int(x // ds), int(y // ds)) for x, y in nodes]

    # Check for cache
    cost_hash = hash((
        round(float(np.median(base_cost)), 4),
        round(float(base_cost.mean()), 4),
        n, ds,
    ))
    cache_file = f'cost_matrix_cache_ds{ds}_n{n}_{abs(cost_hash) % 100000}.npz'

    # Clean old caches
    for f in os.listdir('.'):
        if (f.startswith('cost_matrix_cache') and f.endswith('.npz')
                and f != cache_file):
            try:
                os.remove(f)
                print(f"    🗑️ Removed old cache: {f}")
            except OSError:
                pass

    if cache_enabled and os.path.exists(cache_file):
        with np.load(cache_file) as data:
            cm = data['cm'].copy()
        print(f"    ✅ Loaded cached cost matrix ({cache_file})")
        return cm

    print(f"    DS grid: {w_ds}x{h_ds}, {n} nodes, ds={ds}")

    if NUMBA_AVAILABLE:
        cm = _compute_cost_matrix_numba(ds_base, ds_sx, ds_sy, dn, n, h_ds, w_ds)
    else:
        cm = _compute_cost_matrix_python(ds_base, ds_sx, ds_sy, dn, n, h_ds, w_ds)

    np.savez_compressed(cache_file, cm=cm)
    print(f"    💾 Saved cost matrix cache ({cache_file})")
    return cm


def _compute_cost_matrix_python(ds_base, ds_sx, ds_sy, dn, n, h_ds, w_ds):
    """Pure Python fallback for cost matrix computation."""
    import heapq
    cm = np.full((n, n), np.inf)
    np.fill_diagonal(cm, 0.0)

    ndx = [-1, 1, 0, 0, -1, 1, -1, 1]
    ndy = [0, 0, -1, 1, -1, -1, 1, 1]
    nsl = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]

    for i in range(n):
        sx = min(max(dn[i][0], 0), w_ds - 1)
        sy = min(max(dn[i][1], 0), h_ds - 1)
        dist = np.full((h_ds, w_ds), np.inf)
        dist[sy, sx] = 0.0
        vis = np.zeros((h_ds, w_ds), dtype=bool)
        pq = [(0.0, sx, sy)]

        while pq:
            d, x, y = heapq.heappop(pq)
            if vis[y, x]:
                continue
            vis[y, x] = True
            for k in range(8):
                nx_, ny_ = x + ndx[k], y + ndy[k]
                if not (0 <= nx_ < w_ds and 0 <= ny_ < h_ds):
                    continue
                if vis[ny_, nx_]:
                    continue
                bc = ds_base[ny_, nx_]
                if not np.isfinite(bc):
                    continue
                step_len = nsl[k]
                if bc >= 9.0:
                    nd = d + step_len * 10.0
                else:
                    gx = 0.5 * (ds_sx[y, x] + ds_sx[ny_, nx_])
                    gy = 0.5 * (ds_sy[y, x] + ds_sy[ny_, nx_])
                    ml = math.sqrt(ndx[k] ** 2 + ndy[k] ** 2)
                    dir_slope = (gx * ndx[k] + gy * ndy[k]) / ml
                    sf = float(minetti_factor(dir_slope))
                    avg_base = 0.5 * (ds_base[y, x] + bc)
                    nd = d + step_len * avg_base * sf
                if nd < dist[ny_, nx_]:
                    dist[ny_, nx_] = nd
                    heapq.heappush(pq, (nd, nx_, ny_))

        for j in range(n):
            if i != j:
                tx = min(max(dn[j][0], 0), w_ds - 1)
                ty = min(max(dn[j][1], 0), h_ds - 1)
                cm[i, j] = dist[ty, tx]
        if (i + 1) % 10 == 0:
            print(f"    Node {i + 1}/{n}")

    return cm


def _compute_cost_matrix_numba(ds_base, ds_sx, ds_sy, dn, n, h_ds, w_ds):
    """Numba-accelerated cost matrix computation."""
    from numba import njit
    import time as time_module

    _MC = np.array([280.5, -58.7, -76.8, 51.9, 19.6, 2.5])
    _MF = np.polyval(_MC, 0.0)

    @njit(cache=True)
    def _minetti_factor_nb(slope_signed):
        i = max(-0.45, min(0.45, slope_signed))
        c = _MC[0]
        for k in range(1, 6):
            c = c * i + _MC[k]
        return max(c, 0.5) / _MF

    @njit(cache=True)
    def _heap_push(hd, hx, hy, hn, d, x, y, mx):
        if hn >= mx:
            return hn
        pos = hn
        hd[pos] = d; hx[pos] = x; hy[pos] = y
        hn += 1
        while pos > 0:
            parent = (pos - 1) // 2
            if hd[pos] < hd[parent]:
                hd[pos], hd[parent] = hd[parent], hd[pos]
                hx[pos], hx[parent] = hx[parent], hx[pos]
                hy[pos], hy[parent] = hy[parent], hy[pos]
                pos = parent
            else:
                break
        return hn

    @njit(cache=True)
    def _heap_pop(hd, hx, hy, hn):
        d = hd[0]; x = hx[0]; y = hy[0]
        hn -= 1
        hd[0] = hd[hn]; hx[0] = hx[hn]; hy[0] = hy[hn]
        pos = 0
        while True:
            child = 2 * pos + 1
            if child >= hn:
                break
            if child + 1 < hn and hd[child + 1] < hd[child]:
                child += 1
            if hd[child] < hd[pos]:
                hd[pos], hd[child] = hd[child], hd[pos]
                hx[pos], hx[child] = hx[child], hx[pos]
                hy[pos], hy[child] = hy[child], hy[pos]
                pos = child
            else:
                break
        return d, x, y, hn

    @njit(cache=True)
    def _dijkstra_nb(base_cost, slope_x, slope_y, sx, sy):
        h, w = base_cost.shape
        INF = 1e18
        dist = np.full((h, w), INF, dtype=np.float64)
        dist[sy, sx] = 0.0
        vis = np.zeros((h, w), dtype=np.bool_)
        MAX_HEAP = h * w * 2
        hd = np.empty(MAX_HEAP, dtype=np.float64)
        hx = np.empty(MAX_HEAP, dtype=np.int32)
        hy = np.empty(MAX_HEAP, dtype=np.int32)
        hn = 0
        hn = _heap_push(hd, hx, hy, hn, 0.0, sx, sy, MAX_HEAP)
        ndx = np.array([-1, 1, 0, 0, -1, 1, -1, 1], dtype=np.int32)
        ndy = np.array([0, 0, -1, 1, -1, -1, 1, 1], dtype=np.int32)
        nsl = np.array([1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414])
        while hn > 0:
            dd, x, y, hn = _heap_pop(hd, hx, hy, hn)
            if vis[y, x]:
                continue
            vis[y, x] = True
            for k in range(8):
                nx_ = x + ndx[k]; ny_ = y + ndy[k]
                if nx_ < 0 or nx_ >= w or ny_ < 0 or ny_ >= h:
                    continue
                if vis[ny_, nx_]:
                    continue
                bc = base_cost[ny_, nx_]
                if bc != bc:
                    continue
                step_len = nsl[k]
                if bc >= 9.0:
                    nd = dd + step_len * 10.0
                else:
                    gx = 0.5 * (slope_x[y, x] + slope_x[ny_, nx_])
                    gy = 0.5 * (slope_y[y, x] + slope_y[ny_, nx_])
                    ml = math.sqrt(float(ndx[k]) ** 2 + float(ndy[k]) ** 2)
                    ds = (gx * float(ndx[k]) + gy * float(ndy[k])) / ml
                    sf = _minetti_factor_nb(ds)
                    nd = dd + step_len * 0.5 * (base_cost[y, x] + bc) * sf
                if nd < dist[ny_, nx_]:
                    dist[ny_, nx_] = nd
                    hn = _heap_push(hd, hx, hy, hn, nd, nx_, ny_, MAX_HEAP)
        return dist

    # JIT warmup
    print("    JIT compiling (first run only)...")
    t0 = time_module.time()
    _dijkstra_nb(ds_base[:10, :10], ds_sx[:10, :10], ds_sy[:10, :10], 0, 0)
    print(f"    JIT ready in {time_module.time() - t0:.1f}s")

    cm = np.full((n, n), np.inf)
    np.fill_diagonal(cm, 0.0)
    for i in range(n):
        t0 = time_module.time()
        sx = min(max(dn[i][0], 0), w_ds - 1)
        sy = min(max(dn[i][1], 0), h_ds - 1)
        dist = _dijkstra_nb(ds_base, ds_sx, ds_sy, sx, sy)
        for j in range(n):
            if i != j:
                tx = min(max(dn[j][0], 0), w_ds - 1)
                ty = min(max(dn[j][1], 0), h_ds - 1)
                cm[i, j] = dist[ty, tx]
        elapsed = time_module.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Node {i + 1}/{n} ({elapsed:.2f}s, ETA {elapsed * (n - i - 1):.0f}s)")
    return cm
