"""
Control point placement: hash house selection and various distribution strategies.
"""
import numpy as np
import math
import random
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure, binary_fill_holes, label


def create_valid_mask(img_rgb):
    """Detect valid map area by flood-filling white background from edges."""
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    h, w = r.shape
    struct = generate_binary_structure(2, 2)
    is_white = (r > 245) & (g > 245) & (b > 245)
    bg = np.zeros_like(is_white)
    bg[0, :] = is_white[0, :]
    bg[-1, :] = is_white[-1, :]
    bg[:, 0] = is_white[:, 0]
    bg[:, -1] = is_white[:, -1]
    for _ in range(max(h, w)):
        new = binary_dilation(bg, structure=struct) & is_white
        if np.array_equal(new, bg):
            break
        bg = new
    valid = ~bg
    valid = binary_fill_holes(valid)
    interior_white = is_white & valid
    labeled, n_features = label(interior_white)
    for i in range(1, n_features + 1):
        cluster = labeled == i
        if cluster.sum() > 500:
            valid[cluster] = False
    valid = binary_erosion(valid, iterations=8)
    pct = 100 * valid.sum() / valid.size
    print(f"  Valid map: {valid.sum():,} px ({pct:.0f}%)")
    return valid


def find_valid_hh(valid, veg, slope, cell_meters):
    """Find a suitable hash house (start/finish) location."""
    candidates = valid & (veg > 0.3) & (slope < 15)
    if candidates.sum() < 100:
        candidates = valid & (veg > 0.1) & (slope < 25)
    cy, cx = np.where(candidates)
    if len(cy) == 0:
        cy, cx = np.where(valid)
    center_y, center_x = valid.shape[0] // 2, valid.shape[1] // 2
    dist = (cx - center_x) ** 2 + (cy - center_y) ** 2
    best = np.argmin(dist)
    return int(cx[best]), int(cy[best])


def place_controls_standard(veg, elev, slope, valid, num, hh, cell_meters, seed=42):
    """Grid-zone uniform spread of controls."""
    random.seed(seed); np.random.seed(seed)
    h, w = veg.shape
    hh_x, hh_y = hh
    min_sep = int(80 / cell_meters)
    hh_sep = int(60 / cell_meters)
    candidates_mask = valid & (veg > 0.05) & (slope < 40) & ~np.isnan(elev)
    cy, cx = np.where(candidates_mask)
    if len(cy) == 0:
        return []
    print(f"  Candidate pixels: {len(cy):,}")
    vx_min, vx_max = cx.min(), cx.max()
    vy_min, vy_max = cy.min(), cy.max()
    cols = 6
    rows = max(3, int(cols * (vy_max - vy_min) / max(vx_max - vx_min, 1)))
    zw = max(1, (vx_max - vx_min) // cols)
    zh = max(1, (vy_max - vy_min) // rows)
    ctrls = []
    for zr in range(rows):
        for zc in range(cols):
            if len(ctrls) >= num:
                break
            zx0 = vx_min + zc * zw
            zy0 = vy_min + zr * zh
            zx1 = min(zx0 + zw, vx_max)
            zy1 = min(zy0 + zh, vy_max)
            zone_mask = (cx >= zx0) & (cx < zx1) & (cy >= zy0) & (cy < zy1)
            zone_cx = cx[zone_mask]
            zone_cy = cy[zone_mask]
            if len(zone_cx) == 0:
                continue
            indices = np.random.permutation(len(zone_cx))[:300]
            for idx in indices:
                x, y = int(zone_cx[idx]), int(zone_cy[idx])
                if not valid[y, x]:
                    continue
                too_close = any(math.sqrt((x - px) ** 2 + (y - py) ** 2) < min_sep
                                for px, py, _, _ in ctrls)
                if too_close:
                    continue
                if math.sqrt((x - hh_x) ** 2 + (y - hh_y) ** 2) < hh_sep:
                    continue
                dist_hh = math.sqrt((x - hh_x) ** 2 + (y - hh_y) ** 2) * cell_meters
                max_dist = math.sqrt(w ** 2 + h ** 2) * cell_meters / 2
                diff = ((dist_hh / max_dist) * 2.0 +
                        max(0, 1.0 - veg[y, x]) * 3.0 +
                        min(slope[y, x] / 30.0, 1.0) * 2.0)
                if diff < 0.8:     pts = 10
                elif diff < 1.5:   pts = 20
                elif diff < 2.2:   pts = 30
                elif diff < 3.0:   pts = 40
                else:              pts = 50
                tier = pts // 10
                tc = sum(1 for (_, _, _, p) in ctrls if p == pts)
                ctrls.append((x, y, tier * 10 + tc + 1, pts))
                break
    ctrls = [c for c in ctrls if valid[c[1], c[0]]]
    print(f"  Placed {len(ctrls)} controls (all verified valid)")
    dist_pts = {}
    for _, _, _, p in ctrls:
        dist_pts[p] = dist_pts.get(p, 0) + 1
    print(f"  Points: {dict(sorted(dist_pts.items()))}")
    return ctrls


def place_controls_clustered(veg, elev, slope, valid, num, hh, cell_meters, seed=123):
    """5 hotspot clusters of controls."""
    random.seed(seed); np.random.seed(seed)
    h, w = veg.shape
    hh_x, hh_y = hh
    min_sep = int(60 / cell_meters)
    candidates_mask = valid & (veg > 0.05) & (slope < 40) & ~np.isnan(elev)
    cy, cx = np.where(candidates_mask)
    if len(cy) == 0:
        return []
    n_clusters = 5
    centers = []
    for _ in range(n_clusters):
        for _ in range(200):
            idx = random.randint(0, len(cx) - 1)
            x, y = int(cx[idx]), int(cy[idx])
            if math.sqrt((x - hh_x) ** 2 + (y - hh_y) ** 2) < int(40 / cell_meters):
                continue
            too_close = any(math.sqrt((x - cx2) ** 2 + (y - cy2) ** 2) < int(200 / cell_meters)
                            for cx2, cy2 in centers)
            if too_close:
                continue
            centers.append((x, y)); break
    ctrls = []
    cluster_radius = int(250 / cell_meters)
    per_cluster = num // max(len(centers), 1) + 1
    for ccx, ccy in centers:
        placed = 0
        for _ in range(500):
            if placed >= per_cluster or len(ctrls) >= num:
                break
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(int(30 / cell_meters), cluster_radius)
            x = int(ccx + dist * math.cos(angle))
            y = int(ccy + dist * math.sin(angle))
            if not (0 <= x < w and 0 <= y < h) or not valid[y, x]:
                continue
            too_close = any(math.sqrt((x - px) ** 2 + (y - py) ** 2) < min_sep
                            for px, py, _, _ in ctrls)
            if too_close:
                continue
            if math.sqrt((x - hh_x) ** 2 + (y - hh_y) ** 2) < int(40 / cell_meters):
                continue
            diff = min(slope[y, x] / 25.0, 1.0) * 2 + max(0, 1.0 - veg[y, x]) * 2
            pts = 10 + int(diff / 0.8) * 10
            pts = (max(10, min(pts, 50)) // 10) * 10
            tier = pts // 10; tc = sum(1 for (_, _, _, p) in ctrls if p == pts)
            ctrls.append((x, y, tier * 10 + tc + 1, pts)); placed += 1
    return [c for c in ctrls if valid[c[1], c[0]]]


def place_controls_ring(veg, elev, slope, valid, num, hh, cell_meters, seed=77):
    """Concentric rings from depot."""
    random.seed(seed); np.random.seed(seed)
    h, w = veg.shape
    hh_x, hh_y = hh
    min_sep = int(60 / cell_meters)
    candidates_mask = valid & (veg > 0.05) & (slope < 40) & ~np.isnan(elev)
    cy, cx = np.where(candidates_mask)
    if len(cy) == 0:
        return []
    rings = [
        (int(150 / cell_meters), int(300 / cell_meters), 10, 20),
        (int(300 / cell_meters), int(600 / cell_meters), 20, 30),
        (int(600 / cell_meters), int(1000 / cell_meters), 30, 40),
        (int(1000 / cell_meters), int(1500 / cell_meters), 40, 50),
    ]
    ctrls = []
    per_ring = num // len(rings) + 1
    dist_sq = (cx - hh_x) ** 2 + (cy - hh_y) ** 2
    for r_min, r_max, pts_lo, pts_hi in rings:
        ring_mask = (dist_sq >= r_min ** 2) & (dist_sq < r_max ** 2)
        rcx, rcy = cx[ring_mask], cy[ring_mask]
        if len(rcx) == 0:
            continue
        placed = 0
        for idx in np.random.permutation(len(rcx)):
            if placed >= per_ring or len(ctrls) >= num:
                break
            x, y = int(rcx[idx]), int(rcy[idx])
            if not valid[y, x]:
                continue
            if any(math.sqrt((x - px) ** 2 + (y - py) ** 2) < min_sep for px, py, _, _ in ctrls):
                continue
            diff = min(slope[y, x] / 25.0, 1.0) * 2 + max(0, 1.0 - veg[y, x]) * 2
            pts = pts_lo + int((pts_hi - pts_lo) * min(diff / 3.0, 1.0))
            pts = (max(pts_lo, min(pts, pts_hi)) // 10) * 10
            tier = pts // 10; tc = sum(1 for (_, _, _, p) in ctrls if p == pts)
            ctrls.append((x, y, tier * 10 + tc + 1, pts)); placed += 1
    return [c for c in ctrls if valid[c[1], c[0]]]


def place_controls_sparse_far(veg, elev, slope, valid, num, hh, cell_meters, seed=31):
    """Few controls, all far from depot."""
    random.seed(seed); np.random.seed(seed)
    h, w = veg.shape
    hh_x, hh_y = hh
    min_sep = int(150 / cell_meters)
    far_min = int(600 / cell_meters)
    candidates_mask = valid & (veg > 0.05) & (slope < 40) & ~np.isnan(elev)
    cy, cx = np.where(candidates_mask)
    if len(cy) == 0:
        return []
    dist_sq = (cx - hh_x) ** 2 + (cy - hh_y) ** 2
    far_mask = dist_sq >= far_min ** 2
    cy, cx = cy[far_mask], cx[far_mask]
    if len(cy) == 0:
        return []
    ctrls = []
    for idx in np.random.permutation(len(cx)):
        if len(ctrls) >= num:
            break
        x, y = int(cx[idx]), int(cy[idx])
        if not valid[y, x]:
            continue
        if any(math.sqrt((x - px) ** 2 + (y - py) ** 2) < min_sep for px, py, _, _ in ctrls):
            continue
        diff = min(slope[y, x] / 25.0, 1.0) * 2 + max(0, 1.0 - veg[y, x]) * 2
        pts = (max(30, min(30 + int(diff / 0.8) * 10, 50)) // 10) * 10
        tier = pts // 10; tc = sum(1 for (_, _, _, p) in ctrls if p == pts)
        ctrls.append((x, y, tier * 10 + tc + 1, pts))
    return [c for c in ctrls if valid[c[1], c[0]]]


def place_controls_mixed_density(veg, elev, slope, valid, num, hh, cell_meters, seed=55):
    """Dense near depot (low pts), sparse far (high pts)."""
    random.seed(seed); np.random.seed(seed)
    h, w = veg.shape
    hh_x, hh_y = hh
    candidates_mask = valid & (veg > 0.05) & (slope < 40) & ~np.isnan(elev)
    cy, cx = np.where(candidates_mask)
    if len(cy) == 0:
        return []
    dist_sq = (cx - hh_x) ** 2 + (cy - hh_y) ** 2
    near_r = int(300 / cell_meters)
    ctrls = []
    n_near = num // 2
    min_sep_near = int(50 / cell_meters)
    near_mask = dist_sq < near_r ** 2
    cy_n, cx_n = cy[near_mask], cx[near_mask]
    for idx in np.random.permutation(len(cx_n)):
        if len(ctrls) >= n_near:
            break
        x, y = int(cx_n[idx]), int(cy_n[idx])
        if not valid[y, x]:
            continue
        if math.sqrt((x - hh_x) ** 2 + (y - hh_y) ** 2) < int(40 / cell_meters):
            continue
        if any(math.sqrt((x - px) ** 2 + (y - py) ** 2) < min_sep_near for px, py, _, _ in ctrls):
            continue
        pts = 10; tier = 1; tc = sum(1 for (_, _, _, p) in ctrls if p == pts)
        ctrls.append((x, y, tier * 10 + tc + 1, pts))
    min_sep_far = int(120 / cell_meters)
    far_mask = dist_sq >= near_r ** 2
    cy_f, cx_f = cy[far_mask], cx[far_mask]
    for idx in np.random.permutation(len(cx_f)):
        if len(ctrls) >= num:
            break
        x, y = int(cx_f[idx]), int(cy_f[idx])
        if not valid[y, x]:
            continue
        if any(math.sqrt((x - px) ** 2 + (y - py) ** 2) < min_sep_far for px, py, _, _ in ctrls):
            continue
        diff = min(slope[y, x] / 25.0, 1.0) * 2 + max(0, 1.0 - veg[y, x]) * 2
        pts = (max(30, min(30 + int(diff / 0.8) * 10, 50)) // 10) * 10
        tier = pts // 10; tc = sum(1 for (_, _, _, p) in ctrls if p == pts)
        ctrls.append((x, y, tier * 10 + tc + 1, pts))
    return [c for c in ctrls if valid[c[1], c[0]]]


def place_controls_path_biased(veg, elev, slope, valid, num, hh, cell_meters, seed=99):
    """60% near paths, 40% far from paths."""
    from scipy.ndimage import distance_transform_edt
    random.seed(seed); np.random.seed(seed)
    h, w = veg.shape
    hh_x, hh_y = hh
    min_sep = int(60 / cell_meters)
    # Need path_grid — approximate from cost (paths have low cost)
    path_mask = veg > 0.8  # high veg proxy = near paths
    if path_mask.sum() == 0:
        return place_controls_standard(veg, elev, slope, valid, num, hh, cell_meters, seed)
    path_dist = distance_transform_edt(~path_mask) * cell_meters
    near_path = (path_dist < 150) & (path_dist > 20)
    far_path = path_dist >= 150
    candidates_near = valid & near_path & (slope < 35) & ~np.isnan(elev)
    candidates_far = valid & far_path & (slope < 40) & ~np.isnan(elev)
    cy_n, cx_n = np.where(candidates_near)
    cy_f, cx_f = np.where(candidates_far)
    ctrls = []
    n_near = int(num * 0.6); n_far = num - n_near
    for cx_arr, cy_arr, target, pts_range in [
        (cx_n, cy_n, n_near, (10, 30)), (cx_f, cy_f, n_far, (30, 50))]:
        if len(cx_arr) == 0:
            continue
        placed = 0
        for idx in np.random.permutation(len(cx_arr))[:1000]:
            if placed >= target or len(ctrls) >= num:
                break
            x, y = int(cx_arr[idx]), int(cy_arr[idx])
            if not valid[y, x]:
                continue
            if any(math.sqrt((x - px) ** 2 + (y - py) ** 2) < min_sep for px, py, _, _ in ctrls):
                continue
            if math.sqrt((x - hh_x) ** 2 + (y - hh_y) ** 2) < int(40 / cell_meters):
                continue
            diff = max(0, 1.0 - veg[y, x]) * 2 + min(slope[y, x] / 25.0, 1.0) * 2
            pts = pts_range[0] + int((pts_range[1] - pts_range[0]) * min(diff / 3.0, 1.0))
            pts = (max(pts_range[0], min(pts, pts_range[1])) // 10) * 10
            tier = pts // 10; tc = sum(1 for (_, _, _, p) in ctrls if p == pts)
            ctrls.append((x, y, tier * 10 + tc + 1, pts)); placed += 1
    return [c for c in ctrls if valid[c[1], c[0]]]


def place_controls_elev_biased(veg, elev, slope, valid, num, hh, cell_meters, seed=17):
    """Weighted toward high elevation controls."""
    random.seed(seed); np.random.seed(seed)
    h, w = veg.shape
    hh_x, hh_y = hh
    min_sep = int(70 / cell_meters)
    candidates_mask = valid & (veg > 0.05) & (slope < 45) & ~np.isnan(elev)
    cy, cx = np.where(candidates_mask)
    if len(cy) == 0:
        return []
    elevs = elev[cy, cx]
    weights = ((elevs - elevs.min()) / max(elevs.max() - elevs.min(), 1)) ** 2
    weights /= weights.sum()
    ctrls = []
    for idx in np.random.choice(len(cx), size=min(2000, len(cx)), replace=False, p=weights):
        if len(ctrls) >= num:
            break
        x, y = int(cx[idx]), int(cy[idx])
        if not valid[y, x]:
            continue
        if any(math.sqrt((x - px) ** 2 + (y - py) ** 2) < min_sep for px, py, _, _ in ctrls):
            continue
        if math.sqrt((x - hh_x) ** 2 + (y - hh_y) ** 2) < int(40 / cell_meters):
            continue
        diff = min(slope[y, x] / 25.0, 1.0) * 2 + max(0, 1.0 - veg[y, x]) * 2
        pts = (max(10, min(10 + int(diff / 0.8) * 10, 50)) // 10) * 10
        tier = pts // 10; tc = sum(1 for (_, _, _, p) in ctrls if p == pts)
        ctrls.append((x, y, tier * 10 + tc + 1, pts))
    return [c for c in ctrls if valid[c[1], c[0]]]
