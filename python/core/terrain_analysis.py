"""
Terrain analysis: HCR (Hypsometry Cost Raster), TRI, plan curvature, slope.
IQR-based raster fusion for ACR + HCR combination.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from .cost_functions import WT


def compute_tri(elev):
    """Terrain Ruggedness Index: std dev of elevation in 3×3 window."""
    mean_elev = uniform_filter(elev.astype(np.float64), size=3, mode='nearest')
    mean_sq = uniform_filter((elev.astype(np.float64)) ** 2, size=3, mode='nearest')
    variance = np.maximum(mean_sq - mean_elev ** 2, 0.0)
    tri = np.sqrt(variance)
    return np.nan_to_num(tri, nan=0.0, posinf=0.0, neginf=0.0)


def compute_plan_curvature(elev, cell_m):
    """Plan curvature (absolute value) — detects ridges, spurs, side valleys."""
    e = elev.astype(np.float64)
    p = np.zeros_like(e); q = np.zeros_like(e)
    p[:, 1:-1] = (e[:, 2:] - e[:, :-2]) / (2.0 * cell_m)
    p[:, 0] = (e[:, 1] - e[:, 0]) / cell_m
    p[:, -1] = (e[:, -1] - e[:, -2]) / cell_m
    q[1:-1, :] = (e[2:, :] - e[:-2, :]) / (2.0 * cell_m)
    q[0, :] = (e[1, :] - e[0, :]) / cell_m
    q[-1, :] = (e[-1, :] - e[-2, :]) / cell_m
    r = np.zeros_like(e); t = np.zeros_like(e); s = np.zeros_like(e)
    r[:, 1:-1] = (e[:, 2:] - 2 * e[:, 1:-1] + e[:, :-2]) / (cell_m ** 2)
    t[1:-1, :] = (e[2:, :] - 2 * e[1:-1, :] + e[:-2, :]) / (cell_m ** 2)
    s[1:-1, 1:-1] = ((e[2:, 2:] - e[2:, :-2] - e[:-2, 2:] + e[:-2, :-2])
                      / (4.0 * cell_m ** 2))
    denom = np.maximum(p ** 2 + q ** 2, 1e-12) ** 1.5
    plan_curv = np.abs(-(p ** 2 * r - 2 * p * q * s + q ** 2 * t) / denom)
    return np.nan_to_num(plan_curv, nan=0.0, posinf=0.0, neginf=0.0)


def compute_slope_magnitude_deg(elev, cell_m):
    """Slope angle in degrees."""
    e = elev.astype(np.float64)
    dx = np.zeros_like(e); dy = np.zeros_like(e)
    dx[:, 1:-1] = (e[:, 2:] - e[:, :-2]) / (2.0 * cell_m)
    dx[:, 0] = (e[:, 1] - e[:, 0]) / cell_m
    dx[:, -1] = (e[:, -1] - e[:, -2]) / cell_m
    dy[1:-1, :] = (e[2:, :] - e[:-2, :]) / (2.0 * cell_m)
    dy[0, :] = (e[1, :] - e[0, :]) / cell_m
    dy[-1, :] = (e[-1, :] - e[-2, :]) / cell_m
    return np.nan_to_num(np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2))), nan=0.0)


def normalize_to_range(arr, low=0.1, high=1.0, percentile_clip=(1, 99)):
    """Normalize array to [low, high] with robust percentile clipping."""
    valid_mask = np.isfinite(arr)
    if not valid_mask.any():
        return np.full_like(arr, low, dtype=np.float32)
    valid_vals = arr[valid_mask]
    if len(valid_vals) == 0:
        return np.full_like(arr, low, dtype=np.float32)
    vmin = np.percentile(valid_vals, percentile_clip[0])
    vmax = np.percentile(valid_vals, percentile_clip[1])
    if vmax <= vmin:
        return np.full_like(arr, (low + high) / 2.0, dtype=np.float32)
    normalized = low + (high - low) * (arr - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, low, high)
    normalized[~valid_mask] = low
    return normalized.astype(np.float32)


def iqr_ratio_constant(acr_flat, hcr_flat):
    """Scale HCR to ACR using IQR ratio: c = IQR(ACR) / IQR(HCR)."""
    acr_iqr = np.percentile(acr_flat, 75) - np.percentile(acr_flat, 25)
    hcr_iqr = np.percentile(hcr_flat, 75) - np.percentile(hcr_flat, 25)
    c = acr_iqr / hcr_iqr if hcr_iqr > 1e-10 else 1.0
    print(f"    IQR matching: ACR IQR={acr_iqr:.4f}, HCR IQR={hcr_iqr:.4f}, c={c:.2f}")
    return c


def compute_hcr(elev, cell_m, hcr_tri_power=1.0, hcr_pc_power=0.5,
                hcr_slo_power=1.0, norm_low=0.1, norm_high=1.0):
    """Eq.1: HCR = TRI_norm^a × |PC_norm|^b × SLO_norm^c"""
    print("    Computing morphometric variables...")
    tri_raw = compute_tri(elev)
    tri_norm = normalize_to_range(tri_raw, norm_low, norm_high)
    print(f"      TRI:  {tri_raw.min():.3f} / {np.median(tri_raw):.3f} / {tri_raw.max():.3f}")

    pc_raw = compute_plan_curvature(elev, cell_m)
    pc_norm = normalize_to_range(pc_raw, norm_low, norm_high)
    print(f"      |PC|: {pc_raw.min():.6f} / {np.median(pc_raw):.6f} / {pc_raw.max():.6f}")

    slo_raw = compute_slope_magnitude_deg(elev, cell_m)
    slo_norm = normalize_to_range(slo_raw, norm_low, norm_high)
    print(f"      SLO:  {slo_raw.min():.1f}° / {np.median(slo_raw):.1f}° / {slo_raw.max():.1f}°")

    hcr = (tri_norm ** hcr_tri_power *
           pc_norm ** hcr_pc_power *
           slo_norm ** hcr_slo_power)
    print(f"      HCR:  {hcr.min():.4f} / {np.median(hcr):.4f} / {hcr.max():.4f}")

    components = {
        'tri_raw': tri_raw, 'tri_norm': tri_norm,
        'pc_raw': pc_raw, 'pc_norm': pc_norm,
        'slo_raw': slo_raw, 'slo_norm': slo_norm,
        'hcr_raw': hcr.copy()
    }
    return hcr, components


def apply_hcr_to_base_cost(cost_omap, hcr_raw, valid):
    """Scale HCR via IQR ratio, then final = ACR + HCR_a (with path protection)."""
    print("    Applying HCR to base cost...")
    acr = cost_omap.copy()

    mask = valid & (acr > 0.01) & (acr < 5.0) & np.isfinite(acr)
    acr_flat = acr[mask].ravel()
    hcr_mask = valid & (hcr_raw > 0.0) & np.isfinite(hcr_raw)
    hcr_flat = hcr_raw[hcr_mask].ravel()

    c = iqr_ratio_constant(acr_flat, hcr_flat)
    hcr_a = c * hcr_raw

    path_protection = np.ones_like(acr, dtype=np.float32)
    path_protection[acr < 0.15] = 0.1
    path_protection[(acr >= 0.15) & (acr < 0.25)] = 0.3
    path_protection[acr >= 5.0] = 0.0

    hcr_contribution = hcr_a * path_protection
    combined = acr + hcr_contribution
    combined[~valid] = WT['out_of_bounds']
    combined = np.maximum(combined, 0.05)
    combined[acr >= 5.0] = acr[acr >= 5.0]

    print(f"      Combined: {combined[combined < 5].min():.4f} – {combined[combined < 5].max():.4f}")
    print(f"      Mean HCR add: {hcr_contribution[valid].mean():.4f}")
    return combined.astype(np.float32), hcr_a, hcr_contribution


def build_base_cost_with_hcr(cost_omap, elev, cell_m, valid,
                              hcr_tri_power=1.0, hcr_pc_power=0.5,
                              hcr_slo_power=1.0, norm_low=0.1, norm_high=1.0):
    """Build base cost grid with HCR enhancement."""
    print("  Computing Hypsometry Cost Raster (HCR)...")
    hcr_raw, components = compute_hcr(elev, cell_m, hcr_tri_power, hcr_pc_power,
                                       hcr_slo_power, norm_low, norm_high)
    combined, hcr_a, hcr_contrib = apply_hcr_to_base_cost(cost_omap, hcr_raw, valid)
    components['hcr_a'] = hcr_a
    components['hcr_contribution'] = hcr_contrib
    components['combined'] = combined
    return combined, components
