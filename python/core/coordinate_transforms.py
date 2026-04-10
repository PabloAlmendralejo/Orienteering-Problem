"""
Coordinate transformations: pixel ↔ UTM, paper ↔ UTM, UTM ↔ grid.
"""
import numpy as np
import math


def make_pixel_to_utm(A, B, C, D, E, F):
    """Create pixel→UTM transform from world file parameters."""
    def pixel_to_utm(col, row):
        x = A * col + B * row + C
        y = D * col + E * row + F
        return x, y
    return pixel_to_utm


def make_utm_to_pixel(A, B, C, D, E, F):
    """Create UTM→pixel transform (inverse of world file)."""
    det = A * E - B * D
    def utm_to_pixel(x, y):
        col = (E * (x - C) - B * (y - F)) / det
        row = (-D * (x - C) + A * (y - F)) / det
        return col, row
    return utm_to_pixel


def make_paper_to_utm_omap(proj_center_x, proj_center_y, scale, grivation_deg,
                            paper_ref_x_mm, paper_ref_y_mm):
    """Create OMAP paper coordinates (mm) → UTM transform."""
    grivation_rad = math.radians(grivation_deg)
    cos_g = math.cos(grivation_rad)
    sin_g = math.sin(grivation_rad)

    def paper_to_utm(coords_mm):
        result = []
        for px_mm, py_mm in coords_mm:
            dx_mm = px_mm - paper_ref_x_mm
            dy_mm = py_mm - paper_ref_y_mm
            dx_m = dx_mm * scale / 1000.0
            dy_m = dy_mm * scale / 1000.0
            utm_x = proj_center_x + dx_m * cos_g - dy_m * sin_g
            utm_y = proj_center_y - dx_m * sin_g - dy_m * cos_g
            result.append((utm_x, utm_y))
        return result
    return paper_to_utm


def make_utm_to_grid(bounds, resolution):
    """Create UTM → grid pixel transform."""
    x_min, y_min, x_max, y_max = bounds

    def utm_to_grid(utm_coords):
        result = []
        for ux, uy in utm_coords:
            gx = (ux - x_min) / resolution
            gy = (y_max - uy) / resolution
            result.append((gx, gy))
        return result
    return utm_to_grid
