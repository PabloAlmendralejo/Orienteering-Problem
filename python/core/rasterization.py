"""
Rasterization: OMAP symbol objects → cost/grid arrays.
Line, polygon, and point rasterization with buffer support.
"""
import numpy as np
import math
from PIL import Image, ImageDraw


def rast_line_omap(grid, coords, val, buf):
    """Rasterize a line onto a grid (max value)."""
    h, w = grid.shape
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        steps = max(int(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) * 2), 1)
        for s in range(steps + 1):
            t = s / steps
            ci = int(round(x0 + t * (x1 - x0)))
            ri = int(round(y0 + t * (y1 - y0)))
            for dr in range(-buf, buf + 1):
                for dc in range(-buf, buf + 1):
                    r, c = ri + dr, ci + dc
                    if 0 <= r < h and 0 <= c < w:
                        grid[r, c] = max(grid[r, c], val)


def rast_line_cost_omap(cost, coords, val, buf):
    """Rasterize a line — keeps the LOWER value (traversable surfaces)."""
    h, w = cost.shape
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        steps = max(int(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) * 2), 1)
        for s in range(steps + 1):
            t = s / steps
            ci = int(round(x0 + t * (x1 - x0)))
            ri = int(round(y0 + t * (y1 - y0)))
            for dr in range(-buf, buf + 1):
                for dc in range(-buf, buf + 1):
                    r, c = ri + dr, ci + dc
                    if 0 <= r < h and 0 <= c < w:
                        if cost[r, c] < 0 or val < cost[r, c]:
                            cost[r, c] = val


def rast_line_barrier_omap(cost, coords, val, buf):
    """Rasterize a barrier line — keeps the HIGHER value (obstacles)."""
    h, w = cost.shape
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        steps = max(int(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) * 2), 1)
        for s in range(steps + 1):
            t = s / steps
            ci = int(round(x0 + t * (x1 - x0)))
            ri = int(round(y0 + t * (y1 - y0)))
            for dr in range(-buf, buf + 1):
                for dc in range(-buf, buf + 1):
                    r, c = ri + dr, ci + dc
                    if 0 <= r < h and 0 <= c < w:
                        if cost[r, c] < val:
                            cost[r, c] = val


def rast_poly_omap(cost, coords, val):
    """Polygon fill — keeps the LOWER value."""
    h, w = cost.shape
    if len(coords) < 3:
        return
    img_tmp = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img_tmp).polygon(
        [(int(round(x)), int(round(y))) for x, y in coords], fill=255)
    m = np.array(img_tmp) > 0
    cost[m & ((cost < 0) | (cost > val))] = val


def rast_poly_barrier_omap(cost, coords, val):
    """Polygon fill for barriers — keeps the HIGHER value."""
    h, w = cost.shape
    if len(coords) < 3:
        return
    img_tmp = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img_tmp).polygon(
        [(int(round(x)), int(round(y))) for x, y in coords], fill=255)
    m = np.array(img_tmp) > 0
    cost[m & (cost < val)] = val


def rast_poly_bool_omap(grid, coords, val=True):
    """Polygon fill on a boolean grid."""
    h, w = grid.shape
    if len(coords) < 3:
        return
    img_tmp = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img_tmp).polygon(
        [(int(round(x)), int(round(y))) for x, y in coords], fill=255)
    grid[np.array(img_tmp) > 0] = val
