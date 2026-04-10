"""
Configuration for La Muela study area.
"""
import os

# ── CRS and Projection ──
MAP_CRS = "EPSG:25830"  # ETRS89 / UTM zone 30N

# ── World file parameters (from TIF) ──
WORLD_A = 1.693333333
WORLD_B = 0.0
WORLD_C = 262112.15003333348
WORLD_D = 0.0
WORLD_E = -1.693333333
WORLD_F = 4470152.349666666

# ── Data files ──
DATA_DIR = os.environ.get('ORIENTEERING_DATA_DIR',
                          os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'la_muela'))
TIF_PATH = os.path.join(DATA_DIR, 'La_muela_tiff.tif')
MDT_PATH = os.path.join(DATA_DIR, 'MDT02_candelario_merged.tif')
OMAP_PATH = os.path.join(DATA_DIR, 'LAMUELA.omap')
TERRAIN_CACHE = os.path.join(DATA_DIR, 'la_muela_terrain.npz')

# ── Grid ──
TARGET_RESOLUTION = 2.0
DOWNSAMPLE = 8

# ── Race parameters ──
SEED = 42
NUM_CONTROLS = 40
BASE_SPEED = 2.5
REFERENCE_WEIGHT = 0.3
RACE_DURATION_HOURS = 1
FATIGUE_RATE = 0.20

# ── HCR Configuration ──
HCR_TRI_POWER = 1.0
HCR_PC_POWER = 0.5
HCR_SLO_POWER = 1.0
HCR_NORM_LOW = 0.1
HCR_NORM_HIGH = 1.0

# ── Derived constants ──
RACE_DURATION_SECONDS = RACE_DURATION_HOURS * 3600
COST_TO_SECONDS = TARGET_RESOLUTION / (BASE_SPEED * REFERENCE_WEIGHT)
ROUTE_BUDGET = RACE_DURATION_SECONDS / COST_TO_SECONDS

# ── JSON export distributions ──
DISTRIBUTIONS = {
    'standard':      {'func': 'standard',      'seed': 42,  'num': 40},
    'clustered':     {'func': 'clustered',     'seed': 123, 'num': 35},
    'ring':          {'func': 'ring',          'seed': 77,  'num': 30},
    'path_biased':   {'func': 'path_biased',   'seed': 99,  'num': 35},
    'elev_biased':   {'func': 'elev_biased',   'seed': 17,  'num': 35},
    'sparse_far':    {'func': 'sparse_far',    'seed': 31,  'num': 25},
    'mixed_density': {'func': 'mixed_density', 'seed': 55,  'num': 40},
}
