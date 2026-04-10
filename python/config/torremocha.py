"""
Configuration for Torremocha study area.
"""
import os

# ── CRS and Projection ──
MAP_CRS = "EPSG:25829"  # ETRS89 / UTM zone 29N

# ── World file parameters (from TIF) ──
WORLD_A = 0.4232651207
WORLD_B = 0.01322769604
WORLD_C = 742311.5154
WORLD_D = 0.01322769604
WORLD_E = -0.4232651207
WORLD_F = 4355760.522

# ── Data files ──
DATA_DIR = os.environ.get('ORIENTEERING_DATA_DIR',
                          os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'torremocha'))
TIF_PATH = os.path.join(DATA_DIR, 'Torremocha_tiff.tif')
MDT_PATH = os.path.join(DATA_DIR, 'MDT02-WGS84-0730-1-COB2.tif')
OMAP_PATH = os.path.join(DATA_DIR, 'torremocha_omap.omap')
TERRAIN_CACHE = os.path.join(DATA_DIR, 'torremocha_terrain.npz')

# ── Grid ──
TARGET_RESOLUTION = 2.0  # metres per cell
DOWNSAMPLE = 8

# ── Race parameters ──
SEED = 42
NUM_CONTROLS = 40
BASE_SPEED = 2.5  # m/s on flat open terrain
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
    'torremocha_standard':      {'func': 'standard',      'seed': 42,  'num': 40},
    'torremocha_clustered':     {'func': 'clustered',     'seed': 123, 'num': 35},
    'torremocha_ring':          {'func': 'ring',          'seed': 77,  'num': 30},
    'torremocha_path_biased':   {'func': 'path_biased',   'seed': 99,  'num': 35},
    'torremocha_elev_biased':   {'func': 'elev_biased',   'seed': 17,  'num': 35},
    'torremocha_sparse_far':    {'func': 'sparse_far',    'seed': 31,  'num': 25},
    'torremocha_mixed_density': {'func': 'mixed_density', 'seed': 55,  'num': 40},
}
