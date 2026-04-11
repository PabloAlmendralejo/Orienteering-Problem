# Optimal Route Planning for Orienteering on Real Terrain

End-to-end framework for solving the Orienteering Problem (OP) on real terrain,
combining GIS-based cost surface generation with exact Branch-and-Cut optimisation.

## Overview

The pipeline has two stages:

1. **Python** — Preprocesses orienteering map files (OMAP), digital elevation models (DEM),
   and map imagery into an asymmetric cost matrix using:
   - IOF symbol-based cost raster (ACR)
   - Hypsometry Cost Raster (HCR) with IQR scaling
   - Minetti metabolic slope model (directional)
   - Cumulative fatigue model

2. **C++** — Solves the Orienteering Problem exactly via Branch-and-Cut with HiGHS,
   warm-started by Simulated Annealing.

## Project Structure

```
├── python/
│   ├── run_pipeline.py              # Main entry point
│   ├── requirements.txt
│   ├── core/
│   │   ├── preprocessing.py         # Raw files → terrain cache (.npz)
│   │   ├── cost_functions.py        # Minetti, fatigue, ISOM weights
│   │   ├── terrain_analysis.py      # HCR (TRI, PC, SLO), IQR scaling
│   │   ├── omap_parser.py           # OMAP XML parsing, symbol classification
│   │   ├── rasterization.py         # Line/polygon rasterization
│   │   ├── coordinate_transforms.py # Pixel ↔ UTM ↔ grid transforms
│   │   ├── pathfinding.py           # Anisotropic Dijkstra, cost matrix
│   │   ├── route_optimizer.py       # Greedy + SA route optimization
│   │   └── control_placement.py     # Control point generation (7 strategies)
│   └── config/
│       ├── torremocha.py            # Torremocha-specific settings
│       └── la_muela.py             # La Muela-specific settings
├── cpp/
│   ├── op_bnc_with_highs.cpp        # B&C solver (La Muela / generic)
│   └── op_bnc_with_highs_torremocha.cpp  # B&C solver (Torremocha)
├── paper/
│   └── orienteering_paper.tex       # LaTeX paper
├── data/                            # Input data (not tracked in git)
│   ├── torremocha/
│   │   ├── Torremocha_tiff.tif      # Map image
│   │   ├── MDT02-WGS84-0730-1-COB2.tif  # DEM
│   │   └── torremocha_omap.omap     # Orienteering map (OMAP XML)
│   └── la_muela/
│       ├── La_muela_tiff.tif
│       ├── MDT02_candelario_merged.tif
│       └── LAMUELA.omap
└── .gitignore
```

## Setup

### Python dependencies
```bash
cd python
pip install -r requirements.txt
```

### C++ dependencies
- MSVC or GCC with C++20 support
- [HiGHS](https://highs.dev/) LP solver library

## Usage

### Full pipeline (preprocessing + optimization + JSON export)
```bash
cd python

# First run — preprocesses raw files into terrain cache:
python run_pipeline.py torremocha --preprocess

# Subsequent runs — uses cached .npz (faster):
python run_pipeline.py torremocha

# La Muela:
python run_pipeline.py la_muela --preprocess
```

### C++ Branch-and-Cut solver
```bash
# Compile (Windows/MSVC example):
cl.exe /O2 /std:c++20 /EHsc op_bnc_with_highs_torremocha.cpp ^
  /I<highs_include_path> /link /LIBPATH:<highs_lib_path> highs.lib ^
  /out:op_bnc_with_highs_torremocha.exe

# Run (from directory containing op_input_*.json files):
op_bnc_with_highs_torremocha.exe
```

## Study Areas

| | Torremocha | La Muela |
|---|---|---|
| Location | Extremadura, Spain | Salamanca, Spain |
| CRS | EPSG:25829 | EPSG:25830 |
| Grid | 1056 × 1463 | 1376 × 1622 |
| Resolution | 2.0 m | 2.0 m |
| Elevation | 447–501 m | 1164–1628 m |
| Relief | 55 m | 464 m |
| Cost asymmetry | 16.1% | 50.0% |
| HCR scaling (c) | 0.87 | 2.18 |
| Proven optimal | 5/7 instances | 5/7 instances |

## Key Results

Across 14 instances (7 per terrain), the Simulated Annealing heuristic matches
the Branch-and-Cut optimum on all 10 instances where optimality is proven
within the 15-minute time limit.

## Data

Input data files (TIF maps, DEMs, OMAP files) are included in the repository
under `data/torremocha/` and `data/la_muela/`, tracked via Git LFS for large files.

## Citation

See `paper/orienteering_paper.tex` for the full methodology and results.
