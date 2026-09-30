"""Project paths. Kept free of heavy imports so light tools (evaluate.py, plots.ipynb) load fast."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"
