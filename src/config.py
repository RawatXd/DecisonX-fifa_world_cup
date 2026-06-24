"""
src/config.py
-------------
Central settings for the entire project.
Every other file imports from here so you only
ever change a value in one place.

Usage in any file:
    from src.config import cfg
    print(cfg.DATA_RAW)
"""

from pathlib import Path


class Config:

    # ------------------------------------------------------------------
    # Folder paths
    # ROOT is the project root (the folder that contains src/)
    # ------------------------------------------------------------------
    ROOT       = Path(__file__).resolve().parent.parent
    DATA_RAW   = ROOT / "data" / "raw"
    DATA_PROC  = ROOT / "data" / "processed"
    OUTPUTS    = ROOT / "outputs"

    # ------------------------------------------------------------------
    # Elo rating settings
    # Starting rating given to every team before their first match
    # K controls how much a result moves the rating
    # ------------------------------------------------------------------
    ELO_START    = 1500   # all teams begin equal
    ELO_K_GROUP  = 30     # group stage: moderate update
    ELO_K_KO     = 50     # knockout stage: larger update (higher stakes)
    ELO_HOME_ADV = 75     # bonus points for the host nation

    # ------------------------------------------------------------------
    # Feature engineering
    # How many recent matches to look back when computing form
    # ------------------------------------------------------------------
    FORM_WINDOWS = (3, 5)   # last 3 matches and last 5 matches

    # ------------------------------------------------------------------
    # Model settings (used in Phase 2)
    # ------------------------------------------------------------------
    RANDOM_SEED = 42
    TEST_YEARS  = [2018, 2022]   # tournaments held out for evaluation

    # ------------------------------------------------------------------
    # Simulation settings (used in Phase 3)
    # ------------------------------------------------------------------
    N_SIMULATIONS = 10_000

    # ------------------------------------------------------------------
    # Stage name lookup (stage_code → readable label)
    # ------------------------------------------------------------------
    STAGES = {
        1: "Group stage",
        2: "Round of 16",
        3: "Quarter-finals",
        4: "Semi-finals",
        5: "Final",
    }


# Single instance imported everywhere
cfg = Config()
