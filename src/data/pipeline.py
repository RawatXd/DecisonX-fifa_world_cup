"""
src/data/pipeline.py
---------------------
Phase 1 — Data pipeline.

What this script does, in order:
  1. Loads the 4 raw CSV files from data/raw/
  2. Filters to men's World Cup matches only
  3. Computes Elo ratings for every team across all tournaments
  4. Computes rolling form features (last 3 and last 5 matches)
  5. Saves the final feature table to data/processed/features.csv

Run with (from project root):
  python -m src.data.pipeline
"""

import sys
from pathlib import Path

# Make sure Python can find the src/ folder when running from root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from src.config import cfg


# ===========================================================================
# STEP 1 — LOAD RAW DATA
# ===========================================================================

def load_raw():
    """
    Read the four CSV files that download_data.py saved.
    Returns four DataFrames: matches, goals, teams, tournaments.

    parse_dates turns the date column into a proper datetime object
    so we can sort by it and do date arithmetic later.
    """
    print("  Loading raw CSV files ...")

    matches = pd.read_csv(
        cfg.DATA_RAW / "results.csv",
        parse_dates=["match_date"]
    )
    goals = pd.read_csv(
        cfg.DATA_RAW / "goals.csv",
        parse_dates=["match_date"]
    )
    teams       = pd.read_csv(cfg.DATA_RAW / "teams.csv")
    tournaments = pd.read_csv(cfg.DATA_RAW / "tournaments.csv")

    print(f"    Matches loaded   : {len(matches):,}")
    print(f"    Goals loaded     : {len(goals):,}")
    print(f"    Teams loaded     : {len(teams):,}")
    print(f"    Tournaments      : {len(tournaments):,}")

    return matches, goals, teams, tournaments


# ===========================================================================
# STEP 2 — CLEAN AND FILTER
# ===========================================================================

def clean_matches(matches, tournaments):
    """
    Keep only men's World Cup matches and add useful columns.

    Why filter? The raw data includes women's tournaments too.
    We only want the men's competition for this project.

    New columns added:
      result_code  — 1 = home win, 0 = draw, -1 = away win
      stage_code   — 1 = group stage, up to 5 = final
      went_to_et   — True if the match went to extra time
      went_to_pens — True if the match went to penalties
      year         — which World Cup year this match belongs to
    """
    print("  Filtering to men's World Cup matches ...")

    # Get the tournament IDs for men's competitions only
    mens_ids = tournaments.loc[
        tournaments["tournament_name"].str.contains("Men", na=False),
        "tournament_id"
    ]

    # Keep only rows where tournament_id is in that list
    df = matches[matches["tournament_id"].isin(mens_ids)].copy()

    # Attach the tournament year so we can group by year later
    df = df.merge(
        tournaments[["tournament_id", "year"]],
        on="tournament_id",
        how="left"
    )

    # Convert the text result into a number
    result_map = {
        "home team win" : 1,
        "draw"          : 0,
        "away team win" : -1,
    }
    df["result_code"] = df["result"].map(result_map)

    # Encode the tournament stage as a number (1 = earliest, 5 = final)
    def encode_stage(stage_name):
        s = str(stage_name).lower()
        if "final" in s and "semi" not in s and "quarter" not in s and "third" not in s:
            return 5
        elif "semi" in s:
            return 4
        elif "third" in s:
            return 4
        elif "quarter" in s:
            return 3
        elif "round of 16" in s or "second round" in s:
            return 2
        else:
            return 1   # group stage

    df["stage_code"]   = df["stage_name"].apply(encode_stage)
    df["went_to_et"]   = df["extra_time"].astype(bool)
    df["went_to_pens"] = df["penalty_shootout"].astype(bool)

    # Sort chronologically — this is critical for Elo calculation
    df = df.sort_values("match_date").reset_index(drop=True)

    print(f"    Men's WC matches : {len(df):,}")
    print(f"    Year range       : {df['year'].min()} – {df['year'].max()}")
    print(f"    Teams            : {df['home_team_name'].nunique()} unique")

    return df


# ===========================================================================
# STEP 3 — ELO RATINGS
# ===========================================================================

def build_elo_ratings(df):
    """
    Compute Elo ratings for every team by replaying all matches in order.

    What is Elo?
    ------------
    Elo is a number representing team strength (starts at 1500 for everyone).
    After each match:
      - The winner gains points
      - The loser loses the same points
      - The amount gained/lost depends on how surprising the result was
        (beating a much stronger team = big gain, beating a weaker team = small gain)

    This function adds four new columns to df:
      home_elo_before  — home team's Elo rating BEFORE this match
      away_elo_before  — away team's Elo rating BEFORE this match
      elo_diff         — home_elo_before minus away_elo_before
      home_elo_after   — home team's Elo rating AFTER this match
      away_elo_after   — away team's Elo rating AFTER this match

    We use "before" ratings as features in Phase 2 because at prediction
    time you only know the rating BEFORE the match, not after.
    """
    print("  Computing Elo ratings ...")

    # Dictionary holding the current Elo for each team
    # Every team starts at the same value (1500)
    ratings = {}

    home_elo_before_list = []
    away_elo_before_list = []
    home_elo_after_list  = []
    away_elo_after_list  = []

    for _, row in df.iterrows():
        home_team = row["home_team_name"]
        away_team = row["away_team_name"]

        # Get current ratings (default to starting value if first match)
        r_home = ratings.get(home_team, cfg.ELO_START)
        r_away = ratings.get(away_team, cfg.ELO_START)

        # Record the BEFORE ratings (these become features)
        home_elo_before_list.append(r_home)
        away_elo_before_list.append(r_away)

        # -----------------------------------------------------------
        # Calculate expected result using the Elo formula
        # If elo_diff is large and positive, home team is expected to win
        # -----------------------------------------------------------
        expected_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))

        # Convert result_code to a score from home team's perspective
        # 1 = home win, 0.5 = draw, 0 = away win
        if row["result_code"] == 1:
            actual_home = 1.0
        elif row["result_code"] == 0:
            actual_home = 0.5
        else:
            actual_home = 0.0

        # K controls how much ratings shift per match
        # Knockout matches shift ratings more (higher stakes)
        k = cfg.ELO_K_KO if row["stage_code"] > 1 else cfg.ELO_K_GROUP

        # Update ratings
        delta = k * (actual_home - expected_home)
        r_home_new = r_home + delta
        r_away_new = r_away - delta

        # Save updated ratings back to the dictionary
        ratings[home_team] = r_home_new
        ratings[away_team] = r_away_new

        home_elo_after_list.append(r_home_new)
        away_elo_after_list.append(r_away_new)

    # Attach all four columns to the dataframe
    df = df.copy()
    df["home_elo_before"] = home_elo_before_list
    df["away_elo_before"] = away_elo_before_list
    df["home_elo_after"]  = home_elo_after_list
    df["away_elo_after"]  = away_elo_after_list
    df["elo_diff"]        = df["home_elo_before"] - df["away_elo_before"]

    # Show the top 5 teams by final Elo rating
    top5 = sorted(ratings.items(), key=lambda x: -x[1])[:5]
    print("    Top 5 Elo ratings after 2022 WC:")
    for team, rating in top5:
        print(f"      {team:<20} {rating:.0f}")

    # Save the final Elo snapshot (useful for Phase 3 simulator)
    cfg.DATA_PROC.mkdir(parents=True, exist_ok=True)
    elo_snapshot = pd.DataFrame(
        list(ratings.items()),
        columns=["team_name", "elo_rating"]
    ).sort_values("elo_rating", ascending=False).reset_index(drop=True)
    elo_snapshot.to_csv(cfg.DATA_PROC / "elo_snapshot.csv", index=False)
    print(f"    Elo snapshot saved → data/processed/elo_snapshot.csv")

    return df, ratings


# ===========================================================================
# STEP 4 — ROLLING FORM FEATURES
# ===========================================================================

def build_form_features(df):
    """
    For each match, compute how well each team has been performing
    in their recent World Cup matches (last 3 and last 5 games).

    Why do we need this?
    --------------------
    A team's Elo rating tells you their long-term strength.
    Form features tell you their short-term momentum going into a match.
    Both signals together give the model a richer picture.

    Features added for each window (3 and 5):
      home_form_{n}  — average points per match (3=win, 1=draw, 0=loss)
      away_form_{n}
      home_gf_{n}    — average goals scored
      away_gf_{n}
      home_ga_{n}    — average goals conceded
      away_ga_{n}
      home_gd_{n}    — average goal difference (scored minus conceded)
      away_gd_{n}
    """
    print("  Building rolling form features ...")

    # --- Build a long-format record: one row per team per match ---
    records = []
    for _, row in df.iterrows():
        for side, other in [("home", "away"), ("away", "home")]:
            goals_for     = row[f"{side}_team_score"]
            goals_against = row[f"{other}_team_score"]

            if row["result_code"] == (1 if side == "home" else -1):
                points = 3    # win
            elif row["result_code"] == 0:
                points = 1    # draw
            else:
                points = 0    # loss

            records.append({
                "match_id"   : row["match_id"],
                "match_date" : row["match_date"],
                "team"       : row[f"{side}_team_name"],
                "side"       : side,
                "goals_for"  : goals_for,
                "goals_against": goals_against,
                "goal_diff"  : goals_for - goals_against,
                "points"     : points,
            })

    team_hist = pd.DataFrame(records).sort_values("match_date").reset_index(drop=True)

    # --- For each team and window, compute the rolling averages ---
    # shift(1) means: look at the PREVIOUS matches only, not the current one
    # (we can't use today's result as a feature for today's prediction)

    # Store results in a dictionary: (team, side, window) → Series indexed by match_id
    form_lookup = {}

    for team, grp in team_hist.groupby("team"):
        grp = grp.sort_values("match_date").copy()
        for n in cfg.FORM_WINDOWS:
            shifted = grp[["points", "goals_for", "goals_against", "goal_diff"]].shift(1)
            rolled  = shifted.rolling(window=n, min_periods=1).mean()
            grp[f"form_{n}"] = rolled["points"]
            grp[f"gf_{n}"]   = rolled["goals_for"]
            grp[f"ga_{n}"]   = rolled["goals_against"]
            grp[f"gd_{n}"]   = rolled["goal_diff"]

        for side in ["home", "away"]:
            sub = grp[grp["side"] == side].set_index("match_id")
            for n in cfg.FORM_WINDOWS:
                for stat in ["form", "gf", "ga", "gd"]:
                    form_lookup[(team, side, n, stat)] = sub[f"{stat}_{n}"]

    # --- Merge form features back onto the main match dataframe ---
    for n in cfg.FORM_WINDOWS:
        for stat in ["form", "gf", "ga", "gd"]:
            df[f"home_{stat}_{n}"] = np.nan
            df[f"away_{stat}_{n}"] = np.nan

    for idx, row in df.iterrows():
        for side in ["home", "away"]:
            team = row[f"{side}_team_name"]
            mid  = row["match_id"]
            for n in cfg.FORM_WINDOWS:
                for stat in ["form", "gf", "ga", "gd"]:
                    key = (team, side, n, stat)
                    if key in form_lookup and mid in form_lookup[key].index:
                        df.at[idx, f"{side}_{stat}_{n}"] = form_lookup[key][mid]

    print(f"    Form features added for windows: {cfg.FORM_WINDOWS}")
    return df


# ===========================================================================
# STEP 5 — ASSEMBLE AND SAVE FEATURE TABLE
# ===========================================================================

def save_features(df):
    """
    Select only the columns we need for Phase 2 (the ML model),
    fill any remaining missing values with 0, and save to CSV.

    Why fill with 0?
    ----------------
    Early matches have no prior form data (a team's first match
    has no previous 5 games to average). We fill these with 0,
    which is equivalent to saying "unknown / no history."
    """
    print("  Assembling final feature table ...")

    feature_columns = [
        # Identifiers (not used for training, but useful for debugging)
        "match_id", "match_date", "year",
        "home_team_name", "away_team_name",
        "home_team_score", "away_team_score",

        # TARGET — this is what the model learns to predict
        "result_code",

        # Elo features
        "home_elo_before", "away_elo_before", "elo_diff",

        # Form over last 3 matches
        "home_form_3", "away_form_3",
        "home_gf_3",   "away_gf_3",
        "home_ga_3",   "away_ga_3",
        "home_gd_3",   "away_gd_3",

        # Form over last 5 matches
        "home_form_5", "away_form_5",
        "home_gf_5",   "away_gf_5",
        "home_ga_5",   "away_ga_5",
        "home_gd_5",   "away_gd_5",

        # Match context
        "stage_code", "went_to_et", "went_to_pens",
    ]

    # Keep only columns that actually exist in df
    available = [c for c in feature_columns if c in df.columns]
    features  = df[available].copy()

    # Fill missing form values (early matches) with 0
    features = features.fillna(0)

    # Remove any duplicate rows
    features = features.drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    # Save
    cfg.DATA_PROC.mkdir(parents=True, exist_ok=True)
    out_path = cfg.DATA_PROC / "features.csv"
    features.to_csv(out_path, index=False)

    print(f"    Rows            : {len(features):,}")
    print(f"    Columns         : {len(features.columns)}")
    print(f"    Null values     : {features.isnull().sum().sum()}")
    print(f"    Saved to        : data/processed/features.csv")

    return features


# ===========================================================================
# MAIN — runs all steps in order
# ===========================================================================

def run_pipeline():
    print("\n" + "=" * 50)
    print("  FIFA WC Optimizer — Phase 1: Data Pipeline")
    print("=" * 50 + "\n")

    # Step 1 — Load
    matches, goals, teams, tournaments = load_raw()
    print()

    # Step 2 — Clean
    df = clean_matches(matches, tournaments)
    print()

    # Step 3 — Elo
    df, final_ratings = build_elo_ratings(df)
    print()

    # Step 4 — Form features
    df = build_form_features(df)
    print()

    # Step 5 — Save
    features = save_features(df)

    # Final summary
    print()
    print("=" * 50)
    print("  Phase 1 complete. Files saved:")
    print("    data/processed/features.csv")
    print("    data/processed/elo_snapshot.csv")
    print()
    print("  Result distribution (what the model will predict):")
    counts = features["result_code"].value_counts().sort_index()
    labels = {1: "Home win", 0: "Draw", -1: "Away win"}
    for code, count in counts.items():
        pct = count / len(features) * 100
        print(f"    {labels[code]:<12} : {count:>3}  ({pct:.1f}%)")
    print("=" * 50 + "\n")

    return features, final_ratings


if __name__ == "__main__":
    features, ratings = run_pipeline()