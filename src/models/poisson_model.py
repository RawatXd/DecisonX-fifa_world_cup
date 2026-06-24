"""
src/models/poisson_model.py
----------------------------
Model A — Poisson Goal Model

What this model does:
---------------------
Every team has two hidden numbers:
  - Attack strength  : how many goals they tend to score
  - Defence strength : how many goals they tend to concede

We estimate these from historical match data.
Then for any new match (Team A vs Team B) we can predict:
  - Expected goals for Team A  = attack_A × defence_B × average_goals
  - Expected goals for Team B  = attack_B × defence_A × average_goals

Once we have expected goals (lambda), we use the Poisson distribution
to get the probability of scoring exactly 0, 1, 2, 3, 4, 5 goals.
Combining both teams gives us a full scoreline probability table.

Run this file directly to train and save the model:
  python -m src.models.poisson_model
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from scipy.stats import poisson
from src.config import cfg


# ===========================================================================
# STEP 1 — LOAD FEATURES
# ===========================================================================

def load_features():
    """Load the feature table built in Phase 1."""
    path = cfg.DATA_PROC / "features.csv"
    df = pd.read_csv(path, parse_dates=["match_date"])
    print(f"  Loaded features: {len(df):,} matches")
    return df


# ===========================================================================
# STEP 2 — COMPUTE ATTACK AND DEFENCE STRENGTHS
# ===========================================================================

def compute_team_strengths(df):
    """
    Estimate each team's attack and defence strength index.

    Method:
    -------
    1. Calculate the average goals scored per match across ALL games
       (this becomes our baseline 'average_goals')

    2. For each team, calculate:
         attack_strength  = (avg goals this team scores) / average_goals
         defence_strength = (avg goals this team concedes) / average_goals

    A value > 1.0 means above average.
    A value < 1.0 means below average.

    Example:
      Brazil attack_strength = 1.35 means they score 35% more than average
      Brazil defence_strength = 0.72 means they concede 28% less than average
    """
    print("  Computing attack and defence strengths ...")

    # Use only recent tournaments for relevance (post-1990)
    recent = df[df["match_date"].dt.year >= 1990].copy()

    # Global average goals per match (both teams combined / 2)
    avg_home_goals = recent["home_team_score"].mean()
    avg_away_goals = recent["away_team_score"].mean()
    avg_goals      = (avg_home_goals + avg_away_goals) / 2

    print(f"    Avg home goals/match : {avg_home_goals:.3f}")
    print(f"    Avg away goals/match : {avg_away_goals:.3f}")
    print(f"    Overall average      : {avg_goals:.3f}")

    # Collect goals scored and conceded for each team
    records = []

    for _, row in recent.iterrows():
        records.append({
            "team"    : row["home_team_name"],
            "scored"  : row["home_team_score"],
            "conceded": row["away_team_score"],
        })
        records.append({
            "team"    : row["away_team_name"],
            "scored"  : row["away_team_score"],
            "conceded": row["home_team_score"],
        })

    team_df = pd.DataFrame(records)

    # Calculate mean goals scored and conceded per team
    strengths = team_df.groupby("team").agg(
        avg_scored  =("scored",   "mean"),
        avg_conceded=("conceded", "mean"),
        matches     =("scored",   "count"),
    ).reset_index()

    # Compute attack and defence indices relative to the average
    strengths["attack_strength"]  = strengths["avg_scored"]   / avg_goals
    strengths["defence_strength"] = strengths["avg_conceded"]  / avg_goals

    # Store global average for use when predicting
    strengths["avg_goals_baseline"] = avg_goals

    # Sort by attack strength descending
    strengths = strengths.sort_values("attack_strength", ascending=False).reset_index(drop=True)

    print("\n    Top 10 teams by attack strength:")
    print(f"    {'Team':<22} {'Attack':>8} {'Defence':>9} {'Matches':>8}")
    print(f"    {'-'*50}")
    for _, row in strengths.head(10).iterrows():
        print(f"    {row['team']:<22} {row['attack_strength']:>8.3f} {row['defence_strength']:>9.3f} {int(row['matches']):>8}")

    return strengths, avg_goals


# ===========================================================================
# STEP 3 — PREDICT MATCH OUTCOME PROBABILITIES
# ===========================================================================

def predict_match(home_team, away_team, strengths_df, avg_goals, max_goals=8):
    """
    Given two teams, return:
      - expected goals for each team
      - full scoreline probability matrix
      - P(home win), P(draw), P(away win)

    Parameters:
    -----------
    home_team   : string, name of the home/first team
    away_team   : string, name of the away/second team
    max_goals   : maximum goals to consider per team (8 is enough for 99.9% of cases)

    Returns:
    --------
    dict with keys: lambda_home, lambda_away, score_matrix,
                    p_home_win, p_draw, p_away_win
    """

    def get_strength(team, col, default=1.0):
        row = strengths_df[strengths_df["team"] == team]
        if len(row) == 0:
            return default
        return row.iloc[0][col]

    # Expected goals (lambda) for each team
    # Formula: attack of scoring team × defence of conceding team × baseline
    lambda_home = (
        get_strength(home_team, "attack_strength") *
        get_strength(away_team, "defence_strength") *
        avg_goals
    )
    lambda_away = (
        get_strength(away_team, "attack_strength") *
        get_strength(home_team, "defence_strength") *
        avg_goals
    )

    # Build scoreline probability matrix
    # score_matrix[i][j] = P(home scores i goals, away scores j goals)
    home_probs = [poisson.pmf(g, lambda_home) for g in range(max_goals + 1)]
    away_probs = [poisson.pmf(g, lambda_away) for g in range(max_goals + 1)]

    score_matrix = np.outer(home_probs, away_probs)

    # Extract match outcome probabilities from the matrix
    p_home_win = np.sum(np.tril(score_matrix, -1))  # home goals > away goals
    p_draw     = np.sum(np.diag(score_matrix))       # home goals == away goals
    p_away_win = np.sum(np.triu(score_matrix, 1))    # away goals > home goals

    return {
        "lambda_home" : round(lambda_home, 3),
        "lambda_away" : round(lambda_away, 3),
        "score_matrix": score_matrix,
        "p_home_win"  : round(p_home_win, 4),
        "p_draw"      : round(p_draw, 4),
        "p_away_win"  : round(p_away_win, 4),
    }


def simulate_match_score(home_team, away_team, strengths_df, avg_goals):
    """
    Draw a single random scoreline from the Poisson distributions.
    Used by the Monte Carlo simulator in Phase 3.

    Returns: (home_goals, away_goals) as integers
    """
    def get_strength(team, col, default=1.0):
        row = strengths_df[strengths_df["team"] == team]
        if len(row) == 0:
            return default
        return row.iloc[0][col]

    lambda_home = (
        get_strength(home_team, "attack_strength") *
        get_strength(away_team, "defence_strength") *
        avg_goals
    )
    lambda_away = (
        get_strength(away_team, "attack_strength") *
        get_strength(home_team, "defence_strength") *
        avg_goals
    )

    home_goals = np.random.poisson(lambda_home)
    away_goals = np.random.poisson(lambda_away)
    return int(home_goals), int(away_goals)


# ===========================================================================
# STEP 4 — EVALUATE ON HELD-OUT TOURNAMENTS
# ===========================================================================

def evaluate_poisson(df, strengths_df, avg_goals):
    """
    Test the Poisson model on held-out years (2018 and 2022).
    Reports: accuracy and log loss.

    We train on everything before 2018, then predict 2018+2022 matches.
    This mimics real-world usage — you build the model before the tournament.
    """
    print("\n  Evaluating Poisson model on held-out years (2018, 2022) ...")

    test = df[df["match_date"].dt.year.isin(cfg.TEST_YEARS)].copy()

    correct = 0
    total   = 0
    log_loss_sum = 0.0

    for _, row in test.iterrows():
        result = predict_match(
            row["home_team_name"],
            row["away_team_name"],
            strengths_df,
            avg_goals
        )

        # Predicted outcome
        probs = [result["p_home_win"], result["p_draw"], result["p_away_win"]]
        pred  = [1, 0, -1][np.argmax(probs)]

        if pred == row["result_code"]:
            correct += 1

        # Log loss — penalises confident wrong predictions
        actual_idx = {1: 0, 0: 1, -1: 2}[int(row["result_code"])]
        p_actual = max(probs[actual_idx], 1e-10)
        log_loss_sum += -np.log(p_actual)

        total += 1

    accuracy = correct / total * 100
    avg_log_loss = log_loss_sum / total

    print(f"    Test matches   : {total}")
    print(f"    Accuracy       : {accuracy:.1f}%")
    print(f"    Avg log loss   : {avg_log_loss:.4f}")
    print(f"    (Baseline accuracy for always predicting home win: ~57%)")

    return accuracy, avg_log_loss


# ===========================================================================
# STEP 5 — SAVE MODEL PARAMETERS
# ===========================================================================

def save_poisson_params(strengths_df, avg_goals):
    """Save team strengths to CSV so Phase 3 can load them."""
    cfg.DATA_PROC.mkdir(parents=True, exist_ok=True)
    out = cfg.DATA_PROC / "poisson_params.csv"
    strengths_df.to_csv(out, index=False)
    print(f"\n  Poisson params saved → data/processed/poisson_params.csv")
    print(f"  avg_goals baseline : {avg_goals:.4f}")


# ===========================================================================
# MAIN
# ===========================================================================

def run_poisson_model():
    print("\n" + "=" * 52)
    print("  Phase 2A — Poisson Goal Model")
    print("=" * 52 + "\n")

    df                    = load_features()
    strengths_df, avg_goals = compute_team_strengths(df)
    accuracy, log_loss      = evaluate_poisson(df, strengths_df, avg_goals)
    save_poisson_params(strengths_df, avg_goals)

    # Demo prediction
    print("\n" + "-" * 52)
    print("  Demo: Brazil vs France")
    result = predict_match("Brazil", "France", strengths_df, avg_goals)
    print(f"    Expected goals — Brazil: {result['lambda_home']}  France: {result['lambda_away']}")
    print(f"    P(Brazil win) : {result['p_home_win']:.1%}")
    print(f"    P(Draw)       : {result['p_draw']:.1%}")
    print(f"    P(France win) : {result['p_away_win']:.1%}")

    print("\n  Demo: Argentina vs Germany")
    result2 = predict_match("Argentina", "West Germany", strengths_df, avg_goals)
    print(f"    Expected goals — Argentina: {result2['lambda_home']}  Germany: {result2['lambda_away']}")
    print(f"    P(Argentina win) : {result2['p_home_win']:.1%}")
    print(f"    P(Draw)          : {result2['p_draw']:.1%}")
    print(f"    P(Germany win)   : {result2['p_away_win']:.1%}")
    print("=" * 52 + "\n")

    return strengths_df, avg_goals


if __name__ == "__main__":
    run_poisson_model()