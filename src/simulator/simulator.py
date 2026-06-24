"""
src/simulation/simulator.py
----------------------------
Phase 3 — Monte Carlo Tournament Simulator (FIXED VERSION)

Run with:
  python -m src.simulation.simulator
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from src.config import cfg
from src.models.poisson_model import simulate_match_score


GROUPS = {
    "A": ["Netherlands", "Senegal", "Ecuador", "Qatar"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}


def simulate_group_stage(strengths_df, avg_goals, rng):
    """Simulate group stage. Returns dict: {group: [winner, runner-up]}"""
    advanced = {}

    for group, teams in GROUPS.items():
        points = {team: 0 for team in teams}
        goals_for = {team: 0 for team in teams}
        goals_against = {team: 0 for team in teams}

        # All pairwise matchups
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home = teams[i]
                away = teams[j]

                home_goals, away_goals = simulate_match_score(
                    home, away, strengths_df, avg_goals
                )

                goals_for[home] += home_goals
                goals_for[away] += away_goals
                goals_against[home] += away_goals
                goals_against[away] += home_goals

                if home_goals > away_goals:
                    points[home] += 3
                elif home_goals < away_goals:
                    points[away] += 3
                else:
                    points[home] += 1
                    points[away] += 1

        # Rank by points, then goal difference, then goals for
        ranking = sorted(
            teams,
            key=lambda t: (
                -points[t],
                -(goals_for[t] - goals_against[t]),
                -goals_for[t]
            )
        )

        advanced[group] = ranking[:2]

    return advanced


def simulate_knockout(teams_list, strengths_df, avg_goals):
    """
    Simulate knockout stage. Takes 2^n teams, returns n winners.
    Pairs consecutive teams: [0 vs 1, 2 vs 3, ...]
    """
    winners = []

    for i in range(0, len(teams_list), 2):
        if i + 1 < len(teams_list):
            home = teams_list[i]
            away = teams_list[i + 1]

            home_goals, away_goals = simulate_match_score(
                home, away, strengths_df, avg_goals
            )

            winner = home if home_goals > away_goals else away
            winners.append(winner)

    return winners


def simulate_tournament(strengths_df, avg_goals, rng):
    """Simulate entire tournament. Returns {team: final_stage_reached}"""
    stage_reached = {}

    # Initialize
    for teams in GROUPS.values():
        for team in teams:
            stage_reached[team] = 1  # group stage

    # GROUP STAGE
    advanced_teams = simulate_group_stage(strengths_df, avg_goals, rng)
    r16_teams = []
    for group in sorted(advanced_teams.keys()):
        for team in advanced_teams[group]:
            r16_teams.append(team)
            stage_reached[team] = 2  # reached R16

    # ROUND OF 16 (16 → 8)
    qf_teams = simulate_knockout(r16_teams, strengths_df, avg_goals)
    for team in qf_teams:
        stage_reached[team] = 3  # reached QF

    # QUARTER-FINALS (8 → 4)
    sf_teams = simulate_knockout(qf_teams, strengths_df, avg_goals)
    for team in sf_teams:
        stage_reached[team] = 4  # reached SF

    # SEMI-FINALS (4 → 2, these are the finalists)
    final_teams = simulate_knockout(sf_teams, strengths_df, avg_goals)
    
    for team in final_teams:
        stage_reached[team] = 5  # reached final

    # FINAL — determine the champion
    if len(final_teams) == 2:
        home = final_teams[0]
        away = final_teams[1]
        home_goals, away_goals = simulate_match_score(
            home, away, strengths_df, avg_goals
        )
        champion = home if home_goals > away_goals else away
        stage_reached[champion] = 6  # won tournament

    return stage_reached


def run_simulations(strengths_df, avg_goals, n_sims=10_000):
    """Run tournament n_sims times and track results."""
    print(f"  Running {n_sims:,} tournament simulations ...\n")

    results = {}
    all_teams = set()
    for teams in GROUPS.values():
        all_teams.update(teams)

    for team in all_teams:
        results[team] = {
            "group": 0,
            "r16": 0,
            "qf": 0,
            "sf": 0,
            "final": 0,
            "winner": 0,
        }

    rng = np.random.RandomState(cfg.RANDOM_SEED)
    for sim_num in range(n_sims):
        if (sim_num + 1) % 2000 == 0:
            print(f"    Completed: {sim_num + 1:,} / {n_sims:,}")

        stage_reached = simulate_tournament(strengths_df, avg_goals, rng)

        for team, stage in stage_reached.items():
            if stage >= 1:
                results[team]["group"] += 1
            if stage >= 2:
                results[team]["r16"] += 1
            if stage >= 3:
                results[team]["qf"] += 1
            if stage >= 4:
                results[team]["sf"] += 1
            if stage >= 5:
                results[team]["final"] += 1
            if stage >= 6:
                results[team]["winner"] += 1

    # Convert to probabilities
    rows = []
    for team in sorted(results.keys()):
        counts = results[team]
        rows.append({
            "team": team,
            "p_group": counts["group"] / n_sims,
            "p_r16": counts["r16"] / n_sims,
            "p_qf": counts["qf"] / n_sims,
            "p_sf": counts["sf"] / n_sims,
            "p_final": counts["final"] / n_sims,
            "p_win": counts["winner"] / n_sims,
        })

    df = pd.DataFrame(rows).sort_values("p_win", ascending=False).reset_index(drop=True)
    return df


def save_and_display_results(df):
    """Save results and print summary."""
    cfg.OUTPUTS.mkdir(parents=True, exist_ok=True)
    out_path = cfg.OUTPUTS / "tournament_probs.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Results saved → outputs/tournament_probs.csv")

    print("\n" + "=" * 70)
    print("  TOURNAMENT PROBABILITIES (2022 World Cup Simulation)")
    print("=" * 70)
    print()
    print(f"  {'Team':<20} {'Win %':>10} {'Final %':>10} {'SF %':>10} {'QF %':>10}")
    print(f"  {'-'*70}")

    for _, row in df.iterrows():
        print(
            f"  {row['team']:<20} {row['p_win']:>9.1%} {row['p_final']:>10.1%} "
            f"{row['p_sf']:>9.1%} {row['p_qf']:>9.1%}"
        )

    print("\n" + "=" * 70)

    # Top 5 win
    print("\n  Top 5 teams most likely to WIN:")
    for idx, row in df.head(5).iterrows():
        print(f"    {idx+1}. {row['team']:<25} {row['p_win']:>7.2%}")

    # Top 5 final
    print("\n  Top 5 teams most likely to reach FINAL:")
    top_final = df.nlargest(5, "p_final")
    for idx, (_, row) in enumerate(top_final.iterrows(), 1):
        print(f"    {idx}. {row['team']:<25} {row['p_final']:>7.2%}")


def run_simulator():
    print("\n" + "=" * 70)
    print("  Phase 3 — Monte Carlo Tournament Simulator")
    print("=" * 70 + "\n")

    print("  Loading Poisson parameters ...")
    strengths_df = pd.read_csv(cfg.DATA_PROC / "poisson_params.csv")
    avg_goals = strengths_df["avg_goals_baseline"].iloc[0]
    print(f"    Teams in model: {len(strengths_df)}")
    print(f"    Baseline avg goals: {avg_goals:.4f}")
    print()

    results = run_simulations(strengths_df, avg_goals, n_sims=cfg.N_SIMULATIONS)
    save_and_display_results(results)

    print("\n" + "=" * 70 + "\n")

    return results


if __name__ == "__main__":
    run_simulator()