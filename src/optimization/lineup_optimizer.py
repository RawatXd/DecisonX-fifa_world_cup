"""
src/optimization/lineup_optimizer.py
-------------------------------------
Phase 4 — Lineup Optimizer (Integer Programming)

What this script does:
  1. Loads a team's squad of 23 players
  2. Formulates an integer program: select 11 starters that maximize
     the team's tournament win probability
  3. Solves it using PuLP + CBC solver
  4. Runs sensitivity analysis: for each player in the optimal lineup,
     remove them and re-solve to measure their impact

Run with:
  python -m src.optimization.lineup_optimizer
  
Or with a specific team:
  python -m src.optimization.lineup_optimizer --team brazil
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import argparse
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD
from src.config import cfg


# ===========================================================================
# STEP 1 — LOAD SQUAD AND TOURNAMENT PROBS
# ===========================================================================

def load_squad(team_name):
    """Load the squad CSV for a team."""
    path = cfg.OUTPUTS.parent / "data" / "squads" / f"{team_name.lower()}.csv"
    
    if not path.exists():
        raise FileNotFoundError(f"Squad file not found: {path}")
    
    squad = pd.read_csv(path)
    squad["injured"] = squad["injured"].astype(bool)
    
    print(f"  Loaded {len(squad)} players for {team_name}")
    print(f"    GK:  {len(squad[squad['position']=='GK'])}")
    print(f"    DEF: {len(squad[squad['position']=='DEF'])}")
    print(f"    MID: {len(squad[squad['position']=='MID'])}")
    print(f"    FWD: {len(squad[squad['position']=='FWD'])}")
    
    return squad


def load_tournament_probs(team_name):
    """Load the tournament win probability for a team."""
    path = cfg.OUTPUTS / "tournament_probs.csv"
    df = pd.read_csv(path)
    
    row = df[df["team"] == team_name]
    if len(row) == 0:
        raise ValueError(f"Team {team_name} not found in tournament_probs.csv")
    
    base_win_prob = row.iloc[0]["p_win"]
    print(f"  Base win probability for {team_name}: {base_win_prob:.2%}")
    
    return base_win_prob


# ===========================================================================
# STEP 2 — FORMULATE OPTIMIZATION PROBLEM
# ===========================================================================

def compute_player_strength(row):
    """
    Estimate a player's contribution to win probability.
    
    Simple heuristic:
      - Higher market value = stronger player
      - Defenders reduce opponent goals (defence strength)
      - Forwards increase own goals (attack strength)
    
    This is a simplified model. In a real system, you'd use
    xG data, defensive actions, etc.
    """
    mv = row["market_value_millions"]
    pos = row["position"]
    
    # Base contribution from market value (normalized)
    base = mv / 100.0  # scale to 0-1 range roughly
    
    # Position bonus
    if pos == "FWD":
        return base * 1.2  # forwards have high impact
    elif pos == "MID":
        return base * 1.0
    elif pos == "DEF":
        return base * 0.9
    else:  # GK
        return base * 0.8
    
    return base


def formulate_problem(squad, base_win_prob):
    """
    Formulate the integer programming problem.
    
    Maximize: weighted sum of player strengths
    Subject to:
      - Exactly 11 starters
      - Exactly 1 GK
      - 3-5 DEF
      - 3-4 MID
      - 1-3 FWD
      - At most 5 from same club
      - No injured players
    """
    
    # Create the problem
    prob = LpProblem("Lineup_Optimizer", LpMaximize)
    
    # Decision variables: binary for each player
    player_vars = {}
    for idx, row in squad.iterrows():
        var = LpVariable(f"player_{idx}", cat="Binary")
        player_vars[idx] = var
    
    # Compute player strengths
    strengths = []
    for idx, row in squad.iterrows():
        if row["injured"]:
            strength = 0  # injured players can't play
        else:
            strength = compute_player_strength(row)
        strengths.append(strength)
    
    # Objective: maximize sum of selected player strengths
    prob += lpSum([strengths[idx] * player_vars[idx] for idx in range(len(squad))]), "Total_Strength"
    
    # Constraint 1: Exactly 11 starters
    prob += lpSum([player_vars[idx] for idx in range(len(squad))]) == 11, "Total_Players"
    
    # Constraint 2: Exactly 1 goalkeeper
    gk_indices = squad[squad["position"] == "GK"].index.tolist()
    prob += lpSum([player_vars[idx] for idx in gk_indices]) == 1, "Goalkeeper"
    
    # Constraint 3: 3-5 defenders
    def_indices = squad[squad["position"] == "DEF"].index.tolist()
    prob += lpSum([player_vars[idx] for idx in def_indices]) >= 3, "Defenders_Min"
    prob += lpSum([player_vars[idx] for idx in def_indices]) <= 5, "Defenders_Max"
    
    # Constraint 4: 3-4 midfielders
    mid_indices = squad[squad["position"] == "MID"].index.tolist()
    prob += lpSum([player_vars[idx] for idx in mid_indices]) >= 3, "Midfielders_Min"
    prob += lpSum([player_vars[idx] for idx in mid_indices]) <= 4, "Midfielders_Max"
    
    # Constraint 5: 1-3 forwards
    fwd_indices = squad[squad["position"] == "FWD"].index.tolist()
    prob += lpSum([player_vars[idx] for idx in fwd_indices]) >= 1, "Forwards_Min"
    prob += lpSum([player_vars[idx] for idx in fwd_indices]) <= 3, "Forwards_Max"
    
    # Constraint 6: At most 5 from the same club
    clubs = squad["club"].unique()
    for club in clubs:
        club_indices = squad[squad["club"] == club].index.tolist()
        if len(club_indices) > 5:
            prob += lpSum([player_vars[idx] for idx in club_indices]) <= 5, f"Club_{club}"
    
    return prob, player_vars


# ===========================================================================
# STEP 3 — SOLVE AND EXTRACT SOLUTION
# ===========================================================================

def solve_and_extract(prob, player_vars, squad):
    """Solve the problem and return the optimal lineup."""
    
    # Solve using CBC solver
    prob.solve(PULP_CBC_CMD(msg=0))
    
    # Extract solution
    selected_indices = [idx for idx, var in player_vars.items() if var.value() == 1]
    selected_squad = squad.iloc[selected_indices].copy()
    
    return selected_squad


def print_lineup(selected_squad):
    """Pretty print the selected lineup."""
    print("\n" + "─" * 60)
    print("  OPTIMAL LINEUP (11 starters)")
    print("─" * 60)
    
    for pos in ["GK", "DEF", "MID", "FWD"]:
        players = selected_squad[selected_squad["position"] == pos]
        if len(players) > 0:
            print(f"\n  {pos}:")
            for _, row in players.iterrows():
                print(f"    • {row['player_name']:<25} ({row['club']})")


# ===========================================================================
# STEP 4 — SENSITIVITY ANALYSIS
# ===========================================================================

def sensitivity_analysis(selected_squad, squad, base_win_prob):
    """
    For each player in the optimal lineup, remove them and re-solve.
    Measure the drop in objective value (proxy for impact on win prob).
    """
    print("\n" + "─" * 60)
    print("  SENSITIVITY ANALYSIS")
    print("  (Impact of removing each player)")
    print("─" * 60)
    
    base_value = None
    impacts = []
    
    for drop_idx, drop_row in selected_squad.iterrows():
        # Create a modified squad without this player
        modified_squad = squad.drop(drop_idx).reset_index(drop=True)
        
        # Re-index because we dropped a row
        modified_squad_copy = modified_squad.copy()
        prob_modified, vars_modified = formulate_problem(modified_squad_copy, base_win_prob)
        selected_modified = solve_and_extract(prob_modified, vars_modified, modified_squad_copy)
        
        # Compute new objective value
        strengths_mod = [
            compute_player_strength(modified_squad_copy.iloc[i])
            if i not in [j for j, p in enumerate(modified_squad_copy.itertuples()) if p.injured]
            else 0
            for i in range(len(modified_squad_copy))
        ]
        
        new_value = sum([
            strengths_mod[i] 
            for i in selected_modified.index if i < len(strengths_mod)
        ])
        
        if base_value is None:
            # First iteration: compute base value
            prob_base, vars_base = formulate_problem(squad, base_win_prob)
            selected_base = solve_and_extract(prob_base, vars_base, squad)
            strengths_base = [compute_player_strength(squad.iloc[i]) for i in range(len(squad))]
            base_value = sum([strengths_base[i] for i in selected_base.index])
        
        impact = base_value - new_value
        impacts.append({
            "player": drop_row["player_name"],
            "position": drop_row["position"],
            "club": drop_row["club"],
            "impact": impact,
        })
    
    # Sort by impact descending
    impacts_df = pd.DataFrame(impacts).sort_values("impact", ascending=False)
    
    print(f"\n  {'Player':<25} {'Position':>10} {'Impact':>12}")
    print(f"  {'-'*50}")
    for _, row in impacts_df.iterrows():
        print(f"  {row['player']:<25} {row['position']:>10} {row['impact']:>11.3f}")
    
    return impacts_df


# ===========================================================================
# MAIN
# ===========================================================================

def run_optimizer(team_name="Argentina"):
    print("\n" + "=" * 60)
    print(f"  Phase 4 — Lineup Optimizer ({team_name})")
    print("=" * 60 + "\n")
    
    # Load inputs
    print("  Loading squad and tournament probabilities ...")
    squad = load_squad(team_name)
    base_win_prob = load_tournament_probs(team_name)
    print()
    
    # Formulate and solve
    print("  Formulating integer programming problem ...")
    prob, player_vars = formulate_problem(squad, base_win_prob)
    
    print("  Solving ...")
    selected_squad = solve_and_extract(prob, player_vars, squad)
    print_lineup(selected_squad)
    
    # Sensitivity analysis
    print("\n  Running sensitivity analysis ...")
    impacts = sensitivity_analysis(selected_squad, squad, base_win_prob)
    
    print("\n" + "=" * 60 + "\n")
    
    return selected_squad, impacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="Argentina", help="Team name (e.g., Argentina, Brazil, France)")
    args = parser.parse_args()
    
    run_optimizer(args.team)