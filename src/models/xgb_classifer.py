"""
src/models/xgb_classifier.py
-----------------------------
Model B — XGBoost Match Outcome Classifier

What this model does:
---------------------
Takes the 30-column feature table from Phase 1 and learns to predict
whether the result will be a Home Win, Draw, or Away Win.

Why XGBoost?
------------
The relationship between Elo difference and match outcome is non-linear.
XGBoost builds many decision trees and combines them, which handles
non-linear patterns much better than logistic regression.

Training strategy:
------------------
We train on all matches up to 2014, then evaluate on 2018 and 2022.
This prevents data leakage — the model never sees future tournaments
during training, just like a real prediction scenario.

Run with:
  python -m src.models.xgb_classifier
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    confusion_matrix,
)
from xgboost import XGBClassifier
from src.config import cfg


# ===========================================================================
# STEP 1 — LOAD AND SPLIT DATA
# ===========================================================================

def load_and_split():
    """
    Load features.csv and split into train / test sets.

    Split strategy: TIME-BASED (not random)
    ----------------------------------------
    We use tournaments from 2018 and 2022 as the test set.
    Everything before 2018 is training data.

    Why not random split?
    Because football is time-series data. If you randomly split,
    a 2022 match might appear in training and its 2021 form data
    leaks into the model. Time-based split prevents this.
    """
    print("  Loading features ...")
    df = pd.read_csv(cfg.DATA_PROC / "features.csv", parse_dates=["match_date"])

    # The features our model actually uses (drop identifiers and target)
    feature_cols = [
        "home_elo_before", "away_elo_before", "elo_diff",
        "home_form_3", "away_form_3",
        "home_gf_3",   "away_gf_3",
        "home_ga_3",   "away_ga_3",
        "home_gd_3",   "away_gd_3",
        "home_form_5", "away_form_5",
        "home_gf_5",   "away_gf_5",
        "home_ga_5",   "away_ga_5",
        "home_gd_5",   "away_gd_5",
        "stage_code",
    ]

    # XGBoost needs integer class labels starting from 0
    # result_code: -1 (away win), 0 (draw), 1 (home win)
    # We map these to:   0 (away win), 1 (draw),  2 (home win)
    label_map     = {-1: 0, 0: 1, 1: 2}
    label_reverse = {0: "Away win", 1: "Draw", 2: "Home win"}

    df["target"] = df["result_code"].map(label_map)

    # Time-based split
    test_mask = df["match_date"].dt.year.isin(cfg.TEST_YEARS)
    train_df  = df[~test_mask].copy()
    test_df   = df[ test_mask].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test  = test_df[feature_cols]
    y_test  = test_df["target"]

    print(f"    Training matches : {len(X_train):,}  (up to 2014)")
    print(f"    Test matches     : {len(X_test):,}  (2018 + 2022)")
    print(f"    Features used    : {len(feature_cols)}")

    return X_train, y_train, X_test, y_test, feature_cols, label_reverse


# ===========================================================================
# STEP 2 — TRAIN XGBOOST
# ===========================================================================

def train_xgb(X_train, y_train):
    """
    Train the XGBoost classifier.

    Key hyperparameters explained:
    ------------------------------
    n_estimators    : number of trees to build (200 is a safe default)
    max_depth       : how deep each tree can go (3-5 prevents overfitting)
    learning_rate   : how much each tree corrects the previous one
    subsample       : fraction of training data used per tree (reduces overfitting)
    use_label_encoder: set False to suppress a deprecation warning
    eval_metric     : mlogloss = multi-class log loss (standard for 3-class problems)
    """
    print("\n  Training XGBoost classifier ...")

    model = XGBClassifier(
        n_estimators      = 200,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        random_state      = cfg.RANDOM_SEED,
        eval_metric       = "mlogloss",
        verbosity         = 0,
    )

    model.fit(X_train, y_train)
    print("    Training complete.")
    return model


# ===========================================================================
# STEP 3 — EVALUATE
# ===========================================================================

def evaluate_model(model, X_test, y_test, label_reverse):
    """
    Evaluate the trained model on the held-out test set.

    Metrics explained:
    ------------------
    Accuracy    : % of matches where we predicted the correct outcome
    Log loss    : penalises confident wrong predictions (lower is better)
    Confusion matrix : shows where the model gets confused
                       (e.g. predicts draw when actual result is home win)
    """
    print("\n  Evaluating on test set (2018 + 2022) ...")

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    ll  = log_loss(y_test, y_pred_prob)

    print(f"    Accuracy  : {acc:.1f}%")
    print(f"    Log loss  : {ll:.4f}")

    print("\n    Classification report:")
    target_names = [label_reverse[i] for i in sorted(label_reverse)]
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        digits=3
    )
    for line in report.split("\n"):
        print(f"      {line}")

    print("    Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"      {'':12} {'Away win':>10} {'Draw':>10} {'Home win':>10}")
    for i, row_name in enumerate(["Away win", "Draw", "Home win"]):
        print(f"      {row_name:<12} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")

    return acc, ll


# ===========================================================================
# STEP 4 — FEATURE IMPORTANCE
# ===========================================================================

def show_feature_importance(model, feature_cols):
    """
    Show which features matter most to the model.
    This is useful for understanding what drives match outcomes.
    """
    print("\n  Feature importance (top 10):")
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print(f"    {'Feature':<22} {'Importance':>12}")
    print(f"    {'-'*36}")
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp * 200)
        print(f"    {feat:<22} {imp:>8.4f}  {bar}")

    return importance


# ===========================================================================
# STEP 5 — SAVE MODEL
# ===========================================================================

def save_model(model, feature_cols, accuracy, log_loss_val):
    """Save the trained model and a text report."""
    models_dir = cfg.OUTPUTS / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save the model
    model_path = models_dir / "xgb_model.pkl"
    joblib.dump(model, model_path)

    # Save a plain-text report
    report_path = models_dir / "model_report.txt"
    with open(report_path, "w") as f:
        f.write("FIFA WC Optimizer — XGBoost Model Report\n")
        f.write("=" * 45 + "\n")
        f.write(f"Test years   : {cfg.TEST_YEARS}\n")
        f.write(f"Features     : {len(feature_cols)}\n")
        f.write(f"Accuracy     : {accuracy:.1f}%\n")
        f.write(f"Log loss     : {log_loss_val:.4f}\n")
        f.write("\nFeatures used:\n")
        for c in feature_cols:
            f.write(f"  {c}\n")

    print(f"\n  Model saved → outputs/models/xgb_model.pkl")
    print(f"  Report saved → outputs/models/model_report.txt")


# ===========================================================================
# MAIN
# ===========================================================================

def run_xgb_model():
    print("\n" + "=" * 52)
    print("  Phase 2B — XGBoost Classifier")
    print("=" * 52 + "\n")

    X_train, y_train, X_test, y_test, feature_cols, label_reverse = load_and_split()
    model      = train_xgb(X_train, y_train)
    acc, ll    = evaluate_model(model, X_test, y_test, label_reverse)
    importance = show_feature_importance(model, feature_cols)
    save_model(model, feature_cols, acc, ll)

    print("\n" + "=" * 52)
    print("  Phase 2B complete.")
    print("=" * 52 + "\n")

    return model, feature_cols


if __name__ == "__main__":
    run_xgb_model()