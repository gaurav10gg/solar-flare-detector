"""
multiday_eval.py — TRUE held-out-DAY forecast evaluation across all dates.

Loads every internally-consistent observation day (the loader's accuracy gate
auto-rejects mismatched days), builds features **per day** (so no rolling
statistic ever bleeds across a day boundary), concatenates the per-day feature
matrices, then:

  * trains XGBoost on the EARLIER day(s) and reports TSS / HSS / AUC on the
    held-out LAST day (the credible, generalisation-revealing split);
  * prints per-day flare counts (confirmed vs HXR candidates);
  * prints a single-day (within-day) baseline for comparison;
  * compares the windowed-Neupert and cross-correlation feature importances
    between the single-day and multi-day models.

Strictly time-ordered, no shuffling, no-peek forward labels. Nothing is
hardcoded to a date or flare.

Usage::

    python multiday_eval.py                 # auto: root *.zip + ./raw_data
    python multiday_eval.py raw_data .       # explicit source roots
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np
import pandas as pd

from loader import load_multi_day
from features import build_features
from nowcast import detect_soft, detect_hard, merge_catalog, catalog_summary
from forecast import (make_labels, make_feature_matrix, time_train_test_split,
                      train_forecaster, predict_proba_curve)
from evaluate import optimal_threshold, confusion, tss, hss, roc_curve_data

HORIZON_MIN = 30.0
MIN_CLASS = "C"
TOP_N = 6


def build_day(df_day: pd.DataFrame):
    """Features + catalogue + (X, y) for one day's clean frame."""
    feats = build_features(df_day.drop(columns=["day"], errors="ignore"))
    catalog = merge_catalog(detect_soft(feats), detect_hard(feats))
    labelled = make_labels(feats, catalog, horizon_min=HORIZON_MIN, min_class=MIN_CLASS)
    X, _ = make_feature_matrix(labelled)
    y = labelled.loc[X.index, "y_binary"]
    return feats, catalog, X, y


def fit_and_score(X: pd.DataFrame, y: pd.Series, split_mode: str, test_frac: float):
    """Time-ordered split -> train -> max-TSS threshold on train -> score on test."""
    X_tr, X_te, y_tr, y_te = time_train_test_split(
        X, y, test_frac=test_frac, split_mode=split_mode)
    train_days = sorted({str(d.date()) for d in X_tr.index.normalize().unique()})
    test_days = sorted({str(d.date()) for d in X_te.index.normalize().unique()})

    model = train_forecaster(X_tr, y_tr, mode="binary")
    prob_tr = predict_proba_curve(model, X_tr)
    thr = optimal_threshold(y_tr.to_numpy(), prob_tr.to_numpy())["threshold"]
    prob_te = predict_proba_curve(model, X_te)
    y_pred = (prob_te.to_numpy() >= thr).astype(int)
    cm = confusion(y_te.to_numpy(), y_pred)
    auc = roc_curve_data(y_te.to_numpy(), prob_te.to_numpy())["auc"]
    imp = dict(sorted(zip(model.feature_names,
                          [round(float(v), 4) for v in model.feature_importances_]),
                      key=lambda kv: -kv[1]))
    return {
        "TSS": round(tss(cm=cm), 4), "HSS": round(hss(cm=cm), 4),
        "AUC": round(float(auc), 4) if auc == auc else None,
        "threshold": thr, "confusion": cm, "importances": imp,
        "train_days": train_days, "test_days": test_days,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "pos_test": int(y_te.sum()),
    }


def main():
    roots = sys.argv[1:] if len(sys.argv) > 1 else (glob.glob("*.zip") + ["raw_data"])
    print(f"\nSOURCES: {roots}")
    combined, info = load_multi_day(roots, cadence="1s")

    print("\n" + "=" * 70)
    print("  MULTI-DAY INGEST")
    print("=" * 70)
    print(f"  accepted days : {info['days_accepted']}")
    if info["days_rejected"]:
        for r in info["days_rejected"]:
            print(f"  REJECTED {r['date']}: {r['reason']}")
    else:
        print("  rejected days : none")

    # ---- per-day features / catalogue / matrices ----
    per_day_X, per_day_y = {}, {}
    print("\n" + "=" * 70)
    print("  PER-DAY FLARE COUNTS")
    print("=" * 70)
    for date, g in combined.groupby("day"):
        feats, catalog, X, y = build_day(g)
        per_day_X[date], per_day_y[date] = X, y
        s = catalog_summary(catalog)
        print(f"  {date}: {s['headline']}  | labelled-positive samples={int(y.sum())}")

    days = sorted(per_day_X)
    n_days = len(days)
    if n_days < 2:
        print("\n!! Only one consistent day — cannot do a held-out-DAY split.")
        return

    X_all = pd.concat([per_day_X[d] for d in days]).sort_index()
    y_all = pd.concat([per_day_y[d] for d in days]).reindex(X_all.index)

    # ---- single-day baseline (earliest day, within-day split) ----
    base_day = days[0]
    base = fit_and_score(per_day_X[base_day], per_day_y[base_day],
                         split_mode="within_day", test_frac=0.3)

    # ---- multi-day held-out-DAY (hold out exactly the last day) ----
    test_frac = 1.0 / n_days  # -> ceil(test_frac*n_days) == 1 held-out day
    multi = fit_and_score(X_all, y_all, split_mode="by_day", test_frac=test_frac)

    # ---- report ----
    def block(title, r):
        print("\n" + "-" * 70)
        print(f"  {title}")
        print("-" * 70)
        print(f"  train days : {r['train_days']}  ({r['n_train']:,} samples)")
        print(f"  test  days : {r['test_days']}  ({r['n_test']:,} samples, "
              f"{r['pos_test']} positive)")
        print(f"  TSS={r['TSS']:+.4f}  HSS={r['HSS']:+.4f}  AUC={r['AUC']}  "
              f"thr={r['threshold']:.3f}")
        print(f"  confusion  : {r['confusion']}")
        print(f"  top-{TOP_N} importances:")
        for k, v in list(r["importances"].items())[:TOP_N]:
            print(f"      {k:24s} {v:.4f}")

    print("\n" + "=" * 70)
    print("  HELD-OUT EVALUATION")
    print("=" * 70)
    block(f"SINGLE-DAY baseline ({base_day}, within-day split)", base)
    block(f"MULTI-DAY ({n_days} days, TRUE held-out-day split)", multi)

    # ---- new-feature importance comparison ----
    print("\n" + "=" * 70)
    print("  NEW-FEATURE IMPORTANCE: single-day vs multi-day")
    print("=" * 70)
    for f in ("neupert_windowed", "neupert_residual", "hxr_sxr_lag", "hxr_sxr_xcorr"):
        print(f"  {f:20s} single={base['importances'].get(f, 0):.4f}  "
              f"multi={multi['importances'].get(f, 0):.4f}")

    out = {"ingest": info, "single_day": base, "multi_day": multi}
    with open("data/catalog/multiday_metrics.json", "w") as fh:
        import os
        os.makedirs("data/catalog", exist_ok=True)
        json.dump(out, fh, indent=2, default=str)
    print("\n[written] data/catalog/multiday_metrics.json")


if __name__ == "__main__":
    main()
