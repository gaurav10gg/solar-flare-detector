"""
multiday_eval.py — leave-one-day-out (LODO) cross-validation across all days.

The credible, generalisation-revealing evaluation for a multi-day dataset:

  1. Load every internally-consistent day (loader accuracy gate auto-rejects
     instrument-date mismatches), build features **per day** (no rolling
     statistic ever crosses a day boundary), and cache the per-day matrices.
  2. Empirically fit the Neupert constant ``K_NEUPERT`` (dF_SXR/dt ~= K*F_HXR)
     on pooled active samples, and recompute ``neupert_residual`` with it.
  3. **Leave-one-day-out CV**: for each day, train XGBoost on ALL other days,
     pick the max-TSS threshold on the training data only, and score the
     held-out day. Report TSS/HSS/AUC as mean +/- std across folds plus a
     pooled confusion matrix, and aggregate event-level recall / lead / FAR.
  4. A chronological headline split (earliest ~80% of days train, last ~20%
     test) for a single, time-honest headline number.

Strictly time-ordered, no shuffling, no-peek forward labels. Nothing hardcoded
to a date or flare.

Usage::

    python multiday_eval.py                  # auto sources: root *.zip + ./raw_data
    python multiday_eval.py --use-cache       # reuse cached per-day matrices
    python multiday_eval.py --rebuild raw_data .
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import features as featmod
from loader import load_multi_day
from features import build_features
from nowcast import detect_soft, detect_hard, merge_catalog, catalog_summary
from forecast import (make_labels, make_feature_matrix, train_forecaster,
                      predict_proba_curve, extract_alerts)
from evaluate import (optimal_threshold, confusion, tss, hss, roc_curve_data,
                      event_level_metrics)

HORIZON_MIN = 30.0
MIN_CLASS = "C"
TOP_N = 8
CACHE_DIR = "data/processed/perday"


# --------------------------------------------------------------------------- #
# Per-day build + cache
# --------------------------------------------------------------------------- #
def build_day(df_day: pd.DataFrame):
    feats = build_features(df_day.drop(columns=["day"], errors="ignore"))
    catalog = merge_catalog(detect_soft(feats), detect_hard(feats))
    labelled = make_labels(feats, catalog, horizon_min=HORIZON_MIN, min_class=MIN_CLASS)
    X, _ = make_feature_matrix(labelled)
    y = labelled.loc[X.index, "y_binary"].astype(int)
    return catalog, X, y


def _cache_paths(date):
    return (os.path.join(CACHE_DIR, f"{date}_X.parquet"),
            os.path.join(CACHE_DIR, f"{date}_cat.parquet"))


def load_or_build_days(roots, use_cache):
    """Return {date: (catalog, X, y)} building (and caching) per-day matrices."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    manifest = os.path.join(CACHE_DIR, "manifest.json")
    if use_cache and os.path.exists(manifest):
        days = json.load(open(manifest))["days"]
        out = {}
        for d in days:
            xp, cp = _cache_paths(d)
            X = pd.read_parquet(xp)
            y = X.pop("__y__").astype(int)
            cat = pd.read_parquet(cp) if os.path.exists(cp) else pd.DataFrame()
            out[d] = (cat, X, y)
        print(f"[cache] loaded {len(out)} per-day matrices from {CACHE_DIR}")
        return out, {"days_accepted": days, "days_rejected": [], "from_cache": True}

    combined, info = load_multi_day(roots, cadence="1s")
    out = {}
    print("\n" + "=" * 72 + "\n  PER-DAY FLARE COUNTS\n" + "=" * 72)
    for date, g in combined.groupby("day"):
        cat, X, y = build_day(g)
        out[date] = (cat, X, y)
        xp, cp = _cache_paths(date)
        Xc = X.copy(); Xc["__y__"] = y.to_numpy()
        Xc.to_parquet(xp)
        if len(cat):
            cat.to_parquet(cp)
        s = catalog_summary(cat)
        print(f"  {date}: {s['headline']}  | positive samples={int(y.sum())}")
    json.dump({"days": sorted(out)}, open(os.path.join(CACHE_DIR, "manifest.json"), "w"))
    return out, info


# --------------------------------------------------------------------------- #
# Neupert constant fit
# --------------------------------------------------------------------------- #
def fit_k_neupert(per_day):
    """Least-squares-through-origin K for dF_SXR/dt ~= K * F_HXR on active samples."""
    hs, ds = [], []
    for _, X, _ in per_day.values():
        if {"hxr_broad", "deriv_soft"} <= set(X.columns):
            h = X["hxr_broad"].to_numpy(); d = X["deriv_soft"].to_numpy()
            m = np.isfinite(h) & np.isfinite(d) & (h > 0)
            hs.append(h[m]); ds.append(d[m])
    h = np.concatenate(hs); d = np.concatenate(ds)
    k = float(np.sum(h * d) / np.sum(h * h)) if h.size else 1.0
    return k, int(h.size)


def apply_k(per_day, k):
    """Recompute neupert_residual = deriv_soft - k*hxr_broad in each day's X."""
    for _, X, _ in per_day.values():
        if {"hxr_broad", "deriv_soft", "neupert_residual"} <= set(X.columns):
            X["neupert_residual"] = X["deriv_soft"] - k * X["hxr_broad"]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_fold(X_tr, y_tr, X_te, y_te, catalog_te):
    model = train_forecaster(X_tr, y_tr, mode="binary")
    thr = optimal_threshold(y_tr.to_numpy(),
                            predict_proba_curve(model, X_tr).to_numpy())["threshold"]
    prob_te = predict_proba_curve(model, X_te)
    y_pred = (prob_te.to_numpy() >= thr).astype(int)
    cm = confusion(y_te.to_numpy(), y_pred)
    auc = roc_curve_data(y_te.to_numpy(), prob_te.to_numpy())["auc"]
    alerts = extract_alerts(prob_te.dropna(), catalog_te, model, X_te,
                            threshold=thr, horizon_min=HORIZON_MIN,
                            min_class=MIN_CLASS, refractory_min=20.0, rearm_frac=0.7)
    ev = event_level_metrics(catalog_te, alerts, horizon_min=HORIZON_MIN,
                             n_days=1, min_class=MIN_CLASS)
    imp = {f: float(v) for f, v in zip(model.feature_names, model.feature_importances_)}
    return {"TSS": tss(cm=cm), "HSS": hss(cm=cm),
            "AUC": float(auc) if auc == auc else np.nan, "threshold": thr,
            "cm": cm, "event": ev, "importances": imp,
            "n_test": int(len(y_te)), "pos_test": int(y_te.sum())}


def main():
    args = sys.argv[1:]
    use_cache = "--use-cache" in args
    rebuild = "--rebuild" in args
    roots = [a for a in args if not a.startswith("--")] or (glob.glob("*.zip") + ["raw_data"])
    print(f"\nSOURCES: {roots}  (use_cache={use_cache and not rebuild})")

    per_day, info = load_or_build_days(roots, use_cache and not rebuild)
    days = sorted(per_day)
    n_days = len(days)
    print(f"\nACCEPTED DAYS ({n_days}): {days}")
    if info.get("days_rejected"):
        for r in info["days_rejected"]:
            print(f"  REJECTED {r['date']}: {r['reason']}")

    if n_days < 3:
        print("\n!! Need >=3 consistent days for leave-one-day-out CV.")
        return

    # ---- fit Neupert constant and apply ----
    k_neupert, n_used = fit_k_neupert(per_day)
    featmod.K_NEUPERT = k_neupert
    apply_k(per_day, k_neupert)
    print(f"\nFitted K_NEUPERT = {k_neupert:.4g}  (LS-through-origin on {n_used:,} active samples)")

    # ---- leave-one-day-out CV ----
    print("\n" + "=" * 72 + "\n  LEAVE-ONE-DAY-OUT CROSS-VALIDATION\n" + "=" * 72)
    folds = []
    pooled_cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    imp_acc = {}
    ev_flares = ev_alerted = ev_false = 0
    leads = []
    for test_day in days:
        tr_days = [d for d in days if d != test_day]
        X_tr = pd.concat([per_day[d][1] for d in tr_days]).sort_index()
        y_tr = pd.concat([per_day[d][2] for d in tr_days]).reindex(X_tr.index)
        cat_te, X_te, y_te = per_day[test_day]
        if y_te.sum() == 0 or (len(y_te) - y_te.sum()) == 0:
            print(f"  {test_day}: skipped (single-class test day)")
            continue
        r = score_fold(X_tr, y_tr, X_te, y_te, cat_te)
        folds.append((test_day, r))
        for k in pooled_cm:
            pooled_cm[k] += r["cm"][k]
        for f, v in r["importances"].items():
            imp_acc[f] = imp_acc.get(f, 0.0) + v
        ev = r["event"]
        ev_flares += ev["n_flares"] or 0
        ev_alerted += ev["n_alerted"] or 0
        ev_false += ev["false_alarm_count"] or 0
        if ev["mean_lead"] is not None and ev["n_alerted"]:
            leads += [ev["mean_lead"]] * ev["n_alerted"]
        print(f"  test {test_day}: TSS={r['TSS']:+.3f} HSS={r['HSS']:+.3f} "
              f"AUC={r['AUC']:.3f} thr={r['threshold']:.3f} "
              f"| events {ev['n_alerted']}/{ev['n_flares']} lead~{ev['mean_lead']}")

    tss_arr = np.array([r["TSS"] for _, r in folds])
    hss_arr = np.array([r["HSS"] for _, r in folds])
    auc_arr = np.array([r["AUC"] for _, r in folds])
    n_imp = max(1, len(folds))
    imp_mean = dict(sorted(((f, v / n_imp) for f, v in imp_acc.items()),
                           key=lambda kv: -kv[1]))

    print("\n" + "=" * 72 + "\n  AGGREGATE (mean +/- std over %d folds)\n" % len(folds) + "=" * 72)
    print(f"  TSS = {tss_arr.mean():+.3f} +/- {tss_arr.std():.3f}")
    print(f"  HSS = {hss_arr.mean():+.3f} +/- {hss_arr.std():.3f}")
    print(f"  AUC = {auc_arr.mean():.3f} +/- {auc_arr.std():.3f}")
    print(f"  pooled confusion : {pooled_cm}")
    print(f"  pooled TSS = {tss(cm=pooled_cm):+.3f}   pooled HSS = {hss(cm=pooled_cm):+.3f}")
    print(f"  EVENT-LEVEL: alerted {ev_alerted}/{ev_flares} confirmed flares "
          f"(recall={ev_alerted / ev_flares:.2f} )" if ev_flares else "  EVENT-LEVEL: no flares")
    print(f"  mean lead = {np.mean(leads):.1f} min  |  FAR = {ev_false / n_days:.2f}/day"
          if leads else f"  FAR = {ev_false / n_days:.2f}/day")
    print(f"\n  top-{TOP_N} mean feature importances:")
    for f, v in list(imp_mean.items())[:TOP_N]:
        print(f"      {f:24s} {v:.4f}")

    # ---- chronological headline split ----
    n_test = max(1, round(0.2 * n_days))
    tr_days, te_days = days[:-n_test], days[-n_test:]
    X_tr = pd.concat([per_day[d][1] for d in tr_days]).sort_index()
    y_tr = pd.concat([per_day[d][2] for d in tr_days]).reindex(X_tr.index)
    X_te = pd.concat([per_day[d][1] for d in te_days]).sort_index()
    y_te = pd.concat([per_day[d][2] for d in te_days]).reindex(X_te.index)
    cat_te = pd.concat([per_day[d][0] for d in te_days if len(per_day[d][0])], ignore_index=True) \
        if any(len(per_day[d][0]) for d in te_days) else pd.DataFrame()
    head = score_fold(X_tr, y_tr, X_te, y_te, cat_te)
    print("\n" + "=" * 72 + "\n  CHRONOLOGICAL HEADLINE SPLIT\n" + "=" * 72)
    print(f"  TRAIN {tr_days[0]}..{tr_days[-1]} ({len(X_tr):,})  ->  TEST {te_days} ({len(X_te):,})")
    print(f"  TSS={head['TSS']:+.3f}  HSS={head['HSS']:+.3f}  AUC={head['AUC']:.3f}  "
          f"thr={head['threshold']:.3f}  cm={head['cm']}")

    out = {
        "n_days": n_days, "days": days, "k_neupert": k_neupert,
        "lodo": {
            "n_folds": len(folds),
            "TSS_mean": round(float(tss_arr.mean()), 4), "TSS_std": round(float(tss_arr.std()), 4),
            "HSS_mean": round(float(hss_arr.mean()), 4), "HSS_std": round(float(hss_arr.std()), 4),
            "AUC_mean": round(float(auc_arr.mean()), 4), "AUC_std": round(float(auc_arr.std()), 4),
            "pooled_cm": pooled_cm, "pooled_TSS": round(tss(cm=pooled_cm), 4),
            "pooled_HSS": round(hss(cm=pooled_cm), 4),
            "event_recall": round(ev_alerted / ev_flares, 3) if ev_flares else None,
            "n_flares": ev_flares, "n_alerted": ev_alerted,
            "mean_lead": round(float(np.mean(leads)), 2) if leads else None,
            "far_per_day": round(ev_false / n_days, 2),
            "importances_mean": {k: round(v, 4) for k, v in imp_mean.items()},
            "per_fold": {d: {"TSS": round(r["TSS"], 4), "HSS": round(r["HSS"], 4),
                             "AUC": round(r["AUC"], 4)} for d, r in folds},
        },
        "headline_split": {"train_days": tr_days, "test_days": te_days,
                           "TSS": round(head["TSS"], 4), "HSS": round(head["HSS"], 4),
                           "AUC": round(head["AUC"], 4), "cm": head["cm"]},
    }
    os.makedirs("data/catalog", exist_ok=True)
    json.dump(out, open("data/catalog/lodo_metrics.json", "w"), indent=2, default=str)
    print("\n[written] data/catalog/lodo_metrics.json")


if __name__ == "__main__":
    main()
