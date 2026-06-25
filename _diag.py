"""
_diag.py — deep diagnostic battery on the cached 23-day dataset.

Answers: does the model beat trivial single-feature detectors? Are its
probabilities calibrated? What does the FAR-vs-recall trade-off look like, and
where is the sweet operating point? How are lead times distributed? Uses the
per-day cache written by multiday_eval.py (fast, no re-ingest).
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import features as featmod
from forecast import train_forecaster, predict_proba_curve, extract_alerts
from evaluate import confusion, tss, hss, roc_curve_data, optimal_threshold, event_level_metrics

CACHE = "data/processed/perday"
OUT = "data/catalog"
HORIZON = 30.0


def load_cache():
    days = json.load(open(f"{CACHE}/manifest.json"))["days"]
    per = {}
    for d in days:
        X = pd.read_parquet(f"{CACHE}/{d}_X.parquet")
        y = X.pop("__y__").astype(int)
        cp = f"{CACHE}/{d}_cat.parquet"
        cat = pd.read_parquet(cp) if os.path.exists(cp) else pd.DataFrame()
        per[d] = (cat, X, y)
    return days, per


def fit_k(per):
    hs, ds = [], []
    for _, X, _ in per.values():
        h, d = X["hxr_broad"].to_numpy(), X["deriv_soft_c"].to_numpy()
        m = np.isfinite(h) & np.isfinite(d) & (h > 0)
        hs.append(h[m]); ds.append(d[m])
    h, d = np.concatenate(hs), np.concatenate(ds)
    k = float(np.sum(h * d) / np.sum(h * h))
    for _, X, _ in per.values():
        X["neupert_residual_c"] = X["deriv_soft_c"] - k * X["hxr_broad"]
    return k


def best_single_feature_tss(X, y, col):
    """Max TSS achievable by a single threshold on one feature (both directions)."""
    v = X[col].to_numpy(); yt = y.to_numpy()
    m = np.isfinite(v)
    v, yt = v[m], yt[m]
    if yt.sum() == 0 or yt.sum() == len(yt):
        return np.nan
    qs = np.quantile(v, np.linspace(0.01, 0.99, 60))
    best = -1.0
    for thr in qs:
        for pred in ((v >= thr).astype(int), (v <= thr).astype(int)):
            best = max(best, tss(cm=confusion(yt, pred)))
    return best


def main():
    days, per = load_cache()
    k = fit_k(per)
    featmod.K_NEUPERT = k

    # ---------- 1. dataset overview ----------
    print("=" * 74 + "\n  DATASET OVERVIEW (23-day cache)\n" + "=" * 74)
    tot_samp = tot_pos = 0
    rows = []
    for d in days:
        _, X, y = per[d]
        rows.append((d, len(y), int(y.sum()), 100 * y.mean()))
        tot_samp += len(y); tot_pos += int(y.sum())
    for d, n, p, r in rows:
        print(f"  {d}: {n:>7,} samples  {p:>6,} pos ({r:4.1f}%)")
    print(f"  TOTAL: {tot_samp:,} samples, {tot_pos:,} positive "
          f"({100*tot_pos/tot_samp:.2f}% base rate), {len(days)} days")

    X_all = pd.concat([per[d][1] for d in days]).sort_index()
    y_all = pd.concat([per[d][2] for d in days]).reindex(X_all.index)

    # ---------- 2. single-feature baselines vs model ----------
    print("\n" + "=" * 74 + "\n  BASELINE: best single-feature TSS (pooled, in-sample upper bound)\n" + "=" * 74)
    base_feats = ["deriv_soft_c", "var_soft_c", "neupert_residual_c", "neupert_windowed",
                  "hxr_sxr_lag", "hxr_broad", "hr_slope_c"]
    sf = {c: best_single_feature_tss(X_all, y_all, c) for c in base_feats}
    for c, t in sorted(sf.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -9)):
        print(f"  {c:24s} best-threshold TSS = {t:+.3f}")
    best_single = max(v for v in sf.values() if v == v)
    print(f"  -> best single feature (in-sample, optimistic) = {best_single:+.3f}")

    # ---------- 3. chronological holdout: fit, probs, calibration, operating points ----------
    n_test = max(1, round(0.2 * len(days)))
    tr, te = days[:-n_test], days[-n_test:]
    X_tr = pd.concat([per[d][1] for d in tr]).sort_index()
    y_tr = pd.concat([per[d][2] for d in tr]).reindex(X_tr.index)
    X_te = pd.concat([per[d][1] for d in te]).sort_index()
    y_te = pd.concat([per[d][2] for d in te]).reindex(X_te.index)
    cat_te = pd.concat([per[d][0] for d in te if len(per[d][0])], ignore_index=True) \
        if any(len(per[d][0]) for d in te) else pd.DataFrame()

    model = train_forecaster(X_tr, y_tr, mode="binary")
    p_tr = predict_proba_curve(model, X_tr).to_numpy()
    p_te = predict_proba_curve(model, X_te)
    yv = y_te.to_numpy(); pv = p_te.to_numpy()
    thr_star = optimal_threshold(y_tr.to_numpy(), p_tr)["threshold"]
    roc = roc_curve_data(yv, pv)

    print("\n" + "=" * 74 + f"\n  HELD-OUT MODEL  (train {tr[0]}..{tr[-1]} -> test {te})\n" + "=" * 74)
    print(f"  test samples={len(yv):,}  positives={int(yv.sum())} ({100*yv.mean():.2f}%)")
    print(f"  ROC-AUC = {roc['auc']:.3f}   |   model TSS@max-TSS-thr vs best single feature:")
    cm_star = confusion(yv, (pv >= thr_star).astype(int))
    print(f"     model TSS = {tss(cm=cm_star):+.3f} (thr*={thr_star:.3f})   "
          f"best-single-feature(in-sample) = {best_single:+.3f}")

    # operating-point sweep: FAR/day vs recall/precision
    test_days_span = len(te)
    print("\n  OPERATING-POINT SWEEP (per-sample):")
    print("   thr     recall   precision   FP/day    TSS")
    op_rows = []
    for thr in [0.05, 0.10, 0.20, 0.30, thr_star, 0.50, 0.70, 0.90]:
        cm = confusion(yv, (pv >= thr).astype(int))
        tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
        rec = tp / (tp + fn) if (tp + fn) else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        fpday = fp / test_days_span
        op_rows.append((thr, rec, prec, fpday, tss(cm=cm)))
        tag = " <-max-TSS" if abs(thr - thr_star) < 1e-9 else ""
        print(f"   {thr:.3f}   {rec:5.2f}    {prec:6.3f}   {fpday:7.0f}   {tss(cm=cm):+.3f}{tag}")

    # calibration / reliability
    print("\n  RELIABILITY (predicted vs observed positive frequency):")
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(pv, bins) - 1, 0, 9)
    rel = []
    for b in range(10):
        m = idx == b
        if m.sum() > 50:
            rel.append((0.5 * (bins[b] + bins[b + 1]), pv[m].mean(), yv[m].mean(), int(m.sum())))
    for center, predm, obs, n in rel:
        print(f"   bin~{center:.2f}: pred={predm:.3f} obs={obs:.3f} (n={n:,})")

    # event-level + lead time at max-TSS thr
    alerts = extract_alerts(p_te.dropna(), cat_te, model, X_te, threshold=thr_star,
                            horizon_min=HORIZON, min_class="C",
                            refractory_min=20.0, rearm_frac=0.7)
    ev = event_level_metrics(cat_te, alerts, horizon_min=HORIZON,
                             n_days=test_days_span, min_class="C")
    print(f"\n  EVENT-LEVEL @max-TSS-thr: alerted {ev['n_alerted']}/{ev['n_flares']} "
          f"(recall {ev['event_recall']}), mean lead {ev['mean_lead']} min, "
          f"FAR {ev['far_per_day']}/day")

    # ---------- plots ----------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), facecolor="#0d1117")
    for a in ax:
        a.set_facecolor("#0d1117")
        for s in a.spines.values():
            s.set_color("#30363d")
        a.tick_params(colors="#8b949e"); a.xaxis.label.set_color("#c9d1d9")
        a.yaxis.label.set_color("#c9d1d9"); a.title.set_color("#e6edf3")
    # ROC
    ax[0].plot(roc["fpr"], roc["tpr"], color="#58a6ff", lw=2, label=f"AUC={roc['auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="#6e7681")
    ax[0].set_title("ROC (held-out days)"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[0].legend(facecolor="#161b22", labelcolor="#c9d1d9", edgecolor="#30363d")
    # FAR vs recall
    rr = [r[1] for r in op_rows]; ff = [r[3] for r in op_rows]
    ax[1].plot(rr, ff, "-o", color="#f0883e")
    ax[1].set_title("FAR/day vs recall"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("false positives/day")
    # reliability
    if rel:
        ax[2].plot([r[1] for r in rel], [r[2] for r in rel], "-o", color="#3fb950")
        ax[2].plot([0, 1], [0, 1], "--", color="#6e7681")
    ax[2].set_title("Reliability"); ax[2].set_xlabel("predicted"); ax[2].set_ylabel("observed")
    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/diag_panels.png", dpi=110, facecolor="#0d1117")
    print(f"\n[written] {OUT}/diag_panels.png")

    json.dump({"k_neupert": k, "single_feature_tss": sf, "best_single": best_single,
               "auc_holdout": roc["auc"], "op_rows": op_rows,
               "reliability": rel, "event_level": ev},
              open(f"{OUT}/diag_metrics.json", "w"), indent=2, default=str)
    print(f"[written] {OUT}/diag_metrics.json")


if __name__ == "__main__":
    main()
