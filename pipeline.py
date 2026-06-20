"""
pipeline.py — End-to-end orchestration for the Aditya-L1 flare system.

Runs the full chain on any uploaded path and returns one artifact bundle used by
both the API and the simulation engine:

    loader -> preprocess -> features -> nowcast -> forecast -> evaluate

Design choices:
    * Two forecaster fits, deliberately:
        - a *metrics* model trained on the time-ordered train split and scored on
          the held-out test (honest TSS/HSS/AUC, no leakage);
        - a *display* model trained on the whole day to produce a smooth full-day
          probability curve + alert sequence for the replay dashboard.
      Metrics stay leakage-free; the demo still shows alerts across the entire day.
    * The decision threshold is selected by maximising TSS on the training split
      and reused everywhere (operationally correct for rare events).

Nothing is hardcoded to a date or flare — every artifact is derived from the
uploaded data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from loader import discover_files, load_unified, cleanup_tempdirs
from preprocess import apply_gti, flag_gaps, quality_report
from features import build_features, infer_cadence_seconds as _cadence_seconds
from nowcast import (detect_soft, detect_hard, merge_catalog, save_catalog,
                     catalog_summary)
from forecast import (make_labels, make_feature_matrix, time_train_test_split,
                      train_forecaster, predict_proba_curve, extract_alerts)
from evaluate import (optimal_threshold, confusion, tss, hss, roc_curve_data,
                      leadtime_stats, event_level_metrics, annotate_goes_match)

logger = logging.getLogger("pipeline")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def _downsample_lightcurve(feats: pd.DataFrame, prob: pd.Series,
                           target_points: int = 1440) -> dict:
    """Downsample full-day curves for fast plotting (~target_points samples)."""
    n = len(feats)
    step = max(1, n // target_points)
    sub = feats.iloc[::step]
    prob_sub = prob.reindex(sub.index)

    def col(name):
        return [None if pd.isna(v) else round(float(v), 3) for v in sub[name]]

    return {
        "time": [t.isoformat() for t in sub.index],
        "solexs_counts": col("solexs_counts"),
        "background_soft": col("background_soft"),
        "hxr_broad": col("hxr_broad"),
        "hxr_20_40": col("hxr_20_40"),
        "hxr_80_150": col("hxr_80_150"),
        "hr": col("hr"),
        "flare_probability": [None if pd.isna(v) else round(float(v), 4) for v in prob_sub],
        "step_seconds": step,
    }


def catalog_to_records(catalog: pd.DataFrame) -> list[dict]:
    """JSON-serialisable catalogue rows (ISO-8601 timestamps)."""
    if catalog is None or len(catalog) == 0:
        return []
    out = []
    for _, r in catalog.iterrows():
        rec = r.to_dict()
        for k in ("start", "peak_time", "end"):
            if k in rec and pd.notna(rec[k]):
                rec[k] = pd.to_datetime(rec[k], utc=True).isoformat()
        for k, v in list(rec.items()):
            if isinstance(v, (np.floating, np.integer)):
                rec[k] = float(v)
            elif pd.isna(v):
                rec[k] = None
        out.append(rec)
    return out


def run_pipeline(path, *, cadence: str = "1s", horizon_min: float = 30.0,
                 min_class: str = "C", alert_threshold: float | None = None,
                 sqlite_path: str | None = None,
                 goes_csv: str | None = "data/catalog/goes_flares.csv") -> dict:
    """Execute the full flare pipeline on *path* and return an artifact bundle.

    Parameters
    ----------
    path : str | list
        Upload location (zip / folder / files / list).
    cadence : str
        Unified-grid cadence.
    horizon_min : float
        Forecast horizon for labelling/alerts.
    min_class : str
        Minimum GOES class treated as a positive forecast target.
    alert_threshold : float, optional
        Override the auto-selected (max-TSS) alert threshold.
    sqlite_path : str, optional
        If given, persist the catalogue to this SQLite file.

    Returns
    -------
    dict
        Keys: ``metadata, quality, catalog (DataFrame), catalog_records,
        metrics, alerts, threshold, prob_curve (Series), feats (DataFrame),
        lightcurve, horizon_min``.
    """
    # ---- ingest -> clean (discover once; reuse to avoid re-extracting) ----
    disc = discover_files(path)
    try:
        df, metadata = load_unified(disc, cadence=cadence)
        df = flag_gaps(apply_gti(df, disc.get("solexs_gti")))
        quality = quality_report(df)
    finally:
        # Data is now in memory; remove temp extraction dirs immediately.
        cleanup_tempdirs(disc)

    # ---- features -> nowcast ----
    feats = build_features(df)
    soft = detect_soft(feats)
    hard = detect_hard(feats)
    catalog = merge_catalog(soft, hard)
    # Per-flare NOAA/GOES ground-truth tagging (✓/✗) when a flare list is
    # available; otherwise classes remain "GOES-equivalent (uncalibrated)".
    catalog, goes_calibrated = annotate_goes_match(catalog, goes_csv)
    cat_summary = catalog_summary(catalog)
    if sqlite_path:
        save_catalog(catalog, sqlite_path)

    # ---- labels -> feature matrix ----
    labelled = make_labels(feats, catalog, horizon_min=horizon_min, min_class=min_class)
    X, _names = make_feature_matrix(labelled)
    y = labelled.loc[X.index, "y_binary"]
    n_days = len(X.index.normalize().unique())

    metrics: dict = {}
    threshold = 0.5 if alert_threshold is None else float(alert_threshold)
    prob_full = pd.Series(np.nan, index=feats.index, name="flare_probability")
    alerts: list[dict] = []

    if y.sum() > 0 and (len(y) - y.sum()) > 0:
        # ---- metrics model (leakage-free held-out test) ----
        X_tr, X_te, y_tr, y_te = time_train_test_split(X, y, test_frac=0.3, by_day=True)
        m_model = train_forecaster(X_tr, y_tr, mode="binary")
        prob_tr = predict_proba_curve(m_model, X_tr)
        opt = optimal_threshold(y_tr.to_numpy(), prob_tr.to_numpy())
        if alert_threshold is None:
            threshold = opt["threshold"]
        prob_te = predict_proba_curve(m_model, X_te)
        y_pred = (prob_te.to_numpy() >= threshold).astype(int)
        cm = confusion(y_te.to_numpy(), y_pred)
        roc = roc_curve_data(y_te.to_numpy(), prob_te.to_numpy())

        # ---- display model (whole-day curve + alerts for the replay) ----
        d_model = train_forecaster(X, y, mode="binary")
        prob_disp = predict_proba_curve(d_model, X)
        # Smooth (~3 min) to remove the step-calibration chatter before alerting.
        smooth_n = max(1, int(round(180 / max(1, _cadence_seconds(X.index)))))
        prob_disp = prob_disp.rolling(smooth_n, center=True, min_periods=1).mean()
        prob_full.loc[prob_disp.index] = prob_disp.to_numpy()
        # Alert operating point: a precision-oriented threshold (high percentile)
        # gives a clean, meaningful alert stream for the replay, while the
        # headline TSS above is still reported at the max-TSS threshold. Bounded
        # below by the TSS threshold so it is never less sensitive than that.
        alert_thr = max(threshold, float(np.nanquantile(prob_disp.to_numpy(), 0.985)))
        alerts = extract_alerts(prob_full.dropna(), catalog, d_model, X,
                                threshold=alert_thr, horizon_min=horizon_min,
                                min_class=min_class, refractory_min=20.0,
                                rearm_frac=0.7)
        metrics_alert_threshold = alert_thr

        metrics = {
            "TSS": round(tss(cm=cm), 4), "HSS": round(hss(cm=cm), 4),
            "ROC": {"auc": round(roc["auc"], 4) if roc["auc"] == roc["auc"] else None,
                    "fpr": [round(v, 4) for v in roc["fpr"]],
                    "tpr": [round(v, 4) for v in roc["tpr"]]},
            "confusion": cm, "threshold": round(threshold, 4),
            "alert_threshold": round(metrics_alert_threshold, 4),
            "n_test_samples": int(len(y_te)), "n_days": n_days,
            "lead_time": leadtime_stats(alerts),
            "event_level": event_level_metrics(catalog, alerts,
                                               horizon_min=horizon_min,
                                               n_days=n_days, min_class=min_class),
            "feature_importances": dict(sorted(
                zip(d_model.feature_names, [round(float(v), 4) for v in d_model.feature_importances_]),
                key=lambda kv: -kv[1])),
        }
        metrics["goes_calibrated"] = goes_calibrated
        # Always-on disclaimer reconciling the two-model evaluation regime so a
        # reader never mistakes the (intentionally honest) held-out TSS for a bug
        # when the demo alert log shows clean hits. Sourced here once and reused
        # verbatim by the PDF report and the dashboard.
        metrics["evaluation_note"] = (
            "TSS, HSS, ROC-AUC and the confusion matrix above are computed on a "
            "held-out time split the model never trained on — this is the honest, "
            "leakage-free skill score. The alert log and event-recall/lead-time "
            "numbers below come from a separate model fitted on the full day to "
            "produce a smooth demo curve for the replay; they are illustrative of "
            "alert *behavior*, not a generalization claim."
        )
        if n_days < 2:
            metrics["data_warning"] = ("Single observation day — metrics from a "
                                       "within-day time split; add more days for "
                                       "robust skill scores.")
    else:
        logger.warning("run_pipeline: not enough positive labels to train a "
                       "forecaster (need >=1 of each class). Skipping forecast.")
        metrics = {"note": "insufficient labels for forecasting on this upload",
                   "n_days": n_days, "goes_calibrated": goes_calibrated}

    bundle = {
        "metadata": metadata,
        "quality": quality,
        "catalog": catalog,
        "catalog_records": catalog_to_records(catalog),
        "catalog_summary": cat_summary,
        "goes_calibrated": goes_calibrated,
        "metrics": metrics,
        "alerts": alerts,
        "threshold": threshold,
        "prob_curve": prob_full,
        "feats": feats,
        "lightcurve": _downsample_lightcurve(feats, prob_full),
        "horizon_min": horizon_min,
    }
    logger.info("run_pipeline: done — %s, %d alerts, TSS=%s",
                cat_summary["headline"], len(alerts), metrics.get("TSS"))
    return bundle


if __name__ == "__main__":
    import sys, json
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    b = run_pipeline(target, sqlite_path="data/catalog/flares.sqlite")
    print("\nMETADATA:", json.dumps(b["metadata"], default=str, indent=2))
    print("\nFLARES:", len(b["catalog"]), "| ALERTS:", len(b["alerts"]),
          "| THRESHOLD:", b["threshold"])
    print("METRICS:", json.dumps(b["metrics"], default=str, indent=2))
