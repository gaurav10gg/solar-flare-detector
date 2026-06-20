"""
evaluate.py — Flare-forecasting evaluation & validation.

Implements the standard skill scores used in operational flare forecasting
(Bloomfield et al. 2012), plus lead-time statistics, per-class detection, and
validation of the nowcast catalogue against the public NOAA/GOES flare catalogue.

Headline metric is the **True Skill Statistic (TSS)** — class-imbalance-robust
and the field standard. HSS, ROC-AUC, lead-time distribution and per-class
recall are reported alongside.

Public API:
    confusion(y_true, y_pred)             -> dict(TP, FP, FN, TN)
    tss(...) / hss(...)                   -> float
    roc_curve_data(y_true, y_prob)        -> dict(fpr, tpr, auc, thresholds)
    leadtime_stats(alerts)               -> dict
    per_class_detection(catalog, truth)  -> dict
    validate_vs_goes(catalog, goes_csv)  -> dict | None
    load_goes_catalog(csv_path)          -> pd.DataFrame | None
    report(...)                          -> dict (also prints + saves JSON)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger("evaluate")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

_CLASS_RANK = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}


# --------------------------------------------------------------------------- #
# 1. Confusion matrix
# --------------------------------------------------------------------------- #
def confusion(y_true, y_pred) -> dict:
    """Binary confusion counts as a dict ``{TP, FP, FN, TN}``."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


# --------------------------------------------------------------------------- #
# 2-3. Skill scores
# --------------------------------------------------------------------------- #
def tss(y_true=None, y_pred=None, *, cm: dict = None) -> float:
    """True Skill Statistic = TP/(TP+FN) - FP/(FP+TN). Range [-1, 1].

    Accepts either label arrays or a precomputed confusion dict. TSS is the
    headline metric: insensitive to class imbalance (unlike accuracy).
    """
    cm = cm or confusion(y_true, y_pred)
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    recall = tp / (tp + fn) if (tp + fn) else 0.0          # sensitivity
    fpr = fp / (fp + tn) if (fp + tn) else 0.0             # false-alarm rate
    return float(recall - fpr)


def optimal_threshold(y_true, y_prob) -> dict:
    """Probability threshold that maximises TSS (Youden's J / optimal ROC point).

    For rare events a fixed 0.5 cut-off is inappropriate; the operationally
    correct threshold maximises the True Skill Statistic. This should be chosen
    on training/validation data and then applied to the held-out test set.

    Returns ``{'threshold', 'tss'}``.
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_prob, dtype="float64")
    if len(np.unique(yt)) < 2:
        return {"threshold": 0.5, "tss": 0.0}
    candidates = np.unique(np.clip(yp, 0.0, 1.0))
    if len(candidates) > 200:  # subsample for speed
        candidates = np.quantile(candidates, np.linspace(0, 1, 200))
    best_thr, best_tss = 0.5, -1.0
    for thr in candidates:
        score = tss(cm=confusion(yt, (yp >= thr).astype(int)))
        if score > best_tss:
            best_thr, best_tss = float(thr), float(score)
    return {"threshold": round(best_thr, 4), "tss": round(best_tss, 4)}


def threshold_for_precision(y_true, y_prob, target_precision: float = 0.6) -> dict:
    """Lowest probability threshold whose precision >= *target_precision*.

    The max-TSS threshold maximises balanced skill but, for a rare event, can
    sit at a very low probability where the alarm rate is operationally absurd
    (thousands of false positives per day). The deployable operating point is
    instead the **least conservative threshold that still meets a precision
    target** — this keeps recall as high as possible while bounding the false
    alarm rate. Among all thresholds meeting the target we pick the one with the
    highest recall; if the target is unreachable we fall back to the
    maximum-precision threshold (and report the precision actually achieved).

    The threshold is chosen on training/validation probabilities and then
    applied to held-out data, exactly like :func:`optimal_threshold`.

    Returns ``{'threshold', 'precision', 'recall', 'target_precision',
    'target_met'}``.
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_prob, dtype="float64")
    if len(np.unique(yt)) < 2:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0,
                "target_precision": target_precision, "target_met": False}
    order = np.argsort(-yp)
    yt_s = yt[order]
    yp_s = yp[order]
    tp = np.cumsum(yt_s)
    fp = np.cumsum(1 - yt_s)
    total_pos = int(yt.sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(total_pos, 1)
    meets = precision >= target_precision
    if meets.any():
        idx = np.where(meets)[0]
        best = idx[int(np.argmax(recall[idx]))]  # highest recall among precise enough
        met = True
    else:
        best = int(np.argmax(precision))         # closest we can get
        met = False
    return {"threshold": round(float(yp_s[best]), 4),
            "precision": round(float(precision[best]), 4),
            "recall": round(float(recall[best]), 4),
            "target_precision": target_precision, "target_met": bool(met)}


def hss(y_true=None, y_pred=None, *, cm: dict = None) -> float:
    """Heidke Skill Score (vs random). Range (-inf, 1]; 0 = no skill."""
    cm = cm or confusion(y_true, y_pred)
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / n
    denom = n - expected
    return float((tp + tn - expected) / denom) if denom else 0.0


# --------------------------------------------------------------------------- #
# 4. ROC
# --------------------------------------------------------------------------- #
def roc_curve_data(y_true, y_prob) -> dict:
    """ROC curve points and AUC. Falls back gracefully on single-class input."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_prob, dtype="float64")
    if len(np.unique(yt)) < 2:
        logger.warning("roc_curve_data: only one class present — AUC undefined.")
        return {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "thresholds": [], "auc": float("nan")}
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, thr = roc_curve(yt, yp)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": thr.tolist(), "auc": float(roc_auc_score(yt, yp))}


# --------------------------------------------------------------------------- #
# 5. Lead time
# --------------------------------------------------------------------------- #
def leadtime_stats(alerts: list[dict], bins=None) -> dict:
    """Summary + histogram of alert lead times (minutes). Target window 15-30 min."""
    leads = [a["lead_time_min"] for a in (alerts or [])
             if a.get("lead_time_min") is not None]
    n_alerts = len(alerts or [])
    matched = len(leads)
    out = {
        "n_alerts": n_alerts,
        "n_matched": matched,
        "n_false": n_alerts - matched,
        "match_rate": round(matched / n_alerts, 3) if n_alerts else 0.0,
    }
    if not leads:
        out.update({"mean": None, "median": None, "std": None, "min": None,
                    "max": None, "histogram": {}, "in_target_15_30_pct": None})
        return out
    arr = np.array(leads, dtype="float64")
    bins = bins if bins is not None else [0, 5, 10, 15, 20, 30, 45, 60, np.inf]
    hist, edges = np.histogram(arr, bins=bins)
    hist_d = {f"{edges[i]:g}-{edges[i+1]:g}min": int(hist[i]) for i in range(len(hist))}
    in_target = float(np.mean((arr >= 15) & (arr <= 30)) * 100.0)
    out.update({
        "mean": round(float(arr.mean()), 2), "median": round(float(np.median(arr)), 2),
        "std": round(float(arr.std()), 2), "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2), "histogram": hist_d,
        "in_target_15_30_pct": round(in_target, 1),
    })
    return out


# --------------------------------------------------------------------------- #
# 5b. Event-level forecast metrics
# --------------------------------------------------------------------------- #
def event_level_metrics(catalog: pd.DataFrame, alerts: list[dict],
                        horizon_min: float = 30.0, n_days: int = 1,
                        min_class: str = "C") -> dict:
    """Per-*flare* (event-level) forecast skill, complementing per-sample scores.

    Per-second precision is pessimistic for a rare, bursty target: a single
    flare spans hundreds of contiguous "positive" seconds, so a few seconds of
    misalignment dominate the sample-wise FP/FN counts. Operationally what
    matters is *did we warn ahead of each real flare, and how often did we cry
    wolf per day* — captured here.

    A confirmed flare (>= *min_class*) counts as **alerted** if any alert was
    raised whose matched-flare peak is that flare (i.e. the alert fired within
    *horizon_min* before the peak). False alarms are alerts that matched no real
    flare; the daily false-alarm rate normalises them by the number of days.

    Returns
    -------
    dict
        ``{n_flares, n_alerted, event_recall, mean_lead, median_lead,
        false_alarm_count, far_per_day, horizon_min}``.
    """
    # Confirmed, forecastable flares (a soft/thermal response, >= min_class).
    min_rank = _CLASS_RANK.get(min_class.upper(), 2)
    n_flares = 0
    if catalog is not None and len(catalog):
        for _, fl in catalog.iterrows():
            cat = fl.get("category")
            is_confirmed = (cat == "confirmed_flare") if cat is not None else \
                (fl.get("provenance") in ("both", "soft_only"))
            if is_confirmed and _CLASS_RANK.get(str(fl.get("goes_class"))[:1].upper(), -1) >= min_rank:
                n_flares += 1

    alerts = alerts or []
    matched_peaks, leads = set(), []
    false_alarms = 0
    for a in alerts:
        mf = a.get("matched_flare")
        if mf and a.get("lead_time_min") is not None:
            matched_peaks.add(str(mf.get("peak_time")))
            leads.append(a["lead_time_min"])
        else:
            false_alarms += 1

    n_alerted = len(matched_peaks)
    arr = np.array(leads, dtype="float64") if leads else np.array([])
    return {
        "n_flares": n_flares,
        "n_alerted": min(n_alerted, n_flares) if n_flares else n_alerted,
        "event_recall": round(min(n_alerted, n_flares) / n_flares, 3) if n_flares else None,
        "mean_lead": round(float(arr.mean()), 2) if arr.size else None,
        "median_lead": round(float(np.median(arr)), 2) if arr.size else None,
        "false_alarm_count": int(false_alarms),
        "far_per_day": round(false_alarms / max(1, n_days), 2),
        "horizon_min": horizon_min,
    }


# --------------------------------------------------------------------------- #
# 6. Per-class detection
# --------------------------------------------------------------------------- #
def per_class_detection(catalog: pd.DataFrame, truth: pd.DataFrame,
                        match_window_min: float = 5.0) -> dict:
    """Recall for weak (B/C) vs strong (M/X) flares against a *truth* catalogue.

    A truth flare is "detected" if a catalogue flare peaks within
    *match_window_min*. Reports recall separately for B/C and M/X (large dynamic
    range is an explicit grading criterion).
    """
    if truth is None or len(truth) == 0:
        logger.warning("per_class_detection: no truth catalogue — skipped.")
        return {}
    tol = pd.Timedelta(minutes=match_window_min)
    det_peaks = (pd.to_datetime(catalog["peak_time"], utc=True).tolist()
                 if catalog is not None and len(catalog) else [])

    groups = {"B/C": {"B", "C"}, "M/X": {"M", "X"}}
    result = {}
    for label, letters in groups.items():
        truth_sub = [pd.to_datetime(r["peak_time"], utc=True)
                     for _, r in truth.iterrows()
                     if str(r.get("goes_class", ""))[:1].upper() in letters]
        if not truth_sub:
            result[label] = {"n_truth": 0, "n_detected": 0, "recall": None}
            continue
        detected = sum(any(abs(tp - dp) <= tol for dp in det_peaks) for tp in truth_sub)
        result[label] = {"n_truth": len(truth_sub), "n_detected": detected,
                         "recall": round(detected / len(truth_sub), 3)}
    return result


# --------------------------------------------------------------------------- #
# 7. GOES validation
# --------------------------------------------------------------------------- #
def load_goes_catalog(csv_path: str) -> pd.DataFrame | None:
    """Load a NOAA/GOES flare-list CSV into a normalised ``(peak_time, goes_class)`` frame.

    Accepts the common NOAA/SWPC export shapes (column matching is
    case-insensitive and flexible):

    * a direct peak-time column — one of ``peak_time``, ``peak``, ``max_time``,
      ``max``, ``time_peak``; OR
    * a separate ``date`` column combined with a peak/max time-of-day column
      (e.g. ``date`` + ``max``), which are concatenated;

    plus a class column (``goes_class``, ``class``, ``goes``). Times are parsed as
    UTC. Returns ``None`` (with a clear, actionable warning) if the file is
    absent or unparseable, so callers can skip GOES validation gracefully.
    """
    if not csv_path or not os.path.exists(csv_path):
        logger.warning("load_goes_catalog: '%s' not found — GOES validation will "
                       "be skipped and GOES classes shown as UNCALIBRATED. "
                       "(Provide the NOAA/GOES flare list for the uploaded "
                       "date(s) to enable real precision/recall.)", csv_path)
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("load_goes_catalog: failed to read %s (%s).", csv_path, exc)
        return None
    cols = {c.lower(): c for c in df.columns}
    class_col = next((cols[k] for k in ("goes_class", "class", "goes") if k in cols), None)
    time_col = next((cols[k] for k in ("peak_time", "peak", "max_time", "max", "time_peak")
                     if k in cols), None)

    if time_col is not None:
        peak = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    elif "date" in cols:
        # Combine a date column with the best available peak/start time-of-day.
        tod = next((cols[k] for k in ("max", "peak", "start", "begin") if k in cols), None)
        combo = df[cols["date"]].astype(str) + (" " + df[tod].astype(str) if tod else "")
        peak = pd.to_datetime(combo, utc=True, errors="coerce")
    else:
        peak = None

    if peak is None or class_col is None:
        logger.warning("load_goes_catalog: could not find peak-time/class columns in %s "
                       "(have: %s).", csv_path, list(df.columns))
        return None
    out = pd.DataFrame({"peak_time": peak,
                        "goes_class": df[class_col].astype(str)}
                       ).dropna(subset=["peak_time"]).sort_values("peak_time")
    logger.info("load_goes_catalog: loaded %d GOES flares from %s", len(out), csv_path)
    return out


def annotate_goes_match(catalog: pd.DataFrame, goes_csv: str,
                        match_window_min: float = 5.0) -> tuple[pd.DataFrame, bool]:
    """Tag each catalogue row with a ✓/✗ NOAA-GOES match (per-flare ground truth).

    Adds two columns to a copy of *catalog*:

    * ``goes_match``       — ``True`` if a NOAA/GOES flare peaks within
      *match_window_min* of this detection, ``False`` if not, ``None`` if no
      GOES list is available (uncalibrated);
    * ``goes_truth_class`` — the matched NOAA class string (or ``None``).

    Returns ``(annotated_catalog, calibrated)`` where ``calibrated`` is ``True``
    only when a GOES list was actually loaded. When uncalibrated, downstream UIs
    should label our classes as "GOES-equivalent (uncalibrated)".
    """
    cat = catalog.copy() if catalog is not None else pd.DataFrame()
    goes = load_goes_catalog(goes_csv)
    if goes is None or len(cat) == 0:
        if len(cat):
            cat["goes_match"] = None
            cat["goes_truth_class"] = None
        return cat, False

    tol = pd.Timedelta(minutes=match_window_min)
    truth_peaks = list(zip(pd.to_datetime(goes["peak_time"], utc=True),
                           goes["goes_class"].astype(str)))
    matches, truth_cls = [], []
    for _, fl in cat.iterrows():
        pk = pd.to_datetime(fl["peak_time"], utc=True)
        best = min(((abs(pk - tp), cls) for tp, cls in truth_peaks),
                   default=(None, None), key=lambda r: r[0] if r[0] is not None else tol * 1e9)
        if best[0] is not None and best[0] <= tol:
            matches.append(True)
            truth_cls.append(best[1])
        else:
            matches.append(False)
            truth_cls.append(None)
    cat["goes_match"] = matches
    cat["goes_truth_class"] = truth_cls
    logger.info("annotate_goes_match: %d/%d detections matched a NOAA/GOES flare.",
                int(sum(matches)), len(cat))
    return cat, True


def validate_vs_goes(catalog: pd.DataFrame, goes_csv: str,
                     match_window_min: float = 5.0, min_class: str = "C") -> dict | None:
    """Precision/recall of our nowcast catalogue vs the NOAA/GOES catalogue.

    Matches detected and GOES flares whose peaks fall within *match_window_min*.
    Only flares >= *min_class* (in both) are considered. Skips gracefully
    (returns ``None``) if the GOES CSV is unavailable.
    """
    goes = load_goes_catalog(goes_csv)
    if goes is None:
        return None
    min_rank = _CLASS_RANK.get(min_class.upper(), 2)

    def _rank(c):
        return _CLASS_RANK.get(str(c)[:1].upper(), -1)

    det = [pd.to_datetime(r["peak_time"], utc=True)
           for _, r in (catalog.iterrows() if catalog is not None else [])
           if _rank(r.get("goes_class")) >= min_rank]
    tru = [t for t in goes["peak_time"]
           if _rank(goes.loc[goes["peak_time"] == t, "goes_class"].iloc[0]) >= min_rank]

    tol = pd.Timedelta(minutes=match_window_min)
    tp = sum(any(abs(d - t) <= tol for t in tru) for d in det)
    fp = len(det) - tp
    fn = sum(not any(abs(d - t) <= tol for d in det) for t in tru)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    res = {"n_detected": len(det), "n_goes": len(tru), "TP": tp, "FP": fp, "FN": fn,
           "precision": round(precision, 3), "recall": round(recall, 3),
           "f1": round(f1, 3), "match_window_min": match_window_min}
    logger.info("validate_vs_goes: P=%.3f R=%.3f F1=%.3f (TP=%d FP=%d FN=%d)",
                precision, recall, f1, tp, fp, fn)
    return res


# --------------------------------------------------------------------------- #
# 8. Report
# --------------------------------------------------------------------------- #
def report(y_true, y_pred, y_prob, alerts: list[dict], *,
           catalog: pd.DataFrame = None, truth: pd.DataFrame = None,
           goes_csv: str = None, n_days: int = 1,
           json_path: str = "data/catalog/metrics.json", title: str = "") -> dict:
    """Assemble, print and persist the full metrics summary; returns the dict."""
    cm = confusion(y_true, y_pred)
    metrics = {
        "title": title or "flare forecast evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_test_samples": int(len(np.asarray(y_true))),
        "confusion": cm,
        "TSS": round(tss(cm=cm), 4),
        "HSS": round(hss(cm=cm), 4),
        "ROC": {"auc": round(roc_curve_data(y_true, y_prob)["auc"], 4)},
        "lead_time": leadtime_stats(alerts),
        "event_level": event_level_metrics(catalog, alerts, n_days=n_days),
        "per_class_detection": per_class_detection(catalog, truth),
        "goes_validation": validate_vs_goes(catalog, goes_csv) if goes_csv else None,
        "n_days": n_days,
    }
    if n_days < 2:
        metrics["data_warning"] = ("Only 1 observation day available — metrics are "
                                   "from a within-day time split and are indicative "
                                   "only. Add more days for robust, generalisable "
                                   "skill scores.")
        logger.warning(metrics["data_warning"])

    print("\n" + "=" * 64)
    print(f"  {metrics['title'].upper()}")
    print("=" * 64)
    print("  -- per-sample (per-second) --")
    print(f"  test samples      : {metrics['n_test_samples']:,}")
    print(f"  confusion         : {cm}")
    print(f"  TSS (headline)    : {metrics['TSS']:+.4f}   [TP/(TP+FN) - FP/(FP+TN)]")
    print(f"  HSS               : {metrics['HSS']:+.4f}")
    print(f"  ROC-AUC           : {metrics['ROC']['auc']}")
    ev = metrics["event_level"]
    print("  -- event-level (per-flare) --")
    if ev["n_flares"]:
        print(f"  alerted flares    : {ev['n_alerted']}/{ev['n_flares']} "
              f"(event recall={ev['event_recall']})")
        print(f"  lead time (min)   : mean={ev['mean_lead']} median={ev['median_lead']}")
    else:
        print("  alerted flares    : no confirmed >=C flares in catalogue")
    print(f"  false alarms      : {ev['false_alarm_count']} "
          f"(FAR={ev['far_per_day']}/day)")
    lt = metrics["lead_time"]
    print(f"  alerts            : {lt['n_alerts']} ({lt['n_matched']} matched, "
          f"{lt['n_false']} false; match_rate={lt['match_rate']})")
    if lt.get("mean") is not None:
        print(f"  lead time (min)   : mean={lt['mean']} median={lt['median']} "
              f"range=[{lt['min']}, {lt['max']}]  in 15-30min={lt['in_target_15_30_pct']}%")
        print(f"  lead-time hist    : {lt['histogram']}")
    if metrics["per_class_detection"]:
        print(f"  per-class recall  : {metrics['per_class_detection']}")
    if metrics["goes_validation"]:
        gv = metrics["goes_validation"]
        print(f"  vs GOES catalog   : P={gv['precision']} R={gv['recall']} F1={gv['f1']}")
    else:
        print("  vs GOES catalog   : skipped (no GOES CSV supplied)")
    if "data_warning" in metrics:
        print(f"\n  ** {metrics['data_warning']}")
    print("=" * 64)

    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info("report: metrics written to %s", json_path)
    return metrics


# --------------------------------------------------------------------------- #
# Smoke test / acceptance (end-to-end)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    from loader import discover_files, load_unified
    from preprocess import apply_gti, flag_gaps
    from features import build_features
    from nowcast import detect_soft, detect_hard, merge_catalog
    from forecast import (make_labels, make_feature_matrix, time_train_test_split,
                          train_forecaster, predict_proba_curve, extract_alerts)

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    goes_csv = sys.argv[2] if len(sys.argv) > 2 else "data/catalog/goes_flares.csv"

    disc = discover_files(target)
    df, meta = load_unified(target, cadence="1s")
    df = flag_gaps(apply_gti(df, disc.get("solexs_gti")))
    feats = build_features(df)
    catalog = merge_catalog(detect_soft(feats), detect_hard(feats))

    labelled = make_labels(feats, catalog, horizon_min=30, min_class="C")
    X, names = make_feature_matrix(labelled)
    y = labelled.loc[X.index, "y_binary"]

    X_tr, X_te, y_tr, y_te = time_train_test_split(X, y, test_frac=0.3, by_day=True)
    n_days = len(X.index.normalize().unique())

    model = train_forecaster(X_tr, y_tr, mode="binary")

    # Choose the operating threshold on TRAIN (max-TSS), then apply to TEST.
    prob_tr = predict_proba_curve(model, X_tr)
    opt = optimal_threshold(y_tr.to_numpy(), prob_tr.to_numpy())
    thr = opt["threshold"]
    logger.info("operating threshold (max-TSS on train) = %.4f (train TSS=%.4f)",
                thr, opt["tss"])

    prob_te = predict_proba_curve(model, X_te)
    y_pred = (prob_te.to_numpy() >= thr).astype(int)

    alerts = extract_alerts(prob_te, catalog, model, X_te,
                            threshold=thr, horizon_min=30, min_class="C")

    metrics = report(y_te.to_numpy(), y_pred, prob_te.to_numpy(), alerts,
                     catalog=catalog, truth=None, goes_csv=goes_csv, n_days=n_days,
                     title=f"flare forecast — held-out test ({meta.get('date')})")
    print(f"\n  operating threshold (max-TSS, picked on train): {thr:.3f}")

    print("\n[alert list — held-out test]")
    for a in alerts:
        lt = a["lead_time_min"]
        feats_str = ", ".join(f"{c['feature']}" for c in a["contributing_features"][:3])
        print(f"  {a['alert_time']}  p={a['probability']:.2f}  "
              f"lead={lt if lt is not None else 'FALSE':>6}  [{feats_str}]")
