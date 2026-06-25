"""
forecast.py — ML flare *forecaster* for Aditya-L1 SoLEXS + HEL1OS.

Where ``nowcast.py`` detects flares that are *happening*, this module predicts
flares that are *about to happen*: for each second it estimates the probability
that a >= C-class flare peaks within the next N minutes, emits a calibrated
probability curve, and raises explainable alerts with a measured **lead time**
(flare peak time - alert time). Longer reliable lead time is explicitly better
(per the organisers).

Primary model: gradient-boosted trees (XGBoost) on engineered physics features —
robust on small data, gives feature importances for explainability, and
calibrated probabilities. Upgrade path (CNN/LSTM/TFT on raw series) is left open.

Methodology notes:
    * Labels use a forward sliding window (no peeking): label(t)=1 iff a
      qualifying flare peaks in (t, t+horizon].
    * Splits are strictly time-ordered (NO shuffle): we train on earlier time /
      earlier days and test on later, held-out time / days. Shuffling a time
      series leaks the future and inflates scores.
    * Probabilities are calibrated on a time-ordered holdout (cv='prefit').

References: Bobra & Couvidat (2015); Bloomfield et al. (2012); Nishizuka et al.
(2018, DeFN).

Public API:
    make_labels(df, catalog, horizon_min=30, min_class='C')   -> pd.DataFrame
    add_time_since_last_flare(df, catalog)                    -> pd.Series
    make_feature_matrix(df)                                   -> (X, names)
    time_train_test_split(X, y, ...)                          -> splits
    train_forecaster(X, y, mode='binary')                     -> ForecastModel
    predict_proba_curve(model, X, index)                      -> pd.Series
    extract_alerts(prob_curve, catalog, model, X, ...)        -> list[dict]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError("forecast.py requires xgboost: pip install xgboost") from exc
from sklearn.calibration import CalibratedClassifierCV
try:  # sklearn >= 1.6 prefers wrapping a frozen estimator over cv='prefit'
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover
    FrozenEstimator = None

logger = logging.getLogger("forecast")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# GOES class ordering and intensity buckets used for labelling.
_CLASS_RANK = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
_INTENSITY = {0: "none", 1: "low", 2: "low", 3: "med", 4: "high"}  # A/B/C->low, M->med, X->high
_CLASS_ORDER = ["none", "low", "med", "high"]

# The engineered features, with one-line rationale each.
# NOTE: every windowed forecasting feature here is CAUSAL (trailing, no
# look-ahead) so the model can run live and the reported skill is honest. The
# centred twins (deriv_soft, var_soft, hr_slope, deriv_hard, neupert_residual)
# peek ~half a window into the future and are reserved for retrospective
# detection/cataloguing only — they are deliberately NOT used here.
FEATURE_COLUMNS = [
    "hr",                     # hardness ratio (60-150)/(20-40 keV): instantaneous, causal
    "hr_slope_c",             # trailing dHR/dt: rising hardness fires before broadband threshold
    "hxr_broad",              # HEL1OS 18-160 keV flux (raw input): non-thermal energy proxy
    "deriv_hard_c",           # trailing d(HXR)/dt: impulsive onset rate
    "neupert_windowed",       # trailing windowed HXR fluence (local Neupert): predicts SXR rise
    "neupert_residual_c",     # deriv_soft_c - k*hxr_broad: divergence => imminent SXR peak
    "hxr_sxr_lag",            # measured HXR->SXR lead lag (trailing windows; tightening = imminent)
    "hxr_sxr_xcorr",          # peak HXR->SXR cross-correlation (coupling strength)
    "deriv_soft_c",           # trailing d(SoLEXS)/dt: thermal rise rate
    "var_soft_c",             # trailing short-term SoLEXS variance: pre-flare variability
    "time_since_last_flare",  # recency of last detected flare (sympathetic/quiet-Sun context)
]

# Features that carry a meaningful "zero" when undefined (no hard emission /
# quiescence) and are filled rather than dropped. The windowed-Neupert,
# residual and cross-correlation features are 0/"no-coupling" in quiescence.
_FILL_ZERO = ["hr", "hr_slope_c", "var_soft_c", "deriv_hard_c", "deriv_soft_c",
              "neupert_windowed", "neupert_residual_c", "hxr_sxr_lag", "hxr_sxr_xcorr"]
# Cap (seconds) for "time since last flare" when none has occurred yet.
_TSLF_CAP = 24 * 3600.0


def _class_rank(goes_class) -> int:
    """Rank of a GOES class string ('C9.2'->2). Returns -1 if not parseable."""
    if not isinstance(goes_class, str) or not goes_class:
        return -1
    return _CLASS_RANK.get(goes_class[0].upper(), -1)


# --------------------------------------------------------------------------- #
# 1. Labels
# --------------------------------------------------------------------------- #
def add_time_since_last_flare(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.Series:
    """Seconds since the most recent catalogued flare *peak* at each timestamp.

    Uses every catalogued flare (any provenance). Before the first flare the
    value is capped at 24 h. Returns a Series aligned to ``df.index``.
    """
    tsl = pd.Series(_TSLF_CAP, index=df.index, dtype="float64")
    if catalog is None or len(catalog) == 0:
        return tsl.rename("time_since_last_flare")
    peaks = pd.to_datetime(catalog["peak_time"], utc=True).sort_values()
    idx_s = df.index.view("int64") / 1e9
    for pk in peaks:
        pk_s = pk.value / 1e9
        after = idx_s >= pk_s
        tsl.values[after] = np.minimum(tsl.values[after], idx_s[after] - pk_s)
    return tsl.rename("time_since_last_flare")


def make_labels(df: pd.DataFrame, catalog: pd.DataFrame,
                horizon_min: float = 30.0, min_class: str = "C") -> pd.DataFrame:
    """Attach forward-looking forecast labels via a sliding horizon window.

    For each timestamp ``t``:
        * ``y_binary`` = 1 iff a flare with class >= *min_class* peaks in
          ``(t, t+horizon]``.
        * ``y_class``  = intensity of the strongest such flare in the horizon:
          ``none`` / ``low`` (B,C) / ``med`` (M) / ``high`` (X).

    Only flares carrying a parseable GOES class (soft / both provenance) count;
    hard-only detections have no GOES class and are ignored for labelling.
    Also (re)fills ``time_since_last_flare`` from the catalogue.

    Returns a copy of *df* with ``y_binary`` and ``y_class`` columns added.
    """
    out = df.copy()
    out["time_since_last_flare"] = add_time_since_last_flare(out, catalog)

    horizon = pd.Timedelta(minutes=horizon_min)
    y_bin = np.zeros(len(out), dtype="int8")
    y_rank = np.zeros(len(out), dtype="int8")  # 0=none else max class rank in horizon

    min_rank = _CLASS_RANK.get(min_class.upper(), 2)
    idx = out.index

    if catalog is not None and len(catalog):
        for _, fl in catalog.iterrows():
            rank = _class_rank(fl.get("goes_class"))
            if rank < min_rank:
                continue
            peak = pd.to_datetime(fl["peak_time"], utc=True)
            # t qualifies iff t < peak <= t+horizon  ->  t in [peak-horizon, peak)
            lo = peak - horizon
            mask = (idx >= lo) & (idx < peak)
            if not mask.any():
                continue
            y_bin[mask] = 1
            y_rank[mask] = np.maximum(y_rank[mask], rank)

    out["y_binary"] = y_bin
    out["y_class"] = pd.Categorical(
        [_INTENSITY.get(int(r), "none") for r in y_rank],
        categories=_CLASS_ORDER, ordered=True,
    )
    pos = int(y_bin.sum())
    logger.info("make_labels: horizon=%gmin min_class=%s -> %d/%d positive (%.2f%%); "
                "class dist=%s", horizon_min, min_class, pos, len(out),
                100.0 * pos / max(len(out), 1),
                out["y_class"].value_counts().to_dict())
    return out


# --------------------------------------------------------------------------- #
# 2. Feature matrix
# --------------------------------------------------------------------------- #
def make_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the model design matrix from engineered features.

    Features (see ``FEATURE_COLUMNS`` for per-feature rationale): hardness ratio
    and its slope, HEL1OS broadband flux and derivative, Neupert integral,
    SoLEXS derivative, short-term variance, and time since last flare.

    Quiescence-meaningful NaNs (no hard emission) are filled with 0; rows still
    NaN in core features (genuine data gaps in Neupert / broadband) are dropped
    so the model never trains across a gap.

    Returns
    -------
    (X, feature_names)
        ``X`` is indexed by the surviving timestamps.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"make_feature_matrix: missing feature columns {missing}. "
                       "Run features.build_features() first.")
    X = df[FEATURE_COLUMNS].copy()
    for c in _FILL_ZERO:
        X[c] = X[c].fillna(0.0)
    X["time_since_last_flare"] = X["time_since_last_flare"].fillna(_TSLF_CAP)
    before = len(X)
    X = X.dropna()  # remaining NaN = real gaps (neupert / hxr_broad)
    logger.info("make_feature_matrix: %d rows -> %d after dropping gap rows (%d features)",
                before, len(X), len(FEATURE_COLUMNS))
    return X, list(FEATURE_COLUMNS)


# --------------------------------------------------------------------------- #
# 3. Training
# --------------------------------------------------------------------------- #
@dataclass
class ForecastModel:
    """Fitted forecaster bundle: calibrated classifier + metadata."""
    classifier: object                       # calibrated estimator (predict_proba)
    base: object                             # raw XGBClassifier (feature_importances_)
    feature_names: list[str]
    mode: str                                # 'binary' | 'multiclass'
    classes_: np.ndarray
    feature_importances_: np.ndarray = field(default=None)


def time_train_test_split(X: pd.DataFrame, y: pd.Series, test_frac: float = 0.3,
                          by_day: bool = True, split_mode: str = "auto"):
    """Strictly time-ordered train/test split (NEVER shuffled).

    Two regimes, selected by *split_mode*:

    * ``'by_day'`` — **true held-out-day** evaluation: the last
      ``ceil(test_frac * n_days)`` UTC day(s) form the test set and the model
      only ever sees *earlier* days in training. This is the credible,
      generalisation-revealing split and requires >1 observation day.
    * ``'within_day'`` — single-day fallback: the last *test_frac* of the
      time-ordered rows is the test set (train on the morning, test on the
      evening). Honest about ordering but cannot show day-to-day generalisation.
    * ``'auto'`` (default) — pick ``'by_day'`` when >1 day is present, else
      ``'within_day'`` with a loud warning.

    The legacy *by_day* flag is honoured only when ``split_mode='auto'``
    (``by_day=False`` forces the within-day path) for backward compatibility.

    Returns ``(X_tr, X_te, y_tr, y_te)``.
    """
    X = X.sort_index()
    y = y.reindex(X.index)
    days = X.index.normalize()
    unique_days = days.unique()
    n_days = len(unique_days)

    if split_mode not in ("auto", "by_day", "within_day"):
        raise ValueError("split_mode must be 'auto', 'by_day' or 'within_day'")

    if split_mode == "auto":
        mode = "by_day" if (by_day and n_days > 1) else "within_day"
    else:
        mode = split_mode

    if mode == "by_day" and n_days > 1:
        n_test_days = max(1, int(np.ceil(test_frac * n_days)))
        test_days = set(unique_days[-n_test_days:])
        te_mask = days.isin(test_days)
        train_days = [str(d.date()) for d in sorted(set(unique_days) - test_days)]
        logger.info("time split [by_day]: %d days -> TRAIN on %s | TEST (held-out) on %s",
                    n_days, train_days, [str(d.date()) for d in sorted(test_days)])
    else:
        if mode == "by_day":
            logger.warning("time split: 'by_day' requested but only %d UTC day(s) "
                           "available — falling back to within-day tail split. MORE "
                           "DAYS NEEDED for robust, generalisable metrics.", n_days)
        cut = int(len(X) * (1.0 - test_frac))
        te_mask = np.zeros(len(X), dtype=bool)
        te_mask[cut:] = True
        logger.info("time split [within_day]: tail %.0f%% of %d rows held out as test.",
                    100 * test_frac, len(X))

    return X[~te_mask], X[te_mask], y[~te_mask], y[te_mask]


def train_forecaster(X: pd.DataFrame, y: pd.Series, mode: str = "binary",
                     *, calibrate: bool = True, calib_frac: float = 0.25,
                     random_state: int = 42) -> ForecastModel:
    """Train a calibrated XGBoost flare forecaster.

    Handles class imbalance (``scale_pos_weight`` for binary; balanced
    ``sample_weight`` for multiclass) and calibrates probabilities on a
    time-ordered tail holdout (``cv='prefit'``), never by shuffling.

    Parameters
    ----------
    X : pd.DataFrame
        Time-indexed feature matrix (already the *training* portion).
    y : pd.Series
        Labels: ``y_binary`` (0/1) or ``y_class`` (ordered categorical).
    mode : {'binary', 'multiclass'}
    calibrate : bool
        Probability calibration on a held-out tail (recommended).
    calib_frac : float
        Fraction of the (time-ordered) training tail reserved for calibration.

    Returns
    -------
    ForecastModel
    """
    X = X.sort_index()
    y = y.reindex(X.index)

    if mode == "binary":
        y_enc = y.astype(int).to_numpy()
        classes = np.array([0, 1])
        n_pos = int(y_enc.sum())
        n_neg = int(len(y_enc) - n_pos)
        spw = (n_neg / n_pos) if n_pos > 0 else 1.0
        base = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, scale_pos_weight=spw, eval_metric="logloss",
            tree_method="hist", random_state=random_state, n_jobs=0,
        )
        sample_weight = None
        logger.info("train_forecaster[binary]: n=%d pos=%d neg=%d scale_pos_weight=%.2f",
                    len(y_enc), n_pos, n_neg, spw)
    elif mode == "multiclass":
        cats = _CLASS_ORDER
        code = {c: i for i, c in enumerate(cats)}
        y_enc = np.array([code[str(v)] for v in y], dtype=int)
        classes = np.array(cats)
        # Balanced sample weights ~ inverse class frequency.
        counts = np.bincount(y_enc, minlength=len(cats)).astype("float64")
        inv = np.where(counts > 0, len(y_enc) / (len(cats) * counts), 0.0)
        sample_weight = inv[y_enc]
        base = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, objective="multi:softprob", num_class=len(cats),
            eval_metric="mlogloss", tree_method="hist",
            random_state=random_state, n_jobs=0,
        )
        logger.info("train_forecaster[multiclass]: n=%d class_counts=%s",
                    len(y_enc), dict(zip(cats, counts.astype(int))))
    else:
        raise ValueError("mode must be 'binary' or 'multiclass'")

    # Time-ordered fit / calibration holdout.
    do_calib = calibrate and len(X) > 50 and len(np.unique(y_enc)) > 1
    if do_calib:
        cut = int(len(X) * (1.0 - calib_frac))
        cut = min(max(cut, 1), len(X) - 1)
        Xf, Xc = X.iloc[:cut], X.iloc[cut:]
        yf, yc = y_enc[:cut], y_enc[cut:]
        sw_f = sample_weight[:cut] if sample_weight is not None else None
        if len(np.unique(yf)) < 2 or len(np.unique(yc)) < 2:
            do_calib = False  # cannot calibrate a single-class fold
        else:
            base.fit(Xf, yf, sample_weight=sw_f)
            method = "isotonic" if len(Xc) >= 200 else "sigmoid"
            if FrozenEstimator is not None:
                calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=method)
            else:
                calibrated = CalibratedClassifierCV(base, method=method, cv="prefit")
            calibrated.fit(Xc, yc)
            logger.info("train_forecaster: calibrated (%s) on %d tail rows", method, len(Xc))
            clf = calibrated

    if not do_calib:
        logger.warning("train_forecaster: skipping calibration (insufficient/"
                       "single-class data); using raw XGBoost probabilities.")
        base.fit(X, y_enc, sample_weight=sample_weight)
        clf = base

    # Always (re)fit base on ALL training data for stable feature importances.
    base_full = base
    if do_calib:
        base_full = XGBClassifier(**base.get_params())
        base_full.fit(X, y_enc, sample_weight=sample_weight)

    return ForecastModel(
        classifier=clf, base=base_full, feature_names=list(X.columns),
        mode=mode, classes_=classes,
        feature_importances_=np.asarray(base_full.feature_importances_),
    )


# --------------------------------------------------------------------------- #
# 4. Probability curve
# --------------------------------------------------------------------------- #
def predict_proba_curve(model: ForecastModel, X: pd.DataFrame,
                        index: pd.Index = None) -> pd.Series:
    """Flare probability over time.

    For binary mode this is P(flare in horizon). For multiclass it is the
    probability of *any* flare = ``1 - P(none)``. Returns a Series on *index*
    (defaults to ``X.index``).
    """
    index = X.index if index is None else index
    proba = model.classifier.predict_proba(X)
    classes = list(model.classifier.classes_)
    if model.mode == "binary":
        col = classes.index(1) if 1 in classes else proba.shape[1] - 1
        p = proba[:, col]
    else:
        none_code = 0  # 'none' is encoded as 0 in training
        p = 1.0 - proba[:, classes.index(none_code)] if none_code in classes else proba.max(axis=1)
    return pd.Series(p, index=index, name="flare_probability")


# --------------------------------------------------------------------------- #
# 5. Alerts + lead time
# --------------------------------------------------------------------------- #
def extract_alerts(prob_curve: pd.Series, catalog: pd.DataFrame = None,
                   model: ForecastModel = None, X: pd.DataFrame = None,
                   *, threshold: float = 0.5, horizon_min: float = 30.0,
                   min_class: str = "C", refractory_min: float = 10.0,
                   rearm_frac: float = 0.7) -> list[dict]:
    """Raise alerts on upward threshold crossings and measure lead time.

    An alert fires when the probability crosses *threshold* upward. A refractory
    period suppresses repeat alerts within the same elevated episode. Each alert
    is matched to the first qualifying real flare peaking within *horizon_min*
    after the alert; the lead time is ``peak - alert_time``.

    Parameters
    ----------
    prob_curve : pd.Series
        Output of :func:`predict_proba_curve`.
    catalog : pd.DataFrame, optional
        Nowcast catalogue used to match alerts to real flares (lead time).
    model, X : optional
        If given, the top-3 contributing features (importance x standardised
        value) are attached for explainability.
    threshold : float, default 0.5
    horizon_min : float
        Max look-ahead to associate an alert with a real flare peak.
    min_class : str
        Minimum GOES class of a flare that can satisfy an alert.
    refractory_min : float
        Minimum spacing between successive alerts.
    rearm_frac : float
        Hysteresis: after an alert, the probability must fall below
        ``rearm_frac * threshold`` before another alert can fire. Prevents
        "chattering" when the (step-calibrated) probability sits on the threshold.

    Returns
    -------
    list[dict]
        Each: ``{alert_time, probability, predicted_class, contributing_features,
        matched_flare, lead_time_min}``.
    """
    p = prob_curve.dropna()
    if p.empty:
        return []
    vals = p.to_numpy()
    times = p.index

    # Crossings with hysteresis: fire on upward crossing of `threshold`, then
    # disarm until the probability drops below `rearm_frac * threshold`.
    rearm_level = rearm_frac * threshold
    crossings = []
    armed = True
    for j in range(1, len(vals)):
        if armed and vals[j] >= threshold > vals[j - 1]:
            crossings.append(j)
            armed = False
        elif not armed and vals[j] < rearm_level:
            armed = True
    crossings = np.array(crossings, dtype=int)

    # Prepare qualifying flare peaks for matching.
    min_rank = _CLASS_RANK.get(min_class.upper(), 2)
    flare_peaks = []
    if catalog is not None and len(catalog):
        for _, fl in catalog.iterrows():
            if _class_rank(fl.get("goes_class")) >= min_rank:
                flare_peaks.append((pd.to_datetime(fl["peak_time"], utc=True), fl))
        flare_peaks.sort(key=lambda r: r[0])

    # Precompute standardised feature values for contribution scoring.
    feat_z = None
    if model is not None and X is not None:
        Xa = X.reindex(times)
        mu, sd = Xa.mean(), Xa.std().replace(0, 1.0)
        feat_z = (Xa - mu) / sd

    horizon = pd.Timedelta(minutes=horizon_min)
    refractory = pd.Timedelta(minutes=refractory_min)
    alerts: list[dict] = []
    last_alert_time = None

    for i in crossings:
        t = times[i]
        if last_alert_time is not None and (t - last_alert_time) < refractory:
            continue
        last_alert_time = t

        # Match to first qualifying flare peaking within the horizon.
        matched, lead = None, None
        for pk, fl in flare_peaks:
            if t < pk <= t + horizon:
                matched = {
                    "peak_time": str(pk), "goes_class": fl.get("goes_class"),
                    "provenance": fl.get("provenance"),
                }
                lead = round((pk - t).total_seconds() / 60.0, 2)
                break

        # Predicted class (multiclass) or binary positive.
        predicted_class = "flare"
        if model is not None and model.mode == "multiclass" and X is not None:
            row = X.reindex([t])
            if row.notna().all(axis=1).iloc[0]:
                code = int(model.classifier.predict(row)[0])
                predicted_class = list(model.classes_)[code]

        # Top-3 contributing features = importance x |standardised value|.
        contributing = []
        if feat_z is not None and t in feat_z.index:
            imp = pd.Series(model.feature_importances_, index=model.feature_names)
            contrib = (imp * feat_z.loc[t].abs()).sort_values(ascending=False)
            contributing = [
                {"feature": f, "value": round(float(X.loc[t, f]), 4),
                 "importance": round(float(imp[f]), 4)}
                for f in contrib.head(3).index
            ]

        alerts.append({
            "alert_time": str(t),
            "probability": round(float(vals[i]), 4),
            "predicted_class": predicted_class,
            "contributing_features": contributing,
            "matched_flare": matched,
            "lead_time_min": lead,
        })

    n_matched = sum(a["matched_flare"] is not None for a in alerts)
    logger.info("extract_alerts: %d alerts @thr=%.2f (%d matched a real flare, %d false)",
                len(alerts), threshold, n_matched, len(alerts) - n_matched)
    return alerts


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    from loader import discover_files, load_unified
    from preprocess import apply_gti, flag_gaps
    from features import build_features
    from nowcast import detect_soft, detect_hard, merge_catalog

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    disc = discover_files(target)
    df, _ = load_unified(target, cadence="1s")
    df = flag_gaps(apply_gti(df, disc.get("solexs_gti")))
    feats = build_features(df)
    catalog = merge_catalog(detect_soft(feats), detect_hard(feats))

    labelled = make_labels(feats, catalog, horizon_min=30, min_class="C")
    X, names = make_feature_matrix(labelled)
    y = labelled.loc[X.index, "y_binary"]

    X_tr, X_te, y_tr, y_te = time_train_test_split(X, y, test_frac=0.3)
    model = train_forecaster(X_tr, y_tr, mode="binary")

    print("\n[feature importances]")
    for f, imp in sorted(zip(model.feature_names, model.feature_importances_),
                         key=lambda kv: -kv[1]):
        print(f"  {f:24s} {imp:.4f}")

    prob = predict_proba_curve(model, X_te)
    alerts = extract_alerts(prob, catalog, model, X_te, threshold=0.5, horizon_min=30)
    print(f"\n[test probability] mean={prob.mean():.3f} max={prob.max():.3f}")
    print(f"[alerts] {len(alerts)} on held-out test")
    for a in alerts[:8]:
        print(" ", {k: a[k] for k in ("alert_time", "probability", "lead_time_min")},
              "matched" if a["matched_flare"] else "FALSE")
