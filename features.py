"""
features.py — Feature engine for Aditya-L1 SoLEXS + HEL1OS flare forecasting.

Turns the unified, gap-flagged frame from ``loader``/``preprocess`` into a rich
feature set that powers both the classical nowcast detector and the ML
forecaster. Everything is **NaN-aware** and **gap-respecting**: no statistic is
ever computed across a data gap (false flares hide there).

Physics encoded here (our competitive edge):
    * Hardness ratio HR = (60-80 + 80-150 keV) / (20-40 keV) and its slope —
      spectral hardening is an early precursor that fires before the broadband
      flux threshold is crossed.
    * Neupert effect — F_SXR(t) ~ k * integral of F_HXR(t').  We track the
      cumulative trapezoidal integral of the hard X-ray broadband flux; its
      divergence from the soft X-ray shape foretells an imminent SXR peak.
      (Dennis & Zarro 1993; Veronig et al. 2002.)

Public API:
    background(series, window='30min', stat='median')   -> pd.Series
    rolling_sigma(series, window='30min')               -> pd.Series
    hardness_ratio(df)                                  -> pd.Series
    hr_slope(hr, window='5min')                         -> pd.Series
    neupert_integral(df)                                -> pd.Series
    derivative(series, window='2min')                   -> pd.Series
    build_features(df)                                  -> pd.DataFrame
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger("features")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

_MAD_TO_SIGMA = 1.4826  # MAD -> Gaussian-equivalent sigma

# Neupert coupling constant in dF_SXR/dt ~ k * F_HXR(t).  This sets the scale of
# the *predicted* soft-X-ray rise rate implied by the instantaneous hard-X-ray
# flux.  It is a documented, TUNABLE module constant: the physically correct
# value is obtained by regressing the observed SoLEXS rise rate (deriv_soft)
# against hxr_broad across confirmed flares (least-squares slope through the
# origin).  The default below was fitted empirically by `multiday_eval.py` on
# ~1.7M active samples across 23 observation days (LS-through-origin); it puts
# `predicted_sxr_rise` on the same count-rate scale as `deriv_soft` so the
# residual's zero-crossing genuinely flags HXR/SXR divergence rather than being
# dominated by the raw unit mismatch.
K_NEUPERT = 1.13e-3  # (cts/s of SXR-rate) per (ct/s of HXR-flux); fit on 23 days


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def infer_cadence_seconds(index: pd.DatetimeIndex) -> float:
    """Median sample spacing of *index* in seconds (defaults to 1.0)."""
    if len(index) < 2:
        return 1.0
    diffs = pd.Series(index).diff().dropna().dt.total_seconds()
    med = float(diffs.median()) if len(diffs) else 1.0
    return med if med > 0 else 1.0


def _window_samples(window: str, cadence_s: float, *, odd: bool = False) -> int:
    """Convert a time-offset *window* to a number of samples at *cadence_s*."""
    n = max(1, int(round(pd.Timedelta(window).total_seconds() / cadence_s)))
    if odd and n % 2 == 0:
        n += 1
    return n


def _segment_labels(series: pd.Series, cadence_s: float) -> pd.Series:
    """Integer labels for contiguous runs of valid samples (NaN/gap = boundary).

    A new segment begins at every NaN value and at every time jump larger than
    1.5x the nominal cadence, so no downstream rolling/integral spans a gap.
    """
    valid = series.notna()
    dt = series.index.to_series().diff().dt.total_seconds()
    gap = dt > 1.5 * cadence_s
    boundary = (~valid) | gap.fillna(True)
    return boundary.cumsum()


# --------------------------------------------------------------------------- #
# 1. background
# --------------------------------------------------------------------------- #
def background(series: pd.Series, window: str = "30min", stat: str = "median",
               q: float = 0.1) -> pd.Series:
    """Rolling adaptive baseline, robust to quiescent drift and NaN-aware.

    A centred rolling statistic tracks the slowly varying quiescent level while
    staying insensitive to the flares riding on top of it. NaNs are skipped;
    gaps stay NaN.

    Parameters
    ----------
    series : pd.Series
        Count-rate series on a (near-)uniform UTC grid.
    window : str, default '30min'
        Baseline averaging window.
    stat : {'median', 'mean', 'quantile'}, default 'median'
        Baseline statistic. ``'quantile'`` (a low percentile, see *q*) estimates
        the *quiescent floor* and is robust even to long-duration flares that a
        median would absorb — this is what the detector uses as its baseline.
    q : float, default 0.1
        Quantile used when ``stat='quantile'`` (e.g. 0.1 = 10th percentile).

    Returns
    -------
    pd.Series
        Baseline aligned to *series*.
    """
    cadence_s = infer_cadence_seconds(series.index)
    w = _window_samples(window, cadence_s)
    roll = series.rolling(w, center=True, min_periods=max(1, w // 4))
    if stat == "mean":
        base = roll.mean()
    elif stat == "quantile":
        base = roll.quantile(q)
    else:
        base = roll.median()
    return base.rename(f"{series.name}_bg" if series.name else "background")


# --------------------------------------------------------------------------- #
# 2. rolling_sigma
# --------------------------------------------------------------------------- #
def rolling_sigma(series: pd.Series, window: str = "30min") -> pd.Series:
    """Rolling robust standard deviation via the Median Absolute Deviation.

    sigma = 1.4826 * median(|x - median(x)|) over the window. Robust to the
    flares themselves, so the detection threshold is not inflated by signal.
    Floored at a tiny positive value to avoid a zero threshold on flat stretches.

    Returns
    -------
    pd.Series
        Robust sigma aligned to *series*.
    """
    cadence_s = infer_cadence_seconds(series.index)
    w = _window_samples(window, cadence_s)
    med = series.rolling(w, center=True, min_periods=max(1, w // 4)).median()
    dev = (series - med).abs()
    mad = dev.rolling(w, center=True, min_periods=max(1, w // 4)).median()
    sigma = _MAD_TO_SIGMA * mad
    sigma = sigma.where(sigma > 0)  # 0 -> NaN, then floor below
    floor = max(1e-9, float(np.nanmedian(sigma.values)) * 1e-3) if sigma.notna().any() else 1e-9
    sigma = sigma.fillna(floor).clip(lower=floor)
    return sigma.rename(f"{series.name}_sigma" if series.name else "sigma")


# --------------------------------------------------------------------------- #
# 3. hardness_ratio
# --------------------------------------------------------------------------- #
def hardness_ratio(df: pd.DataFrame, eps: float = 1.0) -> pd.Series:
    """Spectral hardness ratio HR = (60-80 + 80-150 keV) / (20-40 keV).

    Rising HR signals non-thermal spectral hardening — an early precursor that
    can fire before the broadband flux crosses threshold. Safe-divide: where the
    low-energy denominator is at/near zero (< *eps* cts/s) the ratio is NaN
    rather than exploding.

    Returns
    -------
    pd.Series
        Hardness ratio named ``hr``.
    """
    num = df["hxr_60_80"].astype("float64") + df["hxr_80_150"].astype("float64")
    den = df["hxr_20_40"].astype("float64")
    hr = num / den.where(den > eps)
    return hr.rename("hr")


# --------------------------------------------------------------------------- #
# 4. hr_slope
# --------------------------------------------------------------------------- #
def hr_slope(hr: pd.Series, window: str = "5min") -> pd.Series:
    """Rolling least-squares slope of the hardness ratio (the precursor signal).

    Closed-form OLS slope (units: HR per second) over a sliding *window*,
    computed from NaN-aware rolling sums so it is both fast and gap-safe. A
    sustained positive slope is an early hardening warning.

    Returns
    -------
    pd.Series
        dHR/dt named ``hr_slope``.
    """
    cadence_s = infer_cadence_seconds(hr.index)
    w = _window_samples(window, cadence_s)
    mp = max(5, w // 6)

    # Seconds since the series start. Using a local origin (not absolute Unix
    # epoch ~1.8e9) avoids catastrophic cancellation in the OLS denominator.
    t0 = hr.index[0]
    t = pd.Series((hr.index - t0).total_seconds(), index=hr.index)
    y = hr.astype("float64")
    mask = y.notna()
    tv = t.where(mask)  # time only where y is valid (NaN-aware sums)

    roll = lambda s: s.rolling(w, center=True, min_periods=mp)
    n = roll(mask.astype("float64")).sum()
    sum_t = roll(tv).sum()
    sum_y = roll(y).sum()
    sum_tt = roll(tv * tv).sum()
    sum_ty = roll(tv * y).sum()

    denom = n * sum_tt - sum_t * sum_t
    slope = (n * sum_ty - sum_t * sum_y) / denom.where(denom.abs() > 1e-12)
    return slope.rename("hr_slope")


# --------------------------------------------------------------------------- #
# 5. neupert_integral
# --------------------------------------------------------------------------- #
def _trapz_increments(y: pd.Series, cadence_s: float) -> tuple[pd.Series, pd.Series]:
    """Per-sample trapezoidal area increments and their segment labels (gap-safe).

    ``incr[i] = 0.5*(y[i]+y[i-1])*dt`` between consecutive *valid* samples that
    lie in the same contiguous segment; ``0`` across NaNs / gaps so no area is
    ever integrated over missing time. Returns ``(incr, seg)``.
    """
    seg = _segment_labels(y, cadence_s)
    dt = y.index.to_series().diff().dt.total_seconds()
    incr = 0.5 * (y + y.shift(1)) * dt
    same_seg = seg == seg.shift(1)
    incr = incr.where(same_seg & y.notna() & y.shift(1).notna(), 0.0)
    return incr, seg


def neupert_windowed(df: pd.DataFrame, window: str = "10min",
                     column: str = "hxr_broad") -> pd.Series:
    """Trailing **windowed** trapezoidal integral of recent hard X-ray flux.

    This is the physically-correct Neupert precursor. The Neupert effect is
    *local in time* — ``dF_SXR/dt ~ k * F_HXR(t)`` — so what predicts the next
    soft-X-ray rise is the **recent** accumulated non-thermal energy, not the
    flux integrated since start-of-day. We therefore integrate ``hxr_broad`` only
    over a trailing *window* (default 10 min). This deliberately replaces the old
    cumulative-from-midnight integral, which grew monotonically through the day
    and merely encoded time-of-day (a leakage proxy correlated with
    ``time_since_last_flare``), not physics.

    Causal & gap-safe: the window is *trailing* (no look-ahead) and **resets at
    every data gap** (the rolling sum never spans a segment boundary).

    Parameters
    ----------
    df : pd.DataFrame
        Frame containing *column*.
    window : str, default '10min'
        Trailing integration window.
    column : str, default 'hxr_broad'
        Hard X-ray flux column to integrate.

    Returns
    -------
    pd.Series
        Recent HXR fluence (cts/s x s) named ``neupert_windowed``; NaN where the
        source sample is missing.
    """
    cadence_s = infer_cadence_seconds(df.index)
    w = _window_samples(window, cadence_s)
    y = df[column].astype("float64")
    incr, seg = _trapz_increments(y, cadence_s)
    # Trailing rolling sum WITHIN each segment, so it resets at every gap and
    # never integrates across missing time.
    windowed = incr.groupby(seg).transform(
        lambda s: s.rolling(w, min_periods=1).sum())
    windowed = windowed.where(y.notna())
    return windowed.rename("neupert_windowed")


def neupert_integral(df: pd.DataFrame, column: str = "hxr_broad") -> pd.Series:
    """DEPRECATED: cumulative-from-start integral of the hard X-ray flux.

    Retained for backward compatibility only. Its value grows monotonically over
    the observation day and is therefore correlated with time-of-day /
    ``time_since_last_flare`` — i.e. it leaks temporal position rather than
    encoding the local Neupert relation. Use :func:`neupert_windowed` instead;
    :func:`build_features` no longer emits this column.

    Returns
    -------
    pd.Series
        Cumulative integral (cts/s x s) named ``neupert``.
    """
    cadence_s = infer_cadence_seconds(df.index)
    y = df[column].astype("float64")
    incr, seg = _trapz_increments(y, cadence_s)
    neupert = incr.groupby(seg).cumsum()
    neupert = neupert.where(y.notna())
    return neupert.rename("neupert")


# --------------------------------------------------------------------------- #
# 6. derivative
# --------------------------------------------------------------------------- #
def derivative(series: pd.Series, window: str = "2min", polyorder: int = 2) -> pd.Series:
    """Savitzky-Golay smoothed first derivative (per second), gap-respecting.

    The S-G filter is applied independently within each contiguous valid segment
    so the derivative is never computed across a NaN or a time gap. Segments too
    short for the S-G window fall back to a simple finite difference.

    Returns
    -------
    pd.Series
        d(series)/dt named ``<series>_deriv``.
    """
    cadence_s = infer_cadence_seconds(series.index)
    win = _window_samples(window, cadence_s, odd=True)
    if win <= polyorder:
        win = polyorder + 1 + ((polyorder + 1) % 2 == 0)

    y = series.astype("float64")
    seg = _segment_labels(y, cadence_s)
    out = pd.Series(np.nan, index=series.index, dtype="float64")

    for _label, grp in y.groupby(seg):
        grp = grp.dropna()
        n = len(grp)
        if n == 0:
            continue
        vals = grp.values
        if n >= win and win > polyorder:
            d = savgol_filter(vals, window_length=win, polyorder=polyorder,
                              deriv=1, delta=cadence_s, mode="interp")
        elif n >= 3:
            d = np.gradient(vals, cadence_s)
        else:
            d = np.zeros(n)
        out.loc[grp.index] = d
    return out.rename(f"{series.name}_deriv" if series.name else "deriv")


# --------------------------------------------------------------------------- #
# 7. HXR -> SXR cross-correlation lag (measured Neupert coupling)
# --------------------------------------------------------------------------- #
def hxr_sxr_lag(df: pd.DataFrame, window: str = "15min", max_lag: str = "5min",
                lag_step: str | None = None, min_corr: float = 0.3,
                smooth: str = "1min",
                hxr_col: str = "hxr_broad", sxr_rate_col: str = "deriv_soft",
                ) -> tuple[pd.Series, pd.Series]:
    """Rolling lag at which the hard X-ray flux best leads the soft-X-ray rise.

    The Neupert relation implies the hard-X-ray flux ``F_HXR`` *leads* the
    soft-X-ray *rise rate* ``dF_SXR/dt``. Measuring that lead time directly is a
    strong, rarely-used precursor: as an active region approaches a flare the
    HXR->SXR lag tightens. In each trailing *window* we slide ``hxr_broad`` ahead
    of ``deriv_soft`` by candidate lags ``0..max_lag`` and take the lag of peak
    Pearson cross-correlation.

    Implementation is vectorised: for each candidate lag ``L`` we compute one
    rolling correlation of ``deriv_soft`` against ``hxr_broad`` shifted forward by
    ``L`` (so HXR leads), then take the per-sample arg-max across lags. Trailing
    windows ⇒ causal (no look-ahead). The rolling correlation requires a full,
    gap-free window (``min_periods = window``) so the feature is **NaN wherever
    the window spans a data gap**, and it is also NaN where the signal is too
    weak (peak ``|corr| < min_corr`` or negligible HXR activity in the window).

    Parameters
    ----------
    df : pd.DataFrame
        Frame with *hxr_col* and *sxr_rate_col* (run after :func:`derivative`).
    window : str, default '15min'
        Trailing correlation window.
    max_lag : str, default '5min'
        Largest HXR-lead lag tested.
    lag_step : str, optional
        Spacing between candidate lags (default: ~``max_lag/20``, >= cadence).
    min_corr : float, default 0.3
        Minimum peak correlation for the lag to be considered meaningful.
    smooth : str, default '1min'
        Trailing mean applied to BOTH channels before correlating. At 1 s cadence
        the raw soft-X-ray derivative is Poisson-noise-dominated (peak |corr|
        ~0.05); a short pre-smoothing recovers the genuine Neupert coupling
        (peak |corr| ~0.6) without look-ahead. Set ``'0s'`` to disable.

    Returns
    -------
    (pd.Series, pd.Series)
        ``hxr_sxr_lag`` (seconds; HXR-lead lag at peak correlation) and
        ``hxr_sxr_xcorr`` (the peak correlation value in [-1, 1]).
    """
    cadence_s = infer_cadence_seconds(df.index)
    w = _window_samples(window, cadence_s)
    corr_min_periods = max(1, int(0.9 * w))  # tolerate tiny missingness, NaN over real gaps
    max_lag_n = _window_samples(max_lag, cadence_s)
    if lag_step is None:
        step_n = max(1, max_lag_n // 20)
    else:
        step_n = max(1, _window_samples(lag_step, cadence_s))
    lags = list(range(0, max_lag_n + 1, step_n))

    hxr = df[hxr_col].astype("float64")
    sxr_rate = df[sxr_rate_col].astype("float64")

    # Causal denoising: a short trailing mean on both channels. min_periods=1
    # keeps single-sample dropouts from punching holes, but windows spanning a
    # real (multi-sample) gap still collapse to NaN downstream.
    sm_n = _window_samples(smooth, cadence_s) if smooth and smooth != "0s" else 1
    if sm_n > 1:
        hxr = hxr.rolling(sm_n, min_periods=1).mean()
        sxr_rate = sxr_rate.rolling(sm_n, min_periods=1).mean()

    # Only correlate where the window actually contains hard-X-ray activity;
    # otherwise corr is dominated by noise. Gate on trailing windowed HXR energy.
    hxr_energy = hxr.rolling(w, min_periods=corr_min_periods).sum()
    active = hxr_energy > 0

    # Track the running best correlation. Initialise to -inf (NOT NaN) so the
    # first finite candidate always wins the comparison; NaN comparisons are
    # always False and would leave the result empty.
    best_corr = pd.Series(-np.inf, index=df.index, dtype="float64")
    best_lag = pd.Series(np.nan, index=df.index, dtype="float64")
    for L in lags:
        # hxr.shift(L): the HXR value from L samples earlier aligns with the
        # current SXR rise -> positive L means HXR leads SXR.
        corr_L = sxr_rate.rolling(w, min_periods=corr_min_periods).corr(hxr.shift(L))
        better = (corr_L > best_corr).fillna(False) & corr_L.notna()
        best_corr = best_corr.mask(better, corr_L)
        best_lag = best_lag.mask(better, float(L) * cadence_s)

    best_corr = best_corr.replace(-np.inf, np.nan)
    weak = (~active) | (best_corr < min_corr)
    best_lag = best_lag.mask(weak)
    best_corr = best_corr.mask(weak)
    return (best_lag.rename("hxr_sxr_lag"), best_corr.rename("hxr_sxr_xcorr"))


# --------------------------------------------------------------------------- #
# 8. build_features
# --------------------------------------------------------------------------- #
def detection_baseline(series: pd.Series, quiescent_window: str = "1h",
                       sigma_window: str = "30min") -> tuple[pd.Series, pd.Series]:
    """Quiescent-floor background + robust sigma tuned for flare detection.

    * Background = rolling low percentile (quiescent floor) over a long window,
      so even long-duration flares do not inflate their own baseline.
    * Sigma = max(MAD-sigma, Poisson sqrt(background)) — the Poisson floor keeps
      the threshold sane on sparse channels (e.g. the hard X-ray band is 0 cts/s
      most of the time, where MAD degenerates to 0).
    """
    bg = background(series, window=quiescent_window, stat="quantile", q=0.1)
    mad_sigma = rolling_sigma(series, window=sigma_window)
    poisson = np.sqrt(np.clip(bg, 1.0, None))
    sigma = pd.concat([mad_sigma, poisson], axis=1).max(axis=1)
    return bg, sigma.rename(mad_sigma.name)


def _hr_liveness_report(hr: pd.Series, hr_slope_series: pd.Series) -> None:
    """Diagnostic: confirm the hardness-ratio features are alive, not broken.

    HR is *defined* only when the 20-40 keV denominator carries counts, so for
    weak (C-class) flares it is legitimately NaN most of the day. This logs the
    finite fraction and the HR spread so we can honestly distinguish "weak
    because the flares are soft-dominated C-class" (finite & varying during
    flares) from "silently computed wrong" (all-NaN or constant). A clear
    warning is raised only in the genuinely-broken cases.
    """
    n = len(hr)
    finite = hr.notna()
    frac = float(finite.mean()) if n else 0.0
    slope_frac = float(hr_slope_series.notna().mean()) if n else 0.0
    if finite.any():
        vals = hr[finite]
        lo, mid, hi = (float(vals.min()), float(vals.median()), float(vals.max()))
        nuniq = int(vals.round(6).nunique())
    else:
        lo = mid = hi = float("nan")
        nuniq = 0
    logger.info("build_features[HR diagnostic]: finite=%.2f%% (hr_slope finite=%.2f%%), "
                "HR min/median/max=%.3g/%.3g/%.3g, distinct=%d",
                100.0 * frac, 100.0 * slope_frac, lo, mid, hi, nuniq)
    if frac == 0.0:
        logger.warning("build_features[HR diagnostic]: hardness ratio is ALL NaN — "
                       "the 20-40 keV channel is empty/missing, or HR is broken. "
                       "Verify hxr_20_40 / hxr_60_80 / hxr_80_150 are populated.")
    elif nuniq <= 1:
        logger.warning("build_features[HR diagnostic]: hardness ratio is CONSTANT "
                       "where finite — likely broken, not physics.")
    elif frac < 0.10:
        logger.info("build_features[HR diagnostic]: HR finite <10%% of the day — "
                    "expected for C-class-dominated activity (little 20-40 keV "
                    "emission); the feature is alive, just sparse, NOT broken.")


def build_features(
    df: pd.DataFrame,
    *,
    bg_window: str = "30min",
    quiescent_window: str = "1h",
    deriv_window: str = "2min",
    hr_slope_window: str = "5min",
    var_window: str = "1min",
    neupert_window: str = "10min",
    xcorr_window: str = "15min",
    xcorr_max_lag: str = "5min",
) -> pd.DataFrame:
    """Enrich the unified frame with all detection/forecast features.

    Adds, NaN- and gap-safely:
        background_soft, sigma_soft, deriv_soft, var_soft   (SoLEXS channel)
        background_hard, sigma_hard, deriv_hard, var_hard   (HEL1OS broadband)
        hr, hr_slope                                        (spectral hardening)
        neupert_windowed                                    (recent-HXR fluence,
                                                             local Neupert proxy)
        predicted_sxr_rise, neupert_residual                (dF_SXR/dt ~ k*F_HXR
                                                             relation + divergence)
        hxr_sxr_lag, hxr_sxr_xcorr                          (measured HXR->SXR lead)
        time_since_last_flare                               (placeholder; filled
                                                             by nowcast)

    The detection backgrounds use a quiescent-floor percentile over
    *quiescent_window* with a Poisson-floored sigma, which is robust to both
    long-duration flares and sparse hard X-ray counts. The Neupert feature is the
    *windowed* (local) integral, not a cumulative-from-midnight one, to avoid
    encoding time-of-day.

    The original columns are preserved. Returns a new (copied) dataframe.
    """
    out = df.copy()
    cadence_s = infer_cadence_seconds(out.index)
    logger.info("build_features: %d rows @ %.3gs cadence", len(out), cadence_s)

    soft = out["solexs_counts"]
    hard = out["hxr_broad"]

    out["background_soft"], out["sigma_soft"] = detection_baseline(
        soft, quiescent_window, bg_window)
    out["deriv_soft"] = derivative(soft, window=deriv_window)

    out["background_hard"], out["sigma_hard"] = detection_baseline(
        hard, quiescent_window, bg_window)
    out["deriv_hard"] = derivative(hard, window=deriv_window)

    out["hr"] = hardness_ratio(out)
    out["hr_slope"] = hr_slope(out["hr"], window=hr_slope_window)
    _hr_liveness_report(out["hr"], out["hr_slope"])

    # Local (windowed) Neupert proxy + the direct dF_SXR/dt ~ k*F_HXR relation.
    out["neupert_windowed"] = neupert_windowed(out, window=neupert_window)
    out["predicted_sxr_rise"] = K_NEUPERT * out["hxr_broad"]
    out["neupert_residual"] = out["deriv_soft"] - out["predicted_sxr_rise"]

    # Measured HXR->SXR lead lag (tightening lag = imminent flare).
    out["hxr_sxr_lag"], out["hxr_sxr_xcorr"] = hxr_sxr_lag(
        out, window=xcorr_window, max_lag=xcorr_max_lag)

    w_var = _window_samples(var_window, cadence_s)
    mp = max(2, w_var // 4)
    out["var_soft"] = soft.rolling(w_var, center=True, min_periods=mp).var()
    out["var_hard"] = hard.rolling(w_var, center=True, min_periods=mp).var()

    # Filled by the nowcast layer once flares are catalogued.
    out["time_since_last_flare"] = np.nan

    return out


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    from loader import discover_files, load_unified
    from preprocess import apply_gti, flag_gaps

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    disc = discover_files(target)
    df, meta = load_unified(target, cadence="1s")
    df = flag_gaps(apply_gti(df, disc.get("solexs_gti")))

    feats = build_features(df)
    print("\n[feature columns]")
    print(list(feats.columns))
    print("\n[describe — feature subset]")
    cols = ["solexs_counts", "background_soft", "sigma_soft", "deriv_soft",
            "hxr_broad", "hr", "hr_slope", "neupert_windowed", "neupert_residual",
            "hxr_sxr_lag", "hxr_sxr_xcorr"]
    with pd.option_context("display.float_format", lambda x: f"{x:,.4f}",
                           "display.max_columns", None, "display.width", 200):
        print(feats[cols].describe())
    print("\n[head around the strongest soft sample]")
    pk = feats["solexs_counts"].idxmax()
    print(f"peak solexs sample at {pk}: {feats.loc[pk, 'solexs_counts']:.1f} cts/s "
          f"(bg~{feats.loc[pk, 'background_soft']:.1f})")
