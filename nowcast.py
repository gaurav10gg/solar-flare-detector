"""
nowcast.py — Classical (non-ML) flare detection & cataloguing.

Detects flares *as they happen* on the soft (SoLEXS) and hard (HEL1OS broadband)
channels independently, classifies soft-channel peaks onto the GOES A/B/C/M/X
scale, then time-matches the two channels into a single explainable catalogue.

Detection philosophy:
    Onset rule (per PS): flux > background + k*sigma  AND  derivative > 0,
    *sustained* for longer than a persistence window. The persistence rule kills
    single-sample spikes and SAA particle hits. A lower hysteresis threshold
    defines the flare envelope so we capture the full rise+decay.

Large dynamic range is a grading criterion, so ``sigma_mult`` and
``persistence`` are configurable: small B/C bumps just above noise *and* huge
M/X flares are both caught, and big flares never "saturate" the logic (the
envelope simply spans the whole elevated region).

Public API:
    detect_flares(series, background, sigma, deriv, ...)  -> list[dict]
    classify_goes(peak_flux_solexs)                       -> str
    detect_soft(df) / detect_hard(df)                     -> list[dict]
    merge_catalog(soft, hard, match_window='2min')        -> pd.DataFrame
    save_catalog(df, sqlite_path)                         -> None
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

from features import (build_features, infer_cadence_seconds,
                      detection_baseline, derivative)

logger = logging.getLogger("nowcast")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# GOES cross-calibration (documented placeholder — tune against NOAA catalog)
# --------------------------------------------------------------------------- #
# Linear map from SoLEXS soft X-ray count-rate (cts/s, ~1-15 keV) to GOES
# 1-8 A long-channel irradiance (W/m^2). This is a first-order cross-calibration
# anchor; the proper value is obtained by regressing simultaneous SoLEXS peaks
# against the public NOAA/GOES flare catalogue (done in the cross-cal step).
#
# TODO-TUNABLE: replace with the fitted slope once GOES matches are available.
COUNTS_TO_WM2 = 5.0e-8  # W/m^2 per (ct/s)

_GOES_BANDS = [
    ("A", 1e-8, 1e-7),
    ("B", 1e-7, 1e-6),
    ("C", 1e-6, 1e-5),
    ("M", 1e-5, 1e-4),
    ("X", 1e-4, np.inf),
]


def classify_goes(peak_flux_solexs: float) -> str:
    """Map a SoLEXS peak count-rate to a GOES class string (e.g. ``'C9.2'``).

    Applies the documented linear cross-calibration (``COUNTS_TO_WM2``) to get
    GOES 1-8 A irradiance in W/m^2, then assigns the standard letter + magnitude.

    Parameters
    ----------
    peak_flux_solexs : float
        Peak SoLEXS count-rate (cts/s), background-inclusive.

    Returns
    -------
    str
        GOES class, or ``'?'`` if the input is NaN/negative.
    """
    if peak_flux_solexs is None or not np.isfinite(peak_flux_solexs) or peak_flux_solexs < 0:
        return "?"
    flux = float(peak_flux_solexs) * COUNTS_TO_WM2
    for letter, lo, hi in _GOES_BANDS:
        if flux < hi:
            base = lo if letter != "A" else 1e-8
            magnitude = flux / base
            # GOES magnitudes are conventionally quoted 1.0-9.9 within a band.
            return f"{letter}{max(magnitude, 0.0):.1f}"
    return "X"


# --------------------------------------------------------------------------- #
# Core detector
# --------------------------------------------------------------------------- #
def detect_flares(
    series: pd.Series,
    background: pd.Series,
    sigma: pd.Series,
    deriv: pd.Series,
    *,
    sigma_mult: float = 3.0,
    persistence: str = "30s",
    end_sigma_mult: float = 1.0,
    rise_frac: float = 0.25,
    channel: str = "soft",
) -> list[dict]:
    """Detect flares on a single count-rate series. Works on soft OR hard data.

    Algorithm
    ---------
    1. **Envelope** — contiguous regions where ``series > background +
       end_sigma_mult*sigma`` define candidate flares (hysteresis: a low exit
       threshold so the full rise+decay is captured, big flares never clip).
    2. **Persistence** — a region is accepted only if the flux stays above the
       onset threshold ``background + sigma_mult*sigma`` for at least
       *persistence* of continuous time somewhere inside it. This rejects single
       spikes / SAA particle hits while still admitting weak but real B/C bumps.
    3. **Rise gate** — the onset must be a genuine rise: a fraction >= *rise_frac*
       of the samples on the leading edge (region start -> peak) must have a
       positive smoothed derivative. This separates flares from flat elevated
       plateaus and works on both the smooth soft and the spiky hard channel.

    Parameters
    ----------
    series, background, sigma, deriv : pd.Series
        Aligned channel value, baseline, robust sigma and smoothed derivative.
    sigma_mult : float, default 3.0
        Onset threshold in sigmas above background (configurable for dynamic range).
    persistence : str, default '30s'
        Minimum sustained above-threshold duration.
    end_sigma_mult : float, default 1.0
        Envelope (exit) threshold in sigmas above background.
    rise_frac : float, default 0.25
        Minimum fraction of leading-edge samples with positive derivative.
    channel : str
        Provenance label stored on each flare ('soft' or 'hard').

    Returns
    -------
    list[dict]
        Each: ``{start, peak_time, peak_flux, peak_flux_bg_subtracted, end,
        duration, fluence, snr, channel}``.
    """
    idx = series.index
    cadence_s = infer_cadence_seconds(idx)
    persist_n = max(1, int(round(pd.Timedelta(persistence).total_seconds() / cadence_s)))

    s = series.to_numpy(dtype="float64")
    bg = background.reindex(idx).to_numpy(dtype="float64")
    sg = sigma.reindex(idx).to_numpy(dtype="float64")
    dv = deriv.reindex(idx).to_numpy(dtype="float64")

    valid = np.isfinite(s) & np.isfinite(bg) & np.isfinite(sg)
    onset_thr = bg + sigma_mult * sg
    exit_thr = bg + end_sigma_mult * sg

    env = valid & (s > exit_thr)
    onset_mask = valid & (s > onset_thr)

    # Persistence: rolling count of consecutive above-onset samples >= persist_n.
    om_i = onset_mask.astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(om_i)])
    window_sum = csum[persist_n:] - csum[:-persist_n]      # length n-persist_n+1
    sustained = np.zeros(len(s), dtype=bool)
    ends = np.where(window_sum >= persist_n)[0] + (persist_n - 1)
    sustained[ends] = True

    pos_deriv = np.nan_to_num(dv, nan=-1.0) > 0

    flares: list[dict] = []
    # Iterate contiguous envelope regions.
    in_region = False
    a = 0
    for i in range(len(s) + 1):
        cur = env[i] if i < len(s) else False
        if cur and not in_region:
            in_region, a = True, i
        elif not cur and in_region:
            in_region = False
            b = i - 1  # inclusive region end
            if not sustained[a:b + 1].any():
                continue  # spike / particle hit -> rejected by persistence
            region = slice(a, b + 1)
            seg = s[region]
            seg_bg = bg[region]
            k = int(np.nanargmax(seg))
            peak_pos = a + k

            # Rise gate: leading edge (start -> peak) must be a genuine rise.
            rise = pos_deriv[a:peak_pos + 1]
            if len(rise) >= 2 and float(np.mean(rise)) < rise_frac:
                continue

            peak_flux = float(seg[k])
            bg_at_peak = float(seg_bg[k]) if np.isfinite(seg_bg[k]) else float(np.nanmedian(seg_bg))
            peak_bg_sub = peak_flux - bg_at_peak
            above = np.clip(seg - seg_bg, a_min=0.0, a_max=None)
            fluence = float(np.nansum(above) * cadence_s)
            sig_at_peak = sg[peak_pos] if np.isfinite(sg[peak_pos]) else np.nan
            snr = float(peak_bg_sub / sig_at_peak) if sig_at_peak and np.isfinite(sig_at_peak) else np.nan
            flares.append({
                "start": idx[a],
                "peak_time": idx[peak_pos],
                "peak_flux": peak_flux,
                "peak_flux_bg_subtracted": peak_bg_sub,
                "end": idx[b],
                "duration": (idx[b] - idx[a]).total_seconds(),
                "fluence": fluence,
                "snr": snr,
                "channel": channel,
            })

    logger.info("detect_flares[%s]: %d flares (k=%.1f, persist=%s)",
                channel, len(flares), sigma_mult, persistence)
    return flares


# --------------------------------------------------------------------------- #
# Per-channel convenience wrappers
# --------------------------------------------------------------------------- #
def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"background_soft", "sigma_soft", "deriv_soft",
              "background_hard", "sigma_hard", "deriv_hard"}
    return df if needed <= set(df.columns) else build_features(df)


def detect_soft(df: pd.DataFrame, **kwargs) -> list[dict]:
    """Detect thermal/gradual flares on the SoLEXS soft X-ray channel."""
    f = _ensure_features(df)
    return detect_flares(f["solexs_counts"], f["background_soft"], f["sigma_soft"],
                         f["deriv_soft"], channel="soft", **kwargs)


def detect_hard(df: pd.DataFrame, smooth: str = "30s", **kwargs) -> list[dict]:
    """Detect flares on the HEL1OS hard X-ray broadband channel.

    HEL1OS broadband at 1 s cadence is genuinely impulsive — it alternates
    between 0 and short photon bursts, so a raw continuous-persistence rule can
    never trigger. We therefore smooth (rebin) the broadband over *smooth* to
    recover the elevated-activity envelope, recompute the quiescent baseline,
    Poisson-floored sigma and derivative on that smoothed series, then run the
    standard detector.

    Parameters
    ----------
    df : pd.DataFrame
        Unified/feature frame (must contain ``hxr_broad``).
    smooth : str, default '30s'
        Rolling-mean window used to recover the hard X-ray activity envelope.
    **kwargs
        Forwarded to :func:`detect_flares` (e.g. ``sigma_mult``, ``persistence``).
    """
    f = _ensure_features(df)
    hard = f["hxr_broad"]
    cadence_s = infer_cadence_seconds(hard.index)
    n = max(1, int(round(pd.Timedelta(smooth).total_seconds() / cadence_s)))
    smoothed = hard.rolling(n, center=True, min_periods=max(1, n // 3)).mean()
    smoothed = smoothed.rename("hxr_broad")

    bg, sigma = detection_baseline(smoothed, quiescent_window="1h", sigma_window="30min")
    deriv = derivative(smoothed, window="2min")

    kwargs.setdefault("persistence", "30s")
    return detect_flares(smoothed, bg, sigma, deriv, channel="hard", **kwargs)


# --------------------------------------------------------------------------- #
# Catalogue merge
# --------------------------------------------------------------------------- #
def _confidence(provenance: str, soft_snr: float, hard_snr: float) -> float:
    """Confidence 0-1 from cross-channel agreement and SNR."""
    base = {"both": 0.80, "soft_only": 0.55, "hard_only": 0.40}[provenance]
    snrs = [x for x in (soft_snr, hard_snr) if x is not None and np.isfinite(x)]
    snr_factor = min(1.0, (max(snrs) / 10.0)) if snrs else 0.0
    return float(np.clip(base + (1.0 - base) * snr_factor, 0.0, 1.0))


# Provenance -> physical category. A soft (thermal) response is the defining
# signature of a true solar flare (Neupert coupling), so only soft/both events
# are "confirmed". Hard-only transients lack any thermal counterpart and are far
# more likely particle hits / instrumental spikes / non-thermal microflares —
# they are reported as candidates needing vetting, never counted as flares.
_CATEGORY = {"both": "confirmed_flare", "soft_only": "confirmed_flare",
             "hard_only": "hxr_candidate"}


def merge_catalog(
    soft_flares: list[dict],
    hard_flares: list[dict],
    match_window: str = "2min",
) -> pd.DataFrame:
    """Time-match soft & hard detections into one provenance-tagged catalogue.

    A soft and a hard flare whose peaks fall within *match_window* are merged
    into a single ``'both'`` entry (a real flare, high confidence). Unmatched
    soft flares are ``'soft_only'`` (gradual/thermal); unmatched hard flares are
    ``'hard_only'`` (transient candidates, lower confidence).

    Each row also carries a physical ``category``:

    * ``'confirmed_flare'`` — provenance ``both`` or ``soft_only`` (a thermal
      soft-X-ray response is present, the hallmark of a real flare);
    * ``'hxr_candidate'`` — provenance ``hard_only`` (no thermal counterpart:
      a possible particle hit / instrumental spike / non-thermal microflare that
      needs vetting). These are NOT counted as confirmed flares.

    Returns
    -------
    pd.DataFrame
        Columns: ``start, peak_time, end, duration, peak_flux, goes_class,
        provenance, category, confidence`` (plus ``snr`` diagnostics), sorted by
        peak_time.
    """
    tol = pd.Timedelta(match_window)
    hard_used = [False] * len(hard_flares)

    rows: list[dict] = []

    for sf in soft_flares:
        # Best unused hard flare: peaks within tolerance OR overlapping envelopes
        # (impulsive hard bursts ride inside the slow soft rise — Neupert).
        best_j, best_dt = None, None
        for j, hf in enumerate(hard_flares):
            if hard_used[j]:
                continue
            dt = abs(hf["peak_time"] - sf["peak_time"])
            overlaps = (hf["start"] <= sf["end"]) and (hf["end"] >= sf["start"])
            if (dt <= tol or overlaps) and (best_dt is None or dt < best_dt):
                best_j, best_dt = j, dt
        if best_j is not None:
            hf = hard_flares[best_j]
            hard_used[best_j] = True
            provenance = "both"
            start = min(sf["start"], hf["start"])
            end = max(sf["end"], hf["end"])
            rows.append({
                "start": start,
                "peak_time": sf["peak_time"],
                "end": end,
                "duration": (end - start).total_seconds(),
                "peak_flux": sf["peak_flux"],
                "goes_class": classify_goes(sf["peak_flux"]),
                "provenance": provenance,
                "category": _CATEGORY[provenance],
                "confidence": _confidence(provenance, sf.get("snr"), hf.get("snr")),
                "soft_snr": sf.get("snr"),
                "hard_snr": hf.get("snr"),
            })
        else:
            rows.append({
                "start": sf["start"],
                "peak_time": sf["peak_time"],
                "end": sf["end"],
                "duration": sf["duration"],
                "peak_flux": sf["peak_flux"],
                "goes_class": classify_goes(sf["peak_flux"]),
                "provenance": "soft_only",
                "category": _CATEGORY["soft_only"],
                "confidence": _confidence("soft_only", sf.get("snr"), None),
                "soft_snr": sf.get("snr"),
                "hard_snr": np.nan,
            })

    for j, hf in enumerate(hard_flares):
        if hard_used[j]:
            continue
        rows.append({
            "start": hf["start"],
            "peak_time": hf["peak_time"],
            "end": hf["end"],
            "duration": hf["duration"],
            "peak_flux": hf["peak_flux"],
            "goes_class": "-",  # hard channel does not map to GOES soft band
            "provenance": "hard_only",
            "category": _CATEGORY["hard_only"],
            "confidence": _confidence("hard_only", None, hf.get("snr")),
            "soft_snr": np.nan,
            "hard_snr": hf.get("snr"),
        })

    cols = ["start", "peak_time", "end", "duration", "peak_flux",
            "goes_class", "provenance", "category", "confidence", "soft_snr", "hard_snr"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values("peak_time").reset_index(drop=True)
    return df[cols]


def catalog_summary(catalog: pd.DataFrame) -> dict:
    """Honest headline counts: confirmed flares vs hard-X-ray candidates.

    Never collapses the two into a single inflated "N flares" number. Returns
    ``{n_confirmed, n_confirmed_with_hard, n_candidates, n_total, headline}``
    where ``headline`` is a ready-to-display string such as
    ``"5 confirmed flares (1 hard-X-ray confirmed) + 13 HXR transient candidates"``.
    """
    if catalog is None or len(catalog) == 0:
        return {"n_confirmed": 0, "n_confirmed_with_hard": 0, "n_candidates": 0,
                "n_total": 0, "headline": "no detections"}
    cat = catalog.get("category")
    prov = catalog.get("provenance")
    is_confirmed = (cat == "confirmed_flare") if cat is not None else \
        prov.isin(["both", "soft_only"])
    n_confirmed = int(is_confirmed.sum())
    n_with_hard = int((prov == "both").sum()) if prov is not None else 0
    n_candidates = int(len(catalog) - n_confirmed)
    headline = (f"{n_confirmed} confirmed flare{'s' if n_confirmed != 1 else ''} "
                f"({n_with_hard} hard-X-ray confirmed) + {n_candidates} "
                f"HXR transient candidate{'s' if n_candidates != 1 else ''}")
    return {"n_confirmed": n_confirmed, "n_confirmed_with_hard": n_with_hard,
            "n_candidates": n_candidates, "n_total": int(len(catalog)),
            "headline": headline}


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_catalog(df: pd.DataFrame, sqlite_path: str, table: str = "flares") -> None:
    """Write the flare catalogue to a SQLite table (datetimes as ISO-8601 UTC)."""
    out = df.copy()
    for col in ("start", "peak_time", "end"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True).map(
                lambda t: t.isoformat() if pd.notna(t) else None
            )
    with sqlite3.connect(sqlite_path) as conn:
        out.to_sql(table, conn, if_exists="replace", index=False)
    logger.info("save_catalog: wrote %d rows to %s::%s", len(out), sqlite_path, table)


# --------------------------------------------------------------------------- #
# Smoke test / acceptance
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

    soft = detect_soft(feats, sigma_mult=3.0, persistence="30s")
    hard = detect_hard(feats, sigma_mult=3.0)  # defaults: 30s smoothing, 30s persistence
    catalog = merge_catalog(soft, hard, match_window="2min")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")

    print(f"\n[catalogue] {len(catalog)} flares "
          f"(soft={len(soft)}, hard={len(hard)})")
    print(catalog.to_string(index=False))

    if len(catalog):
        prov = catalog["provenance"].value_counts().to_dict()
        print("\n[provenance]", prov)
        print("[GOES classes]", catalog["goes_class"].value_counts().to_dict())

    # Acceptance: the obvious strong soft flare (peak ~184 cts/s) must be caught.
    pk_time = feats["solexs_counts"].idxmax()
    pk_val = float(feats["solexs_counts"].max())
    hit = catalog[(catalog["peak_time"] - pk_time).abs() <= pd.Timedelta("5min")] \
        if len(catalog) else catalog
    print(f"\n[acceptance] strongest soft sample = {pk_val:.1f} cts/s at {pk_time}")
    if len(hit):
        r = hit.iloc[0]
        print(f"  DETECTED: {r['provenance']} flare, peak {r['peak_flux']:.1f} cts/s, "
              f"class {r['goes_class']}, confidence {r['confidence']:.2f}")
    else:
        print("  !! BIG FLARE MISSED — investigate thresholds.")

    out_db = "data/catalog/flares.sqlite"
    import os
    os.makedirs(os.path.dirname(out_db), exist_ok=True)
    save_catalog(catalog, out_db)
