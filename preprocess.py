"""
preprocess.py — Layer 0 cleaning for the unified SoLEXS + HEL1OS frame.

Operates on the dataframe produced by :func:`loader.load_unified` (tz-aware UTC
``time_utc`` index; columns ``solexs_counts, hxr_broad, hxr_20_40, hxr_40_60,
hxr_60_80, hxr_80_150``).

Golden rule: **never interpolate across data gaps** — false flares hide there.
Gaps, GTI exclusions and missing detectors all stay as NaN so downstream
detectors can treat them explicitly.

Public API:
    apply_gti(df, gti_paths)        -> pd.DataFrame
    flag_gaps(df, max_gap='5s')     -> pd.DataFrame
    resample(df, cadence)           -> pd.DataFrame
    quality_report(df)              -> dict

Run as a script for a smoke test::

    python preprocess.py [PATH]
"""

from __future__ import annotations

import logging
import os
import re
from typing import Sequence

import numpy as np
import pandas as pd
from astropy.io import fits

logger = logging.getLogger("preprocess")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# Columns grouped by instrument, so a SoLEXS GTI never nulls HEL1OS data.
_SOLEXS_COLUMNS = ["solexs_counts"]
_HEL1OS_COLUMNS = ["hxr_broad", "hxr_20_40", "hxr_40_60", "hxr_60_80", "hxr_80_150"]


# --------------------------------------------------------------------------- #
# 1. apply_gti
# --------------------------------------------------------------------------- #
def _read_gti_intervals(path: str) -> tuple[np.ndarray | None, str]:
    """Read a ``.gti.gz`` GTI extension -> (N,2) array of Unix-second intervals.

    Returns ``(intervals, instrument)``. ``instrument`` is lower-cased and used
    to decide which columns the GTI applies to. On any failure returns
    ``(None, '')`` so the caller can skip gracefully.
    """
    try:
        with fits.open(path, memmap=False) as hdul:
            gti_hdu = None
            for hdu in hdul:
                if hdu.name.upper() == "GTI" or (
                    getattr(hdu, "columns", None) is not None
                    and {"START", "STOP"} <= {c.name.upper() for c in hdu.columns}
                ):
                    gti_hdu = hdu
                    break
            if gti_hdu is None or gti_hdu.data is None or len(gti_hdu.data) == 0:
                logger.warning("GTI %s: no usable GTI extension, skipping.",
                               os.path.basename(path))
                return None, ""
            cols = {c.name.upper(): c.name for c in gti_hdu.columns}
            start = np.asarray(gti_hdu.data[cols["START"]], dtype="float64")
            stop = np.asarray(gti_hdu.data[cols["STOP"]], dtype="float64")
            instrument = str(hdul[0].header.get("INSTRUME", "")).lower()
            return np.column_stack([start, stop]), instrument
    except Exception as exc:
        logger.warning("GTI %s: failed to read (%s), skipping.",
                       os.path.basename(path), exc)
        return None, ""


def apply_gti(df: pd.DataFrame, gti_paths: Sequence[str] | str | None) -> pd.DataFrame:
    """Mask samples that fall OUTSIDE every Good-Time-Interval as NaN.

    Each GTI file's instrument is detected from its header; the GTI only masks
    that instrument's columns (a SoLEXS GTI never nulls HEL1OS data, and vice
    versa). If the instrument is unknown, all data columns are masked. With no
    GTI files this is a no-op.

    Parameters
    ----------
    df : pd.DataFrame
        Unified frame with a tz-aware UTC index.
    gti_paths : sequence of str | str | None
        Paths to ``.gti.gz`` files (e.g. ``discover_files(...)['solexs_gti']``).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with out-of-GTI samples set to NaN.
    """
    if not gti_paths:
        logger.info("apply_gti: no GTI files supplied — no-op.")
        return df
    if isinstance(gti_paths, str):
        gti_paths = [gti_paths]

    out = df.copy()
    # Unix seconds for fast interval testing.
    unix = out.index.view("int64") / 1e9

    for path in gti_paths:
        intervals, instrument = _read_gti_intervals(path)
        if intervals is None:
            continue

        if "solexs" in instrument:
            target_cols = [c for c in _SOLEXS_COLUMNS if c in out.columns]
        elif "hel1os" in instrument or "czt" in instrument:
            target_cols = [c for c in _HEL1OS_COLUMNS if c in out.columns]
        else:
            target_cols = list(out.columns)
            logger.warning("GTI %s: unknown instrument %r — masking all columns.",
                           os.path.basename(path), instrument)

        inside = np.zeros(len(out), dtype=bool)
        for start, stop in intervals:
            inside |= (unix >= start) & (unix <= stop)

        masked = (~inside).sum()
        out.loc[~inside, target_cols] = np.nan
        logger.info("apply_gti: %s (%s) -> %d intervals, masked %d/%d rows of %s",
                    os.path.basename(path), instrument or "unknown",
                    len(intervals), int(masked), len(out), target_cols)
    return out


# --------------------------------------------------------------------------- #
# 2. flag_gaps
# --------------------------------------------------------------------------- #
def flag_gaps(df: pd.DataFrame, max_gap: str = "5s") -> pd.DataFrame:
    """Make data gaps explicit on a regular grid; never interpolate across them.

    The cadence is inferred from the median index spacing. The frame is
    reindexed onto a continuous regular grid so that any missing timestamps
    appear as NaN rows (rather than as silent jumps in an irregular index).
    Existing NaNs are preserved. No value is ever filled across a gap.

    Parameters
    ----------
    df : pd.DataFrame
        Unified frame with a tz-aware UTC index.
    max_gap : str, default '5s'
        Spacing above which a jump is logged as a real gap (diagnostic only;
        all gaps are NaN-flagged regardless).

    Returns
    -------
    pd.DataFrame
        Reindexed onto a continuous grid with NaN-filled gaps.
    """
    if df.empty or len(df) < 2:
        return df

    diffs = df.index.to_series().diff().dropna()
    cadence = diffs.median()
    if cadence <= pd.Timedelta(0):
        logger.warning("flag_gaps: non-positive inferred cadence, returning unchanged.")
        return df

    max_gap_td = pd.Timedelta(max_gap)
    big = diffs[diffs > max_gap_td]
    if len(big):
        total_missing = (big - cadence).sum()
        logger.info("flag_gaps: %d gaps > %s (largest %s, ~%s missing total).",
                    len(big), max_gap, big.max(), total_missing)

    grid = pd.date_range(df.index.min(), df.index.max(), freq=cadence,
                         tz="UTC", name=df.index.name or "time_utc")
    out = df.reindex(grid)
    added = len(out) - len(df)
    if added > 0:
        logger.info("flag_gaps: inserted %d NaN rows to complete the %s grid.",
                    added, cadence)
    return out


# --------------------------------------------------------------------------- #
# 3. resample
# --------------------------------------------------------------------------- #
def resample(df: pd.DataFrame, cadence: str) -> pd.DataFrame:
    """Resample to *cadence*, preserving gaps as NaN.

    All channels are count-*rates* (cts/s), so bins are aggregated by mean. An
    all-NaN bin stays NaN (empty bins are never invented as zeros).

    Parameters
    ----------
    df : pd.DataFrame
        Unified frame with a tz-aware UTC index.
    cadence : str
        Target pandas offset alias (e.g. ``'1s'``, ``'10s'``, ``'1min'``).

    Returns
    -------
    pd.DataFrame
        Resampled frame; gaps remain NaN.
    """
    if df.empty:
        return df
    # mean() over a bin returns NaN when the bin is entirely NaN/empty.
    out = df.resample(cadence).mean()
    out.index.name = df.index.name or "time_utc"
    logger.info("resample: %d rows -> %d rows @ %s", len(df), len(out), cadence)
    return out


# --------------------------------------------------------------------------- #
# 4. quality_report
# --------------------------------------------------------------------------- #
def quality_report(df: pd.DataFrame) -> dict:
    """Summarise data quality of a unified frame.

    Returns
    -------
    dict
        ``{n_rows, cadence_seconds, time_range, pct_missing (per column),
        gap_count, largest_gap, detectors_present, total_exposure_seconds
        (per column), duty_cycle (per column)}``.
    """
    if df.empty:
        return {"n_rows": 0, "note": "empty frame"}

    n = len(df)
    diffs = df.index.to_series().diff().dropna()
    cadence = diffs.median() if len(diffs) else pd.Timedelta(seconds=1)
    cadence_s = cadence.total_seconds() if cadence > pd.Timedelta(0) else 1.0

    pct_missing = {c: round(float(df[c].isna().mean()) * 100.0, 3) for c in df.columns}

    # A "gap" = an index step larger than ~1.5x the nominal cadence.
    gap_threshold = cadence * 1.5
    gaps = diffs[diffs > gap_threshold]
    largest_gap = str(gaps.max()) if len(gaps) else "0s"

    detectors_present = [c for c in df.columns if df[c].notna().any()]

    # Exposure = valid samples * cadence (seconds of real signal per channel).
    total_exposure = {
        c: round(float(df[c].notna().sum()) * cadence_s, 1) for c in df.columns
    }
    span_seconds = (df.index.max() - df.index.min()).total_seconds() + cadence_s
    duty_cycle = {
        c: round(total_exposure[c] / span_seconds * 100.0, 2) if span_seconds else 0.0
        for c in df.columns
    }

    return {
        "n_rows": int(n),
        "cadence_seconds": round(cadence_s, 3),
        "time_range": (str(df.index.min()), str(df.index.max())),
        "pct_missing": pct_missing,
        "gap_count": int(len(gaps)),
        "largest_gap": largest_gap,
        "detectors_present": detectors_present,
        "total_exposure_seconds": total_exposure,
        "duty_cycle_pct": duty_cycle,
    }


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    from loader import discover_files, load_unified

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    logger.info("Smoke test preprocessing from: %s", os.path.abspath(target))

    disc = discover_files(target)
    df, meta = load_unified(target, cadence="1s")
    print("\n[loader metadata]")
    for k, v in meta.items():
        print(f"  {k:18s}: {v}")

    print("\n[1] apply_gti")
    df_gti = apply_gti(df, disc.get("solexs_gti"))

    print("\n[2] flag_gaps")
    df_gapped = flag_gaps(df_gti, max_gap="5s")

    print("\n[3] resample -> 10s")
    df_10s = resample(df_gapped, "10s")
    print(df_10s.head())

    print("\n[4] quality_report (1s, post-GTI)")
    qr = quality_report(df_gapped)
    for k, v in qr.items():
        print(f"  {k:24s}: {v}")

    print("\n[describe @1s post-GTI]")
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
        print(df_gapped.describe())
