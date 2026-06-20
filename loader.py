"""
loader.py — Layer 0 ingestion for Aditya-L1 SoLEXS + HEL1OS flare forecasting.

Turns an uploaded archive / folder / file-list (for ANY observation date) into a
single, clean, UTC-aligned dataframe ready for feature engineering.

Design contract (generalization):
    * Nothing is hardcoded to a date, a row count, a number of files, or a flare.
    * Files are auto-discovered inside arbitrarily nested folders / zips.
    * The observation date is auto-detected from filenames or FITS headers.
    * Missing detectors (e.g. SoLEXS SDD1 absent), single-file or N-file HEL1OS
      splits, differing row counts, and data gaps are all handled gracefully.

Confirmed schema (treated as schema, not as fixed values):
    SoLEXS  .lc.gz  -> ext 'RATE'      cols TIME (Unix s), COUNTS (cts/s)
    SoLEXS  .gti.gz -> ext 'GTI'       cols START, STOP (Unix s)
    HEL1OS  lightcurve_czt{1,2}.fits -> 5 band extensions
            named  'CZT{n}_LC_BAND_<lo>KEV_TO_<hi>KEV'
            cols  MJD, ISOT (UTC str), CTR (cts/sec), STAT_ERR

Public API:
    discover_files(path)            -> dict
    read_solexs(paths)              -> pd.DataFrame
    read_hel1os(paths)              -> pd.DataFrame
    load_unified(path, cadence)     -> (pd.DataFrame, dict)

Run as a script for a smoke test::

    python loader.py [PATH]
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

logger = logging.getLogger("loader")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# Canonical output columns, in the EXACT required order.
DATA_COLUMNS = [
    "solexs_counts",
    "hxr_broad",
    "hxr_20_40",
    "hxr_40_60",
    "hxr_60_80",
    "hxr_80_150",
]

# Expected detector sets (used only to report what is *missing*, never assumed present).
_EXPECTED_SOLEXS_DETECTORS = {"SDD1", "SDD2"}
_EXPECTED_HEL1OS_DETECTORS = {"czt1", "czt2"}

_DATE_RE = re.compile(r"(20\d{2})[-_]?(0[1-9]|1[0-2])[-_]?(0[1-9]|[12]\d|3[01])")
_BAND_RE = re.compile(r"(\d+(?:\.\d+)?)\s*KEV[_\s]*TO[_\s]*(\d+(?:\.\d+)?)\s*KEV", re.I)
_SDD_RE = re.compile(r"SDD\s*([12])", re.I)
_CZT_RE = re.compile(r"czt\s*([12])", re.I)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _as_path_list(path) -> list[str]:
    """Normalise the *path* argument to a flat list of existing paths."""
    if isinstance(path, (str, os.PathLike)):
        return [os.fspath(path)]
    if isinstance(path, Iterable):
        return [os.fspath(p) for p in path]
    raise TypeError(f"Unsupported path argument: {type(path)!r}")


# Temp dirs created during extraction; cleaned at process exit as a safety net
# (the API/pipeline also clean up per-job via cleanup_tempdirs()).
_TEMP_DIRS: list[str] = []
_ATEXIT_REGISTERED = False


def _cleanup_all_tempdirs() -> None:
    for d in list(_TEMP_DIRS):
        shutil.rmtree(d, ignore_errors=True)
    _TEMP_DIRS.clear()


def cleanup_tempdirs(discovery: dict | None) -> None:
    """Remove temp extraction dirs created for a discovery result (call when done).

    Safe to call once data has been read into memory. Long-running services
    (the API) should call this after each job to avoid filling the disk.
    """
    if not discovery:
        return
    for d in discovery.get("_tempdirs", []):
        shutil.rmtree(d, ignore_errors=True)
        if d in _TEMP_DIRS:
            _TEMP_DIRS.remove(d)


def _extract_zip(zip_path: str, into: str | None = None) -> str:
    """Extract *zip_path* to a temp dir (or *into*) and return the destination."""
    global _ATEXIT_REGISTERED
    dest = into or tempfile.mkdtemp(prefix="al1_ingest_")
    logger.info("Extracting archive %s -> %s", os.path.basename(zip_path), dest)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    _TEMP_DIRS.append(dest)
    if not _ATEXIT_REGISTERED:
        atexit.register(_cleanup_all_tempdirs)
        _ATEXIT_REGISTERED = True
    return dest


def _walk_files(root: str) -> list[str]:
    """Return every file under *root* (recursively)."""
    out: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            out.append(os.path.join(dirpath, f))
    return out


def _detect_date(paths: Sequence[str]) -> str | None:
    """Infer the YYYYMMDD observation date, preferring filenames then FITS headers."""
    votes: Counter[str] = Counter()
    for p in paths:
        m = _DATE_RE.search(os.path.basename(p))
        if m:
            votes[f"{m.group(1)}{m.group(2)}{m.group(3)}"] += 1
    if votes:
        date = votes.most_common(1)[0][0]
        logger.info("Detected observation date from filenames: %s", date)
        return date

    # Fallback: peek into FITS headers for an ISO date.
    for p in paths:
        try:
            with fits.open(p, memmap=False) as hdul:
                for hdu in hdul:
                    hdr = hdu.header
                    for key in ("DATE-OBS", "DATE_OBS", "TSTART"):
                        val = hdr.get(key)
                        if isinstance(val, str):
                            m = _DATE_RE.search(val)
                            if m:
                                date = f"{m.group(1)}{m.group(2)}{m.group(3)}"
                                logger.info("Detected date from header %s: %s", key, date)
                                return date
        except Exception:
            continue
    logger.warning("Could not auto-detect observation date.")
    return None


def _unix_to_utc_index(unix_seconds) -> pd.DatetimeIndex:
    """Vectorised Unix-epoch-seconds -> tz-aware UTC DatetimeIndex."""
    return pd.to_datetime(np.asarray(unix_seconds, dtype="float64"), unit="s", utc=True)


def _mjd_to_utc_index(mjd) -> pd.DatetimeIndex:
    """Vectorised MJD (UTC scale) -> tz-aware UTC DatetimeIndex via astropy."""
    mjd = np.asarray(mjd, dtype="float64")
    unix = Time(mjd, format="mjd", scale="utc").unix
    return pd.to_datetime(unix, unit="s", utc=True)


def _band_label(extname: str) -> str | None:
    """Map a HEL1OS band extension name to a canonical column, by energy range."""
    m = _BAND_RE.search(extname or "")
    if not m:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    if lo <= 18.0 and hi >= 150.0:  # the wide broadband channel
        return "hxr_broad"
    pairs = {
        (20, 40): "hxr_20_40",
        (40, 60): "hxr_40_60",
        (60, 80): "hxr_60_80",
        (80, 150): "hxr_80_150",
    }
    return pairs.get((round(lo), round(hi)))


def _detector_of(path: str, regex: re.Pattern, prefix: str) -> str | None:
    m = regex.search(os.path.basename(path))
    return f"{prefix}{m.group(1)}" if m else None


def _coadd(series_list: Sequence[pd.Series]) -> pd.Series:
    """Co-add aligned series; NaN only where *all* inputs are NaN at a timestamp."""
    series_list = [s for s in series_list if s is not None and len(s)]
    if not series_list:
        return pd.Series(dtype="float64")
    if len(series_list) == 1:
        return series_list[0]
    frame = pd.concat(series_list, axis=1)
    return frame.sum(axis=1, min_count=1)


# --------------------------------------------------------------------------- #
# 1. discover_files
# --------------------------------------------------------------------------- #
def discover_files(path) -> dict:
    """Auto-discover instrument files from a zip, folder, or list of paths.

    Accepts a ``.zip`` file, a directory, a single data file, or any list /
    mixture of those. Archives (including nested ones) are extracted to a temp
    directory, then the tree is recursively globbed for the instrument files.

    Parameters
    ----------
    path : str | os.PathLike | Iterable[str]
        Upload location(s). Never assumes the number or layout of files.

    Returns
    -------
    dict
        ``{'solexs': [paths], 'hel1os_czt': [paths], 'solexs_gti': [paths],
        'date': 'YYYYMMDD' | None, '_tempdirs': [paths]}``
    """
    roots: list[str] = []
    tempdirs: list[str] = []
    direct_files: list[str] = []

    for item in _as_path_list(path):
        if not os.path.exists(item):
            logger.warning("Path does not exist, skipping: %s", item)
            continue
        if os.path.isfile(item):
            if item.lower().endswith(".zip"):
                d = _extract_zip(item)
                roots.append(d)
                tempdirs.append(d)
            else:
                direct_files.append(item)
        elif os.path.isdir(item):
            roots.append(item)

    # Collect all candidate files from every root, extracting nested zips too.
    all_files: list[str] = list(direct_files)
    pending = list(roots)
    seen_roots: set[str] = set()
    while pending:
        root = pending.pop()
        if root in seen_roots:
            continue
        seen_roots.add(root)
        for f in _walk_files(root):
            if f.lower().endswith(".zip"):
                d = _extract_zip(f)
                tempdirs.append(d)
                pending.append(d)
            else:
                all_files.append(f)

    solexs, gti, hel1os = [], [], []
    for f in all_files:
        name = os.path.basename(f).lower()
        if name.endswith(".lc.gz") and "solexs" in name:
            solexs.append(f)
        elif name.endswith(".gti.gz") and "solexs" in name:
            gti.append(f)
        elif name.startswith("lightcurve_czt") and name.endswith(".fits"):
            hel1os.append(f)

    # De-duplicate by resolved real path (the same physical file can be reached
    # twice). NB: do NOT dedupe by basename — HEL1OS AM/PM share basenames.
    def _dedupe(seq):
        seen, out = set(), []
        for x in seq:
            key = os.path.realpath(x)
            if key not in seen:
                seen.add(key)
                out.append(x)
        return out

    solexs, gti, hel1os = _dedupe(solexs), _dedupe(gti), _dedupe(hel1os)

    result = {
        "solexs": sorted(solexs),
        "hel1os_czt": sorted(hel1os),
        "solexs_gti": sorted(gti),
        "date": _detect_date(solexs + hel1os + gti),
        "_tempdirs": tempdirs,
    }
    logger.info(
        "Discovered: %d SoLEXS LC, %d SoLEXS GTI, %d HEL1OS CZT LC (date=%s)",
        len(solexs), len(gti), len(hel1os), result["date"],
    )
    return result


# --------------------------------------------------------------------------- #
# 2. read_solexs
# --------------------------------------------------------------------------- #
def read_solexs(paths: Sequence[str]) -> pd.DataFrame:
    """Read & co-add SoLEXS soft X-ray light curves into a 1 s UTC frame.

    Reads the ``RATE`` extension (``TIME`` Unix s, ``COUNTS`` cts/s) of each
    ``.lc.gz``. If both SDD1 and SDD2 are present their counts are co-added; if
    one is missing the other is used and a warning is logged.

    Returns
    -------
    pd.DataFrame
        Indexed by tz-aware UTC ``time_utc`` with a single ``solexs_counts``
        column. Empty frame if no readable detector is found.
    """
    per_detector: dict[str, list[pd.Series]] = defaultdict(list)
    for p in paths:
        det = _detector_of(p, _SDD_RE, "SDD") or os.path.basename(p)
        try:
            with fits.open(p, memmap=False) as hdul:
                rate = hdul["RATE"] if "RATE" in [h.name for h in hdul] else None
                if rate is None or rate.data is None or len(rate.data) == 0:
                    logger.warning("SoLEXS %s: empty/missing RATE ext, skipping.", det)
                    continue
                time = np.asarray(rate.data["TIME"], dtype="float64")
                counts = np.asarray(rate.data["COUNTS"], dtype="float64")
        except Exception as exc:  # corrupt file / bad column -> skip, don't crash
            logger.warning("SoLEXS %s: failed to read (%s), skipping.", det, exc)
            continue

        idx = _unix_to_utc_index(time)
        s = pd.Series(counts, index=idx, name="solexs_counts")
        s = s[~s.index.duplicated(keep="first")].sort_index()
        per_detector[det].append(s)
        logger.info("SoLEXS %s: %d rows read.", det, len(s))

    if not per_detector:
        logger.warning("No readable SoLEXS detectors found.")
        return pd.DataFrame(columns=["solexs_counts"]).rename_axis("time_utc")

    # Stitch any per-detector multi-file splits, then co-add detectors.
    det_series = {d: _coadd(series) for d, series in per_detector.items()}
    if len(det_series) == 1:
        only = next(iter(det_series))
        logger.warning("Only one SoLEXS detector present (%s); using it alone.", only)

    combined = _coadd(list(det_series.values()))
    df = combined.to_frame("solexs_counts").sort_index().rename_axis("time_utc")
    df = df[~df.index.duplicated(keep="first")]
    return df


# --------------------------------------------------------------------------- #
# 3. read_hel1os
# --------------------------------------------------------------------------- #
def read_hel1os(paths: Sequence[str]) -> pd.DataFrame:
    """Read, stitch & co-add HEL1OS hard X-ray CZT light curves (all 5 bands).

    Every band extension of every file is read and matched to a canonical band
    by its energy range (robust to the ``CZT1_``/``CZT2_`` name prefix). Files
    (AM + PM or any N split) are concatenated, sorted by time and de-duplicated;
    each detector is regularised to a 1 s UTC grid; finally czt1 + czt2 are
    co-added per band when both are present.

    Returns
    -------
    pd.DataFrame
        UTC-indexed frame with columns ``hxr_broad, hxr_20_40, hxr_40_60,
        hxr_60_80, hxr_80_150``. Empty frame if nothing readable is found.
    """
    band_cols = ["hxr_broad", "hxr_20_40", "hxr_40_60", "hxr_60_80", "hxr_80_150"]
    # (detector, band) -> list of Series spanning AM/PM/N files.
    raw: dict[tuple[str, str], list[pd.Series]] = defaultdict(list)

    for p in paths:
        det = _detector_of(p, _CZT_RE, "czt") or os.path.basename(p)
        try:
            hdul = fits.open(p, memmap=False)
        except Exception as exc:
            logger.warning("HEL1OS %s: failed to open (%s), skipping.", det, exc)
            continue
        with hdul:
            for hdu in hdul:
                band = _band_label(hdu.name)
                if band is None:
                    continue
                try:
                    data = hdu.data
                    if data is None or len(data) == 0:
                        logger.warning("HEL1OS %s/%s: empty ext, skipping.", det, hdu.name)
                        continue
                    mjd = np.asarray(data["MJD"], dtype="float64")
                    ctr = np.asarray(data["CTR"], dtype="float64")
                except Exception as exc:
                    logger.warning("HEL1OS %s/%s: bad columns (%s), skipping.",
                                   det, hdu.name, exc)
                    continue
                s = pd.Series(ctr, index=_mjd_to_utc_index(mjd), name=band)
                raw[(det, band)].append(s)

    if not raw:
        logger.warning("No readable HEL1OS CZT light curves found.")
        return pd.DataFrame(columns=band_cols).rename_axis("time_utc")

    # Stitch per (detector, band) and regularise each detector to a 1 s grid.
    per_det_band: dict[str, dict[str, pd.Series]] = defaultdict(dict)
    for (det, band), series_list in raw.items():
        s = pd.concat(series_list).sort_index()
        s = s[~s.index.duplicated(keep="first")]
        # Floor to whole seconds and mean-aggregate (native cadence ~1 s).
        s = s.groupby(s.index.floor("s")).mean()
        per_det_band[det][band] = s

    detectors = sorted(per_det_band)
    logger.info("HEL1OS detectors present: %s", detectors)

    # Co-add detectors per band on a shared 1 s grid.
    out = {}
    for band in band_cols:
        contributions = [per_det_band[d][band] for d in detectors if band in per_det_band[d]]
        if contributions:
            out[band] = _coadd(contributions)
    if not out:
        return pd.DataFrame(columns=band_cols).rename_axis("time_utc")

    df = pd.concat(out, axis=1).sort_index().rename_axis("time_utc")
    # Guarantee all 5 columns exist & ordered, even if a band was absent.
    for c in band_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[band_cols]


# --------------------------------------------------------------------------- #
# 4. load_unified
# --------------------------------------------------------------------------- #
def _regrid(df: pd.DataFrame, cadence: str, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Place *df* onto the common *grid* at *cadence* (mean rate per bin, NaN-safe)."""
    if df.empty:
        return pd.DataFrame(index=grid, columns=df.columns, dtype="float64")
    resampled = df.resample(cadence).mean()
    return resampled.reindex(grid)


def load_unified(path, cadence: str = "1s") -> tuple[pd.DataFrame, dict]:
    """Build the unified, UTC-aligned SoLEXS + HEL1OS dataframe for any upload.

    Discovers files, reads both instruments, and outer-joins them onto a common
    UTC grid spanning the full union of observed times, so the frame covers the
    whole day even where one instrument is silent.

    Parameters
    ----------
    path : str | os.PathLike | Iterable[str] | dict
        A zip, folder, single file, or list thereof — or a pre-computed
        ``discover_files(...)`` dict (avoids re-extracting archives).
    cadence : str, default '1s'
        Pandas offset alias for the common time grid (e.g. ``'1s'``, ``'10s'``).

    Returns
    -------
    (pd.DataFrame, dict)
        df : columns EXACTLY
            ``solexs_counts | hxr_broad | hxr_20_40 | hxr_40_60 | hxr_60_80 |
            hxr_80_150`` indexed by tz-aware UTC ``time_utc``.
        metadata : ``{date, instruments_found, n_rows, missing_detectors,
            time_range}``.
    """
    disc = path if isinstance(path, dict) else discover_files(path)
    solexs_df = read_solexs(disc["solexs"])
    hel1os_df = read_hel1os(disc["hel1os_czt"])

    if solexs_df.empty and hel1os_df.empty:
        raise ValueError("No readable SoLEXS or HEL1OS data found at the given path.")

    # Determine the union time span across whatever was found.
    starts, stops = [], []
    for d in (solexs_df, hel1os_df):
        if not d.empty:
            starts.append(d.index.min())
            stops.append(d.index.max())
    t0 = min(starts).floor(cadence)
    t1 = max(stops).ceil(cadence)
    grid = pd.date_range(t0, t1, freq=cadence, tz="UTC", name="time_utc")
    logger.info("Common grid: %s -> %s @ %s (%d rows)", t0, t1, cadence, len(grid))

    solexs_re = _regrid(solexs_df, cadence, grid)
    hel1os_re = _regrid(hel1os_df, cadence, grid)
    merged = pd.concat([solexs_re, hel1os_re], axis=1)

    # Enforce exact column set & order.
    for c in DATA_COLUMNS:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[DATA_COLUMNS]
    merged.index.name = "time_utc"

    # ---- metadata ----
    instruments_found = []
    if not solexs_df.empty:
        instruments_found.append("SoLEXS")
    if not hel1os_df.empty:
        instruments_found.append("HEL1OS")

    found_sdd = {d for p in disc["solexs"] if (d := _detector_of(p, _SDD_RE, "SDD"))}
    found_czt = {d for p in disc["hel1os_czt"] if (d := _detector_of(p, _CZT_RE, "czt"))}
    missing_detectors = sorted(
        (_EXPECTED_SOLEXS_DETECTORS - found_sdd) | (_EXPECTED_HEL1OS_DETECTORS - found_czt)
    )

    metadata = {
        "date": disc["date"],
        "instruments_found": instruments_found,
        "n_rows": int(len(merged)),
        "missing_detectors": missing_detectors,
        "time_range": (str(merged.index.min()), str(merged.index.max())),
        "cadence": cadence,
        "detectors_found": sorted(found_sdd | found_czt),
    }
    return merged, metadata


# --------------------------------------------------------------------------- #
# 5. load_multi_day  (multi-day ingestion with an accuracy/consistency gate)
# --------------------------------------------------------------------------- #
def _file_date(path: str) -> str | None:
    """YYYYMMDD parsed from a single filename, or None."""
    m = _DATE_RE.search(os.path.basename(path))
    return f"{m.group(1)}{m.group(2)}{m.group(3)}" if m else None


def _day_is_consistent(df: pd.DataFrame, date: str, require_both: bool
                       ) -> tuple[bool, str]:
    """Gate a single day: SoLEXS and HEL1OS must describe the SAME UTC day.

    Wrong file pairings (e.g. a SoLEXS day zipped with the previous day's HEL1OS)
    place the two instruments on different UTC grids — the merged frame then has
    each instrument covering a different calendar day, which is physically
    meaningless. We verify, *from the data itself* (not filenames), that both
    channels carry counts and that their dominant UTC date agrees. Returns
    ``(ok, reason)``.
    """
    soft = df["solexs_counts"].dropna()
    hard = df["hxr_broad"].dropna()
    if require_both and (soft.empty or hard.empty):
        missing = "SoLEXS" if soft.empty else "HEL1OS"
        return False, f"missing {missing} data"
    if soft.empty or hard.empty:
        return True, "single-instrument day (accepted)"
    soft_day = soft.index.normalize().value_counts().idxmax()
    hard_day = hard.index.normalize().value_counts().idxmax()
    if soft_day != hard_day:
        return (False, f"instrument date mismatch (SoLEXS={soft_day.date()}, "
                       f"HEL1OS={hard_day.date()})")
    return True, "consistent"


def load_multi_day(path, cadence: str = "1s", *, require_both: bool = True,
                   ) -> tuple[pd.DataFrame, dict]:
    """Load several observation days into one frame with a ``day`` column.

    Files anywhere under *path* (nested folders / zips) are discovered once, then
    grouped by their own detected date, so the on-disk folder layout does not
    matter. Each date group is read with the existing :func:`load_unified`, GTI-
    masked and gap-flagged, and then passed through an **accuracy gate**
    (:func:`_day_is_consistent`): only days whose SoLEXS and HEL1OS data fall on
    the same UTC calendar day are kept. Mismatched / single-instrument days are
    skipped with a logged reason — this is how "use only the dates where it is
    accurate" is enforced automatically rather than hardcoded.

    Days may differ in row count and in which detectors are present; everything
    is handled per day. Temporary extraction dirs are cleaned up before return.

    Parameters
    ----------
    path : str | os.PathLike | Iterable[str] | dict
        Root(s) holding one or more days of data (zip / folder / file list).
    cadence : str, default '1s'
        Common-grid cadence per day.
    require_both : bool, default True
        Require both instruments to be present (and date-consistent) for a day to
        be accepted.

    Returns
    -------
    (pd.DataFrame, dict)
        ``df`` : per-day-clean frames concatenated in date order, with all
            ``DATA_COLUMNS`` plus a ``day`` column (``'YYYYMMDD'``), indexed by
            tz-aware UTC ``time_utc``.
        ``info`` : ``{days_accepted, days_rejected, per_day_meta, n_days}``.
    """
    from preprocess import apply_gti, flag_gaps  # local import avoids any cycle

    # Group the *source* files by the date in their (archive) filename, BEFORE
    # extraction — the date lives in the zip name (e.g. 'HLS_20260606_...zip'),
    # not in the extracted inner files (e.g. 'lightcurve_czt1.fits'). Each date
    # group is then discovered + extracted independently.
    grouped: dict[str, list[str]] = defaultdict(list)
    undated: list[str] = []
    for item in _as_path_list(path):
        if not os.path.exists(item):
            logger.warning("load_multi_day: path does not exist: %s", item)
            continue
        candidates = _walk_files(item) if os.path.isdir(item) else [item]
        for f in candidates:
            d = _file_date(f)
            if d:
                grouped[d].append(f)
            else:
                undated.append(f)
    if undated:
        logger.warning("load_multi_day: %d file(s) with no parseable date in their "
                       "name were ignored for grouping.", len(undated))

    frames: list[pd.DataFrame] = []
    accepted: list[str] = []
    rejected: list[dict] = []
    per_day_meta: dict[str, dict] = {}

    for date in sorted(grouped):
        disc = discover_files(grouped[date])  # extracts this day's archives only
        try:
            df_day, meta = load_unified(disc, cadence=cadence)
            ok, reason = _day_is_consistent(df_day, date, require_both)
            if not ok:
                logger.warning("load_multi_day: REJECT %s — %s", date, reason)
                rejected.append({"date": date, "reason": reason})
                continue
            df_day = flag_gaps(apply_gti(df_day, disc.get("solexs_gti")))
            df_day = df_day.copy()
            df_day["day"] = date
            frames.append(df_day)
            accepted.append(date)
            per_day_meta[date] = meta
            logger.info("load_multi_day: ACCEPT %s — %s (%d rows)",
                        date, reason, len(df_day))
        except ValueError as exc:
            rejected.append({"date": date, "reason": f"unreadable ({exc})"})
        finally:
            cleanup_tempdirs(disc)  # free this day's extracted files immediately

    if not frames:
        raise ValueError("load_multi_day: no internally-consistent observation "
                         "days found (all candidate days were rejected).")

    combined = pd.concat(frames).sort_index()
    info = {
        "days_accepted": accepted,
        "days_rejected": rejected,
        "per_day_meta": per_day_meta,
        "n_days": len(accepted),
    }
    logger.info("load_multi_day: %d day(s) accepted %s, %d rejected.",
                len(accepted), accepted, len(rejected))
    return combined, info


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    logger.info("Smoke test loading from: %s", os.path.abspath(target))

    df, meta = load_unified(target, cadence="1s")

    print("\n================ METADATA ================")
    for k, v in meta.items():
        print(f"  {k:18s}: {v}")

    print("\n================ HEAD =====================")
    print(df.head())

    print("\n================ DESCRIBE =================")
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
        print(df.describe())

    print("\n================ NON-NULL COUNTS ==========")
    print(df.notna().sum())
