"""
simulate.py — Whole-day -> ~60s replay engine (the demo centerpiece).

Compresses a full observation day (~86,400 s) into ~60 s of wall-clock time and
emits a timeline of frames. As the replay plays:

    * the X-ray light curves draw in progressively (streaming live),
    * nowcast markers fire at the exact timestamps flares are detected,
    * forecast alerts pop *ahead* of the flare peak, showing the lead time,
    * a running clock reports the simulated UTC time,
    * the current flare probability is reported each frame.

Everything is driven by the uploaded data — a different day yields a different
sequence of flares and alerts. Nothing is hardcoded.

Public API:
    build_simulation(df, catalog, prob_curve, alerts, ...)  -> dict
    frame_generator(simulation, speed=None)                 -> generator of frames
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger("simulate")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def _to_ts(x):
    return pd.to_datetime(x, utc=True)


def build_simulation(df: pd.DataFrame, catalog: pd.DataFrame, prob_curve: pd.Series,
                     alerts: list[dict], *, day_in_seconds: float = 60.0,
                     n_frames: int = 600, lightcurve_points: int = 1200) -> dict:
    """Precompute the replay timeline that maps the full day onto *day_in_seconds*.

    Parameters
    ----------
    df : pd.DataFrame
        Feature/unified frame (UTC index) — source of the light curves.
    catalog : pd.DataFrame
        Nowcast flare catalogue (detection markers).
    prob_curve : pd.Series
        Full-day flare probability (UTC index), aligned to *df*.
    alerts : list[dict]
        Forecast alerts (each with ``alert_time`` and optional ``lead_time_min``).
    day_in_seconds : float, default 60.0
        Wall-clock seconds the whole day is compressed into (the "24h in 60s").
    n_frames : int, default 600
        Number of replay frames (temporal resolution of the playback).
    lightcurve_points : int
        Max light-curve samples streamed by the final frame (downsample budget).

    Returns
    -------
    dict
        ``{t_start, t_end, day_seconds, total_sim_seconds, speed_factor,
        frame_interval_s, n_frames, series (precomputed downsampled arrays),
        frames (list of frame dicts), nowcast_markers, forecast_alerts}``.
    """
    if len(df) == 0:
        raise ValueError("build_simulation: empty dataframe.")

    t_start = df.index.min()
    t_end = df.index.max()
    span_s = max(1.0, (t_end - t_start).total_seconds())
    speed_factor = span_s / day_in_seconds  # simulated seconds per wall second
    frame_interval_s = day_in_seconds / n_frames  # wall seconds between frames

    # Downsample the light curve once; frames reveal a growing prefix of it.
    step = max(1, len(df) // lightcurve_points)
    sub = df.iloc[::step]
    prob_sub = prob_curve.reindex(sub.index)
    sub_times = sub.index
    sub_elapsed = (sub_times - t_start).total_seconds().to_numpy()

    def _arr(name):
        if name not in sub.columns:
            return [None] * len(sub)
        return [None if pd.isna(v) else round(float(v), 3) for v in sub[name]]

    series = {
        "time": [t.isoformat() for t in sub_times],
        "elapsed_s": sub_elapsed.tolist(),
        "solexs_counts": _arr("solexs_counts"),
        "background_soft": _arr("background_soft"),
        "hxr_broad": _arr("hxr_broad"),
        "hxr_20_40": _arr("hxr_20_40"),
        "hxr_80_150": _arr("hxr_80_150"),
        "hr": _arr("hr"),
        "flare_probability": [None if pd.isna(v) else round(float(v), 4) for v in prob_sub],
    }

    # Schedule nowcast detection markers at each flare peak.
    nowcast_markers = []
    if catalog is not None and len(catalog):
        for _, fl in catalog.iterrows():
            pk = _to_ts(fl["peak_time"])
            nowcast_markers.append({
                "sim_time": pk.isoformat(),
                "elapsed_s": float((pk - t_start).total_seconds()),
                "peak_flux": float(fl.get("peak_flux", np.nan)),
                "goes_class": fl.get("goes_class"),
                "provenance": fl.get("provenance"),
                "category": fl.get("category"),
                "confidence": float(fl.get("confidence", np.nan)),
            })

    # Schedule forecast alerts at their (pre-peak) alert times.
    forecast_alerts = []
    for a in (alerts or []):
        at = _to_ts(a["alert_time"])
        forecast_alerts.append({
            "sim_time": at.isoformat(),
            "elapsed_s": float((at - t_start).total_seconds()),
            "probability": a.get("probability"),
            "predicted_class": a.get("predicted_class"),
            "lead_time_min": a.get("lead_time_min"),
            "contributing_features": a.get("contributing_features", []),
            "matched_flare": a.get("matched_flare"),
        })

    markers_elapsed = np.array([m["elapsed_s"] for m in nowcast_markers]) if nowcast_markers else np.array([])
    alerts_elapsed = np.array([a["elapsed_s"] for a in forecast_alerts]) if forecast_alerts else np.array([])

    # Precompute frames: each frame reveals data up to its simulated time and
    # carries the events that have fired by then (new events flagged).
    frames = []
    prev_marker_count = 0
    prev_alert_count = 0
    for k in range(1, n_frames + 1):
        elapsed = span_s * k / n_frames
        sim_time = t_start + pd.Timedelta(seconds=elapsed)
        n_pts = int(np.searchsorted(sub_elapsed, elapsed, side="right"))

        fired_markers = int(np.searchsorted(markers_elapsed, elapsed, side="right")) if len(markers_elapsed) else 0
        fired_alerts = int(np.searchsorted(alerts_elapsed, elapsed, side="right")) if len(alerts_elapsed) else 0

        # Current probability = last revealed non-null probability value.
        cur_prob = None
        if n_pts > 0:
            for v in reversed(series["flare_probability"][:n_pts]):
                if v is not None:
                    cur_prob = v
                    break

        frames.append({
            "frame": k,
            "simulated_utc": sim_time.isoformat(),
            "elapsed_s": round(elapsed, 1),
            "progress": round(k / n_frames, 4),
            "n_points": n_pts,                       # reveal series[:n_points]
            "current_probability": cur_prob,
            "n_markers": fired_markers,              # reveal nowcast_markers[:n_markers]
            "n_alerts": fired_alerts,                # reveal forecast_alerts[:n_alerts]
            "new_markers": [nowcast_markers[i] for i in range(prev_marker_count, fired_markers)],
            "new_alerts": [forecast_alerts[i] for i in range(prev_alert_count, fired_alerts)],
        })
        prev_marker_count, prev_alert_count = fired_markers, fired_alerts

    logger.info("build_simulation: %s -> %s (%.0fs) compressed to %.0fs wall "
                "(%.0fx), %d frames, %d markers, %d alerts",
                t_start, t_end, span_s, day_in_seconds, speed_factor,
                n_frames, len(nowcast_markers), len(forecast_alerts))

    return {
        "t_start": t_start.isoformat(),
        "t_end": t_end.isoformat(),
        "day_seconds": span_s,
        "total_sim_seconds": day_in_seconds,
        "speed_factor": speed_factor,
        "frame_interval_s": frame_interval_s,
        "n_frames": n_frames,
        "series": series,
        "frames": frames,
        "nowcast_markers": nowcast_markers,
        "forecast_alerts": forecast_alerts,
    }


def frame_generator(simulation: dict, speed: float = None):
    """Yield frames in real time for streaming, pacing to the replay clock.

    Parameters
    ----------
    simulation : dict
        Output of :func:`build_simulation`.
    speed : float, optional
        Wall-clock seconds for the whole replay. If given, overrides the
        simulation's ``total_sim_seconds`` (e.g. 60 -> day in 60 s, 30 -> faster).
        Pause/scrub are handled by the client; this generator paces frames.

    Yields
    ------
    dict
        Frame dicts (see :func:`build_simulation`), sleeping between them so the
        whole sequence takes ~``speed`` (or ``total_sim_seconds``) wall seconds.
    """
    frames = simulation["frames"]
    n = len(frames)
    total_wall = float(speed) if speed else simulation["total_sim_seconds"]
    interval = total_wall / max(1, n)
    logger.info("frame_generator: streaming %d frames over ~%.0fs (%.3fs/frame)",
                n, total_wall, interval)
    for fr in frames:
        t0 = time.time()
        yield fr
        dt = interval - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    from pipeline import run_pipeline

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    b = run_pipeline(target)
    sim = build_simulation(b["feats"], b["catalog"], b["prob_curve"], b["alerts"],
                           day_in_seconds=60, n_frames=600)

    print(f"\nReplay: {sim['t_start']} -> {sim['t_end']}")
    print(f"  {sim['day_seconds']:.0f}s of Sun compressed to "
          f"{sim['total_sim_seconds']:.0f}s wall ({sim['speed_factor']:.0f}x)")
    print(f"  {sim['n_frames']} frames, {len(sim['nowcast_markers'])} nowcast markers, "
          f"{len(sim['forecast_alerts'])} forecast alerts")

    # Show the event schedule (in replay wall-seconds) without sleeping.
    print("\n[event schedule — wall-clock second within the 60s replay]")
    sf = sim["speed_factor"]
    for m in sim["nowcast_markers"]:
        print(f"  t+{m['elapsed_s']/sf:5.1f}s  NOWCAST  {m['goes_class'] or '-':>5} "
              f"{m['provenance']:>10}  peak={m['peak_flux']:.0f}")
    for a in sim["forecast_alerts"]:
        lt = a["lead_time_min"]
        tag = f"lead {lt}min" if lt is not None else "no-match"
        print(f"  t+{a['elapsed_s']/sf:5.1f}s  ALERT    p={a['probability']:.2f}  {tag}")
