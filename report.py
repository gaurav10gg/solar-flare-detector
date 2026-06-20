"""
report.py — one-click PDF mission report for a processed Aditya-L1 day.

Renders EVERYTHING the dashboard shows into a single, self-contained PDF:
title page + headline, ingest/data-quality table, full-day soft & hard X-ray
light curves with nowcast flare markers, the forecast probability curve with
alert markers and lead times, feature-importance and confusion-matrix figures,
the confirmed-flare and HXR-candidate catalogue tables, and the explainable
forecast-alert log.

Figures are rendered with matplotlib (Agg, headless) and composed with
reportlab. Call :func:`build_pdf_report(bundle, out_path)` with a bundle from
``pipeline.run_pipeline``.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

# ---- mission-control palette (dark, matches the dashboard) ---------------- #
BG = "#0B0F1A"
PANEL = "#121A2B"
GRID = "#24324A"
TEXT = "#E6EDF7"
MUTED = "#8A97AD"
CYAN = "#38BDF8"     # soft X-ray
VIOLET = "#A78BFA"   # hard X-ray
GREEN = "#34D399"    # probability / good
AMBER = "#FBBF24"    # alerts / threshold
RED = "#F87171"      # false / miss

PAGE_W, PAGE_H = A4
MARGIN = 14 * mm
CONTENT_W = PAGE_W - 2 * MARGIN


# --------------------------------------------------------------------------- #
# matplotlib helpers
# --------------------------------------------------------------------------- #
def _style_ax(ax, title=None):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color=TEXT, fontsize=10, loc="left", pad=6, fontweight="bold")


def _fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    return fig, ax


def _to_image(fig, width=CONTENT_W):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    img.drawWidth = width
    img.drawHeight = width * img.imageHeight / img.imageWidth
    return img


def _times(index):
    return [pd.Timestamp(t).to_pydatetime() for t in index]


def _hhmm(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))


def _downsample(feats, n=2400):
    step = max(1, len(feats) // n)
    return feats.iloc[::step]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig_soft(feats, catalog):
    sub = _downsample(feats)
    t = _times(sub.index)
    fig, ax = _fig((10, 2.9))
    _style_ax(ax, "Soft X-ray (SoLEXS 1-15 keV) - thermal")
    ax.plot(t, sub["solexs_counts"], color=CYAN, lw=0.7, label="counts/s")
    if "background_soft" in sub:
        ax.plot(t, sub["background_soft"], color=MUTED, lw=0.9, ls="--", label="background")
    if catalog is not None and len(catalog):
        first_c = first_x = True
        for _, fl in catalog.iterrows():
            confirmed = (fl.get("category") == "confirmed_flare") or \
                fl.get("provenance") in ("both", "soft_only")
            pk = pd.Timestamp(fl["peak_time"]).to_pydatetime()
            if confirmed:
                ax.axvspan(pd.Timestamp(fl["start"]).to_pydatetime(),
                           pd.Timestamp(fl["end"]).to_pydatetime(),
                           color=AMBER, alpha=0.10,
                           label="confirmed flare" if first_c else None)
                ax.scatter([pk], [fl.get("peak_flux", np.nan)], s=22, color=AMBER,
                           edgecolor="#000", lw=0.4, zorder=5)
                first_c = False
            else:
                ax.axvline(pk, color=VIOLET, lw=0.6, alpha=0.35,
                           label="HXR candidate" if first_x else None)
                first_x = False
    ax.set_ylabel("counts/s", color=MUTED, fontsize=8)
    _hhmm(ax)
    ax.legend(loc="upper right", fontsize=6.5, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, framealpha=0.85)
    return _to_image(fig)


def _fig_hard(feats):
    sub = _downsample(feats)
    t = _times(sub.index)
    fig, ax = _fig((10, 2.6))
    _style_ax(ax, "Hard X-ray (HEL1OS 18-160 keV) - non-thermal / impulsive")
    for col, c, lab in (("hxr_broad", VIOLET, "broad"),
                        ("hxr_20_40", CYAN, "20-40 keV"),
                        ("hxr_80_150", AMBER, "80-150 keV")):
        if col in sub:
            ax.plot(t, sub[col], color=c, lw=0.6, alpha=0.9, label=lab)
    ax.set_ylabel("counts/s", color=MUTED, fontsize=8)
    _hhmm(ax)
    ax.legend(loc="upper right", fontsize=6.5, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, framealpha=0.85)
    return _to_image(fig)


def _fig_forecast(feats, prob, alerts, catalog, threshold):
    sub_p = prob.reindex(_downsample(feats).index)
    t = _times(sub_p.index)
    fig, ax = _fig((10, 2.9))
    _style_ax(ax, "Forecast flare probability + alerts")
    ax.plot(t, sub_p.to_numpy(), color=GREEN, lw=0.9, label="P(flare)")
    if threshold is not None:
        ax.axhline(threshold, color=AMBER, ls="--", lw=0.8,
                   label=f"alert threshold {threshold:.2f}")
    if catalog is not None and len(catalog):
        first = True
        for _, fl in catalog.iterrows():
            if (fl.get("category") == "confirmed_flare") or \
                    fl.get("provenance") in ("both", "soft_only"):
                pk = pd.Timestamp(fl["peak_time"]).to_pydatetime()
                ax.axvline(pk, color=CYAN, lw=0.7, alpha=0.5,
                           label="actual flare peak" if first else None)
                first = False
    for k, a in enumerate(alerts or []):
        at = pd.Timestamp(a["alert_time"]).to_pydatetime()
        matched = a.get("matched_flare") is not None
        ax.scatter([at], [a.get("probability", 0)], s=30, marker="v",
                   color=(GREEN if matched else RED), edgecolor="#000", lw=0.4,
                   zorder=6, label=("alert (hit)" if matched else "alert (false)")
                   if k == 0 else None)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("probability", color=MUTED, fontsize=8)
    _hhmm(ax)
    ax.legend(loc="upper right", fontsize=6.5, facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, framealpha=0.85)
    return _to_image(fig)


def _fig_importance(metrics):
    imp = metrics.get("feature_importances", {})
    if not imp:
        return None
    items = list(imp.items())[:12][::-1]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = _fig((5.0, 3.2))
    _style_ax(ax, "Feature importances (top 12)")
    ax.barh(names, vals, color=CYAN, edgecolor=GRID)
    ax.tick_params(axis="y", labelsize=7, colors=TEXT)
    return _to_image(fig, width=CONTENT_W * 0.52)


def _fig_confusion(metrics):
    cm = metrics.get("confusion")
    if not cm:
        return None
    M = np.array([[cm.get("TN", 0), cm.get("FP", 0)],
                  [cm.get("FN", 0), cm.get("TP", 0)]], dtype=float)
    fig, ax = _fig((4.4, 3.2))
    _style_ax(ax, "Confusion matrix (held-out test)")
    ax.imshow(M, cmap="cividis")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"], color=TEXT, fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"], color=TEXT, fontsize=8)
    ax.grid(False)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{int(v):,}", ha="center", va="center",
                color="#fff", fontsize=11, fontweight="bold")
    return _to_image(fig, width=CONTENT_W * 0.45)


# --------------------------------------------------------------------------- #
# Text / table styles
# --------------------------------------------------------------------------- #
def _styles():
    ss = getSampleStyleSheet()
    base = dict(textColor=colors.HexColor(TEXT), fontName="Helvetica")
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], textColor=colors.HexColor(CYAN),
                                fontSize=22, leading=26, alignment=TA_LEFT),
        "sub": ParagraphStyle("s", textColor=colors.HexColor(MUTED), fontSize=10,
                              leading=14, alignment=TA_LEFT),
        "h2": ParagraphStyle("h2", textColor=colors.HexColor(TEXT), fontSize=13,
                             leading=16, spaceBefore=10, spaceAfter=6,
                             fontName="Helvetica-Bold"),
        "body": ParagraphStyle("b", fontSize=9, leading=13, **base),
        "muted": ParagraphStyle("m", textColor=colors.HexColor(MUTED), fontSize=8,
                               leading=11),
        "headline": ParagraphStyle("hl", textColor=colors.HexColor(AMBER), fontSize=12,
                                   leading=16, fontName="Helvetica-Bold"),
    }


def _kv_table(rows, w=CONTENT_W):
    t = Table(rows, colWidths=[w * 0.32, w * 0.68])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor(MUTED)),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor(TEXT)),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def _data_table(header, data, col_widths):
    rows = [header] + data
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.5),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(BG)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(CYAN)),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(TEXT)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor(PANEL), colors.HexColor("#0E1626")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]
    return Table(rows, colWidths=col_widths, repeatRows=1, style=TableStyle(style))


def _clock(ts):
    try:
        return pd.Timestamp(ts).strftime("%H:%M:%S")
    except Exception:
        return str(ts)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def build_pdf_report(bundle: dict, out_path: str) -> str:
    """Render *bundle* into a polished multi-page PDF at *out_path*; returns the path."""
    st = _styles()
    meta = bundle.get("metadata", {})
    quality = bundle.get("quality", {})
    catalog = bundle.get("catalog")
    summary = bundle.get("catalog_summary", {})
    metrics = bundle.get("metrics", {})
    feats = bundle.get("feats")
    prob = bundle.get("prob_curve")
    alerts = bundle.get("alerts", [])
    threshold = metrics.get("alert_threshold") or bundle.get("threshold")
    calibrated = bundle.get("goes_calibrated", False)
    story = []

    # ---- header ----
    story.append(Paragraph("Aditya-L1 Solar Flare Report", st["title"]))
    story.append(Paragraph(
        "Nowcasting &amp; Forecasting - SoLEXS (soft, thermal) + HEL1OS (hard, non-thermal)",
        st["sub"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(summary.get("headline", "no detections"), st["headline"]))
    if not calibrated:
        story.append(Paragraph(
            "GOES classes are GOES-equivalent (UNCALIBRATED) - no NOAA flare list supplied.",
            st["muted"]))
    story.append(Spacer(1, 10))

    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rng = meta.get("time_range", ["?", "?"])
    story.append(_kv_table([
        ["Observation date", str(meta.get("date", "?"))],
        ["Instruments", " + ".join(meta.get("instruments_found", []) or ["-"])],
        ["Detectors found", ", ".join(meta.get("detectors_found", []) or ["-"])],
        ["Missing detectors", ", ".join(meta.get("missing_detectors", []) or ["none"])],
        ["Time range (UTC)", f"{rng[0]}  ->  {rng[1]}"],
        ["Cadence", str(meta.get("cadence", "?"))],
        ["Data gaps", str(quality.get("gap_count", "-"))],
        ["Confirmed / candidates",
         f"{summary.get('n_confirmed', 0)} confirmed  |  {summary.get('n_candidates', 0)} HXR candidates"],
        ["Forecast alerts", str(len(alerts))],
        ["Report generated", gen],
    ]))
    story.append(Spacer(1, 12))

    # ---- light curves ----
    story.append(Paragraph("Light curves &amp; nowcast detections", st["h2"]))
    if feats is not None and len(feats):
        story.append(_fig_soft(feats, catalog))
        story.append(Spacer(1, 6))
        story.append(_fig_hard(feats))
    story.append(PageBreak())

    # ---- forecast ----
    story.append(Paragraph("Forecast probability &amp; alerts", st["h2"]))
    if feats is not None and prob is not None and prob.notna().any():
        story.append(_fig_forecast(feats, prob, alerts, catalog, threshold))
    else:
        story.append(Paragraph("No forecast (insufficient labels on this upload).", st["muted"]))
    story.append(Spacer(1, 10))

    # ---- metrics ----
    story.append(Paragraph("Forecast skill", st["h2"]))
    ev = metrics.get("event_level", {}) or {}
    skill_rows = [
        ["TSS (headline)", f"{metrics.get('TSS', '-')}"],
        ["HSS", f"{metrics.get('HSS', '-')}"],
        ["ROC-AUC", f"{(metrics.get('ROC') or {}).get('auc', '-')}"],
        ["Event recall", f"{ev.get('n_alerted', '-')}/{ev.get('n_flares', '-')} "
                         f"({ev.get('event_recall', '-')})"],
        ["Mean / median lead", f"{ev.get('mean_lead', '-')} / {ev.get('median_lead', '-')} min"],
        ["False alarms", f"{ev.get('false_alarm_count', '-')} (FAR {ev.get('far_per_day', '-')}/day)"],
        ["GOES calibration", "calibrated (NOAA)" if calibrated else "uncalibrated (placeholder)"],
        ["Observation days", str(metrics.get("n_days", "-"))],
    ]
    story.append(_kv_table(skill_rows))
    note = metrics.get("evaluation_note")
    if note:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Note: " + note, st["muted"]))
    story.append(Spacer(1, 8))
    figs = [f for f in (_fig_importance(metrics), _fig_confusion(metrics)) if f is not None]
    if figs:
        cw = [CONTENT_W * 0.54, CONTENT_W * 0.46][:len(figs)]
        row = Table([figs], colWidths=cw)
        row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
        story.append(row)
    if metrics.get("data_warning"):
        story.append(Spacer(1, 6))
        story.append(Paragraph("Note: " + metrics["data_warning"], st["muted"]))
    story.append(PageBreak())

    # ---- catalogue ----
    story.append(Paragraph("Flare catalogue", st["h2"]))
    if catalog is not None and len(catalog):
        cat = catalog.copy()
        conf = cat[cat.apply(lambda r: (r.get("category") == "confirmed_flare") or
                             r.get("provenance") in ("both", "soft_only"), axis=1)]
        cand = cat[~cat.index.isin(conf.index)]
        cw = [CONTENT_W * x for x in (0.13, 0.13, 0.13, 0.12, 0.13, 0.12, 0.12, 0.12)]
        hdr = ["Peak UTC", "Class", "Provenance", "Start", "End", "Dur(min)",
               "Conf.", "GOES"]

        def rowify(df):
            out = []
            for _, f in df.iterrows():
                gm = f.get("goes_match")
                gtxt = ("OK " + str(f.get("goes_truth_class") or "")) if gm is True else \
                    ("X" if gm is False else "uncal")
                out.append([_clock(f["peak_time"]), str(f.get("goes_class", "-")),
                            str(f.get("provenance", "-")), _clock(f["start"]),
                            _clock(f["end"]), f"{f.get('duration', 0)/60:.1f}",
                            f"{f.get('confidence', 0)*100:.0f}%", gtxt])
            return out

        story.append(Paragraph(
            f"Confirmed flares ({len(conf)}) - thermal soft-X-ray response present", st["body"]))
        story.append(Spacer(1, 3))
        story.append(_data_table(hdr, rowify(conf) or [["-"] * 8], cw))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"HXR transient candidates ({len(cand)}) - hard-only, no thermal counterpart "
            "(possible particle hits; flagged for vetting)", st["body"]))
        story.append(Spacer(1, 3))
        story.append(_data_table(hdr, rowify(cand) or [["-"] * 8], cw))
    else:
        story.append(Paragraph("No flares catalogued.", st["muted"]))
    story.append(Spacer(1, 12))

    # ---- alert log ----
    story.append(Paragraph("Explainable forecast alerts", st["h2"]))
    if alerts:
        cw = [CONTENT_W * x for x in (0.14, 0.12, 0.13, 0.14, 0.47)]
        hdr = ["Alert UTC", "P(flare)", "Lead (min)", "Outcome", "Top contributing features"]
        rows = []
        for a in alerts:
            feats_txt = ", ".join(c["feature"] for c in a.get("contributing_features", [])) or "-"
            matched = a.get("matched_flare")
            outcome = (f"hit -> {matched.get('goes_class', '')}" if matched else "no match")
            rows.append([_clock(a["alert_time"]), f"{a.get('probability', 0):.2f}",
                         str(a.get("lead_time_min", "-")), outcome, feats_txt])
        story.append(_data_table(hdr, rows, cw))
    else:
        story.append(Paragraph("No alerts raised.", st["muted"]))

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "Methodology: windowed Neupert coupling (dF_SXR/dt ~ k*F_HXR), HXR->SXR cross-correlation "
        "lag, robust adaptive backgrounds, XGBoost forecaster with max-TSS thresholding, strictly "
        "time-ordered (no-shuffle) evaluation. Hard-X-ray-only transients are reported as candidates, "
        "never counted as confirmed flares.", st["muted"]))

    # ---- build with dark background ----
    def _paint(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor(BG))
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.setFont("Helvetica", 7)
        canvas.drawString(MARGIN, 8 * mm,
                          "Aditya-L1 - ISRO PS-15 - Solar Flare Nowcasting & Forecasting")
        canvas.drawRightString(PAGE_W - MARGIN, 8 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(out_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=16 * mm, title="Aditya-L1 Flare Report")
    frame = Frame(MARGIN, 16 * mm, CONTENT_W, PAGE_H - MARGIN - 16 * mm, id="main")
    doc.addPageTemplates([PageTemplate(id="dark", frames=[frame], onPage=_paint)])
    doc.build(story)
    return out_path


if __name__ == "__main__":
    import sys, glob
    from pipeline import run_pipeline
    target = sys.argv[1:] if len(sys.argv) > 1 else glob.glob("*.zip")
    b = run_pipeline(target)
    out = build_pdf_report(b, "data/catalog/aditya_report.pdf")
    print("wrote", out)
