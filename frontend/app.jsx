/* Aditya-L1 Solar Flare Forecasting — mission-control dashboard (React + Plotly). */
const { useState, useEffect, useRef, useCallback } = React;

const API = ""; // served same-origin by FastAPI; override with window.API_BASE if needed.
const base = () => (window.API_BASE || API);

// SoLEXS counts -> GOES W/m^2 cross-cal (mirror of nowcast.COUNTS_TO_WM2).
const COUNTS_TO_WM2 = 5.0e-8;
const GOES_BANDS = [
  { cls: "B", lo: 1e-7, hi: 1e-6, color: "rgba(74,158,111,0.10)" },
  { cls: "C", lo: 1e-6, hi: 1e-5, color: "rgba(201,162,39,0.10)" },
  { cls: "M", lo: 1e-5, hi: 1e-4, color: "rgba(232,132,60,0.12)" },
  { cls: "X", lo: 1e-4, hi: 1e-2, color: "rgba(255,77,109,0.12)" },
];
const wm2ToCounts = (f) => f / COUNTS_TO_WM2;

const PLOT_BG = "#0e1622";
const FONT = { family: "JetBrains Mono, monospace", color: "#8ea0b8", size: 11 };
const darkLayout = (over = {}) => ({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: PLOT_BG, font: FONT,
  margin: { l: 56, r: 18, t: 14, b: 36 }, showlegend: true,
  legend: { orientation: "h", x: 0, y: 1.12, font: { size: 10 } },
  xaxis: { gridcolor: "#1d2939", zerolinecolor: "#1d2939", color: "#6b7c93" },
  yaxis: { gridcolor: "#1d2939", zerolinecolor: "#1d2939", color: "#6b7c93" },
  ...over,
});
const PLOT_CFG = { displayModeBar: false, responsive: true };

const fmtClass = (c) => (c && c !== "-" ? c : "—");
const classLetter = (c) => (c && c[0] && "ABCMX".includes(c[0]) ? c[0] : "");
const clockUTC = (iso) => {
  if (!iso) return "--:--:--";
  const d = new Date(iso);
  return d.toISOString().slice(11, 19);
};
const dateUTC = (iso) => (iso ? new Date(iso).toISOString().slice(0, 10) : "");

async function jget(path) {
  const r = await fetch(base() + path);
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}

/* ===================================================================== */
/* Upload screen                                                          */
/* ===================================================================== */
function UploadScreen({ onLoaded }) {
  const [drag, setDrag] = useState(false);
  const [files, setFiles] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [stage, setStage] = useState("");
  const inputRef = useRef();

  const pick = (fl) => setFiles(Array.from(fl));
  const onDrop = (e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files); };

  const process = async () => {
    if (!files.length) return;
    setBusy(true); setErr(null); setStage("Uploading files…");
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      setStage("Ingesting → features → nowcast → forecast (this runs the full pipeline)…");
      const r = await fetch(base() + "/upload", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || "Upload failed");
      const info = await r.json();
      setStage("Fetching results…");
      const [cat, met, lc, sim] = await Promise.all([
        jget(`/catalog/${info.job_id}`), jget(`/metrics/${info.job_id}`),
        jget(`/lightcurve/${info.job_id}`), jget(`/simulation/${info.job_id}`),
      ]);
      onLoaded({ ...info, catalog: cat.flares, catalogSummary: cat.summary,
        goesCalibrated: cat.goes_calibrated, metrics: met.metrics,
        threshold: met.threshold, horizon_min: met.horizon_min,
        lightcurve: lc.lightcurve, simulation: sim });
    } catch (e) {
      setErr(String(e.message || e));
    } finally { setBusy(false); setStage(""); }
  };

  return (
    <div className="upload-wrap">
      <div className={"dropzone" + (drag ? " drag" : "")}
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)} onDrop={onDrop}
        onClick={() => inputRef.current.click()}>
        <div className="dz-icon">🛰️</div>
        <h2>Drop SoLEXS / HEL1OS data</h2>
        <p>Raw <code>.zip</code> from ISSDC PRADAN, or extracted <code>.fits</code> / <code>.lc.gz</code></p>
        <p className="muted">Any observation date · auto-discovers instruments · nothing hardcoded</p>
        <input ref={inputRef} type="file" multiple hidden
          accept=".zip,.fits,.gz" onChange={(e) => pick(e.target.files)} />
      </div>

      {files.length > 0 && (
        <div className="filelist">
          {files.map((f, i) => (
            <div className="filechip" key={i}>
              <span>📄 {f.name}</span>
              <span className="sz">{(f.size / 1e6).toFixed(1)} MB</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginTop: 18, display: "flex", gap: 12, alignItems: "center" }}>
        <button className="btn btn-primary" disabled={!files.length || busy} onClick={process}>
          {busy ? <><span className="spinner-sm" /> &nbsp;Processing…</> : "▶ Process & Build Simulation"}
        </button>
        {files.length > 0 && !busy && (
          <button className="btn btn-ghost" onClick={() => setFiles([])}>Clear</button>
        )}
      </div>
      {stage && <p className="muted" style={{ marginTop: 12, fontFamily: "var(--mono)" }}>{stage}</p>}
      {err && <div className="warn-banner" style={{ marginTop: 14 }}>⚠ {err}</div>}
    </div>
  );
}

function QualityReport({ meta, quality, summary, nAlerts }) {
  if (!meta) return null;
  const det = (meta.detectors_found || []).join(", ") || "—";
  const miss = (meta.missing_detectors || []).join(", ") || "none";
  const gaps = quality ? quality.gap_count : "—";
  const missPct = quality && quality.pct_missing
    ? Math.max(...Object.values(quality.pct_missing)).toFixed(2) : "—";
  const s = summary || { n_confirmed: 0, n_candidates: 0, n_confirmed_with_hard: 0 };
  return (
    <div className="panel" style={{ marginTop: 22, maxWidth: 720, marginLeft: "auto", marginRight: "auto" }}>
      <h3 className="panel-title">Ingest & Data Quality</h3>
      <div className="qgrid">
        <div className="qcard"><div className="ql">Detected Date</div><div className="qv">{meta.date || "?"}</div></div>
        <div className="qcard"><div className="ql">Instruments</div><div className="qv" style={{ fontSize: 16 }}>{(meta.instruments_found || []).join(" + ") || "—"}</div></div>
        <div className="qcard"><div className="ql">Detectors</div><div className="qv" style={{ fontSize: 15 }}>{det}</div></div>
        <div className="qcard"><div className="ql">Missing</div><div className={"qv " + (miss === "none" ? "good" : "warn")} style={{ fontSize: 15 }}>{miss}</div></div>
        <div className="qcard"><div className="ql">Data Gaps</div><div className={"qv " + (gaps === 0 ? "good" : "warn")}>{gaps}</div></div>
        <div className="qcard"><div className="ql">Max Missing %</div><div className="qv">{missPct}</div></div>
        <div className="qcard"><div className="ql">Confirmed Flares</div><div className="qv good">{s.n_confirmed}<span style={{ fontSize: 11, color: "var(--muted)" }}> ({s.n_confirmed_with_hard} HXR-conf)</span></div></div>
        <div className="qcard"><div className="ql">HXR Candidates</div><div className="qv" style={{ color: "var(--violet)" }}>{s.n_candidates}</div></div>
        <div className="qcard"><div className="ql">Forecast Alerts</div><div className="qv">{nAlerts}</div></div>
      </div>
      <p className="muted" style={{ fontFamily: "var(--mono)", fontSize: 11, marginTop: 12 }}>
        {(summary && summary.headline) || ""}. HXR-only transients are flagged as candidates
        (possible particle hits / non-thermal microflares), not counted as confirmed flares.
      </p>
    </div>
  );
}

/* ===================================================================== */
/* Simulation screen (centerpiece)                                        */
/* ===================================================================== */
function SimulationScreen({ sim, job }) {
  const chartRef = useRef();
  const rafRef = useRef();
  const stateRef = useRef({ playing: false, elapsedWall: 0, last: 0 });
  const lastDrawnFrame = useRef(-1);
  const shownAlerts = useRef(0);

  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [wallTarget, setWallTarget] = useState(60);
  const [clock, setClock] = useState(sim ? sim.t_start : null);
  const [prob, setProb] = useState(null);
  const [events, setEvents] = useState([]);
  const [toasts, setToasts] = useState([]);

  const nFrames = sim ? sim.n_frames : 0;
  const daySec = sim ? sim.day_seconds : 86400;

  const drawFrame = useCallback((idx) => {
    if (!sim || !chartRef.current) return;
    const fr = sim.frames[Math.max(0, Math.min(idx, nFrames - 1))];
    const np = fr.n_points;
    const s = sim.series;
    const x = s.time.slice(0, np);
    const soft = s.solexs_counts.slice(0, np);
    const hard = s.hxr_broad.slice(0, np);
    const bg = s.background_soft.slice(0, np);

    const simElapsed = fr.progress * daySec;
    const firedMarkers = sim.nowcast_markers.filter((m) => m.elapsed_s <= simElapsed);
    const firedAlerts = sim.forecast_alerts.filter((a) => a.elapsed_s <= simElapsed);

    const shapes = [];
    firedAlerts.forEach((a) => shapes.push({
      type: "line", x0: a.sim_time, x1: a.sim_time, yref: "paper", y0: 0, y1: 1,
      line: { color: "rgba(255,182,39,0.55)", width: 1, dash: "dot" },
    }));
    firedMarkers.forEach((m) => {
      const col = m.provenance === "both" ? "#3ddc84"
        : m.provenance === "hard_only" ? "#9d7bff" : "#36d1dc";
      shapes.push({ type: "line", x0: m.sim_time, x1: m.sim_time, yref: "paper",
        y0: 0, y1: 1, line: { color: col, width: 1.5 } });
    });

    const traces = [
      { x, y: soft, name: "SoLEXS (soft)", mode: "lines", line: { color: "#36d1dc", width: 1.6 }, yaxis: "y" },
      { x, y: bg, name: "background", mode: "lines", line: { color: "#3a4d66", width: 1, dash: "dot" }, yaxis: "y" },
      { x, y: hard, name: "HEL1OS (hard)", mode: "lines", line: { color: "#9d7bff", width: 1.2 }, yaxis: "y2" },
    ];
    const layout = darkLayout({
      margin: { l: 56, r: 56, t: 26, b: 34 },
      xaxis: { gridcolor: "#1d2939", color: "#6b7c93",
        range: [sim.t_start, sim.t_end], type: "date" },
      yaxis: { title: "SoLEXS cts/s", gridcolor: "#1d2939", color: "#36d1dc", rangemode: "tozero" },
      yaxis2: { title: "HEL1OS cts/s", overlaying: "y", side: "right", color: "#9d7bff", showgrid: false, rangemode: "tozero" },
      shapes,
    });
    Plotly.react(chartRef.current, traces, layout, PLOT_CFG);

    setClock(fr.simulated_utc);
    setProb(fr.current_probability);

    // Event feed (newest first).
    const feed = [
      ...firedMarkers.map((m) => ({ kind: "nowcast", t: m.elapsed_s, m })),
      ...firedAlerts.map((a) => ({ kind: "alert", t: a.elapsed_s, a })),
    ].sort((p, q) => q.t - p.t);
    setEvents(feed);

    // Toasts for newly-fired alerts.
    if (firedAlerts.length > shownAlerts.current) {
      const fresh = firedAlerts.slice(shownAlerts.current);
      shownAlerts.current = firedAlerts.length;
      fresh.forEach((a) => {
        const id = Math.random();
        setToasts((ts) => [...ts, { id, a }]);
        setTimeout(() => setToasts((ts) => ts.filter((t) => t.id !== id)), 5200);
      });
    } else {
      shownAlerts.current = firedAlerts.length;
    }
  }, [sim, nFrames, daySec]);

  // Animation loop.
  useEffect(() => {
    if (!sim) return;
    const loop = (ts) => {
      const st = stateRef.current;
      if (st.last === 0) st.last = ts;
      const dt = (ts - st.last) / 1000;
      st.last = ts;
      if (st.playing) {
        st.elapsedWall += dt;
        let p = st.elapsedWall / wallTarget;
        if (p >= 1) { p = 1; st.playing = false; setPlaying(false); }
        setProgress(p);
        const idx = Math.floor(p * (nFrames - 1));
        if (idx !== lastDrawnFrame.current) { lastDrawnFrame.current = idx; drawFrame(idx); }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    drawFrame(0);
    return () => cancelAnimationFrame(rafRef.current);
  }, [sim, wallTarget, nFrames, drawFrame]);

  const play = () => {
    const st = stateRef.current;
    if (progress >= 1) { st.elapsedWall = 0; shownAlerts.current = 0; }
    st.playing = true; st.last = 0; setPlaying(true);
  };
  const pause = () => { stateRef.current.playing = false; setPlaying(false); };
  const restart = () => {
    const st = stateRef.current;
    st.elapsedWall = 0; st.playing = false; shownAlerts.current = 0;
    lastDrawnFrame.current = -1; setPlaying(false); setProgress(0); drawFrame(0);
  };
  const scrub = (v) => {
    const st = stateRef.current;
    st.elapsedWall = v * wallTarget;
    setProgress(v);
    shownAlerts.current = sim.forecast_alerts.filter((a) => a.elapsed_s <= v * daySec).length;
    const idx = Math.floor(v * (nFrames - 1));
    lastDrawnFrame.current = idx; drawFrame(idx);
  };

  if (!sim) return <div className="empty">No simulation loaded.</div>;
  const speedFactor = Math.round(daySec / wallTarget);
  const probPct = prob != null ? (prob * 100).toFixed(0) : "—";
  const probColor = prob == null ? "#6b7c93" : prob > 0.5 ? "#ff5c5c" : prob > 0.3 ? "#ffb627" : "#3ddc84";

  return (
    <div className="sim-layout">
      <div className="sim-main">
        <div className="sim-clock-bar">
          <div className="sim-clock">{clockUTC(clock)}
            <span className="date">{dateUTC(clock)} UTC</span></div>
          <span className="sim-speedchip">⏩ {speedFactor.toLocaleString()}× · 24h→{wallTarget}s</span>
          <div className="sim-prob">
            <div className="pl">FLARE PROBABILITY</div>
            <div className="pv" style={{ color: probColor }}>{probPct}<span style={{ fontSize: 14 }}>%</span></div>
          </div>
        </div>

        <div className="sim-chart"><div ref={chartRef} style={{ width: "100%", height: "100%" }} /></div>

        <div className="sim-controls">
          <button className="ctrl-btn play" onClick={playing ? pause : play}>{playing ? "❚❚" : "▶"}</button>
          <button className="ctrl-btn" onClick={restart}>↺</button>
          <input className="scrub" type="range" min="0" max="1" step="0.001"
            value={progress} onChange={(e) => scrub(parseFloat(e.target.value))} />
          <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "#6b7c93", minWidth: 40 }}>
            {(progress * 100).toFixed(0)}%</span>
          <div className="speed-sel">
            {[120, 60, 30].map((w) => (
              <button key={w} className={wallTarget === w ? "active" : ""}
                onClick={() => { setWallTarget(w); stateRef.current.elapsedWall = progress * w; }}>
                24h→{w}s</button>
            ))}
          </div>
          <div style={{ flex: 1 }} />
          <ReportButton job={job} />
        </div>
      </div>

      <div className="sim-side">
        <div className="panel" style={{ flexShrink: 0 }}>
          <h3 className="panel-title">Live Event Log</h3>
          <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "#6b7c93" }}>
            {events.length} events · {sim.nowcast_markers.length} flares · {sim.forecast_alerts.length} alerts
          </div>
        </div>
        <div className="event-feed">
          {events.length === 0 && <div className="muted" style={{ fontFamily: "var(--mono)", fontSize: 12 }}>Press ▶ to begin replay…</div>}
          {events.map((e, i) => e.kind === "alert" ? (
            <div className="evt alert" key={"a" + i}>
              <div className="et"><span>⚠ FORECAST</span><span>+{(e.a.elapsed_s / speedFactor).toFixed(1)}s</span></div>
              <div className="eh">Flare likely{e.a.predicted_class && e.a.predicted_class !== "flare" ? ` · ${e.a.predicted_class}` : ""}</div>
              {e.a.lead_time_min != null
                ? <span className="lead-badge">▶ {e.a.lead_time_min} min lead</span>
                : <div className="ef">p={(e.a.probability * 100).toFixed(0)}% · watching</div>}
              {e.a.contributing_features && e.a.contributing_features.length > 0 &&
                <div className="ef" style={{ marginTop: 4 }}>{e.a.contributing_features.map((c) => c.feature).join(" · ")}</div>}
            </div>
          ) : (
            <div className="evt nowcast" key={"m" + i}>
              <div className="et"><span>● NOWCAST</span><span>+{(e.m.elapsed_s / speedFactor).toFixed(1)}s</span></div>
              <div className="eh">{fmtClass(e.m.goes_class)} flare detected</div>
              <div className="ef">peak {Math.round(e.m.peak_flux)} cts/s · <span className={"prov " + e.m.provenance}>{e.m.provenance}</span></div>
            </div>
          ))}
        </div>
      </div>

      <div className="toast-stack">
        {toasts.map((t) => (
          <div className="toast" key={t.id}>
            <div className="th">⚠ {t.a.lead_time_min != null
              ? `Flare likely in ~${Math.round(t.a.lead_time_min)} min` : "Elevated flare probability"}</div>
            <div className="tb">
              p = {(t.a.probability * 100).toFixed(0)}%
              {t.a.contributing_features && t.a.contributing_features.length > 0 &&
                <> · {t.a.contributing_features.map((c) => c.feature).join(", ")}</>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ===================================================================== */
/* Light-curve screen                                                     */
/* ===================================================================== */
function LightCurveScreen({ lc }) {
  const ref = useRef();
  useEffect(() => {
    if (!lc || !ref.current) return;
    const x = lc.time;
    const bands = GOES_BANDS.map((b) => ({
      type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y",
      y0: wm2ToCounts(b.lo), y1: wm2ToCounts(b.hi),
      fillcolor: b.color, line: { width: 0 }, layer: "below",
    }));
    const annos = GOES_BANDS.map((b) => ({
      xref: "paper", x: 0.995, xanchor: "right", yref: "y",
      y: Math.sqrt(wm2ToCounts(b.lo) * wm2ToCounts(b.hi)),
      text: b.cls, showarrow: false, font: { size: 13, color: "#6b7c93", family: "var(--mono)" },
    }));
    const traces = [
      { x, y: lc.solexs_counts, name: "SoLEXS soft X-ray", mode: "lines", line: { color: "#36d1dc", width: 1.4 }, yaxis: "y" },
      { x, y: lc.hxr_broad, name: "HEL1OS 18–160 keV", mode: "lines", line: { color: "#9d7bff", width: 1.2 }, yaxis: "y2" },
      { x, y: lc.hxr_80_150, name: "HEL1OS 80–150 keV", mode: "lines", line: { color: "#ff5c5c", width: 0.9 }, yaxis: "y2" },
    ];
    const layout = darkLayout({
      height: 560,
      margin: { l: 64, r: 64, t: 30, b: 40 },
      xaxis: { gridcolor: "#1d2939", color: "#6b7c93", type: "date" },
      yaxis: { title: "SoLEXS cts/s (log)", type: "log", gridcolor: "#1d2939", color: "#36d1dc", shapes: bands },
      yaxis2: { title: "HEL1OS cts/s (log)", type: "log", overlaying: "y", side: "right", color: "#9d7bff", showgrid: false },
      shapes: bands, annotations: annos,
    });
    Plotly.react(ref.current, traces, layout, PLOT_CFG);
  }, [lc]);
  if (!lc) return <div className="empty">No light-curve data.</div>;
  return (
    <div className="panel">
      <h3 className="panel-title">Full-Day Light Curves · GOES class bands shaded</h3>
      <div ref={ref} style={{ width: "100%", height: 560 }} />
      <p className="muted" style={{ fontFamily: "var(--mono)", fontSize: 11, marginTop: 8 }}>
        Soft (SoLEXS, left axis) overlaid with hard X-ray (HEL1OS, right axis). GOES bands derived from the
        SoLEXS→W/m² cross-calibration (placeholder constant, tunable vs NOAA catalogue).
      </p>
    </div>
  );
}

/* ===================================================================== */
/* Alerts + explainability screen                                         */
/* ===================================================================== */
function AlertsScreen({ sim, importances }) {
  if (!sim || !sim.forecast_alerts) return <div className="empty">No alerts.</div>;
  const alerts = sim.forecast_alerts;
  const matched = alerts.filter((a) => a.lead_time_min != null);
  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr" }}>
      <div className="panel">
        <h3 className="panel-title">Alert Ticker · {alerts.length} alerts ({matched.length} matched a real flare)</h3>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(330px,1fr))" }}>
          {alerts.map((a, i) => (
            <div className={"evt alert"} key={i} style={{ animation: "none" }}>
              <div className="et"><span>⚠ {clockUTC(a.sim_time)} UTC</span>
                <span>p={(a.probability * 100).toFixed(0)}%</span></div>
              <div className="eh">
                {a.lead_time_min != null
                  ? `⚠ ${a.matched_flare ? fmtClass(a.matched_flare.goes_class) : ""} flare likely in ~${Math.round(a.lead_time_min)} min`
                  : "Elevated flare probability"}
              </div>
              {a.lead_time_min != null && <span className="lead-badge">▶ {a.lead_time_min} min lead time</span>}
              {a.contributing_features && a.contributing_features.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  {a.contributing_features.map((c, j) => {
                    const maxImp = Math.max(...a.contributing_features.map((x) => x.importance || 0.0001));
                    const w = ((c.importance || 0) / maxImp) * 100;
                    return (
                      <div key={j} style={{ marginBottom: 4 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--mono)", fontSize: 10, color: "#8ea0b8" }}>
                          <span>{c.feature}</span><span>{c.value}</span>
                        </div>
                        <div className="bar-track" style={{ height: 7 }}>
                          <div className="bar-fill" style={{ width: w + "%" }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ===================================================================== */
/* Catalogue screen                                                       */
/* ===================================================================== */
function FlareTable({ rows, calibrated }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="data">
        <thead><tr>
          <th>Peak (UTC)</th><th>Class{calibrated ? "" : " (uncal.)"}</th><th>Provenance</th>
          <th>Start</th><th>End</th><th>Duration</th><th>Peak cts/s</th><th>Confidence</th>
          <th>GOES match</th>
        </tr></thead>
        <tbody>
          {rows.map((f, i) => (
            <tr key={i}>
              <td>{clockUTC(f.peak_time)}</td>
              <td><span className={"cls " + classLetter(f.goes_class)}>{fmtClass(f.goes_class)}</span></td>
              <td><span className={"prov " + f.provenance}>{f.provenance}</span></td>
              <td>{clockUTC(f.start)}</td>
              <td>{clockUTC(f.end)}</td>
              <td>{(f.duration / 60).toFixed(1)} min</td>
              <td>{Math.round(f.peak_flux)}</td>
              <td><span className="confbar"><i style={{ width: (f.confidence * 100) + "%" }} /></span> {(f.confidence * 100).toFixed(0)}%</td>
              <td>{f.goes_match === true
                ? <span className="match-yes">✓ {f.goes_truth_class || ""}</span>
                : f.goes_match === false
                  ? <span className="match-no">✗</span>
                  : <span className="muted" style={{ fontSize: 10 }}>uncalibrated</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CatalogScreen({ catalog, calibrated }) {
  if (!catalog || !catalog.length) return <div className="empty">No flares catalogued.</div>;
  const confirmed = catalog.filter((f) => (f.category || f.provenance) === "confirmed_flare" ||
    f.provenance === "both" || f.provenance === "soft_only");
  const candidates = catalog.filter((f) => (f.category === "hxr_candidate") || f.provenance === "hard_only");
  const nHard = confirmed.filter((f) => f.provenance === "both").length;
  return (
    <div className="grid" style={{ gridTemplateColumns: "1fr", gap: 18 }}>
      {!calibrated && (
        <div className="warn-banner">⚠ GOES classes are GOES-EQUIVALENT (uncalibrated) — derived from a
          placeholder SoLEXS→W/m² constant. Drop a NOAA/GOES flare-list CSV at
          <code> data/catalog/goes_flares.csv </code> to populate real ✓/✗ matches and calibrate classes.</div>
      )}
      <div className="panel">
        <h3 className="panel-title">Confirmed Flares · {confirmed.length} ({nHard} hard-X-ray confirmed)</h3>
        <p className="muted" style={{ fontFamily: "var(--mono)", fontSize: 11, marginTop: -6, marginBottom: 12 }}>
          Thermal soft-X-ray response present (provenance both / soft_only) — the hallmark of a real solar flare.
        </p>
        {confirmed.length ? <FlareTable rows={confirmed} calibrated={calibrated} />
          : <div className="empty">None.</div>}
      </div>
      <div className="panel">
        <h3 className="panel-title" style={{ }}>
          <span style={{ display: "inline-flex", alignItems: "center" }}>HXR Transient Candidates · {candidates.length}</span>
        </h3>
        <p className="muted" style={{ fontFamily: "var(--mono)", fontSize: 11, marginTop: -6, marginBottom: 12 }}>
          Hard-X-ray only, NO thermal counterpart — possible particle hits / instrumental spikes / non-thermal
          microflares. Flagged for vetting, NOT counted as confirmed flares.
        </p>
        {candidates.length ? <FlareTable rows={candidates} calibrated={calibrated} />
          : <div className="empty">None.</div>}
      </div>
    </div>
  );
}

/* ===================================================================== */
/* Metrics screen                                                         */
/* ===================================================================== */
function MetricsScreen({ metrics }) {
  const rocRef = useRef();
  const ltRef = useRef();
  useEffect(() => {
    if (!metrics || !metrics.ROC || !rocRef.current) return;
    const roc = metrics.ROC;
    Plotly.react(rocRef.current, [
      { x: roc.fpr, y: roc.tpr, mode: "lines", fill: "tozeroy", name: "ROC",
        line: { color: "#36d1dc", width: 2 }, fillcolor: "rgba(54,209,220,0.08)" },
      { x: [0, 1], y: [0, 1], mode: "lines", name: "chance", line: { color: "#3a4d66", width: 1, dash: "dash" } },
    ], darkLayout({
      height: 300, showlegend: false,
      xaxis: { title: "False Positive Rate", gridcolor: "#1d2939", color: "#6b7c93", range: [0, 1] },
      yaxis: { title: "True Positive Rate", gridcolor: "#1d2939", color: "#6b7c93", range: [0, 1] },
      annotations: [{ x: 0.6, y: 0.2, text: `AUC = ${roc.auc ?? "n/a"}`, showarrow: false, font: { color: "#36d1dc", size: 16, family: "var(--mono)" } }],
    }), PLOT_CFG);
  }, [metrics]);

  useEffect(() => {
    if (!metrics || !metrics.lead_time || !metrics.lead_time.histogram || !ltRef.current) return;
    const h = metrics.lead_time.histogram;
    Plotly.react(ltRef.current, [
      { x: Object.keys(h), y: Object.values(h), type: "bar",
        marker: { color: "#3ddc84" } },
    ], darkLayout({
      height: 300, showlegend: false,
      xaxis: { title: "lead time", gridcolor: "#1d2939", color: "#6b7c93" },
      yaxis: { title: "alerts", gridcolor: "#1d2939", color: "#6b7c93" },
    }), PLOT_CFG);
  }, [metrics]);

  if (!metrics || metrics.TSS == null) {
    return <div className="empty">No forecast metrics (insufficient labelled flares on this upload).</div>;
  }
  const lt = metrics.lead_time || {};
  const ev = metrics.event_level || {};
  const imp = metrics.feature_importances || {};
  const impMax = Math.max(...Object.values(imp), 0.0001);
  return (
    <div>
      {metrics.data_warning && <div className="warn-banner">⚠ {metrics.data_warning}</div>}
      <div className="panel-title" style={{ marginBottom: 10 }}>Per-sample skill (per-second)</div>
      <div className="metric-cards">
        <div className="mcard headline"><div className="ml">TSS · headline</div><div className="mv">{metrics.TSS >= 0 ? "+" : ""}{metrics.TSS}</div><div className="mh">True Skill Statistic</div></div>
        <div className="mcard"><div className="ml">HSS</div><div className="mv">{metrics.HSS}</div><div className="mh">Heidke Skill Score</div></div>
        <div className="mcard"><div className="ml">ROC-AUC</div><div className="mv">{metrics.ROC?.auc ?? "—"}</div><div className="mh">discrimination</div></div>
        <div className="mcard"><div className="ml">Median Lead</div><div className="mv">{lt.median ?? "—"}<span style={{ fontSize: 14 }}> min</span></div><div className="mh">{lt.in_target_15_30_pct ?? 0}% in 15–30 min</div></div>
        <div className="mcard"><div className="ml">Alerts</div><div className="mv">{lt.n_alerts ?? 0}</div><div className="mh">{lt.n_matched ?? 0} matched · {lt.n_false ?? 0} false</div></div>
      </div>
      <div className="panel-title" style={{ margin: "18px 0 10px" }}>Event-level skill (per-flare) — the operational story</div>
      <div className="metric-cards">
        <div className="mcard headline"><div className="ml">Event Recall</div><div className="mv">{ev.event_recall != null ? (ev.event_recall * 100).toFixed(0) + "%" : "—"}</div><div className="mh">alerted {ev.n_alerted ?? 0}/{ev.n_flares ?? 0} confirmed flares</div></div>
        <div className="mcard"><div className="ml">Mean Lead</div><div className="mv">{ev.mean_lead ?? "—"}<span style={{ fontSize: 14 }}> min</span></div><div className="mh">median {ev.median_lead ?? "—"} min</div></div>
        <div className="mcard"><div className="ml">False Alarms</div><div className="mv">{ev.false_alarm_count ?? "—"}</div><div className="mh">FAR {ev.far_per_day ?? "—"}/day</div></div>
        <div className="mcard"><div className="ml">GOES calibration</div><div className="mv" style={{ fontSize: 18 }}>{metrics.goes_calibrated ? "calibrated" : "uncal."}</div><div className="mh">{metrics.goes_calibrated ? "matched to NOAA" : "placeholder constant"}</div></div>
      </div>

      <div className="row" style={{ marginBottom: 16 }}>
        <div className="col panel"><h3 className="panel-title">ROC Curve</h3><div ref={rocRef} style={{ height: 300 }} /></div>
        <div className="col panel"><h3 className="panel-title">Lead-Time Distribution</h3><div ref={ltRef} style={{ height: 300 }} /></div>
      </div>

      <div className="row">
        <div className="col panel">
          <h3 className="panel-title">Feature Importances (XGBoost)</h3>
          <div className="bars">
            {Object.entries(imp).map(([k, v]) => (
              <div className="bar-row" key={k}>
                <span>{k}</span>
                <div className="bar-track"><div className="bar-fill" style={{ width: (v / impMax * 100) + "%" }} /></div>
                <span>{v.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="col panel">
          <h3 className="panel-title">Confusion (held-out test)</h3>
          {metrics.confusion && (
            <div style={{ fontFamily: "var(--mono)" }}>
              <div className="kv"><span className="k">True Positives</span><span style={{ color: "#3ddc84" }}>{metrics.confusion.TP}</span></div>
              <div className="kv"><span className="k">False Positives</span><span style={{ color: "#ffb627" }}>{metrics.confusion.FP}</span></div>
              <div className="kv"><span className="k">False Negatives</span><span style={{ color: "#ff5c5c" }}>{metrics.confusion.FN}</span></div>
              <div className="kv"><span className="k">True Negatives</span><span>{metrics.confusion.TN}</span></div>
              <div className="kv"><span className="k">Operating threshold</span><span>{metrics.threshold}</span></div>
              <div className="kv"><span className="k">Test samples</span><span>{metrics.n_test_samples?.toLocaleString()}</span></div>
              <div className="kv"><span className="k">Observation days</span><span>{metrics.n_days}</span></div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ===================================================================== */
/* App shell                                                              */
/* ===================================================================== */
const TABS = [
  ["upload", "Upload"], ["sim", "Simulation"], ["light", "Light Curves"],
  ["alerts", "Alerts"], ["catalog", "Catalogue"], ["metrics", "Metrics"],
];

function ReportButton({ job, big }) {
  const [busy, setBusy] = useState(false);
  if (!job || !job.job_id) return null;
  const dl = async () => {
    setBusy(true);
    try {
      const r = await fetch(`${base()}/report/${job.job_id}`);
      if (!r.ok) throw new Error("HTTP " + r.status);
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `Aditya-L1_Flare_Report_${job.metadata?.date || ""}.pdf`;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert("Report generation failed: " + e.message);
    } finally { setBusy(false); }
  };
  return (
    <button className={"report-btn" + (big ? " big" : "") + (busy ? " busy" : "")}
      onClick={dl} disabled={busy} title="Download a full PDF report — graphs, catalogue, metrics, alerts">
      {busy ? <><span className="spin" /> Generating PDF…</> : <>⬇ Download Report (PDF)</>}
    </button>
  );
}

function App() {
  const [tab, setTab] = useState("upload");
  const [job, setJob] = useState(null);

  const onLoaded = (j) => { setJob(j); setTab("sim"); };
  const ready = !!job;

  return (
    <div className="app">
      <div className="topbar">
        <div className="logo">
          <div className="logo-badge" />
          <div>
            <div className="logo-title">ADITYA-L1 · Solar Flare Forecasting</div>
            <div className="logo-sub">SoLEXS + HEL1OS · Neupert-coupled nowcast & forecast</div>
          </div>
        </div>
        <div className="topbar-spacer" />
        {job && <ReportButton job={job} />}
        {job && <span className="status-pill"><span className="dot live" /> {job.metadata?.date} · {(job.metadata?.instruments_found || []).join("+")}</span>}
        <span className="status-pill"><span className={"dot" + (ready ? " live" : "")} /> {ready ? "DATA LOADED" : "AWAITING UPLOAD"}</span>
      </div>

      <div className="tabs">
        {TABS.map(([id, label], i) => (
          <div key={id}
            className={"tab" + (tab === id ? " active" : "") + (!ready && id !== "upload" ? " disabled" : "")}
            onClick={() => (ready || id === "upload") && setTab(id)}>
            <span className="tnum">{String(i).padStart(2, "0")}</span> {label}
          </div>
        ))}
      </div>

      <div className="content">
        {tab === "upload" && (
          <>
            <UploadScreen onLoaded={onLoaded} />
            {job && <QualityReport meta={job.metadata} quality={job.quality}
              summary={job.catalogSummary || job.catalog_summary} nAlerts={job.n_alerts} />}
          </>
        )}
        {tab === "sim" && <SimulationScreen sim={job?.simulation} job={job} />}
        {tab === "light" && <LightCurveScreen lc={job?.lightcurve} />}
        {tab === "alerts" && <AlertsScreen sim={job?.simulation} importances={job?.metrics?.feature_importances} />}
        {tab === "catalog" && <CatalogScreen catalog={job?.catalog}
          calibrated={job?.goesCalibrated ?? job?.goes_calibrated} />}
        {tab === "metrics" && <MetricsScreen metrics={job?.metrics} />}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
