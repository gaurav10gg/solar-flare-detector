"""
api.py — FastAPI backend for the Aditya-L1 flare forecasting dashboard.

Endpoints
---------
POST /upload                 : accept file(s)/zip, run the full pipeline,
                               persist results, return job_id + metadata + quality.
GET  /catalog/{job_id}       : the flare catalogue table.
GET  /metrics/{job_id}       : TSS / HSS / AUC / lead-time stats + ROC points.
GET  /lightcurve/{job_id}    : downsampled full-day curves for plotting.
GET  /simulation/{job_id}    : full prebuilt replay payload (series + frames +
                               markers + alerts) for client-side animation.
GET  /report/{job_id}        : full PDF mission report (figures + tables).
WS   /simulate/{job_id}?speed=60 : stream replay frames in real time.
GET  /jobs                   : list processed jobs.
GET  /health                 : liveness.

The built frontend (``frontend/``) is served as static files at ``/``.
CORS is open for local development.

Run::

    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import run_pipeline
from simulate import build_simulation
from report import build_pdf_report

logger = logging.getLogger("api")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

UPLOAD_DIR = "data/uploads"
FRONTEND_DIR = "frontend"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Aditya-L1 Solar Flare Forecasting", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=False,
)

# In-memory job store: job_id -> {bundle, simulation, metadata, quality}.
# (For a demo this is sufficient; a production deployment would use a DB/cache.)
JOBS: dict[str, dict] = {}


def _get(job_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    return JOBS[job_id]


# --------------------------------------------------------------------------- #
# Upload + processing
# --------------------------------------------------------------------------- #
@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    """Accept one or more SoLEXS/HEL1OS files (zip/fits/lc.gz), process, return job."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    saved = []
    for f in files:
        dest = os.path.join(job_dir, os.path.basename(f.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    logger.info("upload[%s]: saved %d file(s) -> %s", job_id, len(saved), job_dir)

    # Run the full pipeline off the event loop (it is CPU-bound and ~20s).
    try:
        bundle = await asyncio.to_thread(
            run_pipeline, job_dir,
            sqlite_path=os.path.join(job_dir, "flares.sqlite"),
        )
    except Exception as exc:  # surface a clean error to the client
        logger.exception("upload[%s]: pipeline failed", job_id)
        raise HTTPException(status_code=422, detail=f"Processing failed: {exc}")

    simulation = build_simulation(
        bundle["feats"], bundle["catalog"], bundle["prob_curve"],
        bundle["alerts"], day_in_seconds=60, n_frames=600,
    )

    JOBS[job_id] = {
        "bundle": bundle, "simulation": simulation,
        "metadata": bundle["metadata"], "quality": bundle["quality"],
    }
    return {
        "job_id": job_id,
        "metadata": bundle["metadata"],
        "quality": bundle["quality"],
        "catalog_summary": bundle["catalog_summary"],   # confirmed vs candidates
        "goes_calibrated": bundle["goes_calibrated"],
        "n_confirmed": bundle["catalog_summary"]["n_confirmed"],
        "n_candidates": bundle["catalog_summary"]["n_candidates"],
        "n_alerts": len(bundle["alerts"]),
        "metrics_summary": {k: bundle["metrics"].get(k)
                            for k in ("TSS", "HSS", "ROC", "n_days", "data_warning",
                                      "event_level", "goes_calibrated")},
    }


# --------------------------------------------------------------------------- #
# Data endpoints
# --------------------------------------------------------------------------- #
@app.get("/catalog/{job_id}")
async def catalog(job_id: str):
    """Flare catalogue rows + honest confirmed/candidate summary and GOES-cal flag."""
    b = _get(job_id)["bundle"]
    return {"job_id": job_id, "flares": b["catalog_records"],
            "summary": b["catalog_summary"], "goes_calibrated": b["goes_calibrated"]}


@app.get("/metrics/{job_id}")
async def metrics(job_id: str):
    """Forecast skill metrics: TSS/HSS/AUC, ROC points, lead-time stats, importances."""
    b = _get(job_id)["bundle"]
    return {"job_id": job_id, "metrics": b["metrics"],
            "threshold": b["threshold"], "horizon_min": b["horizon_min"]}


@app.get("/lightcurve/{job_id}")
async def lightcurve(job_id: str):
    """Downsampled full-day light curves + probability for plotting."""
    b = _get(job_id)["bundle"]
    return {"job_id": job_id, "lightcurve": b["lightcurve"]}


@app.get("/simulation/{job_id}")
async def simulation(job_id: str):
    """Full prebuilt replay payload (series, frames, markers, alerts, timing)."""
    sim = _get(job_id)["simulation"]
    return JSONResponse(sim)


@app.get("/report/{job_id}")
async def report(job_id: str):
    """Generate (and cache) a full PDF mission report for the job and return it."""
    job = _get(job_id)
    path = job.get("report_path")
    if not path or not os.path.exists(path):
        os.makedirs("data/reports", exist_ok=True)
        date = job["metadata"].get("date", "report")
        path = os.path.join("data/reports", f"aditya_report_{date}_{job_id[:8]}.pdf")
        try:
            build_pdf_report(job["bundle"], path)
        except Exception as exc:
            logger.exception("report[%s]: PDF generation failed", job_id)
            raise HTTPException(status_code=500, detail=f"Report failed: {exc}")
        job["report_path"] = path
    fname = f"Aditya-L1_Flare_Report_{job['metadata'].get('date', '')}.pdf"
    return FileResponse(path, media_type="application/pdf", filename=fname)


@app.get("/jobs")
async def jobs():
    """List processed jobs with a one-line summary each."""
    return {"jobs": [
        {"job_id": jid,
         "date": j["metadata"].get("date"),
         "instruments": j["metadata"].get("instruments_found"),
         "n_confirmed": j["bundle"]["catalog_summary"]["n_confirmed"],
         "n_candidates": j["bundle"]["catalog_summary"]["n_candidates"],
         "n_alerts": len(j["bundle"]["alerts"])}
        for jid, j in JOBS.items()
    ]}


@app.get("/health")
async def health():
    return {"status": "ok", "n_jobs": len(JOBS)}


# --------------------------------------------------------------------------- #
# WebSocket replay stream
# --------------------------------------------------------------------------- #
@app.websocket("/simulate/{job_id}")
async def simulate_ws(websocket: WebSocket, job_id: str, speed: float = 60.0):
    """Stream replay frames in real time (whole day in ~*speed* wall-seconds).

    Paces frames with ``asyncio.sleep`` so the event loop stays responsive. The
    client may send ``{"action": "stop"}`` to end the stream early.
    """
    await websocket.accept()
    if job_id not in JOBS:
        await websocket.send_json({"error": f"Unknown job_id '{job_id}'"})
        await websocket.close()
        return

    sim = JOBS[job_id]["simulation"]
    frames = sim["frames"]
    total_wall = float(speed) if speed and speed > 0 else sim["total_sim_seconds"]
    interval = total_wall / max(1, len(frames))

    await websocket.send_json({
        "type": "init", "n_frames": len(frames),
        "t_start": sim["t_start"], "t_end": sim["t_end"],
        "speed_factor": sim["speed_factor"], "total_wall_seconds": total_wall,
        "series": sim["series"], "nowcast_markers": sim["nowcast_markers"],
        "forecast_alerts": sim["forecast_alerts"],
    })

    try:
        for fr in frames:
            await websocket.send_json({"type": "frame", **fr})
            await asyncio.sleep(interval)
        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        logger.info("simulate_ws[%s]: client disconnected", job_id)
    except Exception as exc:  # pragma: no cover
        logger.warning("simulate_ws[%s]: %s", job_id, exc)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Static frontend (mounted last so API routes take precedence)
# --------------------------------------------------------------------------- #
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    @app.get("/")
    async def _no_frontend():
        return {"message": "Backend running. Build the frontend/ directory to serve the UI."}
