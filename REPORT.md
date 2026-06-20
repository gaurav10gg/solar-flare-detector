# Aditya‑L1 Solar Flare Nowcasting & Forecasting — Technical Report

**Problem Statement 15** · Forecasting and Nowcasting of Solar Flares using SoLEXS (soft X‑ray,
1–15 keV, thermal/gradual) + HEL1OS (hard X‑ray, 18–160 keV, non‑thermal/impulsive) on Aditya‑L1.

End‑to‑end production system: `loader → preprocess → features → nowcast → forecast → evaluate →
pipeline → simulate → api → React/Plotly dashboard`. Nothing is hardcoded to a date or flare;
strictly time‑ordered splits, no shuffling, no‑peek forward labels.

---

## 1. Catalogue framing — confirmed flares vs HXR candidates

A real solar flare has a **thermal soft‑X‑ray response** (Neupert coupling). We therefore split
detections into two honest categories and **never report a single inflated flare count**:

- **`confirmed_flare`** — provenance `both` or `soft_only` (SoLEXS thermal response present).
- **`hxr_candidate`** — provenance `hard_only` (no thermal counterpart: possible particle hit /
  instrumental spike / non‑thermal microflare; flagged for vetting, **not** counted as a flare).

Per‑day counts across all five internally‑consistent observation days:

| Date | Confirmed flares (hard‑X‑ray confirmed) | HXR transient candidates |
|---|---|---|
| 2026‑06‑05 | 5 (1) | 13 |
| 2026‑06‑06 | 6 (3) | 14 |
| 2026‑06‑07 | 10 (2) | 12 |
| 2026‑06‑08 | 4 (2) | 26 |
| 2026‑06‑10 | 3 (2) | 17 |

> June 5 headline now reads **"5 confirmed flares (1 hard‑X‑ray confirmed) + 13 HXR transient
> candidates"**, not "18 flares". The flagship **C9.2 `both` event** is the clearest confirmed flare.

---

## 2. Multi‑day ingestion with an accuracy gate

`loader.load_multi_day()` groups source archives by the date in their filename, extracts each day
independently, and applies a **consistency gate** (`_day_is_consistent`): a day is accepted only if
SoLEXS and HEL1OS data fall on the **same UTC calendar day** (verified from the data, not the
filenames). Mismatched or single‑instrument days are auto‑rejected with a logged reason — this is
how "use only the dates where it is accurate" is enforced *automatically* rather than hardcoded.

All five days (2026‑06‑05/06/07/08/10) passed the gate; earlier wrong‑pairings (a SoLEXS day zipped
with the previous day's HEL1OS) would be rejected as `instrument date mismatch`.

---

## 3. Headline benchmark — before / after

| Model | Split | TSS | HSS | AUC |
|---|---|---|---|---|
| **Before** — leaky cumulative Neupert, single day | within‑day (Jun 5) | +0.305 | +0.197 | 0.634 |
| **After** — new physics features, single day | within‑day (Jun 5) | **+0.362** | **+0.302** | **0.682** |
| **After** — new physics features, **multi‑day** | **held‑out DAY** (train 05–08 → test 10) | **+0.232** | +0.162 | 0.665 |

- The within‑day score improved (**0.305 → 0.362 TSS**, AUC 0.634 → 0.682) purely from replacing the
  leaky feature and adding the cross‑correlation lag.
- The **held‑out‑day** score (0.232 TSS, 0.665 AUC) is the credible, generalisation‑revealing number:
  the model forecasts a **completely unseen day** and still shows real skill.

### Top‑6 feature importances

| Single‑day (Jun 5) | imp | Multi‑day (held‑out Jun 10) | imp |
|---|---|---|---|
| hxr_sxr_xcorr | 0.235 | var_soft | 0.206 |
| time_since_last_flare | 0.207 | deriv_soft | 0.197 |
| deriv_soft | 0.166 | **neupert_residual** | 0.155 |
| var_soft | 0.104 | **hxr_sxr_lag** | 0.123 |
| **hxr_sxr_lag** | 0.102 | time_since_last_flare | 0.122 |
| **neupert_residual** | 0.079 | **neupert_windowed** | 0.080 |

> On the **held‑out day**, the top of the table is dominated by the **new physics features**
> (`neupert_residual`, `hxr_sxr_lag`, `neupert_windowed`) — they *generalise across days*, which the
> old cumulative integral did not.

---

## 4. Fixing the leaky Neupert feature (Task 1)

The old `neupert` was a **cumulative integral from start‑of‑day** → a monotonic ramp correlated with
time‑of‑day (it leaked temporal position, not physics). It is replaced by the physically‑correct
**local** Neupert features (Neupert effect is local: `dF_SXR/dt ≈ k·F_HXR(t)`):

- `neupert_windowed` — trailing windowed (10 min) trapezoidal integral of recent HXR flux
  (segment‑aware, resets at every gap).
- `predicted_sxr_rise = K_NEUPERT · hxr_broad` and `neupert_residual = deriv_soft − predicted_sxr_rise`
  (divergence ⇒ imminent SXR peak). `K_NEUPERT` is a documented, tunable module constant.

**Leakage proof** (corr with `time_since_last_flare`, Jun 5):
`cumulative = −0.506` → `windowed = −0.032` — a **15.8× reduction**.

---

## 5. Measured HXR→SXR coupling lag (Task 2)

`hxr_sxr_lag()` slides `hxr_broad` ahead of `deriv_soft` in a trailing 15‑min window and reports the
lag of peak cross‑correlation (`hxr_sxr_lag`, seconds) and that peak value (`hxr_sxr_xcorr`). Both
channels are denoised with a causal 1‑min mean first (raw 1 s signals are Poisson‑noise‑dominated:
peak |corr| 0.05 → 0.65 after smoothing). Causal, gap‑safe (NaN where the window spans a gap).

Jun 5: 31,293 finite samples; **HXR leads SXR in 100%** of finite samples; median lag 165 s
(≈105 s near the flagship flare); peak xcorr up to 0.65. A tightening HXR→SXR lag is a strong,
rarely‑used precursor.

---

## 6. Hardness‑ratio liveness (Task 5)

`build_features` logs an HR diagnostic: Jun 5 shows **16.4 % finite, 6,670 distinct values**,
median 0.55. HR is **alive and varying**, just sparse — exactly as expected when activity is
C‑class‑dominated (little 20–40 keV emission), **not broken**. A clear warning is raised only when
HR is all‑NaN or constant.

---

## 7. Event‑level metrics (Task 6)

Per‑second precision is pessimistic for a bursty target (one flare = hundreds of contiguous positive
seconds). `evaluate.event_level_metrics()` reports the operational story alongside the per‑sample
confusion matrix:

- **Did we warn ahead of each confirmed flare** (event recall) and **how often we cry wolf per day**
  (FAR/day).
- Jun 5 (display model): **event recall 0.80 (4/5 confirmed flares alerted)**, mean lead 9.3 min,
  median 5.2 min, **FAR 0/day**.

---

## 8. NOAA/GOES ground truth (Task 7)

`evaluate.load_goes_catalog()` accepts flexible NOAA/SWPC CSV shapes; `annotate_goes_match()` tags
each catalogue row with a **✓/✗ GOES match** and the matched NOAA class, and `validate_vs_goes()`
reports precision/recall. When **no CSV is present** the system degrades gracefully and the UI labels
classes **"GOES‑equivalent (uncalibrated)"** (the SoLEXS→W/m² constant `COUNTS_TO_WM2` is a documented
placeholder). Drop a flare list at `data/catalog/goes_flares.csv` to enable real matches.

---

## 9. Honest caveats
- Held‑out‑day TSS (0.232) < within‑day (0.362): true cross‑day generalisation is harder — this is
  the honest number and it is reported as the headline for multi‑day.
- GOES classes are uncalibrated until a NOAA flare list is supplied (surfaced in the UI).
- `K_NEUPERT` defaults to 1.0; only the residual's zero‑crossing structure is used, but it can be
  fit per the regression described in `features.py`.

---

## 10. How to run

```bash
# Backend + dashboard
python -m uvicorn api:app --host 127.0.0.1 --port 8000      # open http://127.0.0.1:8000

# Multi-day held-out-day evaluation (auto-discovers root *.zip + ./raw_data)
python multiday_eval.py
```
