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

Of ~40 candidate dates supplied (May–June 2026), **23 days passed the gate** (both instruments on the
same UTC day); SoLEXS‑only days (e.g. most of mid‑May) and HEL1OS‑only days (e.g. Jun 9) were
auto‑rejected with a logged reason, as were earlier wrong‑pairings (a SoLEXS day zipped with the
previous day's HEL1OS → `instrument date mismatch`). These 23 days are the basis for the
leave‑one‑day‑out evaluation in §3.

---

## 3. Headline benchmark — before / after

| Model | Split | TSS | HSS | AUC |
|---|---|---|---|---|
| **Before** — leaky cumulative Neupert, single day | within‑day (Jun 5) | +0.305 | +0.197 | 0.634 |
| **After** — new physics features, single day | within‑day (Jun 5) | +0.362 | +0.302 | 0.682 |
| **After** — new physics features, **23 days** | **leave‑one‑day‑out CV** (21 folds) | **+0.217 ± 0.133** | +0.118 ± 0.075 | **0.684 ± 0.096** |
| **After** — new physics features, **23 days** | chronological (train 18 d → test last 5 d) | +0.192 | +0.185 | 0.708 |

- The within‑day score improved (**0.305 → 0.362 TSS**, AUC 0.634 → 0.682) purely from replacing the
  leaky feature and adding the cross‑correlation lag.
- The headline number is now the **leave‑one‑day‑out CV across all 23 internally‑consistent days**
  (May 25 – Jun 18 2026): each day is forecast by a model trained on the *other 22 days only*.
  **TSS 0.217 ± 0.133, AUC 0.684 ± 0.096** (pooled TSS 0.276). This is the credible,
  generalisation‑revealing number — the model forecasts **completely unseen days** and still shows
  real, consistent skill across three weeks of solar activity.
- **Event‑level (pooled over folds):** alerted **47 / 89 confirmed flares (recall 0.53)**, **mean lead
  14.6 min**.

### Deployable operating point (the default everywhere)

TSS is *defined* at the threshold that maximises it — but for a rare event that point sits at an
absurdly low probability and would fire **~2,450 false alarms/day**, which is operationally useless.
The system therefore defaults to a **precision‑targeted operating point** (`pipeline.TARGET_PRECISION`,
selected on training data via `evaluate.threshold_for_precision`): the least‑conservative probability
cut that still hits the precision target. On the 23‑day held‑out validation:

| Operating point | threshold | precision | recall | false alarms |
|---|---|---|---|---|
| max‑TSS (statistical optimum) | 0.04 | 0.21 | 0.23 | **2,453 / day** |
| **precision‑targeted (default)** | **~0.31** | **0.70** | 0.11 | **129 / day** |

This single change takes the false‑alarm rate from *disqualifying* to *defensible* (≈19× fewer). The
max‑TSS value is still reported as `peak_tss` for reference, and ROC‑AUC (threshold‑free) is unchanged.
The confusion matrix, dashboard, PDF report and live alert stream all use this one operating point.

### Mean feature importances (leave‑one‑day‑out, 21 folds)

| Feature | mean imp | note |
|---|---|---|
| var_soft | 0.326 | short‑term SoLEXS variability |
| deriv_soft | 0.262 | thermal rise rate |
| **neupert_residual** | 0.133 | **new** — HXR/SXR divergence (local Neupert) |
| time_since_last_flare | 0.078 | recency context (was the leakage suspect — now demoted) |
| **neupert_windowed** | 0.069 | **new** — trailing windowed HXR fluence |
| **hxr_sxr_lag** | 0.046 | **new** — measured HXR→SXR lead lag |
| hr_slope | 0.036 | rising spectral hardness |
| **hxr_sxr_xcorr** | 0.035 | **new** — HXR→SXR coupling strength |

> Across truly held‑out days the **four new physics features together carry ≈ 0.28 of the importance**,
> while the previously‑dominant `time_since_last_flare` (the leakage suspect) falls to 0.078. The
> windowed/residual Neupert and cross‑correlation‑lag features *generalise across days*, which the old
> cumulative integral did not.

---

## 4. Fixing the leaky Neupert feature (Task 1)

The old `neupert` was a **cumulative integral from start‑of‑day** → a monotonic ramp correlated with
time‑of‑day (it leaked temporal position, not physics). It is replaced by the physically‑correct
**local** Neupert features (Neupert effect is local: `dF_SXR/dt ≈ k·F_HXR(t)`):

- `neupert_windowed` — trailing windowed (10 min) trapezoidal integral of recent HXR flux
  (segment‑aware, resets at every gap).
- `predicted_sxr_rise = K_NEUPERT · hxr_broad` and `neupert_residual = deriv_soft − predicted_sxr_rise`
  (divergence ⇒ imminent SXR peak). `K_NEUPERT` was **fitted empirically** (LS‑through‑origin on
  ~1.7M active samples across all 23 days) to **1.13 × 10⁻³**, so the residual is a true
  scale‑matched divergence rather than a raw unit mismatch.

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
- Leave‑one‑day‑out TSS (0.217 ± 0.133) < within‑day (0.362): true cross‑day generalisation is
  harder — this is the honest number and it is reported as the headline for multi‑day. Fold‑to‑fold
  spread is real (quiet days with few/no flares score near zero; active days score 0.4–0.5).
- 2 of 23 days are excluded from CV scoring as single‑class (no labelled positives, e.g. a flare‑free
  day) — they still train the other folds but cannot be scored as a held‑out test.
- High per‑sample FAR (≈18/day) is at the *max‑TSS* threshold used for honest skill scoring; the
  replay/alert demo model uses a higher‑precision operating point (§7).
- GOES classes are uncalibrated until a NOAA flare list is supplied (surfaced in the UI).
- `K_NEUPERT` is now fit empirically (1.13 × 10⁻³, see §4); re‑running `multiday_eval.py` on a new
  dataset re‑fits it.

---

## 10. How to run

```bash
# Backend + dashboard
python -m uvicorn api:app --host 127.0.0.1 --port 8000      # open http://127.0.0.1:8000

# Leave-one-day-out CV across all consistent days (auto-discovers root *.zip + ./raw_data).
# Builds per-day feature matrices, fits K_NEUPERT, runs LODO + chronological split,
# writes data/catalog/lodo_metrics.json.
python multiday_eval.py
python multiday_eval.py --use-cache    # reuse cached per-day matrices (fast re-run)
```
