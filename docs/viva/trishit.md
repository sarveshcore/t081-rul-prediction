# Trishit's viva notes: data and EDA

## What I contributed

Issue #1 added the original NASA FD001 files, the loader, correct training
RUL targets, validation tests and source documentation. Issue #3 adds a
EDA notebook with training-only decisions, six report figures, CSV evidence tables, sensor
selection, the dataset/EDA report section and these explanations.

`load_fd()` returns `(train, test, test_rul)`; `load_train_fd()` opens only the
training file for EDA. `add_rul()` calculates each training engine's maximum
cycle minus the current cycle and optionally clips it at a cap. Training
engines run to failure; test histories stop earlier, so their last-cycle RUL
comes from the supplied labels, never from `add_rul(test)`.

## Findings I can defend

- Training has 20,631 rows, 100 engines and 26 raw columns. There are no
  missing cells, duplicate rows or duplicate unit-cycle keys. Cycles start
  at one and have no gaps. Lifetimes range from 128 to 362, with median 199
  and mean 206.31 cycles. Longer engines contribute more rows.
- FD001 has one nominal operating condition and HPC degradation according
  to NASA. Settings 1 and 2 form a small cloud around zero; setting 3 is
  always 100. Scatter alone is not a proof of the physical regime count.
- The exact constants are s_1, s_5, s_10, s_16, s_18 and s_19. Unique-value
  counts establish this even when floating-point std is slightly above zero.
- s_6 has only 21.60 (406 rows) and 21.61 (20,225 rows), a 98.03% dominant
  value and std about 0.001389. The share is descriptive, not a cutoff.
  Its narrow two-level range is consistent with rounding at recorded precision;
  excluding it matches Liu et al. (2021), Section 4.2, and their 14-sensor set.
  This does not prove zero predictive value. A low raw std alone is insufficient because
  sensors use different units. s_17 has 13 distinct values and is retained.
- The retained list has 14 sensors: 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15,
  17, 20 and 21. It is exported in `USEFUL_SENSORS` and checked against the
  notebook's computed evidence.
- Engines 9, 44, 65, 76 and 100 are selected with seed 42. Trend plots show
  raw readings and 11-cycle trailing means within each engine. RUL decreases
  left to right; zero is failure. Smoothing is for display, not a fitted
  preprocessing step reused for model evaluation.
- The training target is `min(125, max_cycle - cycle)`. The cap changes
  8,031 rows (38.93%) and removes no observations. Liu, Wang and Yu (2021),
  Section 4.2, use the same upper limit (DOI: 10.1007/s44196-021-00020-1).
  Early sensor trends are often flatter; this motivates, but does not prove
  the optimality of, a capped target. Issue #5 compares an uncapped run.
- Correlations include both uncapped and 125-capped training RUL. They are pooled, row-weighted
  associations, not causal relationships or independent observations.

## Explain each figure

1. `eda_lifetimes_rul.png`: lifetime histogram counts engines once; the RUL
   histogram counts rows; target curves show what caps of 100/125 change.
2. `eda_operating_settings.png`: small settings 1/2 variation and constant
   setting 3 are consistent with the published one-condition description.
3. `eda_sensor_variation.png`: std and unique-value counts support six
   exact-constant removals plus the nearly constant binary s_6 removal.
4. `eda_sensor_trends.png`: five seeded engines, aligned to failure, show
   noisier or flatter early histories and stronger late directional changes.
5. `eda_correlations.png`: several sensors share variation and correlate
   with remaining life; model evaluation decides how useful that is.

6. `eda_test_diagnostics.png`: last observed test cycle per engine and supplied
   true-RUL distribution, added last without revisiting any earlier decision.

## Ten likely questions

1. **What is RUL?** Operating cycles remaining before failure. At cycle 150
   of a training engine that fails at 200, RUL is 50.
2. **Why is the final training RUL zero?** The final observation is the
   dataset's failure endpoint. Subtracting it from itself gives zero.
3. **Why can't you calculate test RUL the same way?** The final observed test
   cycle is not failure. The supplied label gives the remaining cycles after it.
4. **Why group by engine?** Engines have different lifetimes. A global maximum
   would label short-lived engines incorrectly, and rolling across engines
   would combine unrelated histories.
5. **Why drop flat sensors?** They cannot distinguish observations in this
   training set. s_6 is almost constant; its removal is an explicit EDA choice.
6. **Why not drop every low-std sensor?** Std depends on measurement units.
   Distinct-value counts and dominant fractions reveal flatness more directly.
7. **Why cap at 125, and what about 100 or no cap?** It is the team's chosen,
   literature-supported starting point for weak early degradation information.
   A cap of 100 gives a longer plateau; no cap preserves exact labels. Only
   engine-grouped validation can compare their predictive performance.
8. **Why align plots on RUL?** Different engines fail at different cycle
   counts. Alignment lets us compare their behavior at the same distance
   from failure. It is retrospective EDA, not an available prediction input.
9. **Does correlation prove a sensor causes failure?** No. It is descriptive,
   can reflect shared degradation and between-engine effects, and misses
   nonlinear or temporal relationships.
10. **How did you avoid test leakage?** Sensor choice and cap rationale use only the
    training loader. The final cell opens test files for the two descriptive
    plots Sarvesh approved; no earlier choice was revisited. Model scalers must be fitted inside training folds; neighboring
    cycles from an engine must not be split across training and validation.

## Review-approved test diagnostics

The final cell uses `load_fd()`, counts each engine once, and joins supplied true
RUL to its last observed cycle by engine ID. It never calls `add_rul(test)` or caps
test labels. Last observed cycle is not the failure cycle or physical engine age.
Model evaluation remains in issue #6; these plots must not guide tuning.

## Phase 2: dataset choices and ablations (#37)

Code: `src/rul/ablations.py`, `scripts/run_ablations.py`,
`notebooks/08_ablations.ipynb`, `tests/test_ablations.py`. Results:
`results/ablations.csv`, `results/ablation_window_unpadded.csv`. Figures:
`figures/ablation_coverage_padding.png`, `figures/ablation_cv_comparison.png`.
Report: `report/sections/ablations.tex`.

### My part in plain words

Two jobs. First, justify how the sequence data is built: window length,
padding, the RUL cap and the operating regime. Second, take the selected
Transformer and change one thing at a time, scoring each change with the
same five engine folds, to see what each design choice was worth. No test
data is used for any ablation.

### Dataset choices

**Window length and padding.** The first w − 1 cycles of every engine have
less history than the window, so they are left-padded with copies of the
engine's first reading.

| Window | Training windows padded | Test engines shorter than the window |
|---|---|---|
| 15 | 1,400 (6.8% of 20,631 cycles) | 0 |
| 30 (chosen) | 2,900 (14.1%) | 0 (shortest test engine has 31 cycles) |
| 50 | 4,900 (23.8%) | 7 (units 1, 2, 14, 22, 25, 39, 85) |

**RUL cap at 125.** Uncapped training RUL runs from 0 to 361 (mean 107.81 ±
68.88). Early in life the sensors show no wear, so engines with very
different remaining lives look the same. Capping clips 8,031 training cycles
(38.9%) onto a plateau and cuts the target's standard deviation to 41.67.

**Operating regime.** FD001 has one condition: settings 1 and 2 have
standard deviations of 0.0022 and 0.00029, and setting 3 is always 100. So
standardising per regime and standardising globally are the same thing. I
don't claim a clustering result for this; the setting spreads are the
evidence.

### How the ablations were run

- **One change at a time** from the selected Transformer (w = 30, cap 125,
  dropout 0.1, weight decay 1e-4, early stopping with patience 10, 14
  sensors), on the same five engine folds, scored on every cycle against
  RUL capped at 125.
- **Each row is compared with a baseline from the same run.** The thread
  count alone changes training: the same baseline gives 13.00 with default
  threads, 13.17 with one thread, and 13.28 in `dl_cv.csv`. The rows came
  from two runs, so `ablations.csv` has `run`, `reference` and `rmse_delta`
  columns.
- **Evidence rule:** a difference counts only if it is bigger than the fold
  standard deviation of both rows and bigger than that run-to-run variation
  of about 0.3 cycles.

### Results

| Change | CV RMSE | Δ vs own baseline | Late RMSE |
|---|---|---|---|
| Baseline, default threads | 13.00 ± 0.67 | – | 7.85 |
| Window 15 | 16.96 ± 0.48 | +3.96 | 14.18 |
| **Window 50** | **11.34 ± 0.52** | **−1.66** | **6.42** |
| No dropout | 13.54 ± 0.84 | +0.54 | 8.14 |
| No weight decay | 13.47 ± 0.71 | +0.46 | 7.69 |
| Top 3 SHAP sensors | 18.70 ± 1.50 | +5.70 | 15.48 |
| Top 5 SHAP sensors | 15.44 ± 1.12 | +2.44 | 11.64 |
| Sequence branch only | 13.34 ± 0.34 | +0.33 | 7.34 |
| Feature branch only | 15.29 ± 0.91 | +2.29 | 12.33 |
| Both branches | 13.47 ± 0.53 | +0.47 | 7.53 |
| Baseline, one thread | 13.17 ± 0.72 | – | 8.36 |
| **No early stopping** | **15.65 ± 0.66** | **+2.48** | 9.74 |
| All three regularisers off | 16.00 ± 1.13 | +2.83 | 10.11 |
| No cap, predictions clipped at 125 | 15.42 ± 1.61 | +2.25 | 10.79 |
| No cap, scored on true RUL | 38.57 ± 2.25 | not comparable | 11.00 |

What it shows:
- **Window:** 15 is far too short. 50 is clearly better, and it stays better
  on cycles ≥ 50, where no window of either length is padded: 11.96 ± 0.59
  against 14.19 ± 0.77, better on every fold. So the gain is real history,
  not padding telling the model how young the engine is.
- **Regularisation:** early stopping is the one that matters (+2.48 without
  it). Dropout and weight decay are inside the fold spread, so no measurable
  effect.
- **Cap:** training without it is worse even when its predictions are
  clipped to the same scale as every other row.
- **Inputs:** three sensors are not enough; five recover part of the gap.
- **Hybrid branches:** the sensor window carries the signal. Adding the
  engineered features does not help, overall or near failure.

### What changed in review, and why

These are fixed in the merged version, and I should be able to explain each:
1. **"No early stopping" first still early-stopped.** `fit_net` restores the
   best early-stopping epoch at the end, so `patience = epochs` alone wasn't
   enough. `fit_net(..., restore_best=False)` now keeps the last of 40
   epochs. That turned "no effect" into +2.48.
2. **The baseline was copied from `dl_cv.csv`** (13.28) instead of being
   rerun with the rows it's compared against. Now it's rerun in each run.
3. **The no-cap row was scored unfairly.** A model trained without the cap
   correctly predicts RUL above 125 on healthy cycles, but was scored against
   labels capped at 125, so its correct predictions counted as errors. Now
   its predictions are clipped first, and the true-RUL score is a separate,
   labelled row.
4. **The 50-cycle result was first described as "no advantage".** It is the
   biggest improvement in the study, so it's now reported as one.

### 10 likely questions (Phase 2)

**1. Why change only one factor at a time?**
With two changes you can't tell which one caused the difference. Every row
differs from the baseline in exactly one setting, on the same folds.

**2. Why isn't a gap smaller than the fold spread evidence?**
A different split of engines into folds could move the score by that much.
Dropout's +0.54 is smaller than the 0.84 spread, so it could be noise.

**3. Why does each row need a baseline from its own run?**
Thread count alone moves the same baseline from 13.00 to 13.17, because the
order of floating-point operations changes training. Comparing rows from
different runs would mix that noise into the difference.

**4. A 50-cycle window is better. Why did you keep 30?**
30 was fixed before the test set was opened in #38, and the ablations ran
afterwards. Switching now would be a choice made after seeing test results.
It's reported as a limitation and the first thing to change next time.

**5. Couldn't the 50-cycle gain just come from padding?**
That was my worry too: a padded window shows how early in life an engine is.
So we scored only cycles ≥ 50, where nothing is padded for either window.
50 still wins on every fold (11.96 vs 14.19).

**6. Why does window length trade history against padding?**
Longer windows see more history but pad more early cycles (4,900 training
windows at 50), and at 50 seven test engines are shorter than the window.

**7. Why cap RUL at 125?**
Early in life the sensors can't tell 300 cycles left from 150, so the
uncapped target asks for something the data doesn't show. The cap turns
that into a flat "healthy" plateau. Without it, CV RMSE is 2.25 worse even
with predictions clipped.

**8. Which regulariser matters most, and why?**
Early stopping. Training on to 40 epochs and keeping the last weights,
instead of the best early-stopping epoch, makes CV RMSE 2.48 worse; with
only 64 training engines per fold that is the clearest sign of overfitting
in the study. Dropout and weight decay show no measurable effect.

**9. Why test the top SHAP sensors?**
To see whether Phase 1's explanation transfers: if three sensors held the
signal, the network wouldn't need all 14. It does: top 3 is +5.70, top 5 is
+2.44.

**10. What do the hybrid branch ablations show?**
The sequence branch alone matches the plain Transformer (+0.33, within the
spread). The feature branch alone is clearly worse (+2.29). Both together
are no better than the sequence branch alone, so the features add nothing
measurable once the window is there.

### Things I must check before the viva (Phase 2)
- My slide is slide 8 of `slides/phase2_slides.pptx`; the script and answers
  are in its speaker notes. About 1 minute.
- Be able to explain `restore_best` and the same-run baselines without notes.
