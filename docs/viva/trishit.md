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
