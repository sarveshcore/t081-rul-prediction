# Viva notes: Aryan

My part: turning raw sensor readings into features that describe
degradation. Code: `src/rul/features.py`, tests in `tests/test_features.py`.
Notebook: `notebooks/02_features.ipynb`. Results:
`results/features_ablation.csv`, `results/features_hi_correlation.csv`,
figures `figures/features_degradation_trends.png` and
`figures/features_correlation.png`. Report: `report/sections/features.tex`.

## My part in plain words

One row of FD001 is one cycle of one engine. A single sensor reading is
noisy and says little about how worn the engine is. My features look at
each sensor's **recent history inside the same engine** instead.

`make_features(df, sensors, windows=(5, 10, 20), scaler=None)` returns the
same rows, in the same order and with the same index, plus new columns. For
each of the 14 `USEFUL_SENSORS`:

- **Normalised reading** (`s_11_norm`): a z-score using the training mean
  and std of the engine's operating regime.
- **Rolling mean and std** (`s_11_mean_10`, `s_11_std_10`) over the last 5,
  10 and 20 cycles.
- **Rolling slope** (`s_11_slope_10`): the least squares trend over the
  same windows, meaning how fast the sensor is moving.

Plus two columns for the whole row:

- **`regime`**: which operating condition the row belongs to (always 0 in
  FD001).
- **`health_index`**: one number from 0 (failed) to 1 (healthy) that
  combines all 14 sensors.

Rolling stats, slopes and the health index are all computed on the
normalised sensors, so `s_2_mean_5` is the mean of `s_2_norm`, not of raw
`s_2`.

The models use the **engineered set of 142 columns**: cycle + 14 raw
sensors (15), rolling mean and std (14 × 2 × 3 = 84), slopes (14 × 3 = 42)
and the health index (1).

**The scaler.** Everything learned from data (regime clusters, the
per-regime mean and std, the health index regression) lives in a
`FeatureScaler`. With `scaler=None`, `make_features` fits one on the rows it
gets, which must be training engines. With a scaler passed in, it only
applies it and never refits. In cross validation the scaler is refitted in
every fold on that fold's 80 training engines, then applied to the 20
validation engines. For the test set it is fitted once on all 100 training
engines.

**Ablation** (default Random Forest, seed 42, 5-fold `engine_folds`, every
validation cycle against RUL capped at 125):

| Features | n | RMSE | MAE | NASA score | Late RMSE |
|---|---|---|---|---|---|
| Cycle + raw sensors | 15 | 16.81 ± 0.78 | 11.53 | 26573 | 15.54 |
| + rolling mean and std | 99 | 16.36 ± 0.77 | 11.04 | 23478 | 13.86 |
| + rolling slopes | 141 | 14.78 ± 0.49 | 10.04 | 17179 | 11.37 |
| + health index | 142 | 14.74 ± 0.56 | 9.97 | 16928 | 11.24 |

The forest is left at default settings on purpose, so the table measures the
features and not the tuning. (The tuned forest reaches 14.42 on the
engineered set and the tuned XGBoost 13.99, see Husain's notes.)

## The theory

### Rolling mean and std
For window length w, at cycle t of one engine, the window W is the last
min(w, t) cycles of **that engine only**, with n points in it:

- mean m = (1/n) Σ z over W
- std σ = √( Σ (z − m)² / (n − 1) ) over W

The mean filters out sensor noise and estimates the current level of wear.
The std measures how erratic the sensor has become.

### Rolling slope (ordinary least squares)
Fit a straight line z = a + b·τ through the n points (τ = cycle) in the
window. The best b, in closed form:

    b = (n Σ τz − Σ τ Σ z) / (n Σ τ² − (Σ τ)²)

which is the same as Σ(τ − τ̄)(z − z̄) / Σ(τ − τ̄)².

All five sums (n, Στ, Στ², Σz, Στz) are rolling sums, so every slope in
the dataset is computed in one vectorised pass instead of fitting 20,631 ×
14 × 3 regressions. It takes about 0.4 s for all of FD001, and a test
checks it against `np.polyfit`.

**Worked example.** Cycles 1, 2, 3 with z = 0, 0.1, 0.3: n = 3, Στ = 6,
Στ² = 14, Σz = 0.4, Στz = 1.1. Then b = (3.3 − 2.4) / (42 − 36) = 0.15
standard deviations per cycle.

**Why the slope captures degradation.** The mean says *how worn* an engine
is; the slope says *how fast it is wearing*. FD001 sensors drift slowly for
most of a life, then accelerate as the HPC fault develops (see the s_11 and
s_12 panels of `features_degradation_trends.png`). A steep slope marks the
final phase of life, even when the sensor's level overlaps with a healthy
engine that simply started at a different baseline. The slope uses
differences within one engine, so engine-to-engine offsets in initial wear
cancel out.

### Early cycles (`min_periods=1`)
The first w − 1 cycles don't have w past readings. Instead of dropping those
rows or leaving NaN, the window shrinks to the cycles seen so far:

- cycle 1: mean = the reading, std = 0 (undefined for one point), slope = 0
  (one point has no trend; the formula's denominator is 0)
- cycles 2 to w − 1: partial windows, which are noisier

Dropping them would throw away 19 rows per engine for w = 20 (1,900 rows,
about 9% of training), and a test engine with a short history would get no
prediction. The cost: a slope fitted to 2 or 3 noisy points can be large
and meaningless (the spike at the start of the slope panels). That costs
little accuracy, because in FD001 those cycles are at least 109 cycles from
failure, where the capped target is at or near 125.

### Regime normalisation
Sensor levels depend on the flight condition (altitude, Mach, throttle) as
much as on wear. Steps:

1. Standardise the 3 settings (otherwise the largest-scale setting
   dominates the distances).
2. KMeans (seed 42) into K regimes.
3. For each regime r and sensor j, store the **training** mean μ and std σ.
4. Normalise: z = (x − μ of the row's regime) / (σ of the row's regime).

FD001 has one condition, so K = 1 and this is a plain z-score. FD002 and
FD004 have 6 conditions; with `n_regimes=6` the same code removes the
regime offsets and leaves only wear. If a sensor is constant in a regime,
σ is 0 up to float noise (around 1e-15); anything below 1e-8 is treated as
constant and only centred, so noise isn't amplified into fake signal.

**Why training statistics only.** If μ and σ were computed on validation or
test engines, their own failure trajectories would shape the scaling. A
validation engine's late, degraded cycles would shift its mean, so
features would partly encode information about the answer. The CV error
would come out too low. The scaler is fitted once and frozen, and a test
asserts that the regime means and stds, the KMeans centres and the health
index coefficients don't change when the scaler is applied to validation
data.

### Health index
A linear regression from the 14 normalised sensors onto a health target:

    h = 1 − (t − 1) / (T − 1)    (T = the engine's failure cycle)
    HI = clip(β₀ + βᵀz, 0, 1)

The target falls linearly from 1 at cycle 1 to 0 at failure. T is only
needed to *fit* the regression on training engines (they run to failure).
To *apply* it you only need the sensors, so it works on test engines.

**Why regression and not PCA.** The first principal component is the
direction of greatest variance, which need not be the direction of wear,
and its sign and scale are arbitrary. Regression picks the sensor mix that
tracks degradation directly and gives a fixed 0 to 1 scale.

**Limitation.** Real wear is flat for most of a life and then accelerates,
but the target is a straight line. So HI starts around 0.8 instead of 1 and
only falls clearly in the last third of life.

**How good is it?** Correlation with capped RUL is **0.868 ± 0.016 out of
fold** (fitted on each fold's training engines, measured on its validation
engines) and 0.864 in sample. They're almost the same, so it doesn't
overfit. The best single feature, a 20 cycle rolling mean, reaches |r| =
0.80 (in sample).

### The limitation of fixed windows (sets up Phase 2)
Every feature looks back at most 20 cycles, but engines live 128 to 362
cycles. A fixed window:

- can't tell whether a sensor has been drifting for 150 cycles or just
  started moving
- can't see when the drift began to accelerate
- uses one window length for engines that degrade at very different rates

The window lengths (5, 10, 20) are hand-picked and not tuned. Sequence
models (LSTM, temporal CNN) take the whole trajectory and learn which part
of the history matters. The four published models that beat us on FD001
(RMSE 12.61 to 13.18) all learn from windows of the raw sequence. That's
the Phase 2 motivation.

## How my features held up downstream

- **Cross validation (Husain, #5):** engineered features improved every
  model. XGBoost 16.42 → 13.99 RMSE, Random Forest 16.43 → 14.42, Elastic
  Net 20.12 → 17.06.
- **Test set (Sarvesh, #6):** XGBoost 17.58 → 15.36 RMSE, and late-stage RMSE
  13.17 → **7.03**. A paired bootstrap on engineered minus raw gives a 95%
  interval of −4.4 to −0.2 cycles, so the gain is real, not noise. The
  report concludes that most of our improvement over published RF and
  gradient boosting comes from the features rather than the learner.
- **Importance (Husain, #7):** the health index is ranked **#1 by every
  measure** for both ensembles. Shuffling it on held-out engines raises
  XGBoost's RMSE by 10.95 cycles (Random Forest by 7.36). `cycle` is second
  (4.84 and 5.22). By family, rolling means and the HI take most of the
  credit; 10-cycle windows get the most importance in the forest.

## 10 likely questions

**1. Why does the rolling slope capture degradation?**
It's the local rate of change of a sensor, in standard deviations per
cycle. Wear accelerates near failure, so the slope grows in magnitude as
the engine approaches the end of its life. The level of a sensor says how
worn the engine is; the slope says how fast it's wearing. It also cancels
engine-to-engine baseline offsets, since it only uses differences within
one engine. In the ablation, adding slopes cut RMSE by 1.58 and late-stage
RMSE by 2.49, the biggest gain of any family.

**2. Write the slope formula and explain why you compute it that way.**
b = (n Στz − Στ Σz) / (n Στ² − (Στ)²). Every sum is a rolling sum, so the
whole dataset takes one vectorised pass instead of a regression per row. A
test checks it against `np.polyfit`. For n = 1 the denominator is 0, so I
define b = 0.

**3. How do you make sure a window never crosses two engines?**
All rolling operations use `groupby("unit")` on data sorted by (unit,
cycle). The tests check that each engine's features computed together equal
its features computed alone, and that adding +10,000 to one engine's
sensors doesn't change its neighbour's rolling features. Rows can arrive
shuffled with any index; I sort internally, then restore the original order
and index.

**4. Why must normalisation statistics come from training data only?**
Otherwise the validation or test engines' own failure trajectories shape
the scaling, so information about the answer leaks into the features and
CV error looks better than it is. The scaler is fitted on training engines
(per fold in CV) and applied unchanged to the rest. A test checks that none
of its fitted parts change when applied to validation data.

**5. What happens at cycle 1? Why not drop rows with incomplete windows?**
`min_periods=1`: the window is the cycles seen so far. At cycle 1 the mean
is the reading, and std and slope are 0. There are no NaNs anywhere (a test
checks this). Dropping incomplete windows would lose about 9% of training
rows, and a short test engine would get no prediction. The downside is
noisy slopes for the first few cycles, but those cycles are far from
failure, at the capped target.

**6. FD001 has one operating condition. Why build regime clustering?**
So the code works unchanged for FD002 and FD004 (6 conditions), where raw
sensor levels jump between conditions much more than they move with wear.
In FD001 it's just a z-score. I standardise the settings before KMeans so
no single setting dominates the distance, and fit the clusters and
per-regime stats on training data only.

**7. How did you build the health index, and why not PCA?**
A linear regression from the normalised sensors onto a target that goes
linearly from 1 at the first cycle to 0 at failure, clipped to 0 to 1. PCA
finds the direction of most variance, which needn't be wear, and its sign
is arbitrary. Regression targets degradation directly. Out-of-fold
correlation with RUL is 0.868 ± 0.016.

**8. Isn't the health index leaking, since its target uses the failure cycle?**
No. The failure cycle T is used only to *fit* the regression, on training
engines, which really did run to failure. At prediction time the HI is just
β₀ + βᵀz from the sensors, with no T needed. In CV it's refitted per fold,
so validation engines never contribute to it. And the out-of-fold
correlation (0.868) matches the in-sample one (0.864).

**9. The HI adds only 0.04 RMSE in your ablation but is the #1 feature in
SHAP. How?**
They measure different things. The ablation adds the HI last, after
rolling means and slopes already carry most of the same information. Its
*marginal* gain over them is small (within fold noise). Importance shares
credit among correlated features: the HI is one column that summarises all
14 sensors, so the trees split on it first and it collects the credit that
was spread over many sensor columns before. Permutation importance (+10.95
RMSE when shuffled) says the model relies on it *as trained*, not that the
information is unavailable elsewhere. Slopes show the reverse: a big
ablation gain but small importance, because once the HI exists, it and the
means absorb their credit.

**10. What's the main weakness of your features, and what would you do next?**
Fixed windows of at most 20 cycles on lives of 128 to 362 cycles. They
can't see long-range history, like how long a drift has lasted or when it
accelerated, and the window lengths are hand-picked. The HI's linear target
also doesn't match wear that's flat and then accelerates. Next: sequence
models (LSTM or temporal CNN) on the full trajectory, which is what the four
published models that beat us do. A cheaper step would be an HI target that
matches the RUL cap (healthy until 125 cycles from failure, then linear).

## Other questions to be ready for

- *Why windows 5, 10, 20?* Short, medium and long, hand-picked to see
  different time scales, and not tuned. In the forest, 10-cycle windows got
  the most importance. Tuning them on CV is possible but adds another
  search.
- *Why the std, if it matters so little?* Sensors get more erratic as parts
  wear, so it was worth testing. Its importance is small (about 4% in RF,
  1% in XGBoost).
- *Why not use the normalised columns as model inputs?* In FD001 they're a
  monotone rescaling of the raw sensors, and trees don't care about
  scaling. The engineered set keeps the raw sensors and adds rolling
  features of the normalised ones.
- *What is `n_regimes=None`?* When fitting it means 1. When a scaler is
  passed it means "use the scaler's count". A different explicit value
  raises an error so it can't be silently ignored.
- *How do you apply features to a test engine?* Fit the scaler on all 100
  training engines, compute features over each test engine's full observed
  history, and predict at its last cycle (`results.tex`, Sarvesh's #6).

## Things I must check before the viva

- Run "raw + rolling + health index, **no slopes**" once, to show how much of
  the slope gain survives when the HI is present (question 9). It isn't in
  the committed ablation.
- Agree with Husain on one sentence for the HI-importance vs ablation
  question, so we don't contradict each other in the mock viva.
- Re-read `features.tex` against this file after any change to the
  ablation numbers.
