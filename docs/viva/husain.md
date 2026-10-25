# Viva notes: Husain

My part: training and tuning the models. Code: `src/rul/models.py`,
`notebooks/03_models.ipynb`. Results: `results/models_cv.csv`,
`results/models_search.csv`, `results/best_params.json`. Report:
`report/sections/models.tex`.

## My part in plain words

We predict how many cycles an engine has left (RUL), capped at 125. I
compared two baselines and three models, each on two feature sets:

- **Baselines:** always predict the mean RUL; plain linear regression.
- **Models:** Elastic Net (linear with a penalty), Random Forest (average
  of many deep trees), XGBoost (sum of many small trees, each fixing the
  last one's mistakes).
- **Feature sets:** raw (cycle + 14 sensors = 15 columns) and engineered
  (raw + rolling mean, std and slope over 5, 10, 20 cycles + health index
  = 142 columns).

Every model is tuned with the same 5 folds (20 engines held out per fold),
picking the settings with the lowest mean RMSE. The engineered features
are rebuilt inside every fold from that fold's training engines only.

**Results (cross validation RMSE, capped RUL):**

| Model | Raw | Engineered |
|---|---|---|
| Mean baseline | 41.67 | – |
| Linear regression | 20.12 | 17.07 |
| Elastic Net | 20.12 | 17.06 |
| Random Forest | 16.43 | 14.42 |
| **XGBoost** | 16.42 | **13.99** |

XGBoost on engineered features wins, with late stage RMSE (true RUL ≤ 50)
of 9.88 against Random Forest's 10.98. It goes to the test set (#6).
Linear regression on engineered features is not one of the issue's
baselines (that is raw linear regression); it is there to show what the
Elastic Net penalty adds on the same inputs.

## The theory

### Bias and variance
Error on new engines = bias² + variance + noise.
- **Bias:** the model is too simple to follow the real pattern. It is wrong
  in the same way on any training set.
- **Variance:** the model follows the training data too closely, noise
  included. Train it on different engines and you get a different model.

| Model | Bias | Variance | How tuning controls it |
|---|---|---|---|
| Linear / Elastic Net | high (linear in its inputs: it follows curves only if the features already encode them) | low | `alpha` up: less variance, more bias |
| One deep tree | low | high | – |
| Random Forest | low (deep trees) | reduced by averaging | more trees, fewer features per split |
| XGBoost | starts high (shallow trees), each round lowers it | grows with rounds | learning rate, depth, `reg_lambda`, subsampling |

### Elastic Net, and why L1 + L2
Minimise squared error + α·ρ·|β|₁ + α·(1−ρ)/2·|β|₂².
- **L1 (|β|₁):** pushes some coefficients exactly to zero. Feature
  selection.
- **L2 (|β|₂²):** shrinks all coefficients smoothly. Stable, shares weight
  between correlated features.
- **Why both:** L1 alone (Lasso) picks one feature from a correlated group
  more or less at random, and the pick can change with the data. Our
  engineered features are very correlated (the same sensor's mean over 5,
  10 and 20 cycles). L2 keeps the solution stable, L1 keeps it sparse.
- **Scaling:** the penalty depends on coefficient size, which depends on
  feature units. So a `StandardScaler` comes first, in one `Pipeline`, and
  is fitted on each fold's training engines only.

Our result: on engineered features the best setting was the weakest
penalty in the grid, α = 0.001 (ρ = 0.8), with RMSE 17.06, a tie with
plain linear regression on the same features (17.07). The error rises
steadily as α grows: 17.75 at α = 0.1 and 19.39 at α = 1 (pure L1). On
raw features α = 0.1, ρ = 1 ties with linear regression too. So the
penalty does not buy accuracy here. A stronger penalty buys sparsity (at
α = 0.1 the L1 penalty zeroes 38 of 142 coefficients) but costs accuracy.
The improvement over raw features comes from the features, not the
penalty. The grid stops at α = 0.001 because smaller α only moves Elastic
Net closer to least squares, which we already score.

### Bagging vs boosting
| | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| Trees built | independently, in parallel | one after another |
| Each tree sees | a bootstrap sample of rows | all rows, fits the current residuals |
| Trees are | deep (low bias, high variance) | shallow (high bias, low variance) |
| Combining | average | sum, each scaled by the learning rate |
| Mainly reduces | variance | bias |
| More trees | error levels off; adding trees does not increase overfitting, it only costs time | too many rounds can overfit |

Random Forest's variance: with trees of variance σ² and correlation ρ, the
average has variance ρσ² + (1−ρ)σ²/B. More trees (B) shrink the second
term, and Breiman showed the forest's error converges as B grows.
Picking a random subset of features at each split lowers ρ.

Gradient boosting: F_m = F_{m−1} + η·f_m. For squared error, the new tree
f_m is fitted to the residuals y − F_{m−1}. XGBoost adds a penalty
γ·(number of leaves) + λ/2·(sum of squared leaf values) and uses the second
derivative of the loss to pick splits.

### Every hyperparameter I tuned
Selected values in brackets as raw / engineered.

**Elastic Net**
- `alpha` {0.001, 0.01, 0.1, 0.3, 1.0} [0.1 / 0.001]: penalty strength.
- `l1_ratio` {0.2, 0.5, 0.8, 1.0} [1.0 / 0.8]: share of L1 in the penalty.
- Fitted with up to 100,000 coordinate descent iterations on a
  precomputed Gram matrix (XᵀX, 142 × 142), so even the weakest penalty
  converges; with 10,000 iterations α ≤ 0.01 did not.

**Random Forest**
- `n_estimators` {100, 200} [200 / 100]: number of trees. More trees lower
  variance until the error levels off; after that they only cost time.
- `max_depth` {8, 12, none} [12 / none]: how deep each tree can grow.
  Deeper trees have lower bias and higher variance.
- `min_samples_leaf` {1, 5, 10} [1 / 5]: fewest rows allowed in a leaf.
  Bigger leaves give smoother predictions and lower variance.
- `max_features` {0.3, 0.5, 1.0} [0.3 / 0.5]: fraction of features tried
  at each split. Lower means less correlated trees and lower variance, at
  the cost of slightly weaker trees.

**XGBoost**
- `n_estimators` {200, 400} [400]: boosting rounds.
- `learning_rate` {0.03, 0.05, 0.1} [0.03 / 0.05]: how much each tree
  counts. Smaller needs more rounds but generalises better. Trades off
  with `n_estimators`.
- `max_depth` {3, 4, 6} [4]: tree depth, which limits how many features
  one tree can combine.
- `subsample` {0.8, 1.0} [1.0]: fraction of rows per tree.
- `colsample_bytree` {0.5, 0.8, 1.0} [0.5 / 0.8]: fraction of features per
  tree.
- `reg_lambda` {1, 5} [5]: L2 penalty on leaf values, which shrinks leaves
  backed by few rows.

Random Forest and XGBoost use seed 42. Elastic Net has no randomness here:
its coordinate descent updates coefficients in a fixed order.

### Why GroupKFold
Rows from the same engine are near copies of each other. If engine 7's
cycle 150 is in training and cycle 151 in validation, the model can
recognise engine 7 instead of learning degradation, and validation error
comes out too low. Then we would pick settings that overfit. GroupKFold
with groups = `unit` keeps each engine in one fold, so validation always
uses unseen engines, just like the test set. 5 folds = 80 training and 20
validation engines per fold, and every engine is validated once.

The same logic applies to the feature scaler: it is fitted per fold on
training engines only, so held-out engines never inform it.

### The RUL cap (125)
Early in life sensors barely move, so the model cannot tell 300 remaining
cycles from 200. The cap tells it to predict "healthy, 125" until
degradation shows. Evidence, with both versions scored against true RUL:

| Trained on | RMSE all cycles | RMSE where true RUL ≤ 125 | Late RMSE |
|---|---|---|---|
| capped RUL | 47.47 | **15.09** | **9.88** |
| uncapped RUL | **38.26** | 25.99 | 11.20 |

The cap loses on young engines and wins clearly near failure, which is
what maintenance needs. Careful: capping limits the training labels, not
the predictions. XGBoost can still predict above 125 (in cross validation
the capped model's largest prediction was 135.98, and 2,110 of 20,631
predictions were above 125). It never sees a target above 125, so for
young engines its predictions stay near 125 while their true RUL is in
the hundreds.

## 10 likely questions

**1. Why did XGBoost beat Random Forest?**
Boosting reduces bias step by step, and degradation is a curved,
accelerating pattern that shallow trees added in sequence fit well. The
gap is small overall (13.99 vs 14.42, within one fold std of 0.46 to 0.48),
but bigger on late stage cycles (9.88 vs 10.98), which matter most.

**2. Elastic Net on raw features doesn't beat linear regression. Why?**
They are tied: RMSE 0.0006 lower, MAE 0.0015 higher, while folds vary by
about 1.5. A penalty lowers variance, but with 15 features and 16,000+ rows
per fold linear regression has almost no variance to remove. Its problem
is bias (it is linear in 15 raw inputs), and a penalty cannot fix bias. At
small α Elastic Net becomes linear regression. On the 142 engineered
features it is the same story: the search picks the weakest penalty and
ties linear regression (17.06 vs 17.07). What helps is the features
(20.12 → 17.07).

**3. Why combine L1 and L2 instead of just Lasso?**
Lasso picks one feature from a group of correlated features almost at
random, and the pick changes with the data. L2 spreads weight and keeps
the fit stable. Elastic Net lets the data choose the mix; here the search
picked ρ = 1 (pure L1) on raw features and ρ = 0.8 on engineered features,
though with α so small the mix hardly matters.

**4. Why scale features for Elastic Net but not for the trees?**
The penalty adds up coefficient sizes, and a coefficient's size depends on
its feature's units, so unscaled features would be penalised unequally.
Trees split on thresholds, and rescaling a feature just moves the
threshold, so trees don't care.

**5. Why random search for RF and XGBoost but a full grid for Elastic
Net?**
Elastic Net is fast, so all 20 combinations are cheap. RF and XGBoost
grids have 54 and 216 combinations, and a full search with 5 folds on
both feature sets would take hours. A random sample of 6 and 8 covers the
range of every hyperparameter at a fraction of the cost. Usually only a
few hyperparameters matter much, and random search tries more distinct
values of each than a grid of the same size.

**6. How do you know you didn't overfit the hyperparameters to the
validation folds?**
I can't fully. The same folds pick the best configuration and report its
score, so that score is a bit optimistic: out of 34 tries, the winner is
partly the one that got lucky on these folds. This is selection bias, and
it is separate from leakage. Grouping by engine stops leakage between
engines; a small search (34 configurations per feature set) and averaging
over 5 folds keep the optimism small, but neither removes it. Nested cross
validation would (an inner loop to tune, an outer loop to score). We rely
instead on the test set: #6 scores it once with the frozen parameters from
`best_params.json`, and that is our unbiased estimate.

**7. Why did you refit the feature scaler in every fold?**
The regime normalisation and the health index learn statistics from data.
Fitting them on all 100 engines would let validation engines shape their
own features, which is leakage and gives an optimistic error.

**8. Isn't the uncapped model better? It has lower RMSE.**
Only over all cycles, and only because the capped model, which never
saw a target above 125, predicts close to 125 for young engines whose true
RUL is in the hundreds. Where the true RUL is ≤ 125 the capped model
is much better (15.09 vs 25.99), and also on late stage cycles (9.88 vs
11.20). The two models must be scored against the same labels for this to
be a fair comparison.

**9. What does the learning rate do, and why does it go with
n_estimators?**
Each tree's prediction is multiplied by the learning rate before it is
added. A small rate means each tree fixes only a bit of the error, so you
need more trees, but the model generalises better. Halving the rate needs
roughly twice the rounds for the same fit, so the two are tuned together.

**10. Bias and variance of each model?**
Linear: high bias, low variance (it can only use nonlinear patterns the
features already encode). Single deep tree: low bias, high
variance. Random Forest: keeps the low bias, cuts variance by averaging
decorrelated trees. XGBoost: starts with high-bias shallow trees and cuts
bias with each round; variance is controlled by the learning rate, depth,
`reg_lambda` and subsampling.

## Things I must check before the viva
- The test results from #6 (`results/test_metrics.csv`), and whether
  XGBoost's validation lead over Random Forest holds on the test engines.
- How 13.99 CV RMSE compares with published FD001 results (Navodit, #8).
  Note that our CV error is over every cycle, while the test error is at
  each engine's last cycle, so the two are not directly comparable.
- The Random Forest numbers in the feature ablation (Aryan, #4) use
  default settings, so they differ slightly from my tuned forest (14.74 vs
  14.42 on all features).
