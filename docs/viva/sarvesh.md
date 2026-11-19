# Viva notes: Sarvesh

My part: how we split the data without leakage, how we score predictions, and
the NASA score. Code: `src/rul/split.py`, `src/rul/evaluate.py`. Report:
`report/sections/evaluation.tex`.

## My part in plain words

FD001 has 100 training engines (20,631 rows, one per cycle). Each one runs
until it fails, after 128 to 362 cycles (median 199). We predict how many
cycles an engine has left (RUL).

**Splitting.** Each engine goes entirely into training or entirely into
validation, never a mix. `engine_split` holds out 20 of the 100 engines
(seed 42). `engine_folds` makes 5 folds of 20 engines each with
`GroupKFold`, so every engine is validated exactly once. Husain tunes all
models on these folds.

**Scoring.** The 100 test engines are cut off before failure. NASA gives
their true RUL only at the last cycle we see, so `last_cycle` takes one row
per test engine, and we score 100 predictions. True test RUL ranges from 7
to 145 cycles (median 86). 33 of the 100 are in the late stage (RUL ≤ 50).

`evaluate` reports:
- RMSE and MAE over all engines
- the NASA score
- RMSE and MAE separately for early (true RUL > 50) and late (≤ 50) engines

## The theory

### Why a row split leaks
Two rows from the same engine share its manufacturing variation, its
initial wear and almost the same health state. A random row split puts
cycle 150 of engine 7 in training and cycle 151 in validation. The model
can then score well by recognising engine 7 and remembering how far along
it was, instead of learning how degradation looks in general. Validation
error comes out too low, and we pick the wrong hyperparameters. The test
engines are new engines, so only an engine split mimics the real task.

### GroupKFold
K fold cross validation where a "group" (here, an engine) is never split
across folds. With 100 engines and 5 folds, each fold holds out 20 engines.
It does not shuffle, so it needs no seed and gives the same folds every
time. Cross validation averages over 5 validation sets instead of 1, so
the tuning decision depends less on which 20 engines landed in validation.

### RMSE and MAE
With d = predicted − true:
- RMSE = √(mean of d²). Squaring makes big misses count much more.
- MAE = mean of |d|. Every cycle of error counts the same.

RMSE ≥ MAE always. A big gap between them means a few large errors.

### The NASA score (Saxena et al. 2008, PHM08 challenge)
For each engine:
- predicted too early (d < 0): s = e^(−d/13) − 1
- predicted too late (d ≥ 0): s = e^(d/10) − 1

The total score is the sum over engines, and lower is better. Late is
punished harder because the divisor is smaller (10 vs 13):

| Error | Penalty |
|---|---|
| 20 cycles late | e² − 1 ≈ 6.39 |
| 20 cycles early | e^(20/13) − 1 ≈ 3.66 |
| 30 cycles late | e³ − 1 ≈ 19.09 |

The penalty is exponential, so one very late prediction can dominate the
total. It is a sum, so we only compare scores computed on the same 100
engines.

### Why late stage error matters more
Maintenance is planned when an engine is close to failure. An error of 20
cycles when 120 remain changes nothing. An error of 20 when 15 remain
means the engine fails before we act. That is why we report late stage
RMSE and MAE on their own.

## 10 likely questions

**1. Why not split by row?**
Rows of one engine are strongly related. A model can recognise the engine
rather than learn degradation, so validation looks better than it is. Test
engines are unseen engines, so validation must use unseen engines too.

**2. What is GroupKFold and why 5 folds?**
K fold cross validation that keeps each group (engine) inside one fold.
Five folds of 20 engines balance two things: enough training engines per
fold (80) and enough validation folds to average out noise.

**3. Why seed 42 for `engine_split` but no seed for GroupKFold?**
`engine_split` shuffles engine ids at random, so a seed makes it
repeatable. GroupKFold assigns engines to folds deterministically without
shuffling, so there is nothing random to seed.

**4. Why is the test set used only once?**
If we looked at test scores while tuning, we would be tuning on the test
set. It would then overstate how well we do on new engines. Every tuning
choice uses cross validation on training engines only.

**5. Why score test engines only at their last cycle?**
That is the only cycle with a known true RUL (from `RUL_FD001.txt`). It
also gives each engine one vote, so long engines do not count more than
short ones.

**6. Write the NASA score and explain 13 and 10.**
s = e^(−d/13) − 1 if d < 0, else e^(d/10) − 1, with d = predicted − true,
summed over engines. The smaller divisor for late predictions makes them
grow faster: 10 cycles late costs as much as 13 cycles early.

**7. Why punish late predictions more than early ones?**
A late prediction says the engine has more life than it really has. That
risks a failure in service. An early prediction only brings maintenance
forward, which costs money but is safe.

**8. Difference between RMSE and MAE, and which do you trust more?**
RMSE squares errors, so it is sensitive to large misses. MAE treats all
errors equally. We report both. A big gap means a few engines are badly
predicted, and the NASA score shows whether those misses are late.

**9. Why 50 cycles as the late cut, and what if you picked 30?**
50 is a team decision recorded in ROADMAP.md, without a stated reason, so
agree on the reason as a team before the viva. One defensible argument:
50 cycles is roughly a quarter of a median engine life (199 cycles), a
reasonable window for planning maintenance. It also puts 33 of the 100
test engines in the late stage, enough for a stable average. At 30
there would be fewer late engines and a noisier late stage RMSE. The
overall metrics would not change.

**10. Can a model have good RMSE but be bad for maintenance?**
Yes. It can be accurate on healthy engines (most of the data) and miss
badly near failure. That is why we report late stage error and the NASA
score next to RMSE.

## Test results (#6)

Notebook `notebooks/04_results.ipynb`, numbers in `results/test_metrics.csv`,
report `report/sections/results.tex`.

**What I did.** I refitted every frozen configuration from
`results/best_params.json` on all 100 training engines (target capped at
125), fitted the feature statistics on training engines only, built
features over each test engine's whole observed history, and predicted
once per engine at its last cycle with `predict_last_cycle`. Scored once,
against the true RUL. Nothing was tuned after.

**Headline numbers (true RUL, 100 engines).**

| Model (engineered) | RMSE | MAE | NASA score | Late RMSE |
|---|---|---|---|---|
| XGBoost (selected) | 15.36 | 11.26 | 389 | 7.03 |
| Random Forest | 15.23 | 11.33 | 344 | 7.34 |
| Elastic Net | 18.27 | 14.11 | 522 | 14.83 |
| Mean baseline | 43.07 | 35.90 | 33629 | 64.09 |

XGBoost wins on late stage RMSE (7.03), Random Forest on overall RMSE and
the NASA score.

**Likely questions.**

*Random Forest beat your selected model. Why not switch?* Because picking
the winner after looking at the test set turns the test set into a second
validation set, and the reported error stops being an honest estimate for
new engines. Also the gap is noise: a paired bootstrap (resample the 100
engines 10,000 times, score both models on the same resample) gives a 95%
interval of −1.3 to +1.1 cycles for the RMSE difference. Zero is inside, so
they are tied.

*What is a paired bootstrap?* Draw 100 engines with replacement, compute
both models' RMSE on that same draw, take the difference, repeat. "Paired"
means both models see the same engines each time, so engine difficulty
cancels out and only the model difference is left.

*Why is test RMSE (15.36) higher than CV RMSE (13.99)?* They score
different things. CV scores every cycle of held-out engines against the
capped label. The test scores one cycle per engine against the true,
uncapped label. 11 test engines have true RUL above 125, and a model
trained on a cap barely predicts above it, so those cost about 25 cycles
each. Against labels clipped at 125 the test RMSE is 14.08, close to CV.

*Why do the two ensembles differ on the NASA score but not RMSE?* The score
is exponential in the error, so one big miss can outweigh many small ones.
Here it's one early miss: engine 45 (true RUL 114) is predicted at 54 by
XGBoost and 64 by Random Forest, which costs 96.8 vs 47.4 points. That is
more than the whole 44-point gap. The late predictions cost both models
about the same (180.2 vs 179.5), even though XGBoost has 9 predictions more
than 20 cycles late and Random Forest 6.

*How do you compare with published work?* Against labels clipped at 125,
like the papers, using every FD001 row with RMSE and score in Ragab et al.
2020, Table III (except an SVM baseline). By RMSE our RF (14.02) and
XGBoost (14.08) come fifth and sixth out of 14. We beat published RF
(17.91) and gradient boosting (15.67) from Zhang et al. 2017, and six of the
ten deep models, from CNN-LSTM (14.40) to the 2D CNN (18.45). Four deep
models beat us by 0.8 to 1.5 cycles: Li et al.'s 1D CNN (12.61), ATS2S
(12.63), HDNN (13.02) and BLCNN (13.18). On the NASA score we look worse:
six published models score below our RF's 310, including BiLSTM-ED (273)
and CNN-LSTM (290), which have worse RMSE than us but fewer badly late
predictions.

*Where are the errors worst?* Far from failure and above the cap. Close to
failure (true RUL 25 or less) the ensembles are within about 7 cycles. From
51 to 75 cycles all models predict too late, which is where degradation
just starts to show.

## Things I must check before the viva
- That Navodit's literature review (#8) cites the same published numbers
  as `results/test_published_comparison.csv`.


# Phase 2: windows, training loop and DL test results (#33, #38)

My part: turning each engine into sensor windows, the training loop every
network shares, the nested validation that keeps early stopping honest, and
the one-time test set evaluation of every network. Code:
`src/rul/sequences.py`, `src/rul/train.py`, `src/rul/holdout.py`. Notebook:
`notebooks/09_dl_results.ipynb`. Report: `report/sections/dl_setup.tex`,
`report/sections/dl_results.tex`.

## My Phase 2 part in plain words

**Windows.** A network reads the last 30 cycles of the 14 useful sensors
and predicts the RUL at the last of those cycles. Every engine gets one
window ending at every cycle, so the networks are scored on exactly the
same rows and folds as Phase 1. A window never mixes two engines.

**Padding.** The first 29 cycles of an engine don't have 30 cycles of
history. I left pad them with copies of the engine's first reading instead
of dropping them, because dropping them would remove exactly the healthy
early cycles, and a healthy engine's sensors are flat, so repeating cycle 1
adds no fake trend.

**Why 30.** The shortest FD001 test engine has 31 cycles, so every test
engine is scored on real history, never padding. The same window as Li et
al. 2018 and ATS2S.

**Training loop.** Adam on mean squared error, batches of 256, gradient
norm clipped at 1, seeded. The net's output is multiplied by 125 (the cap),
so a new net, whose outputs start near 0, only has to learn values near 1.

**Nested validation.** Early stopping picks the number of epochs by
watching validation loss, so it is tuning. Inside each of the 5 CV folds I
hold out 20% of the fold's training engines (16 of 80) for early stopping.
The fold's 20 scored engines are never used to stop training.

**Test evaluation (#38).** Each network is retrained the same way as inside
a fold (80 engines to fit, 20 to early stop), with 5 seeds (42 to 46), and
predicts once per test engine from the window ending at its last cycle. The
hybrids also get the 142 engineered features at that cycle. I report the
mean and spread over the 5 seeds. XGBoost and Random Forest are not
retrained: their Phase 1 test predictions are read from
`results/test_predictions.csv`.

## The theory

### Why nest the early stopping split
Anything chosen by looking at an engine's error is tuned on that engine. If
early stopping watched the scored engines, it would stop at the epoch that
happens to suit them, and the CV error would come out too low. Holding out
separate engines for early stopping keeps the scored engines untouched,
the same reason the test set is used only once.

### Why 5 seeds
A network's result depends on its random starting weights and the order of
its batches. With one seed, a lucky or unlucky draw could flip a ranking.
Five seeds show how much of a gap is just initialisation. The seed here also
picks which 20 engines are held out for early stopping, so the spread
includes that choice too.

### Why not retrain on all 100 engines with a fixed epoch count
That would need a new hyperparameter (how many epochs) chosen without any
validation. Keeping the exact fold procedure means the test score is for the
same model that cross-validation measured.

### Paired bootstrap, averaged over seeds
Draw 100 engines with replacement, score both models on that same draw,
take the RMSE difference, repeat 10,000 times, report the 2.5th and 97.5th
percentiles. "Paired" means both models see the same engines each time, so
engine difficulty cancels. For a network, every draw is scored with all 5
seeds and their RMSEs averaged, so the interval is for the same number as
the results table (the mean over seeds), not for an averaged prediction,
which would score better than any single network and is not what we report.
`seed_bootstrap` in `src/rul/holdout.py` does this, and its tests check that
it averages RMSE over seeds rather than predictions.

### Why the bootstrap comparisons were fixed in advance
If I chose which pairs to test after seeing the test numbers, I would test
the ones that look interesting, and some would look significant by chance.
The pairs were fixed before the test set was opened: the selected network
(Transformer) against the selected Phase 1 model (XGBoost), and the
Transformer hybrid against both of its parents.

## 10 likely questions

1. *Why pad with the first reading and not zeros?* Zeros after
   standardising mean "average engine", which a healthy engine is not, and a
   jump from 0 to the real values would look like a sudden change. The first
   reading is flat and belongs to the same engine.
2. *Does padding leak anything?* No. It only copies the engine's own first
   cycle, which is in the past of every window.
3. *Why is the scaler fitted inside each fold?* Its mean and standard
   deviation are statistics of the data. Fitting it on scored engines would
   leak their distribution into training.
4. *Why one window per cycle and not one per engine?* One per engine would
   give 100 training rows. One per cycle gives about 20,000 and matches the
   rows Phase 1 was scored on.
5. *Why multiply the output by 125?* Initial outputs are near 0. Without
   scaling, the last layer must grow its weights a lot to reach 125, which
   was slow and made batch 256 fail to train at all (fixed in `83dd63b`).
6. *What does gradient clipping do?* It rescales the gradient when its norm
   is above 1, so one bad batch can't throw the weights far.
7. *Why do the networks train on 64 engines per fold and XGBoost on 80?*
   16 are held out for early stopping. The networks learn from a fifth fewer
   engines, so the comparison slightly favours Phase 1.
8. *Why is test RMSE higher than CV RMSE?* Same as Phase 1: CV scores every
   cycle against the capped label, the test scores one cycle per engine
   against the uncapped label, and 11 test engines have true RUL above 125.
9. *Why report the mean over seeds and not the best seed?* Picking the best
   seed after seeing the test score is selecting on the test set.
10. *Why not average the 5 seeds' predictions into an ensemble?* It would
    be a different model, one cross-validation never measured. It is fine
    for the plot, labelled as such, but not as a reported score.

## DL test results (#38)

Notebook `notebooks/09_dl_results.ipynb`, numbers in
`results/dl_test_metrics.csv` and `results/dl_test_bootstrap.csv`, report
`report/sections/dl_results.tex`.

**Headline numbers (true RUL, 100 engines, mean ± std over 5 seeds).**

| Model | RMSE | NASA score | Late RMSE |
|---|---|---|---|
| XGBoost (Phase 1, selected) | 15.36 | 389 | 7.03 |
| Transformer (selected) | 15.37 ± 0.66 | 422 | 5.88 |
| Residual hybrid | 15.13 ± 0.10 | 348 | 5.03 |
| Hybrid, Transformer + features | 15.92 ± 0.66 | 450 | 5.85 |
| LSTM | 16.02 ± 0.67 | 622 | 6.21 |
| Temporal CNN | 16.84 ± 0.93 | 584 | 8.10 |

**The answer to the brief in one line.** Learning from raw sensor windows
ties hand-engineered features on FD001 (15.37 vs 15.36, bootstrap interval
−2.3 to +2.2), and does better close to failure.

**Likely questions.**

*The Transformer was 0.71 ahead in CV. Why is it tied on the test set?* CV
gaps of under a cycle are small next to test noise: with 100 engines the
bootstrap interval is about ±2.2 cycles wide. The test set also scores one
cycle per engine against uncapped labels, a different job from CV.

*The residual hybrid has the best test RMSE. Why isn't it your final
model?* Because it wasn't selected by CV before the test set was opened.
Choosing it now would make the test set a second validation set, same rule
as RF vs XGBoost in Phase 1. Its lead over XGBoost (0.23) is far inside the
noise anyway.

*Why is the residual hybrid so stable across seeds (±0.10)?* Most of its
prediction comes from XGBoost, which is deterministic. The network only adds
a small correction.

*Where do the networks win and lose?* They win near failure: up to 25
cycles left, Transformer RMSE 2.5 vs 6.6 for XGBoost. They lose from 51 to
100 cycles, where degradation is just starting (19 to 22 vs 16 to 17).
Caveat: the late stage is 33 engines and wasn't in the pre-registered
bootstrap, so I call it a consistent pattern across seeds, not a tested
result.

*Why is the Transformer's NASA score worse at the same RMSE?* It leans late:
mean error +1.9 cycles, 11 predictions more than 20 cycles late vs 9 for
XGBoost, and the score punishes late errors exponentially.

*Did adding engineered features to the Transformer help?* Not on the test
set: 15.92 vs 15.37, interval −0.2 to +1.3. It helped in CV only slightly,
and that didn't carry over.

*How do you compare with published work?* Against labels clipped at 125,
our best four (residual hybrid 13.88, RF 14.02, XGBoost 14.08, Transformer
14.31) rank 7th to 10th of 21, behind six published deep models from BLCNN
(13.18) to GCU-Transformer (11.27).

*Why were the saved predictions trained in parallel?* To finish in 7
minutes instead of 30: one process per seed with 2 threads. A different
thread count can change the last digits, which the notebook states.
