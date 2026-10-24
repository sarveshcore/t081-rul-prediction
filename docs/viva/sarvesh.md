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

## Things I must check before the viva
- Our final numbers from `results/test_metrics.csv` (#6), and which model
  wins on late stage RMSE.
- How our RMSE compares with the published FD001 results (Navodit, #8).
