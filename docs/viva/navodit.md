# Viva notes: Navodit

My part: the literature, how our results compare with published FD001
work, and the report. Report: `report/main.tex`,
`report/sections/related.tex`, the introduction, discussion, conclusion
and contributions sections, and `report/refs.bib`. Numbers I quote come
from `report/sections/results.tex` (Table `tab:published`),
`results/test_metrics.csv` and `results/test_published_comparison.csv`.

## My part in plain words

NASA's C-MAPSS data simulates turbofan engines wearing out. We use the
FD001 subset and predict how many cycles an engine has left (RUL). My job
is to place our work among the published papers on the same data:

- **What others did:** classical ML (Random Forest, gradient boosting)
  and deep sequence models (CNN, LSTM, attention) on FD001.
- **Where we sit:** our Random Forest and XGBoost on engineered features
  rank 5th and 6th of 14 by RMSE, behind four deep models and ahead of
  every published classical model.
- **The report:** an IEEE conference paper, one LaTeX file per section,
  every number taken from `results/` and every figure from `figures/`.

**Our test results against labels clipped at 125 (as most papers do):**

| Model (engineered features) | RMSE | Score S |
|---|---|---|
| Random Forest | 14.02 | 310 |
| XGBoost (selected) | 14.08 | 347 |
| Elastic Net | 17.01 | 469 |

Against the true, unclipped RUL, the selected XGBoost scores RMSE 15.36,
MAE 11.26, S 389 (Random Forest 15.23, S 344).

## The theory

### The dataset (Saxena et al. 2008)
C-MAPSS is a simulator of a turbofan engine. Saxena, Goebel, Simon and
Eklund (2008) used it to make run-to-failure data and released it as a
benchmark. Each row is one engine at one cycle: engine id, cycle number,
3 operating settings, 21 sensor readings.

FD001 is the simplest subset:
- **one operating condition**, **one fault mode** (the
  high-pressure compressor degrades)
- **100 training engines**, run until they fail: 20,631 rows, lifetimes
  128 to 362 cycles (median 199)
- **100 test engines**, cut off at an unknown point before failure. NASA
  gives the true RUL only at each test engine's last cycle (in
  `RUL_FD001.txt`). True test RUL runs from 7 to 145 (median 86).

So we score one prediction per test engine, 100 predictions in total.
Seven of the 21 sensors are dropped by the EDA (six constant, `s_6` almost
constant), leaving 14.

### The capped (piecewise) RUL target
The raw label at cycle t is "failure cycle minus t". A new engine might
have 300 cycles left, but its sensors look exactly like one with 200 left,
because degradation hasn't started to show. Asking a model to tell them
apart only teaches it noise.

So the target is capped: RUL = min(125, failure cycle − t). The label is
flat at 125 while the engine looks healthy, then falls in a straight line
to 0. That shape (flat, then linear) is why it is called piecewise linear.
In practice it means: "say healthy until you see wear, then count down."
The cap of 125 is common in the FD001 papers; Liu et al. (2021) use it. It
changes 8,031 training labels (38.93%).

Many papers also clip the **test** labels at 125 before scoring. That is
why we report two sets of numbers: true RUL (our honest headline) and
clipped at 125 (for comparison with papers).

### RMSE vs the NASA score S
With d = predicted − true RUL for each test engine:
- **RMSE** = √(mean of d²). Big misses count more, but early and late
  misses count the same.
- **Score S** (from the PHM08 challenge, Saxena et al. 2008), summed over
  engines:
  - too early (d < 0): e^(−d/13) − 1
  - too late (d ≥ 0): e^(d/10) − 1

Late is punished harder (divisor 10, not 13), and the penalty is
exponential, so a few badly late engines can dominate S. Late is worse in
real life because the engine is believed healthy when it is about to fail.
Early only brings maintenance forward. Lower S is better, and because it is
a sum, it is only compared on the same 100 engines.

### Classical ML vs deep sequence models
**Classical ML (what we did):** a human decides what history matters. We
summarise each sensor's last 5, 10 and 20 cycles (rolling mean, std,
slope) and add a health index, 142 columns per row. Then a Random Forest
or XGBoost maps that one row to RUL. The model never sees the sequence,
only our summary of it.

**Deep sequence models:** they take a window of raw sensor readings over
many cycles and learn their own features.
- **CNN (1D convolution):** slides small filters along the time axis, like
  a learned moving average or slope detector. Li et al. (2018) is the best
  FD001 result in our table.
- **LSTM** (Hochreiter and Schmidhuber 1997): reads the sequence one
  cycle at a time and keeps a memory cell; gates decide what to remember
  and what to forget, so it can carry information over long stretches.
  Zheng et al. (2017) applied it to C-MAPSS.
- **Attention:** learns a weight for each time step, so the model can
  focus on the cycles that matter most (for example, where a sensor starts
  to drift). Ragab et al. (2020) combine an encoder-decoder with attention
  (ATS2S).

| | Classical ML (ours) | Deep sequence models |
|---|---|---|
| Features | hand made, fixed windows | learned from raw windows |
| Explaining it | easy (importance, SHAP) | harder |
| Data and tuning needed | less | more |
| Sees long history | only up to 20 cycles back | the whole window |

**Why Phase 2 tests sequence models** (`ROADMAP.md`): our features look at
most 20 cycles back, while engines live 128 to 362 cycles. A fixed window
can't tell whether a drift started 15 or 150 cycles ago, and the windows
5, 10, 20 are themselves a hand choice. The four models ahead of us all
learn from sequence windows. Phase 2 plans an LSTM, a temporal CNN or a
Transformer encoder on fixed length sensor windows, an ablation study, and
a hybrid that feeds our engineered features into the sequence model.

### The research gap
Ten of the twelve published rows in our table are deep models. We ask
how far well-made features plus a classical model get, scored the same way
per engine, with error split into early and late stage, and explained.
Phase 1 answers that for classical ML; Phase 2 compares it with learned
sequences.

## Where we sit among published results

Labels clipped at 125. Published numbers as collected in Ragab et al.
(2020), Table III; SVM baseline (RMSE 40.72) left out.

| Rank | Method | RMSE | S |
|---|---|---|---|
| 1 | 1D CNN (Li et al. 2018) | 12.61 | 274 |
| 2 | ATS2S, attention (Ragab et al. 2020) | 12.63 | 243 |
| 3 | HDNN (Al-Dulaimi et al. 2019) | 13.02 | 245 |
| 4 | BLCNN (Liu et al. 2019) | 13.18 | 302 |
| 5 | **Random Forest (ours)** | **14.02** | **310** |
| 6 | **XGBoost (ours)** | **14.08** | **347** |
| 7 | CNN-LSTM (Wu et al. 2019) | 14.40 | 290 |
| 8 | BiLSTM-ED (Yu et al. 2019) | 14.74 | 273 |
| 9 | LSTMBS (Liao et al. 2018) | 14.89 | 481 |
| 10 | MODBNE (Zhang et al. 2017) | 15.04 | 334 |
| 11 | Gradient boosting (Zhang et al. 2017) | 15.67 | 474 |
| 12 | Deep LSTM (Zheng et al. 2017) | 16.14 | 338 |
| – | Elastic Net (ours) | 17.01 | 469 |
| 13 | Random Forest (Zhang et al. 2017) | 17.91 | 480 |
| 14 | 2D CNN (Babu et al. 2016) | 18.45 | 1287 |

The five facts to remember:
1. **Best published:** 1D CNN, 12.61. Best by score: ATS2S, 243.
2. **Us:** RF 14.02 and XGBoost 14.08, 5th and 6th. The four ahead are
   deep models, 0.8 to 1.4 cycles better than our RF.
3. **Classical ML:** we beat published RF (17.91) and gradient boosting
   (15.67), and six deep models (CNN-LSTM down to the 2D CNN).
4. **Score S:** worse picture. Six published models score below our RF's
   310 (all four above us, plus CNN-LSTM 290 and BiLSTM-ED 273). For
   XGBoost's 347 it is eight (add MODBNE 334 and Deep LSTM 338).
5. **Features do the work:** our RF on raw sensors scores 16.58 against
   the same labels (XGBoost raw 16.37). Engineered features take that to
   14.02.

## 10 likely questions

**1. What is C-MAPSS FD001, and why use only FD001?**
Simulated turbofan run-to-failure data from NASA (Saxena et al. 2008).
FD001 has one operating condition and one fault mode, 100 training engines
run to failure and 100 test engines cut off before failure, 21 sensors and
3 settings. It is the simplest subset and the one almost every paper
reports, so it gives the cleanest comparison. The other subsets add more
operating conditions and fault modes; they are a stretch goal.

**2. Why cap RUL at 125? Isn't that changing the answer?**
Early in life the sensors don't move, so 300 and 200 cycles left look the
same. The cap says "healthy, 125" until wear shows, then counts down. It
only changes training labels. On the test set our headline is scored
against the true RUL (XGBoost 15.36). Eleven test engines have true RUL
above 125 (up to 145); the model can't predict much above 125, so it
underestimates them by about 25 cycles. That is the safe direction (early
maintenance).

**3. Why report the NASA score as well as RMSE?**
RMSE treats early and late errors the same. S punishes late predictions
more (e^(d/10) vs e^(−d/13)) and exponentially, because a late prediction
can mean an engine fails in service. Two models with almost the same RMSE
can differ a lot on S: our RF and XGBoost are 14.02 vs 14.08 in RMSE but
310 vs 347 in S. Against the true RUL it is the same story (S 344 vs
389): XGBoost has 9 predictions more than 20 cycles late, RF only 6.

**4. What is the best published FD001 result, and where are you?**
Best RMSE is the 1D CNN of Li et al. (2018), 12.61, then Ragab et al.'s
attention model ATS2S, 12.63, then HDNN 13.02 and BLCNN 13.18. Our RF
(14.02) and XGBoost (14.08) come 5th and 6th of 14, ahead of every
published classical model and six deep ones.

**5. Your test RMSE is 15.36, but you compare 14.08 with papers. Which is
real?**
Both, scored against different labels. 15.36 is against the true RUL,
which is our honest number. Many papers also clip the test labels at 125,
so to compare like for like we rescore the same predictions against
clipped labels: 14.08. The gap is mostly the 11 engines whose true RUL is
above 125. Comparing our 15.36 with a paper's clipped number would be
unfair to us; comparing our 14.08 with an unclipped number would be unfair
to them.

**6. Your Random Forest gets 14.02, the published one 17.91. Why so much
better?**
Mostly the features. Our own RF on raw sensors scores 16.58 on the same
labels, so engineered features (rolling mean, std, slope, health index)
are worth about 2.6 cycles. The rest (16.58 vs 17.91) we can't pin down:
tuning, sensor choice and protocol details differ and are not all
reported. The fair claim is that a classical model with good features
lands close to deep models, not that our forest is a better algorithm.

**7. You beat CNN-LSTM and BiLSTM-ED on RMSE but they beat you on score.
How?**
RMSE is about typical error size; S is dominated by a few badly late
predictions. CNN-LSTM (14.40, 290) and BiLSTM-ED (14.74, 273) make
slightly bigger errors on average but fewer large late ones. For
maintenance, S is arguably the more relevant number, and on S six
published models beat our RF.

**8. Why not just use an LSTM or CNN from the start?**
Phase 1 asks how far classical ML with engineered features gets, scored
per engine and explained with SHAP. That is a baseline any deep model
should be measured against, and it turns out to be strong: 5th of 14. But
the four models ahead of us all learn from sensor sequences, and our
features only look 20 cycles back. So Phase 2 tests LSTM, temporal CNN or
Transformer models on sensor windows, plus a hybrid with our features, to
see whether learned sequences beat hand-made summaries.

**9. Are you really 0.8 cycles behind BLCNN? Could it be noise?**
Treat small gaps with care. Protocols differ in details not always
reported (window length, how labels are clipped), and we copy published
numbers rather than rerun them. On our own side, a paired bootstrap over
the 100 test engines gives a 95% interval of −1.3 to +1.1 cycles for RF
minus XGBoost, so our two ensembles can't be separated. Gaps of a fraction
of a cycle mean little. The safe claim is the ranking in groups: four deep
models clearly ahead, then us, then the rest.

**10. Walk us through the report.**
IEEE conference format (`\documentclass[conference]{IEEEtran}`),
`report/main.tex` with one `\input` per section, built with
`latexmk -pdf main.tex` from `report/`. Every figure comes from
`figures/`, every number from `results/`, and someone else reads it
through to check the numbers agree. Sections and owners are in the table
below.

## Report structure

| Section | File | Owner |
|---|---|---|
| Abstract | `report/main.tex` | Navodit |
| Introduction | `report/sections/intro.tex` | Navodit |
| Related Work | `report/sections/related.tex` | Navodit |
| Dataset and EDA | `report/sections/data.tex` | Trishit |
| Feature Engineering | `report/sections/features.tex` | Aryan |
| Methods and Theory | `report/sections/models.tex` | Husain |
| Evaluation Protocol | `report/sections/evaluation.tex` | Sarvesh |
| Results | `report/sections/results.tex` | Sarvesh |
| Explainability | subsection of `models.tex` (`sec:explain`) | Husain |
| Discussion and Limitations | `report/sections/discussion.tex` | Navodit |
| Conclusion and Phase 2 plan | `report/sections/conclusion.tex` | Navodit |
| Contributions (who did what) | `report/sections/contributions.tex` | Navodit |
| References | `report/refs.bib` | everyone adds theirs |

Building notes (`report/README.md`): compile with `report/` as the working
directory, `\input{sections/data}` style includes, `\bibliography{refs}`,
`graphicx` for figures, figure paths like `../figures/eda_sensor_trends.png`.
Wide multi-panel figures use `figure*` (both columns). The final
`report/main.pdf` is committed on `main`.

## Teammates' areas, enough for cross-questions

**Data and EDA (Trishit).** 20,631 training rows, no missing values.
Settings 1 and 2 are a small cloud near zero, setting 3 is always 100.
Dropped `s_1`, `s_5`, `s_10`, `s_16`, `s_18`, `s_19` (constant) and `s_6`
(two values only). 14 sensors left. Cap at 125 changes 38.93% of labels.

**Features (Aryan).** Per engine, per sensor: rolling mean, std and least
squares slope over 5, 10 and 20 cycles, plus cycle and a health index (one
linear mix of all sensors). 142 columns. Ablation with a default Random
Forest (CV RMSE): raw 16.81, + mean and std 16.36, + slope 14.78, + health
index 14.74. The slope gives the biggest gain.

**Models (Husain).** Elastic Net, Random Forest, XGBoost, tuned with 5 fold
GroupKFold by engine. CV RMSE on engineered features: XGBoost 13.99 (picked),
RF 14.42, Elastic Net 17.06. SHAP: health index and cycle matter most, then
`s_4`, `s_15`, `s_2`.

**Evaluation and results (Sarvesh).** Split by engine, never by row. Test
set used once, after everything was frozen. Late stage = true RUL ≤ 50
(33 test engines). Selected XGBoost late stage RMSE 7.03. RF slightly
better overall (15.23 vs 15.36) but the gap is noise, and switching after
seeing the test set would bias the result.

## Things I must check before the viva
- That `report/sections/related.tex` and `refs.bib` use the same published
  numbers and author names as `results/test_published_comparison.csv`.
- The details of the other subsets (FD002 to FD004) in Saxena et al. 2008,
  in case we are asked why FD001.
