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
underestimates them by about 23 cycles. That is the safe direction (early
maintenance).

**3. Why report the NASA score as well as RMSE?**
RMSE treats early and late errors the same. S punishes late predictions
more (e^(d/10) vs e^(−d/13)) and exponentially, because a late prediction
can mean an engine fails in service. Two models with almost the same RMSE
can differ a lot on S: our RF and XGBoost are 14.02 vs 14.08 in RMSE but
310 vs 347 in S. Against the true RUL it is the same story (S 344 vs
389), and most of that gap is one engine that XGBoost predicts about 9
cycles earlier than RF.

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
clipped labels: 14.08. The gap is entirely the 11 engines whose true RUL is
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
`report/main.tex` with one `\input` per section, built from `report/` with
`pdflatex main`, `bibtex main`, then `pdflatex main` twice. Every figure comes from
`figures/`, every number from `results/`, and someone else reads it
through to check the numbers agree. Sections and owners are in the table
below.

## Report structure

| Section | File | Owner |
|---|---|---|
| Abstract | `report/main.tex` | Navodit |
| Introduction | `report/sections/intro.tex` | Navodit |
| Related Work | `report/sections/related.tex` | Navodit |
| Dataset and Exploratory Analysis | `report/sections/data.tex` | Trishit |
| Evaluation Protocol | `report/sections/evaluation.tex` | Sarvesh |
| Degradation Features | `report/sections/features.tex` | Aryan |
| Models and Tuning | `report/sections/models.tex` | Husain |
| Explainability | subsection of `models.tex` (`sec:explain`) | Husain |
| Test Set Results | `report/sections/results.tex` | Sarvesh |
| Discussion and Limitations | `report/sections/discussion.tex` | Navodit |
| Conclusion and Phase 2 Plan | `report/sections/conclusion.tex` | Navodit |
| Contributions | `report/sections/contributions.tex` | Navodit |
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

## Phase 2: deep learning literature

Report: `report/sections/dl_related.tex` (Table `tab:dl-published`) and the
Deep LSTM row of `tab:related`. Data: `results/dl_published.csv`, one row
per published model with the source of every value in `values_from`. Test:
`tests/test_literature.py`. Our own network numbers come from Husain's
`results/dl_cv.csv` and `report/sections/dl_models.tex`.

### My part in plain words

Phase 2 trains networks that read raw sensor windows instead of our
engineered features. My job was to review the three architectures we use
(LSTM, TCN, Transformer), collect the deep models published on FD001, say
what each architecture assumes about degradation data, and list where
published numbers are scored differently from ours.

- **The published deep results** go in a separate file,
  `results/dl_published.csv`, not in the Phase 1 comparison. Adding rows to
  the Phase 1 table would change the pool our "5th and 6th of 14" ranking is
  counted from.
- **Two values are weaker than the rest.** The GCU-Transformer RMSE of 11.27
  comes from later papers' comparison tables, not from Mo et al. itself
  (Sarvesh found a second, independent comparison that gives the same
  number). Zheng et al.'s training cap could not be confirmed from the
  original paper, so it is shown as `--` in both report tables.
- **No source states whether its test labels were clipped** before scoring,
  so `test_labels_capped` is "not stated" on every row.

### The theory

#### Why plain RNNs fail on long sequences, and the LSTM fix
Backpropagation through time multiplies the gradient by the recurrent
weights once per step it travels back. Over many steps it shrinks towards
zero or blows up, so the network cannot learn links more than a few cycles
apart. The LSTM (Hochreiter and Schmidhuber 1997) keeps a memory cell
updated by *addition*, `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`, not by repeated
multiplication. Gates decide what is written (`i_t`), read (`o_t`) and,
with the forget gate added by Gers et al. (2000), kept (`f_t`). While the
forget gate stays open, the error flows back through the cell unchanged.

#### TCN: causal and dilated
Bai et al. (2018). *Causal*: the output at cycle t uses only cycles up to
t (left padding only). *Dilated*: with dilation d and kernel k, a filter
reads t, t−d, …, t−(k−1)d. Doubling d each block makes the receptive field
grow exponentially with depth while weights grow linearly. Ours (k = 3, two
convolutions per block): `R = 1 + 2(k−1)(2^L − 1)`, so 29 cycles for 3
blocks and 61 for 4. Three blocks miss one cycle of our 30-cycle window;
four see all of it. Across many sequence tasks a generic TCN beat
canonical LSTMs and GRUs.

#### Transformer: self-attention
Vaswani et al. (2017). Each position makes a query, key and value; its
output is an average of all values weighted by how well its query matches
each key. Any two cycles are one step apart, and all positions run in
parallel. Attention ignores order, so a positional encoding is added (we
use the sinusoidal one); without it a window and the same window shuffled
give the same prediction. Cost grows with the square of the window length.

#### What each architecture assumes about a degradation curve
An FD001 trace is flat and noisy while healthy, then drifts as the
compressor degrades.
- **LSTM:** the useful history fits in a fixed-size state updated cycle by
  cycle. Suits accumulating damage, but recent cycles tend to dominate.
- **CNN/TCN:** useful patterns are local and look the same wherever they
  occur (a rise, a step). That is what our Phase 1 rolling means and slopes
  do by hand. Nothing outside the receptive field can matter.
- **Transformer:** assumes nothing about locality or order; the structure
  must be learned, and FD001 has only 100 training engines.

### Published deep results on FD001

From `results/dl_published.csv`. Values for the first group are as
collected in Ragab et al. 2020, Table III; Chen et al. and DAST are from
Zhang et al. 2022. `--` = not stated in the sources we checked.

| Model | Window | Cap | RMSE | S |
|---|---|---|---|---|
| 2D CNN, Babu et al. 2016 | 15 | -- | 18.45 | 1287 |
| Deep LSTM, Zheng et al. 2017 | -- | -- | 16.14 | 338 |
| MODBNE, Zhang et al. 2017 | -- | -- | 15.04 | 334 |
| CNN-LSTM, Wu et al. 2019 | -- | -- | 14.40 | 290 |
| HDNN, Al-Dulaimi et al. 2019 | -- | -- | 13.02 | 245 |
| ATS2S, Ragab et al. 2020 | 30 | 125 | 12.63 | 243 |
| 1D CNN, Li et al. 2018 | 30 | 125 | 12.61 | 274 |
| LSTM-attention + features, Chen et al. 2021 | 30 | -- | 14.53 | 322.44 |
| GCU-Transformer, Mo et al. 2021 | -- | -- | 11.27 | -- |
| DAST, Zhang et al. 2022 | 40 | 125 | 11.43 | 203.15 |

The CSV also has LSTMBS (14.89), BiLSTM-ED (14.74) and BLCNN (13.18).

### How our numbers relate to these, so far

Our networks so far have only cross-validation results (`results/dl_cv.csv`,
five engine folds, RUL capped at 125):

| Model | CV RMSE | Late RMSE (RUL ≤ 50) |
|---|---|---|
| Transformer | 13.28 ± 0.34 | 7.98 |
| LSTM | 14.02 ± 0.43 | 8.84 |
| TCN | 15.91 ± 1.01 | 11.54 |
| XGBoost, engineered (Phase 1) | 13.99 ± 0.46 | 9.88 |

**These cannot be put next to the published table.** Published numbers are
on the 100 test engines; ours are on held-out *training* engines. The only
fair comparison is our test-set result, scored once with everything frozen.

Even then, published numbers differ in protocol, by more than many of the
gaps between models:
- **Test labels:** clipping at the cap moves our XGBoost from 15.36 to
  14.08. No source we checked says which labels it scored.
- **Cap and window:** 125 and 130, windows from 15 to 40 cycles. The
  shortest test engine has 31 cycles, so a 40-cycle window (DAST) needs
  padding or drops engines.
- **Selection:** some papers report the best of several runs or don't say
  how hyperparameters were chosen. Ours are chosen by CV on training engines.
- **S is a sum over engines,** so only comparable on the same 100 engines.
- **Transcribed values:** we copy numbers, often from another paper's table.

So gaps under about one cycle of RMSE should not be read as meaningful.

### 10 likely questions (Phase 2)

**1. Why can't a plain RNN learn long-range dependencies, and how does the
LSTM fix it?**
Backpropagation through time multiplies the gradient by the recurrent
weights once per step back, so it vanishes or explodes. The LSTM's cell
state is updated by addition, and while the forget gate is open the error
passes back unchanged, so a reading many cycles back can still change the
weights.

**2. What is a TCN's receptive field, and why use dilation?**
The number of past cycles the last output can see. With kernel 3 and
dilation doubling per block, ours is 1 + 2(k−1)(2^L − 1): 29 for 3 blocks,
61 for 4. Dilation makes the field grow exponentially with depth while the
number of weights grows only linearly.

**3. Why does the Transformer need a positional encoding?**
Self-attention (and our averaging of its outputs) ignores order, so without
it a window and the same window shuffled give the same prediction; it
couldn't tell a rising sensor from a falling one. The encoding gives each
cycle a distinct code.

**4. What is the best published deep result on FD001?**
In our table the lowest RMSE is GCU-Transformer at 11.27, but that number
is from later comparison tables, not the original paper. The best number
reported in its own paper is DAST, 11.43 with S 203.15 (40-cycle windows,
cap 125). Among the models collected by Ragab et al., 1D CNN 12.61 and
ATS2S 12.63 lead.

**5. Your Transformer gets 13.28. Is that worse than DAST's 11.43?**
They can't be compared. 13.28 is cross validation on training engines;
11.43 is on the test set, under a protocol that doesn't state whether test
labels were clipped. Only our test-set result can be placed in that table.

**6. Why aren't published FD001 numbers directly comparable?**
Test-label clipping (our XGBoost moves from 15.36 to 14.08), caps of 125 vs
130, windows of 15 to 40 cycles (and what happens to short test engines),
best-of-several-runs reporting, and transcription from other papers'
tables. These effects are larger than many gaps between models.

**7. Why a 30-cycle window?**
Li et al., ATS2S and Chen et al. all use 30, and the shortest FD001 test
engine has 31 cycles, so every test engine is scored on real history. A
40-cycle window, as in DAST, would need padding or drop engines. The
shortest training engine (128 cycles) is far longer than the window.

**8. Why cap RUL at 125 for the networks too?**
It follows the piecewise-linear target of Heimes (2008), with the value
used for FD001 by Li et al., Ragab et al. and DAST. Keeping it also makes
the networks comparable with our Phase 1 models.

**9. Mo et al. and DAST say plain attention misses local structure. Does
your data agree?**
Not on FD001 so far. Our plain Transformer has the lowest CV RMSE of our
networks (13.28), and the TCN, which builds in exactly that local
structure, is worst (15.91). Their gains may come from other parts of their
models and training, so in the report we treat the argument as a
hypothesis, not a finding.

**10. Why are the published deep results in a separate file?**
Our Phase 1 claim, 5th and 6th of 14 by RMSE, is counted against the
comparison in `results/test_published_comparison.csv`. Adding newer models
(GCU-Transformer, DAST, Chen et al.) to that table would change the pool and
silently break the claim. Keeping `dl_published.csv` separate leaves Phase 1
true as reported.

### Things I must check before the viva (Phase 2)
- GCU-Transformer 11.27 against Mo et al. 2021 itself, if I get the PDF.
- Zheng et al. 2017's training cap from the original paper, if I get it.
