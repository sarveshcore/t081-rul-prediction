# Remaining Useful Life Prediction for Industrial Engines

Team T081, Section E. CSAI project 7.

We estimate how many operating cycles a turbofan engine has left before it
fails, from its multivariate sensor readings. Phase 1 compares engineered
degradation features with classical regression (Elastic Net, Random Forest,
XGBoost). Phase 2 adds sequence models (LSTM, temporal CNN, Transformer).

**Data:** NASA C-MAPSS Turbofan Engine Degradation Simulation, subset FD001
(Saxena and Goebel, 2008).

## Results

From `results/test_metrics.csv`, made by `notebooks/04_results.ipynb`. Each of the 100
FD001 test engines is scored once, at its last observed cycle, against the true RUL. Models
use the engineered features and the hyperparameters frozen by cross validation. Late means
true RUL of 50 cycles or fewer (33 engines).

| Model | RMSE | MAE | NASA score | RMSE early | RMSE late |
|---|---|---|---|---|---|
| Mean baseline | 43.07 | 35.90 | 33629 | 27.30 | 64.09 |
| Elastic Net | 18.27 | 14.11 | 522 | 19.75 | 14.83 |
| Random Forest | 15.23 | 11.33 | 344 | 17.88 | 7.34 |
| XGBoost (selected) | 15.36 | 11.26 | 389 | 18.11 | 7.03 |

Random Forest and XGBoost are tied: a paired bootstrap over the test engines puts the RMSE
gap between them inside its 95% interval. Against test labels clipped at 125, as many
published FD001 results are scored, Random Forest reaches RMSE 14.02 and XGBoost 14.08.
That beats published Random Forest, gradient boosting and six of ten published deep
models, and is 0.8 to 1.5 cycles behind four deep models (RMSE 12.61 to 13.18). On the
NASA score, six published models do better than Random Forest (310) and eight do better than
XGBoost (347). See `report/sections/results.tex`.

### Phase 2: sequence models

From `results/dl_test_metrics.csv`, made by `notebooks/09_dl_results.ipynb`. The same 100
test engines, scored once against the true RUL. Each network reads a 30-cycle window of the
14 useful sensors, uses the settings frozen by cross validation, and is trained with 5 seeds;
networks show the mean ± standard deviation over seeds. Phase 1 rows are unchanged.

| Model | RMSE | MAE | NASA score | RMSE late |
|---|---|---|---|---|
| XGBoost (Phase 1, selected) | 15.36 | 11.26 | 389 | 7.03 |
| LSTM | 16.02 ± 0.67 | 11.80 | 622 | 6.21 |
| Temporal CNN | 16.84 ± 0.93 | 12.51 | 584 | 8.10 |
| Transformer (selected) | 15.37 ± 0.66 | 11.77 | 422 | 5.88 |
| Hybrid, Transformer + engineered features | 15.92 ± 0.66 | 12.13 | 450 | 5.85 |
| Hybrid, LSTM + engineered features | 15.93 ± 0.67 | 11.90 | 493 | 6.11 |
| Residual hybrid, XGBoost + Transformer | 15.13 ± 0.10 | 11.11 | 348 | 5.03 |

The selected Transformer ties XGBoost on engineered features: a paired bootstrap over the
test engines puts the RMSE difference between −2.3 and +2.2 cycles. Learning from raw
sensor windows matches the engineered features but does not beat them. The networks are
more accurate close to failure and less accurate 51 to 100 cycles out. Adding the
engineered features to the Transformer did not help on the test set. The residual hybrid
has the lowest RMSE but was not the model selected by cross validation. See
`report/sections/dl_results.tex`.

## Reproduce

Needs Python 3.12 to 3.14. PyTorch 2.14.1 has no wheels for 3.15 yet.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Then `make all` runs ruff, pytest and every notebook in `notebooks/` in numeric order,
regenerating `results/` and `figures/`. It stops at the first notebook that fails.
Notebook 06 retrains every network in the cross validation search (about 55 minutes on
CPU). Notebooks 07 and 09 load saved results by default; set `LOAD_SAVED = False` in 07 or
`RETRAIN = True` in 09 to retrain (09 takes about 30 minutes in one process).
Without `make` (for example on Windows), run the notebooks by hand in numeric order.

## Layout

```
data/raw/        C-MAPSS FD001 text files
src/rul/         data, eda, split, features, models, evaluate, explain,
                 sequences, train, nets, hybrid, holdout
notebooks/       01_eda, 02_features, 03_models, 04_results, 05_explain,
                 06_sequence_models, 07_hybrid, 09_dl_results
tests/           pytest checks for leakage and correctness
figures/         plots used by the report and slides
results/         metrics and chosen hyperparameters
report/          IEEE LaTeX report
slides/          presentation PDF
docs/viva/       each member's viva notes
```

## Team

| Member | Phase 1 | Phase 2 |
|---|---|---|
| Sarvesh Srinath | Repository, engine level split, evaluation, final results, review | Sensor windows, shared training loop, DL test results |
| Trishit Swarnakar | Data loading, EDA, RUL target | DL dataset choices and ablations |
| Aryan Kumar | Feature engineering | Hybrid models |
| Husain Khorakiwala | Models, tuning, explainability | LSTM, temporal CNN, Transformer |
| Navodit Sharma | Literature review, report, slides | DL literature review, report, slides |

How we work: [CONTRIBUTING.md](CONTRIBUTING.md). Plan and dates:
[ROADMAP.md](ROADMAP.md).
