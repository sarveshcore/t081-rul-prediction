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
published FD001 results are scored, XGBoost reaches RMSE 14.08. That beats published
Random Forest, gradient boosting and early deep learning results, and is about 1.4 cycles
behind the 1D CNN of Li et al. (12.61). See `report/sections/results.tex`.

## Reproduce

Needs Python 3.12 or newer.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Then `make all` runs ruff, pytest and every notebook in `notebooks/` in numeric order,
regenerating `results/` and `figures/`. It stops at the first notebook that fails.
Without `make` (for example on Windows), run the notebooks by hand in numeric order.

## Layout

```
data/raw/        C-MAPSS FD001 text files
src/rul/         data, split, features, models, evaluate
notebooks/       01_eda, 02_features, 03_models, 04_results
tests/           pytest checks for leakage and correctness
figures/         plots used by the report and slides
results/         metrics and chosen hyperparameters
report/          IEEE LaTeX report
slides/          presentation PDF
docs/viva/       each member's viva notes
```

## Team

| Member | Owns |
|---|---|
| Sarvesh Srinath | Repository, engine level split, evaluation, final results, review |
| Trishit Swarnakar | Data loading, EDA, RUL target |
| Aryan Kumar | Feature engineering |
| Husain Khorakiwala | Models, tuning, explainability |
| Navodit Sharma | Literature review, report, slides |

How we work: [CONTRIBUTING.md](CONTRIBUTING.md). Plan and dates:
[ROADMAP.md](ROADMAP.md).
