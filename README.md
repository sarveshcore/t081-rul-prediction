# Remaining Useful Life Prediction for Industrial Engines

Team T081, Section E. CSAI project 7.

We estimate how many operating cycles a turbofan engine has left before it
fails, from its multivariate sensor readings. Phase 1 compares engineered
degradation features with classical regression (Elastic Net, Random Forest,
XGBoost). Phase 2 adds sequence models (LSTM, temporal CNN, Transformer).

**Data:** NASA C-MAPSS Turbofan Engine Degradation Simulation, subset FD001
(Saxena and Goebel, 2008).

## Results

Filled in by #6 from `results/test_metrics.csv`. Test RUL is scored once per
engine, at its last observed cycle.

| Model | RMSE | MAE | NASA score | RMSE early | RMSE late |
|---|---|---|---|---|---|
| Mean baseline | | | | | |
| Elastic Net | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

## Reproduce

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Then run the notebooks in `notebooks/` in numeric order.

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
