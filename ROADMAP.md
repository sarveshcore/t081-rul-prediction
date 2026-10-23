# Roadmap

Phase 1 evaluation: **Thursday 29 October 2026, 2:45 pm.**
Code freeze: **Tuesday 27 October, 11:59 pm.** After the freeze only the
report, slides and viva notes change.

## Timeline

| Day | Goal | Issues |
|---|---|---|
| Fri 23 Oct | Repo set up, everyone has cloned and opened a test branch | scaffold PR |
| Sat 24 Oct | Data loader and split merged, so nobody is blocked | #1, #2 |
| Sun 25 Oct | EDA, features and literature review merged | #3, #4, #8 |
| Mon 26 Oct | Models tuned with cross validation | #5 |
| Tue 27 Oct | Test results, explainability, reproducibility. Report draft. **Freeze** | #6, #7, #11, #9 |
| Wed 28 Oct | Report and slides final. Rehearsal and mock viva | #9, #10, #12 |
| Thu 29 Oct | Morning buffer only. Evaluation at 2:45 pm | |

## Dependencies

```
#1 data ──┬── #3 EDA
          ├── #4 features ──┐
#2 split ─┴─────────────────┴── #5 models ──┬── #6 test results ──┐
                                            └── #7 SHAP ──────────┤
#8 literature ────────────────────────────────────────────────────┴── #9 report, #10 slides, #12 viva
```

Aryan and Husain do not wait for #3: they start on raw sensors as soon as #1
is merged and switch to the EDA's sensor list when it lands.

## Rubric map

| Phase 1 criterion | Issues | Owner |
|---|---|---|
| Literature Review | #8 | Navodit |
| Dataset Quality and EDA | #1, #3 | Trishit |
| Feature Engineering | #4 | Aryan |
| Theoretical Rigor | theory sections in #2, #4, #5, #7 | each owner |
| Model Application | #5, #6, #7 | Husain, Sarvesh |
| GitHub Repository and Code Quality | #11, every PR | Sarvesh |
| Project Presentation | #10 | Navodit, everyone |
| Viva Voce | #12 | everyone |
| Project Report (LaTeX, IEEE) | #9 | Navodit, everyone |

## Decisions

- **FD001 only** for Phase 1. FD002 to FD004 are a stretch goal.
- **Split by engine**, never by row. Tuning uses 5 fold GroupKFold on `unit`.
- **The test set is used once**, in #6, after hyperparameters are frozen.
- **RUL is capped at 125 for training** and the cap is justified in #3.
  Test scoring uses the true RUL from `RUL_FD001.txt`.
- **Early vs late stage:** late means true RUL of 50 cycles or fewer.
- **Seed 42** everywhere.

## Phase 2 (after Phase 1)

LSTM, temporal CNN or Transformer encoder on fixed length sensor windows,
an ablation study, and a hybrid that feeds engineered features into the
sequence model. Issues will be opened under the `phase 2` label.
