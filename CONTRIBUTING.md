# How we work

Read this once before your first pull request. It is short on purpose.

## The rules

1. **Never push to `main`.** Every change goes through a pull request (PR).
   `main` is protected, so a push will be rejected anyway.
2. **One issue, one branch, one PR.** Pick your issue, work on a branch for
   it, and open a PR that says `Closes #N`.
3. **Sarvesh reviews and merges.** You do not merge your own PR. Sarvesh's
   own PRs are reviewed by any teammate.
4. **Never use the test set** (`test_FD001.txt`, `RUL_FD001.txt`) for anything
   except #6. Tuning uses cross validation on the training engines only.
5. **Split by engine (`unit`), never by row.** Rolling windows never cross
   from one engine into another.
6. **Seed 42** for every random thing.
7. **Notebooks run top to bottom** from a fresh kernel before you commit them.
   Keep the outputs; graders read them.
8. **Write your report section and your viva notes** as part of your work,
   not at the end. Every issue lists the section you own.

## Daily rhythm

- Post a one line update in the group chat by **10 pm**: what merged, what
  is next, what blocks you.
- Blocked? Comment on the issue and tag Sarvesh. Don't wait silently.
- Sarvesh reviews open PRs within **4 hours** between 9 am and 11 pm.
- Small PRs merge fast. Open a draft PR early if you want feedback.

## First time with Git? Step by step

One time setup:

```sh
git clone https://github.com/sarveshcore/t081-rul-prediction.git
cd t081-rul-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

For each issue:

```sh
git switch main
git pull                          # get everyone's merged work
git switch -c eda/3-sensor-trends # <area>/<issue number>-<short name>

# ... work ...

git status                        # check what changed
git add notebooks/01_eda.ipynb src/rul/data.py
git commit -m "feat(eda): plot sensor trends for five engines"
git push -u origin eda/3-sensor-trends
```

Then open the link Git prints, fill in the PR template, and write
`Closes #3` in the description.

Review asked for changes? Commit on the same branch and push again. The PR
updates by itself.

`main` moved while you worked? Bring it in:

```sh
git switch main && git pull
git switch eda/3-sensor-trends
git merge main
```

If Git reports a **conflict**, stop and ask in the group chat before you
try to fix it. Never use `git push --force`.

## Names

- **Branch:** `<area>/<issue>-<short-name>`, for example `features/4-rolling-slope`.
  Areas: `data`, `eda`, `features`, `models`, `eval`, `report`, `slides`, `repo`, `viva`.
- **Commit:** `type(area): what changed`, for example
  `fix(features): keep rolling windows inside one engine`.
  Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`.
- **PR title:** the same style as a commit.

## Code

- Reusable code lives in `src/rul/`. Notebooks import it
  (`from rul.features import make_features`); they don't copy it.
- Every new `.py` file starts with two comment lines saying what it is:

  ```python
  # ABOUTME: Feature engineering for C-MAPSS sensor data.
  # ABOUTME: Rolling statistics, slopes, regime normalisation, health index.
  ```

- Functions get a one line docstring. Names say what the thing does.
- Add a test in `tests/` for anything that could leak data or silently be
  wrong (splits, windows, RUL target, metrics). Run `pytest` before you push.
- Figures go to `figures/<area>_<name>.png`, numbers to `results/`.

## Interfaces

These signatures let everyone build in parallel. Keep them; if you must
change one, say so in your PR and tag whoever uses it.

```python
# src/rul/data.py  (#1, Trishit)
SENSORS: list[str]          # "s_1" ... "s_21"
SETTINGS: list[str]         # "setting_1" ... "setting_3"
USEFUL_SENSORS: list[str]   # set by the EDA in #3
def load_fd(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (train, test, test_rul). Columns: unit, cycle, settings, sensors.
    test_rul is indexed by unit: true RUL at that unit's last test cycle."""
def add_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Add column `rul` = unit's max cycle minus cycle, clipped at cap."""

# src/rul/split.py  (#2, Sarvesh)
def engine_split(df, val_frac=0.2, seed=42) -> tuple[pd.DataFrame, pd.DataFrame]: ...
def engine_folds(df, n_splits=5):  # yields (train_idx, val_idx), grouped by unit
def last_cycle(df) -> pd.DataFrame:  # one row per unit, its final cycle

# src/rul/features.py  (#4, Aryan)
def make_features(df, sensors, windows=(5, 10, 20), scaler=None) -> tuple[pd.DataFrame, object]:
    """Same rows as df, plus engineered columns. Fits scaler if None,
    otherwise applies the given one (pass the training scaler for val/test)."""

# src/rul/models.py  (#5, Husain)
def build_model(name: str, **params):  # "elastic_net", "random_forest", "xgboost"

# src/rul/evaluate.py  (#2, Sarvesh)
def evaluate(y_true, y_pred, late_cut=50) -> dict:
    """rmse, mae, nasa_score, rmse_early, mae_early, rmse_late, mae_late."""
```

## Pull request checklist

The PR template repeats this. Before you ask for review:

- [ ] `pytest` passes and `ruff check .` is clean
- [ ] Notebooks run from a fresh kernel
- [ ] No test set use outside #6
- [ ] Report section for this issue updated
- [ ] I can explain every line of this PR in the viva
