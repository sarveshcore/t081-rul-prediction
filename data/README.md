# NASA C-MAPSS FD001 data

These are the original FD001 text files from NASA's [C-MAPSS Jet Engine
Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data),
distributed in [`CMAPSSData.zip`](https://data.nasa.gov/docs/legacy/CMAPSSData.zip).
They were extracted without changing their contents. NASA's dataset page does
not specify a license.

| File | Contents |
| --- | --- |
| `raw/train_FD001.txt` | 100 engine trajectories, each observed through failure; 20,631 cycle rows. |
| `raw/test_FD001.txt` | 100 engine trajectories, each stopped before failure; 13,096 cycle rows. |
| `raw/RUL_FD001.txt` | One true remaining-life value per test engine, ordered by engine ID, at its final observed cycle. |

Each trajectory row contains an engine ID, cycle number, three operating
settings, and 21 sensor readings. The files have no header and use whitespace
separators. `rul.data.load_fd()` assigns column names and maps the test RUL
values to engine IDs. `rul.data.add_rul()` calculates cycle-level targets for
the run-to-failure **training** trajectories; the test targets come from
`RUL_FD001.txt`.

For training-only analysis, use `rul.data.load_train_fd()`. It reads the same
training table without opening either test file. The executed EDA notebook is
`notebooks/01_eda.ipynb`; it records the evidence for the 14 retained sensors
exported as `rul.data.USEFUL_SENSORS`. Only its final cell opens the test files
for the descriptive plots approved in PR #19; no training decision is revisited.

To reproduce these raw files, download the linked archive and extract only
`train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` into `data/raw/`.

Citation: A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation,” *Proceedings of the
1st International Conference on Prognostics and Health Management*, Denver,
CO, October 2008.
