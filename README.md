# How to run

1. Place the data files (`x_train.csv`, `y_train.csv`, `x_test.csv`) in `data/`.
2. Run `python run.py` to get `submission.csv` (reproduces our best submission, F1 0.445).
3. Run `python make_results.py` to reproduce the tables and the figure data in the report.

## Files

- `helpers.py` is provided by the course.
- `implementations.py` contains the basic algorithms required by the project.
- `make_results.py` uses both files to compare the different algorithms.

## Pipeline

- Data preparation and feature generation are in `preproc` in `run.py` (drop id/date/weight columns, replace special codes, one-hot or numeric encoding, standardize).
- The model is trained by `newton` in `run.py`, and the threshold is chosen by `scan_f1_thres` on the training set.
- Cross-validation (5 folds) is in `cross_validate` in `make_results.py`. Preprocessing, training and threshold use 4 folds, F1 is measured on the 5th.
- More details are in the report.
