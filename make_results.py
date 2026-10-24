"""
Reproduce numbers in Tabs. 2&3 and Figs. 1&2
# usage: run from repo root, about half an hour
python make_results.py    
"""

import numpy as np
import run
from implementations import *

# step sizes, relative to 1 / L (L: the largest eigenvalue of x tx / N),
# for SGD relative to 1 / mean x_n^2
GD_STEP = 1
LOGISTIC_STEP = 4
SGD_STEP = 0.05
# number of steps
GD_ITERS = LOGISTIC_ITERS = 500
SGD_ITERS = 200000

x, _, y, _, names = run.load_data()
folds = np.array_split(np.random.default_rng(1).permutation(len(y)), 5)
print(folds)

def f1(y, y1):
    return np.sum((y == 1) & y1) * 2.0 / (y1.sum() + y.sum())


def cross_validate(model_func, threshold=None):
    scores = []
    for val in folds:
        mask = np.ones(len(y), dtype=bool)
        mask[val] = False
        tr = np.arange(len(y))[mask] 
        transform = run.preproc(x[tr], names)
        tx = transform(x[tr])
        w = model_func(y[tr], tx)
        t = run.scan_f1_thres(y[tr], tx @ w) if threshold is None else threshold
        scores.append(f1(y[val], transform(x[val]) @ w >= t))
    return scores


def step_size(tx):
    # 1 / L, the largest eigenvalue of x x^T / N.
    return 1 / np.linalg.eigvalsh(tx.T @ tx / len(tx)).max()


def sgd(y, tx):
    np.random.seed(1)
    gamma = SGD_STEP / np.mean(np.sum(tx**2, axis=1))
    return mean_squared_error_sgd(y, tx, np.zeros(tx.shape[1]), SGD_ITERS, gamma)[0]


# Table 2: different methods
print("tab2")
methods = [
    ("linear regression, SGD", sgd),
    (
        "linear regression, GD",
        lambda y, tx: mean_squared_error_gd(
            y, tx, np.zeros(tx.shape[1]), GD_ITERS, GD_STEP * step_size(tx)
        )[0],
    ),
    ("least squares", lambda y, tx: least_squares(y, tx)[0]),
    ("ridge regression", lambda y, tx: ridge_regression(y, tx, 1e-4)[0]),
    (
        "logistic regression, GD",
        lambda y, tx: logistic_regression(
            y, tx, np.zeros(tx.shape[1]), LOGISTIC_ITERS, LOGISTIC_STEP * step_size(tx)
        )[0],
    ),
    (
        "reg. logistic regression, GD",
        lambda y, tx: reg_logistic_regression(
            y, tx, run.LAMBDA, np.zeros(tx.shape[1]), LOGISTIC_ITERS, LOGISTIC_STEP * step_size(tx)
        )[0],
    ),
    ("ours: reg. logistic regression, Newton", run.newton),
]
for name, fit in methods:
    try:
        scores = cross_validate(fit)
    except:  # X^T X singular
        print("%s: fails (singular)" % name)
        continue
    print("%s: F1 %.4f +- %.4f" % (name, np.mean(scores), np.std(scores)))
ours = np.array(scores) 

# Table 3: ablations
print("tab3")
ablations = [
    ("threshold 0.5", None, None),
    ("categorical as numbers", "ONEHOT", False),
    ("drop cols > 90% missing", "MAX_MISSING", 0.9),
    ("drop cols > 50% missing", "MAX_MISSING", 0.5),
    ("w/o log transform", "LOG", False),
    ("w/o clamping", "CLIP", False),
    ("penalty 1e-5", "LAMBDA", 1e-5),
    ("penalty 1e-2", "LAMBDA", 1e-2),
]

for name, setting, value in ablations:
    if setting is not None:
        old = getattr(run, setting)
        setattr(run, setting, value)
        scores = cross_validate(run.newton)
        setattr(run, setting, old)
    else:  # the first one
        scores = cross_validate(run.newton, threshold=0.0)  # logit 0 = p 0.5
    d = np.array(scores) - ours
    print(
        "%s: delta F1 %.4f +- %.4f (std error)"
        % (name, d.mean(), d.std(ddof=1) / np.sqrt(len(d)))
    )

# Figure 1: Newton loss
print("fig1")
tx = run.preproc(x, names)(x)
for k in range(11):
    w = run.newton(y, tx, iters=k)
    z = tx @ w
    loss = np.average(np.log(1 + np.exp(z)) - z * y)
    penalty = run.LAMBDA * w @ w
    print("Newton at step %d: loss (w/ penalty) %.6f" % (k, loss + penalty))

# Figure 2: training set F1 against threshold 
print("fig2")
p = run.sigmoid(tx @ run.newton(y, tx))
print(
    "chosen threshold: %.4f"
    % run.sigmoid(run.scan_f1_thres(y, tx @ run.newton(y, tx)))
)
for t in np.arange(0, 1, 0.05):
    print("threshold %.2f: training F1 %.4f" % (t, f1(y, p >= t)))
