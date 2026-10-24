import os
import numpy as np
from helpers import *

LAMBDA = 1e-3  # L2 reg
MISSING_CODES = (7, 9, 77, 99, 777, 999, 7777, 9999, 99900)  # don't know / refused
NONE_CODES = (8, 88, 888)  # none
eps = 1e-3
# preprocessing steps; experiments.py turns some off one at a time
MAX_MISSING = 0.99  # drop columns with more missing values
ONEHOT = True  # one-hot encode columns with few answers
CLIP = True  # clamping numeric columns
LOG = True # log encoding for long tails

def load_data():
    DATADIR = "data"
    x_train, x_test, y_train, not_used, test_ids = load_csv_data(DATADIR)
    filename = os.path.join(DATADIR, "x_train.csv")
    with open(filename) as f:
        colnames = f.readline().strip().split(",")
    return x_train, x_test, (y_train + 1) / 2.0, test_ids, colnames[1:]


def preproc(x, colnames):
    n, descriptors = len(x), []
    def replace_codes(col, train_maximum): # replace weird missing/none codes
        v = col.copy()
        is_missing = [x for x in MISSING_CODES if x > train_maximum]
        is_none = [x for x in NONE_CODES if x > train_maximum]
        v[np.isin(v, is_missing)] = np.nan
        v[np.isin(v, is_none)] = 0.0
        return v
    for i, name in enumerate(colnames):
        col = x[:, i]
        flag_nan = np.isnan(col)
        if name not in ["IDATE", "IMONTH", "IDAY", "IYEAR", "FMONTH", "SEQNO", "_PSU", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_CLLCPWT", "_DUALCOR"] and flag_nan.sum() <= MAX_MISSING * n:
            values, counts = np.unique(col[~flag_nan], return_counts=True)
            if ONEHOT and (name in ["EXRACT11", "EXRACT21", "_STATE"] or values.size <= 15):
                # onehot for highfreq values, most frequent one is the reference
                levels = [v for v, c in zip(values, counts) if c >= eps * n]
                if flag_nan.mean() >= eps:
                    levels.append(np.nan)
                ref = values[np.argmax(counts)] if counts.max() >= flag_nan.sum() else np.nan
                levels = [lv for lv in levels if not np.array_equal(lv, ref, equal_nan=True)]
                descriptors.append((i, levels, None))
            else:
                # numeric: codes above all regular values are special (77/88/99 for 1..30 days)
                regular = values[~np.isin(values, NONE_CODES + MISSING_CODES)]
                maximum = np.inf
                if regular.size:
                    maximum = regular.max()
                v = replace_codes(col, maximum)
                if not np.isnan(v).all(): # ignore all-nan cols
                    lo, hi = np.nanquantile(v, [eps, 1 - eps]) if CLIP else (-np.inf, np.inf)
                    w = np.clip(v[~np.isnan(v)], lo, hi)
                    log = LOG and np.mean(((w - w.mean()) / (w.std() + 1e-9)) ** 3) > 2  # log encode long tails
                    median = np.median(np.log1p(w) if log else w) # used to fill empty
                    descriptors.append((i, None, (maximum, lo, hi, log, median)))

    def encode(x):
        cols = []
        for j, level_desc, numeric_desc in descriptors:
            col = x[:, j]
            if level_desc is None: # for numeric
                maximum, lo, hi, log, median = numeric_desc
                v = np.clip(replace_codes(col, maximum), lo, hi)
                if log:
                    v = np.log1p(v)
                cols.append(np.where(np.isnan(v), median, v))
            else: # for enum
                for level in level_desc:
                    if np.isnan(level):
                        cols.append(np.isnan(col))
                    else:
                        cols.append(col == level)
        return np.column_stack(cols).astype(float)
    
    z = encode(x)
    # make a transform so it can apply to both datasets
    mask = z.std(0) > 0  # drop constant features
    return lambda x: np.column_stack([np.ones(len(x)), (encode(x)[:, mask] - z[:, mask].mean(0)) / z[:, mask].std(0)])


def sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def newton(y, tx, iters=10):
    # newton with l2 reg
    n, d = tx.shape
    w = np.zeros(d) # init
    for it in range(iters):
        p = sigmoid(tx @ w)
        grad = tx.T @ (p - y) / n + 2 * LAMBDA * w
        tmp = tx * np.sqrt(p * (1 - p) / n).reshape(-1, 1)
        hes = tmp.T @ tmp + 2 * LAMBDA * np.eye(d) 
        w -= np.linalg.solve(hes, grad) # g/h
    return w


def scan_f1_thres(y, scores): # Argmax_t F1: [positive iff. score >= t]
    idx = sorted(range(len(scores)), key=lambda i: -scores[i])
    true_pos_num = y.sum()
    sum_true_pos = 0 # how many TP if judge first k as positive
    best_f1 = -1
    best_thres = -1
    for pred_pos_num, i in enumerate(idx, start=1):
        sum_true_pos += y[i]
        f1 = 2 * sum_true_pos / (pred_pos_num + true_pos_num)
        if f1 > best_f1:
            best_f1 = f1
            best_thres = scores[i]
    return best_thres

def main():
    x_train, x_test, y, test_ids, names = load_data()
    transform = preproc(x_train, names)
    tx_train = transform(x_train)
    w = newton(y, tx_train)
    threshold = scan_f1_thres(y, tx_train @ w)
    pred = np.where(transform(x_test) @ w >= threshold, 1, -1)
    out = "submission.csv"
    create_csv_submission(test_ids, pred, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
