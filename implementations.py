import numpy as np

def sigmoid(x):
    """
    Return sigmoid(x)
    Args: x (with arbitrary shape)
    """
    z = np.exp(x)
    return z / (z + 1)
    # return (1.0 + np.tanh(t / 2)) / 2  # more robust


def _mse(y, tx, w):
    """
    Return MSE = 1/2N (xw - y)^2. 
    Args: y[N], tx[N,D], w[D]
    """
    N, D = tx.shape
    e = tx @ w - y
    return np.sum(e * e, axis=0) / 2 / N


def _mse_grad(y, tx, w):
    """
    Return dMSE/dw = e d(xw - y)/dw = ex/N. 
    Args: y[N], tx[N,D], w[D]
    """
    N, D = tx.shape
    e = tx @ w - y
    return e.T @ tx / N


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Return (weights, loss) for gradient descent
    Args: y[N], tx[N,D], w[D], number of iters max_iters, step size gamma
    """
    N, D = tx.shape
    w = initial_w * 1.0
    loss = _mse(y, tx, w)
    for it in range(max_iters):
        grad = _mse_grad(y, tx, w)
        w = w - gamma * grad
        loss = _mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Return (weights, loss) for stoc. gradient descent (minibatch size = 1)
    Args: y[N], tx[N,D], w[D], number of iters max_iters, step size gamma
    """
    N, D = tx.shape
    rng = np.random.default_rng(1) 
    w = initial_w.copy()
    for it in range(max_iters):
        sample = rng.integers(N)
        y_ = np.array([y[sample]], dtype=float)
        tx_ = np.array([tx[sample]], dtype=float)
        grad = _mse_grad(y_, tx_, w)
        w = w - gamma * grad
    loss = _mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """
    Solve normal equations x x^T w = x y
    Args: y[N], tx[N,D]
    """
    x = tx.T 
    lhs = x @ tx 
    rhs = x @ y 
    w = np.linalg.inv(lhs) @ rhs  # or solve(lhs, rhs)
    loss = _mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regress minimize 1/2N (x^T w - y)^2 + lam w^2
    using the closed form: w = x y / (x x^T + 2 N lam I)
    Args: y[N], tx[N,D], lambda (strength)
    """
    N, D = tx.shape
    x = tx.T 
    lhs = x @ tx + np.eye(D) * (2 * N * lambda_)
    rhs = x @ y 
    w = np.linalg.inv(lhs) @ rhs
    loss = _mse(y, tx, w)
    return w, loss

########


def _logist_error(y, tx, w):
    """
    Return logist loss
    Args: y[N], tx[N,D], w[D]
    """
    N, D = tx.shape
    z = tx @ w
    l = np.log(1.0 + np.exp(z))
    return np.sum(l - z * y) / N


def _logist_grad(y, tx, w):
    """
    Return logist grad
    Args: y[N], tx[N,D], w[D]
    """
    N, D = tx.shape 
    z = tx @ w 
    x = tx.T 
    sig = sigmoid(tx @ w)
    return x @ (sig - y) / N


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Return gradient descnet based logistic regression solutions (weights, loss)
    Args: y[N], tx[N,D], w0[D], number of iters, step size
    """
    w = initial_w * 1.0
    loss = _logist_error(y, tx, w)
    for it in range(max_iters):
        grad = _logist_grad(y, tx, w)
        w -= gamma * grad
        loss = _logist_error(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Return L2-regularized gradient descnet based logistic regression solutions (weights, loss w/o penalty)
    The only difference is an additional grad 2 lam w from the extra loss term w^2
    Args: y[N], tx[N,D], penalty strength, w0[D], number of iters, step size
    """
    w = initial_w * 1.0
    loss = _logist_error(y, tx, w)
    for it in range(max_iters):
        grad = _logist_grad(y, tx, w) + w * lambda_ * 2
        w -= gamma * grad
        loss = _logist_error(y, tx, w)
    return w, loss
