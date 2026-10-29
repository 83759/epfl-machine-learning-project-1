from helpers import calculate_mse, batch_iter, compute_loss_mse
import numpy as np


def _compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the final model parameters as a numpy array
        loss: the final loss
    """
    w = initial_w
    for n_iter in range(max_iters):
        w -= gamma * _compute_gradient(y, tx, w)

    return w, compute_loss_mse(y, tx, w)


def _compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the final model parameters as a numpy array
        loss: the final loss
    """
    batch_size = 1
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):
            w -= gamma * _compute_stoch_gradient(y_batch, tx_batch, w)

    return w, compute_loss_mse(y, tx, w)


def least_squares(y, tx):
    """Calculate the least squares solution.
       Returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """

    N = y.shape[0]
    D = tx.shape[1]
    assert y.shape == (N,)
    assert tx.shape == (N, D)

    # X^T (y - X w) = 0
    # <=> X^T y = X^T X w
    # <=> b = A w

    A = tx.T @ tx
    b = tx.T @ y

    w = np.linalg.solve(A, b)

    return w, compute_loss_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    N, D = tx.shape
    assert y.shape == (N,)
    assert tx.shape == (N, D)

    # -X^T (y - X w) + 2 N lambda w = 0
    # <=> -X^T y + X^T X w + 2 N lambda w = 0
    # <=> X^T X w + 2 N lambda w = X^T y
    # <=> (X^T X + 2 N lambda I) w = X^T y
    # <=> A w = b

    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y

    w = np.linalg.solve(A, b)

    return w, compute_loss_mse(y, tx, w)


def _sigmoid(z):
    """Compute the sigmoid function without overflowing for large inputs."""
    return np.exp(-np.logaddexp(0.0, -z))


def _compute_logistic_loss(y, tx, w):
    """Compute the averaged binary logistic loss for labels in {0, 1}."""
    scores = tx @ w
    return np.mean(np.logaddexp(0.0, scores) - y * scores)


def _compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the averaged binary logistic loss."""
    errors = _sigmoid(tx @ w) - y
    return (tx.T @ errors) / y.shape[0]


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit binary logistic regression with full-batch gradient descent.

    Labels must use the ``{0, 1}`` convention. The returned loss is the
    averaged logistic loss evaluated at the final weights.
    """
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        w -= gamma * _compute_logistic_gradient(y, tx, w)

    return w, _compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Fit L2-regularized binary logistic regression by gradient descent.

    The optimized objective adds ``lambda_ * ||w||^2`` to the averaged
    logistic loss. The returned loss excludes the regularization penalty.
    """
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        gradient = _compute_logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w -= gamma * gradient

    return w, _compute_logistic_loss(y, tx, w)
