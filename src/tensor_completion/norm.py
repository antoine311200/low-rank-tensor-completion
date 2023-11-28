import numpy as np

from tensor_completion.utils import unfold


def nucleus_norm(x, truncate=None):
    """Compute the Nucleus Norm of a matrix.

    The Nucleus Norm of a matrix is defined as the sum of its singular values.

    Args:
        x (np.ndarray): Matrix to compute the Nucleus Norm of.
        truncate (int, optional): Number of singular values to keep. Defaults to None.

    Returns:
        float: Nucleus Norm of the matrix."""

    # Compute the singular values of the matrix
    s = np.linalg.svd(x, compute_uv=False)

    # Truncate the singular values if needed
    if truncate is not None:
        s = s[truncate:]

    # Return the sum of the singular values
    return np.sum(s)

def tensor_nuclear_norm(x, weights=None, truncate=None):
    """Compute the Tensor Nuclear Norm of a tensor.

    The Tensor Nuclear Norm of a tensor is defined as the weighted sum of the nucleus norm of each unfolding mode.

    Args:
        x (np.ndarray): Tensor to compute the Tensor Nuclear Norm of.
        truncate (int, optional): Number of singular values to keep. Defaults to None.

    Returns:
        float: Tensor Nuclear Norm of the tensor."""

    # Check if the weights are provided
    if weights is None:
        weights = np.ones(x.ndim) / x.ndim

    # Compute the Tensor Nuclear Norm
    return np.sum([weights[i] * nucleus_norm(unfold(x, i), truncate) for i in range(x.ndim)])

def autoregressive_norm(Z, A, h_indices):
    """Compute the Autoregressive Norm of a tensor.

    The Autoregressive Norm of a matrix is defined as the squred sum of the difference between each element and the sum of its lagged neighbors.\

    Args:
        Z (np.ndarray): Tensor to compute the Autoregressive Norm of.
        A (np.ndarray): Autoregressive coefficients.
        h_indices (np.ndarray): Indices of the lagged neighbors.

    Returns:
        float: Autoregressive Norm of the tensor."""

    # Compute the difference between each element and the sum of its lagged neighbors
    # diff = Z - np.sum([A[i] * Z.take(h_indices[i], axis=0) for i in range(len(A))], axis=0)

    diff = Z.copy()
    A = A.astype(Z.dtype)

    for i in range(len(h_indices)):
        diff -= A[:, i][:, np.newaxis] * np.roll(Z, -h_indices[i], axis=1)

    # Return the squared sum of the difference
    return np.sum(diff ** 2)

