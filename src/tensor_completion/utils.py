import numpy as np


def unfold(x, mode):
    """Unfold a tensor along a given mode.
    Convert a tensor of shape (I_1, ..., I_N) to a matrix of shape (I_mode, I_1 * ... * I_{mode-1} * I_{mode+1} * ... * I_N).

    Args:
        x (np.ndarray): Tensor to unfold.
        mode (int): Mode along which to unfold the tensor.

    Returns:
        np.ndarray: Unfolded tensor."""
    # return np.moveaxis(x, mode, 0).reshape(x.shape[mode], -1)
    return np.moveaxis(x, mode, 0).reshape((x.shape[mode], -1), order = 'F')

def fold(x, mode, shape):
    """Fold a tensor along a given mode.
    Convert a matrix to a tensor from a given shape and a given mode.

    Args:
        x (np.ndarray): Tensor to fold.
        mode (int): Mode along which to fold the tensor.
        shape (tuple): Shape of the folded tensor.

    Returns:
        np.ndarray: Folded tensor."""
    x = x.reshape((shape[mode], *(shape[:mode] + shape[mode+1:])), order = 'F')
    return np.moveaxis(x, 0, mode)

def generalized_singular_threshold(M, theta, threshold=0.1):
    """Generalized Singular Value Thresholding.

    Compute the Generalized Singular Value Thresholding of a matrix.

    Args:
        M (np.ndarray): Matrix to threshold.
        theta (int): Number of singular values to keep.
        threshold (float, optional): Threshold value. Defaults to 0.1.

    Returns:
        np.ndarray: Thresholded matrix."""
    U, s, V = np.linalg.svd(M, full_matrices=False)
    idx = np.sum(s > threshold)
    vec = s.copy()
    vec[theta:idx] -= threshold
    return U[:, :idx] @ np.diag(vec[:idx]) @ V[:idx, :]