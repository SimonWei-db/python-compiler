
import numpy as np


# dense matrix vector multiplication
def dense_mv(A, x):
    """
    Perform a dense matrix-vector multiplication.

    Args:
        A: A numpy array representing the matrix.
        x: A numpy array representing the vector.

    Returns:
        The result of the matrix-vector multiplication.
    """
    y = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            y[i] = A[i, j] * x[j]
    return y


def matmul(A, B):
    """
    Perform a matrix multiplication.

    Args:
        A: A numpy array representing the first matrix.
        B: A numpy array representing the second matrix.

    Returns:
        The result of the matrix multiplication.
    """
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C

