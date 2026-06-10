"""
Finite Element Framework - Vectorize a matrix

"""

# Importing Libraries
import numpy as np

def Vector(A: np.ndarray) -> np.ndarray:
    """
    Stack columns of a matrix into a single column vector.

    Parameters
    ----------
    A : np.ndarray
        Input matrix (m x n)

    Returns
    ----------
    vecA : np.ndarray
        Column vector of size (m*n, 1) containing elements of A stacked column-wise
    """
    # Ensure input is a numpy array
    A = np.array(A)

    # Stack columns into a single column vector
    vecA = A.flatten(order='F')  # 'F' means column-major order

    # Convert to column vector shape (m*n, 1)
    vecA = vecA.reshape(-1, 1)

    return vecA