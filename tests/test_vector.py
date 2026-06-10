import numpy as np

from src.fem.Vector import Vector


def test_vector_column_wise_stacking():
    A = np.array([
        [1, 2],
        [3, 4],
    ])

    vecA = Vector(A)

    expected = np.array([
        [1],
        [3],
        [2],
        [4],
    ])

    assert np.array_equal(vecA, expected)


def test_vector_shape():
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])

    vecA = Vector(A)

    assert vecA.shape == (6, 1)