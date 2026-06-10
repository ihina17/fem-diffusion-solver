import numpy as np
import pytest

from src.fem.create_id import create_id_matrix


def test_create_id_matrix_single_dof():
    constraints = np.array([
        [1, 1, 0.0],
        [4, 1, 0.0],
    ])

    Global_ID, NEqns = create_id_matrix(
        constraints=constraints,
        dofs_per_nodes=1,
        NumNodes=4,
    )

    expected_Global_ID = np.array([
        [-1],
        [ 1],
        [ 2],
        [-2],
    ])

    assert np.array_equal(Global_ID, expected_Global_ID)
    assert NEqns == 2


def test_create_id_matrix_two_dofs():
    constraints = np.array([
        [1, 1, 0.0],
        [1, 2, 5.0],
    ])

    Global_ID, NEqns = create_id_matrix(
        constraints=constraints,
        dofs_per_nodes=2,
        NumNodes=3,
    )

    expected_Global_ID = np.array([
        [-1, -2],
        [ 1,  2],
        [ 3,  4],
    ])

    assert np.array_equal(Global_ID, expected_Global_ID)
    assert NEqns == 4


def test_create_id_matrix_no_constraints():
    constraints = np.empty((0, 3))

    Global_ID, NEqns = create_id_matrix(
        constraints=constraints,
        dofs_per_nodes=1,
        NumNodes=3,
    )

    expected_Global_ID = np.array([
        [1],
        [2],
        [3],
    ])

    assert np.array_equal(Global_ID, expected_Global_ID)
    assert NEqns == 3


def test_create_id_matrix_preserves_constraint_order():
    constraints = np.array([
        [3, 1, 0.0],
        [1, 1, 0.0],
    ])

    Global_ID, NEqns = create_id_matrix(
        constraints=constraints,
        dofs_per_nodes=1,
        NumNodes=4,
    )

    expected_Global_ID = np.array([
        [-2],
        [ 1],
        [-1],
        [ 2],
    ])

    assert np.array_equal(Global_ID, expected_Global_ID)
    assert NEqns == 2