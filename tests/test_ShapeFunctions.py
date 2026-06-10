import numpy as np
import pytest

from src.fem.shape_functions import ShapeFunctions


def check_partition_of_unity(element_type, zeta, expected_nodes, expected_dim):
    N, DN = ShapeFunctions(element_type, np.array(zeta, dtype=float))

    assert N.shape == (1, expected_nodes)
    assert DN.shape == (expected_nodes, expected_dim)

    assert np.isclose(np.sum(N), 1.0)
    assert np.allclose(np.sum(DN, axis=0), np.zeros(expected_dim))


def test_l2_shape_functions():
    check_partition_of_unity("L2", [0.0], expected_nodes=2, expected_dim=1)


def test_l3_shape_functions():
    check_partition_of_unity("L3", [0.0], expected_nodes=3, expected_dim=1)


def test_q4_shape_functions():
    check_partition_of_unity("Q4", [0.0, 0.0], expected_nodes=4, expected_dim=2)


def test_q8_shape_functions():
    check_partition_of_unity("Q8", [0.0, 0.0], expected_nodes=8, expected_dim=2)


def test_t3_shape_functions():
    check_partition_of_unity("T3", [1.0 / 3.0, 1.0 / 3.0], expected_nodes=3, expected_dim=2)


def test_t6_shape_functions():
    check_partition_of_unity("T6", [1.0 / 3.0, 1.0 / 3.0], expected_nodes=6, expected_dim=2)


def test_b8_shape_functions():
    check_partition_of_unity("B8", [0.0, 0.0, 0.0], expected_nodes=8, expected_dim=3)


def test_tet4_shape_functions():
    check_partition_of_unity("Tet4", [0.25, 0.25, 0.25], expected_nodes=4, expected_dim=3)


def test_w6_shape_functions():
    check_partition_of_unity("W6", [1.0 / 3.0, 1.0 / 3.0, 0.0], expected_nodes=6, expected_dim=3)


def test_q4_center_values():
    N, DN = ShapeFunctions("Q4", np.array([0.0, 0.0]))

    assert np.allclose(N, np.array([[0.25, 0.25, 0.25, 0.25]]))
    assert DN.shape == (4, 2)


def test_b8_center_values():
    N, DN = ShapeFunctions("B8", np.array([0.0, 0.0, 0.0]))

    assert np.allclose(
        N,
        np.array([[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]])
    )
    assert DN.shape == (8, 3)


def test_invalid_element_type_raises_error():
    with pytest.raises(ValueError):
        ShapeFunctions("BAD", np.array([0.0, 0.0]))


def test_wrong_number_of_coordinates_raises_error():
    with pytest.raises(ValueError):
        ShapeFunctions("Q4", np.array([0.0]))