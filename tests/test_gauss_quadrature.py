import numpy as np
import pytest

from src.fem.gauss_quadrature import GaussPoints, Gauss_3D


def test_gauss_1d_weights_sum_to_2():
    r, w = GaussPoints(1, "ignored", 8)

    assert np.isclose(np.sum(w), 2.0)
    assert np.isclose(np.sum(r), 0.0)


def test_gauss_2d_q4_weights_sum_to_4():
    r, w = GaussPoints(2, "q4", 8)

    assert np.isclose(np.sum(w), 4.0)
    assert np.isclose(np.sum(r[:, 0]), 0.0)
    assert np.isclose(np.sum(r[:, 1]), 0.0)


def test_gauss_2d_t3_weights_sum_to_half():
    r, w = GaussPoints(2, "t3", 1)

    assert np.isclose(np.sum(w), 0.5)
    assert r.shape == (1, 2)


def test_gauss_3d_b8_weights_sum_to_8():
    r, w = GaussPoints(3, "b8", 2)

    assert np.isclose(np.sum(w), 8.0)
    assert r.shape == (8, 3)


def test_gauss_3d_tet4_weights_sum_to_one_sixth():
    r, w = GaussPoints(3, "tet4", 1)

    assert np.isclose(np.sum(w), 1.0 / 6.0)


def test_invalid_dimension_raises_error():
    with pytest.raises(ValueError):
        GaussPoints(0, "q4", 2)


def test_invalid_element_type_raises_error():
    with pytest.raises(ValueError):
        GaussPoints(2, "Nakshatrala", 2)


def test_invalid_ngpts_type_raises_error():
    with pytest.raises(TypeError):
        GaussPoints(1, "ignored", "a")


def test_gauss_3d_docstring_exists():
    assert Gauss_3D.__doc__ is not None