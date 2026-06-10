"""
Finite Element Framework - Shape Functions

Master function:
    ShapeFunctions

Supported elements:
    1D: L2, L3
    2D: Q4, Q8, T3, T6
    3D: B8, Tet4, W6
"""

import numpy as np
from typing import Tuple


def _as_vector(zeta: np.ndarray, expected_size: int, element_name: str) -> np.ndarray:
    """
    Convert local coordinate input into a flat NumPy vector.

    Accepts:
        np.array([xi, eta])
        np.array([[xi, eta]])
    """
    zeta = np.asarray(zeta, dtype=float).ravel()

    if zeta.size != expected_size:
        raise ValueError(
            f"{element_name} needs {expected_size} local coordinate(s), "
            f"but got {zeta.size}."
        )

    return zeta


def ShapeFunctions(EleType: str, zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return shape functions and local derivatives for a finite element.

    Parameters
    ----------
    EleType : str
        Element type: L2, L3, Q4, Q8, T3, T6, B8, Tet4, W6.

    zeta : np.ndarray
        Local/reference coordinates.

    Returns
    -------
    N : np.ndarray
        Shape function values with shape (1, number_of_nodes).

    DN : np.ndarray
        Derivatives with respect to local coordinates.
        Shape is (number_of_nodes, dimension).
    """
    if not isinstance(EleType, str):
        raise TypeError("EleType must be a string.")

    key = EleType.strip().lower()

    if key == "l2":
        return L2(zeta)
    if key == "l3":
        return L3(zeta)
    if key == "q4":
        return Q4(zeta)
    if key == "q8":
        return Q8(zeta)
    if key == "t3":
        return T3(zeta)
    if key == "t6":
        return T6(zeta)
    if key == "b8":
        return B8(zeta)
    if key == "tet4":
        return Tet4(zeta)
    if key == "w6":
        return W6(zeta)

    raise ValueError(f"Unknown element type '{EleType}'.")


def L2(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-node line element.
    Reference domain: -1 <= xi <= 1
    """
    xi = _as_vector(zeta, 1, "L2")[0]

    N = np.array([
        [0.5 * (1.0 - xi), 0.5 * (1.0 + xi)]
    ])

    DN = np.array([
        [-0.5],
        [ 0.5],
    ])

    return N, DN


def L3(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Three-node quadratic line element.
    Reference domain: -1 <= xi <= 1
    """
    xi = _as_vector(zeta, 1, "L3")[0]

    N1 = 0.5 * xi * (xi - 1.0)
    N2 = 0.5 * xi * (xi + 1.0)
    N3 = 1.0 - xi**2

    N = np.array([[N1, N2, N3]])

    DN = np.array([
        [xi - 0.5],
        [xi + 0.5],
        [-2.0 * xi],
    ])

    return N, DN


def Q4(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Four-node quadrilateral element.
    Reference domain: -1 <= xi <= 1, -1 <= eta <= 1
    """
    xi, eta = _as_vector(zeta, 2, "Q4")

    N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta)

    N = np.array([[N1, N2, N3, N4]])

    DN = np.array([
        [-0.25 * (1.0 - eta), -0.25 * (1.0 - xi)],
        [ 0.25 * (1.0 - eta), -0.25 * (1.0 + xi)],
        [ 0.25 * (1.0 + eta),  0.25 * (1.0 + xi)],
        [-0.25 * (1.0 + eta),  0.25 * (1.0 - xi)],
    ])

    return N, DN


def Q8(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eight-node serendipity quadrilateral element.
    Reference domain: -1 <= xi <= 1, -1 <= eta <= 1
    """
    xi, eta = _as_vector(zeta, 2, "Q8")

    N1 = 0.25 * (1.0 - xi) * (1.0 - eta) * (-1.0 - xi - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta) * ( xi - eta - 1.0)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta) * ( xi + eta - 1.0)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0)

    N5 = 0.5 * (1.0 - xi**2) * (1.0 - eta)
    N6 = 0.5 * (1.0 + xi) * (1.0 - eta**2)
    N7 = 0.5 * (1.0 - xi**2) * (1.0 + eta)
    N8 = 0.5 * (1.0 - xi) * (1.0 - eta**2)

    N = np.array([[N1, N2, N3, N4, N5, N6, N7, N8]])

    DN = np.array([
        [0.25 * (1.0 - eta) * (2.0 * xi + eta),
         0.25 * (1.0 - xi) * (xi + 2.0 * eta)],

        [0.25 * (1.0 - eta) * (2.0 * xi - eta),
        -0.25 * (1.0 + xi) * (xi - 2.0 * eta)],

        [0.25 * (1.0 + eta) * (2.0 * xi + eta),
         0.25 * (1.0 + xi) * (xi + 2.0 * eta)],

        [0.25 * (1.0 + eta) * (2.0 * xi - eta),
         0.25 * (1.0 - xi) * (-xi + 2.0 * eta)],

        [-xi * (1.0 - eta),
         -0.5 * (1.0 - xi**2)],

        [0.5 * (1.0 - eta**2),
         -(1.0 + xi) * eta],

        [-xi * (1.0 + eta),
         0.5 * (1.0 - xi**2)],

        [-0.5 * (1.0 - eta**2),
         -(1.0 - xi) * eta],
    ])

    return N, DN


def T3(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Three-node triangular element.
    Reference triangle: (0,0), (1,0), (0,1)
    """
    r, s = _as_vector(zeta, 2, "T3")

    N1 = 1.0 - r - s
    N2 = r
    N3 = s

    N = np.array([[N1, N2, N3]])

    DN = np.array([
        [-1.0, -1.0],
        [ 1.0,  0.0],
        [ 0.0,  1.0],
    ])

    return N, DN


def T6(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Six-node quadratic triangular element.
    Reference triangle: (0,0), (1,0), (0,1)
    """
    r, s = _as_vector(zeta, 2, "T6")

    L1 = 1.0 - r - s
    L2 = r
    L3 = s

    N1 = L1 * (2.0 * L1 - 1.0)
    N2 = L2 * (2.0 * L2 - 1.0)
    N3 = L3 * (2.0 * L3 - 1.0)
    N4 = 4.0 * L1 * L2
    N5 = 4.0 * L2 * L3
    N6 = 4.0 * L3 * L1

    N = np.array([[N1, N2, N3, N4, N5, N6]])

    DN = np.array([
        [-3.0 + 4.0 * r + 4.0 * s, -3.0 + 4.0 * r + 4.0 * s],
        [ 4.0 * r - 1.0,             0.0],
        [ 0.0,                       4.0 * s - 1.0],
        [ 4.0 * (1.0 - 2.0 * r - s), -4.0 * r],
        [ 4.0 * s,                   4.0 * r],
        [-4.0 * s,                   4.0 * (1.0 - r - 2.0 * s)],
    ])

    return N, DN


def B8(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eight-node brick/hexahedral element.
    Reference domain: -1 <= xi, eta, mu <= 1
    """
    xi, eta, mu = _as_vector(zeta, 3, "B8")

    N1 = 0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - mu)
    N2 = 0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - mu)
    N3 = 0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - mu)
    N4 = 0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - mu)
    N5 = 0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + mu)
    N6 = 0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + mu)
    N7 = 0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + mu)
    N8 = 0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + mu)

    N = np.array([[N1, N2, N3, N4, N5, N6, N7, N8]])

    DN = np.array([
        [-0.125 * (1.0 - eta) * (1.0 - mu),
         -0.125 * (1.0 - xi) * (1.0 - mu),
         -0.125 * (1.0 - xi) * (1.0 - eta)],

        [ 0.125 * (1.0 - eta) * (1.0 - mu),
         -0.125 * (1.0 + xi) * (1.0 - mu),
         -0.125 * (1.0 + xi) * (1.0 - eta)],

        [ 0.125 * (1.0 + eta) * (1.0 - mu),
          0.125 * (1.0 + xi) * (1.0 - mu),
         -0.125 * (1.0 + xi) * (1.0 + eta)],

        [-0.125 * (1.0 + eta) * (1.0 - mu),
          0.125 * (1.0 - xi) * (1.0 - mu),
         -0.125 * (1.0 - xi) * (1.0 + eta)],

        [-0.125 * (1.0 - eta) * (1.0 + mu),
         -0.125 * (1.0 - xi) * (1.0 + mu),
          0.125 * (1.0 - xi) * (1.0 - eta)],

        [ 0.125 * (1.0 - eta) * (1.0 + mu),
         -0.125 * (1.0 + xi) * (1.0 + mu),
          0.125 * (1.0 + xi) * (1.0 - eta)],

        [ 0.125 * (1.0 + eta) * (1.0 + mu),
          0.125 * (1.0 + xi) * (1.0 + mu),
          0.125 * (1.0 + xi) * (1.0 + eta)],

        [-0.125 * (1.0 + eta) * (1.0 + mu),
          0.125 * (1.0 - xi) * (1.0 + mu),
          0.125 * (1.0 - xi) * (1.0 + eta)],
    ])

    return N, DN


def Tet4(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Four-node tetrahedral element.
    Reference tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    """
    r, s, t = _as_vector(zeta, 3, "Tet4")

    N1 = 1.0 - r - s - t
    N2 = r
    N3 = s
    N4 = t

    N = np.array([[N1, N2, N3, N4]])

    DN = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0],
    ])

    return N, DN


def W6(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Six-node wedge/prism element.
    Local coordinates: r, s, t
    """
    r, s, t = _as_vector(zeta, 3, "W6")

    N1 = 0.5 * (1.0 - t) * (1.0 - r - s)
    N2 = 0.5 * (1.0 - t) * r
    N3 = 0.5 * (1.0 - t) * s
    N4 = 0.5 * (1.0 + t) * (1.0 - r - s)
    N5 = 0.5 * (1.0 + t) * r
    N6 = 0.5 * (1.0 + t) * s

    N = np.array([[N1, N2, N3, N4, N5, N6]])

    DN = np.array([
        [-0.5 * (1.0 - t), -0.5 * (1.0 - t), -0.5 * (1.0 - r - s)],
        [ 0.5 * (1.0 - t),  0.0,              -0.5 * r],
        [ 0.0,               0.5 * (1.0 - t), -0.5 * s],
        [-0.5 * (1.0 + t), -0.5 * (1.0 + t),  0.5 * (1.0 - r - s)],
        [ 0.5 * (1.0 + t),  0.0,               0.5 * r],
        [ 0.0,               0.5 * (1.0 + t),  0.5 * s],
    ])

    return N, DN


def per_eletype(etype: str, z: np.ndarray, tol: float = 1e-12) -> None:
    """
    Print a basic partition-of-unity check for one element type.
    """
    N, DN = ShapeFunctions(etype, z)

    sum_N = float(np.sum(N))
    sum_DN = np.sum(DN, axis=0)

    print(f"Testing element type: {etype}")
    print(f"zeta: {z}")
    print(f"N shape: {N.shape}")
    print(f"DN shape: {DN.shape}")
    print(f"sum(N): {sum_N}")
    print(f"column sums of DN: {sum_DN}")

    if abs(sum_N - 1.0) > tol:
        print("Incorrect N: shape functions do not sum to 1.")
    elif not np.all(np.abs(sum_DN) < tol):
        print("Incorrect DN: derivative columns do not sum to 0.")
    else:
        print("Shape function check passed.")


# Lowercase alias
shape_functions = ShapeFunctions
