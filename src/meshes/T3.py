"""
T3 Mesh Generators
==================

Construct triangular meshes from the existing structured
Q4 meshes.

Each Q4 element is divided into two T3 elements.

Node numbering remains 1-based.
"""

import numpy as np

from src.meshes.Q4 import (
    generate_l_shaped_q4_mesh,
    generate_square_hole_q4_mesh,
)

from src.meshes.Q4 import generate_unit_square_q4_mesh


# ============================================================
# 1. Convert Q4 connectivity to T3 connectivity
# ============================================================

def convert_q4_to_t3(Coord, Connectivity):
    """
    Divide every Q4 element into two T3 elements.

    Q4 ordering:
        [n1, n2, n3, n4]

    T3 ordering:
        [n1, n2, n3]
        [n1, n3, n4]

    Returns
    -------
    Coord : np.ndarray
        Original nodal coordinates.

    T3_Connectivity : np.ndarray
        Triangular element connectivity (1-based).
    """

    Connectivity = np.asarray(Connectivity, dtype=int)

    if Connectivity.ndim != 2 or Connectivity.shape[1] != 4:
        raise ValueError("Expected Q4 connectivity with four nodes per element.")

    T3_Connectivity = np.empty(
        (2 * Connectivity.shape[0], 3),
        dtype=int,
    )

    T3_Connectivity[0::2] = Connectivity[:, [0, 1, 2]]
    T3_Connectivity[1::2] = Connectivity[:, [0, 2, 3]]

    return Coord, T3_Connectivity


# ============================================================
# 2. L-shaped domain
# ============================================================

def generate_l_shaped_t3_mesh(n):
    """
    Generate a structured T3 mesh for the L-shaped domain.
    """

    Coord, Connectivity = generate_l_shaped_q4_mesh(n)

    return convert_q4_to_t3(Coord, Connectivity)


# ============================================================
# 3. Square-hole domain
# ============================================================

def generate_square_hole_t3_mesh(n):
    """
    Generate a structured T3 mesh for the square-hole domain.

    n must be divisible by 9.
    """

    Coord, Connectivity = generate_square_hole_q4_mesh(n)

    return convert_q4_to_t3(Coord, Connectivity)

# ============================================================
# 4. Unit-square domain
# ============================================================

def generate_unit_square_t3_mesh(n):
    """
    Generate a structured T3 mesh for the unit square.

    Domain:
        [0, 1] x [0, 1]

    Each Q4 element is divided into two T3 elements.

    Parameters
    ----------
    n : int
        Number of subdivisions along each direction.

    Returns
    -------
    Coord : np.ndarray
        Nodal coordinates.

    Connectivity : np.ndarray
        T3 element connectivity (1-based indexing).
    """

    Coord, Connectivity = generate_unit_square_q4_mesh(n)

    return convert_q4_to_t3(Coord, Connectivity)