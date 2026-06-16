"""
Error analysis utilities for steady-state diffusion FEM problems.

This file computes:
    1. L2 error norm
    2. H1 seminorm error

The function compares the FEM numerical solution against an analytical exact
solution.
"""

import numpy as np
from typing import Tuple

from src.fem.gauss_quadrature import GaussPoints
from src.fem.shape_functions import ShapeFunctions


def Calculate_Error(Connectivity: np.ndarray,
                    Coord: np.ndarray,
                    EleType: str,
                    NGPTS: int,
                    U: np.ndarray,
                    exact_solution,
                    exact_gradient) -> Tuple[float, float]:
    """
    Compute the L2 error and H1-seminorm error of the FEM solution.

    The L2 error measures the difference between the numerical solution and
    the exact solution:

        ||u_h - u_exact||_L2

    The H1 seminorm error measures the difference between the gradients:

        ||grad(u_h) - grad(u_exact)||_L2

    Parameters
    ----------
    Connectivity : np.ndarray
        Element connectivity array of shape (Nele, NodesPerEle).

        In this project, connectivity is usually 1-based:

            [[1, 2, 5, 4],
             [2, 3, 6, 5]]

        The function automatically converts it to 0-based indexing.

    Coord : np.ndarray
        Coordinate matrix of shape (NumNodes, dim).

        Example for 2D:

            [[x1, y1],
             [x2, y2],
             ...]

    EleType : str
        Element type.

        Example:

            "Q4"

    NGPTS : int
        Number of Gauss points per direction.

        For Q4, usually:

            NGPTS = 2

    U : np.ndarray
        Full nodal FEM solution of shape (NumNodes, 1) or (NumNodes,).

    exact_solution : callable
        Analytical exact solution function.

        Example:

            def exact_solution(x, y):
                return x + y

    exact_gradient : callable
        Analytical gradient of exact solution.

        Example:

            def exact_gradient(x, y):
                return np.array([1.0, 1.0])

    Returns
    -------
    error_in_L2 : float
        L2 norm of the solution error.

    error_in_H1_seminorm : float
        H1 seminorm of the gradient error.
    """

    # ------------------------------------------------------------
    # Initialize error values
    # ------------------------------------------------------------
    error_in_L2 = 0.0
    error_in_H1 = 0.0

    # ------------------------------------------------------------
    # Convert connectivity to integer array
    # ------------------------------------------------------------
    Connectivity = np.asarray(Connectivity, dtype=int).copy()

    # Your project usually uses 1-based node numbering.
    # Python needs 0-based indexing.
    if Connectivity.min() >= 1:
        Connectivity = Connectivity - 1

    # ------------------------------------------------------------
    # Basic mesh data
    # ------------------------------------------------------------
    Nele = Connectivity.shape[0]
    dim = Coord.shape[1]

    # Flatten U so U[node_id] gives the scalar nodal value.
    U = np.asarray(U, dtype=float).reshape(-1)

    # ------------------------------------------------------------
    # Get Gauss points and weights
    # ------------------------------------------------------------
    r, w = GaussPoints(dim, EleType, NGPTS)

    # ------------------------------------------------------------
    # Loop over all elements
    # ------------------------------------------------------------
    for ele in range(Nele):

        # Node numbers for current element, already 0-based
        EleNodes = Connectivity[ele, :]

        # Coordinates of current element nodes
        xCap = Coord[EleNodes, :]

        # Nodal solution values of current element
        uCap = U[EleNodes].reshape(-1, 1)

        # --------------------------------------------------------
        # Loop over Gauss points
        # --------------------------------------------------------
        for gpt in range(len(w)):

            # Natural coordinate of current Gauss point
            if r.ndim == 1:
                zeta = np.array([[r[gpt]]], dtype=float)
            else:
                zeta = r[gpt, :].reshape(1, -1)

            # Shape functions and natural derivatives
            N, DN = ShapeFunctions(EleType, zeta)

            # N shape should be (1, NodesPerEle)
            N = np.asarray(N, dtype=float).reshape(1, -1)

            # DN shape should be (NodesPerEle, dim)
            DN = np.asarray(DN, dtype=float)

            # ----------------------------------------------------
            # Map Gauss point from natural coordinates to x-y
            # ----------------------------------------------------
            x = (xCap.T @ N.T).reshape(-1)

            # ----------------------------------------------------
            # Jacobian matrix
            # ----------------------------------------------------
            J = xCap.T @ DN
            detJ = np.linalg.det(J)

            if detJ <= 0.0:
                raise ValueError(
                    f"Element {ele + 1} has non-positive detJ = {detJ}. "
                    "Check element node ordering."
                )

            invJ = np.linalg.inv(J)

            # ----------------------------------------------------
            # Gradient matrix
            # ----------------------------------------------------
            # DN contains derivatives with respect to natural coordinates.
            # B contains derivatives with respect to physical coordinates.
            B = DN @ invJ

            # ----------------------------------------------------
            # Numerical solution at Gauss point
            # ----------------------------------------------------
            u_num = float(N @ uCap)

            # Numerical gradient at Gauss point
            grad_u_num = (B.T @ uCap).reshape(-1)

            # ----------------------------------------------------
            # Exact solution and exact gradient at Gauss point
            # ----------------------------------------------------
            u_exact = float(exact_solution(*x))
            grad_u_exact = np.asarray(exact_gradient(*x), dtype=float).reshape(-1)

            # Use only the active spatial dimensions
            grad_u_exact = grad_u_exact[:dim]

            # ----------------------------------------------------
            # Add L2 error contribution
            # ----------------------------------------------------
            error_in_L2 += w[gpt] * (u_num - u_exact) ** 2 * detJ

            # ----------------------------------------------------
            # Add H1 seminorm error contribution
            # ----------------------------------------------------
            grad_error = grad_u_num - grad_u_exact
            error_in_H1 += w[gpt] * (grad_error @ grad_error) * detJ

    return np.sqrt(error_in_L2), np.sqrt(error_in_H1)

Lx = 2.0
Ly = 2.0

def exact_solution(x, y):
    return (2.0 / np.pi**2) * np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly)

def exact_gradient(x, y):
    dc_dx = (1.0 / np.pi) * np.cos(np.pi * x / Lx) * np.sin(np.pi * y / Ly)
    dc_dy = (1.0 / np.pi) * np.sin(np.pi * x / Lx) * np.cos(np.pi * y / Ly)
    return np.array([dc_dx, dc_dy])

L2_error, H1_error = Calculate_Error(
    Connectivity,
    Coord,
    "Q4",
    2,
    U,
    exact_solution,
    exact_gradient
)

print("L2 error =", L2_error)
print("H1 error =", H1_error)