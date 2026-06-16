import numpy as np

from src.fem.gauss_quadrature import GaussPoints
from src.fem.shape_functions import ShapeFunctions


def Calculate_Error(Connectivity, Coord, EleType, NGPTS, U, exact_solution, exact_gradient):
    """
    Returns (L2_error, H1_seminorm_error)
    exact_solution(x,y) -> scalar
    exact_gradient(x,y) -> array_like (length >= dim)
    """

    # Work on copies to avoid mutating caller arrays
    Connectivity = np.array(Connectivity, copy=True)
    Coord = np.asarray(Coord)
    U = np.asarray(U).reshape(-1)

    # Convert to zero-based indexing if mesh uses 1-based indices
    if Connectivity.min() >= 1:
        Connectivity = Connectivity - 1

    Nele = Connectivity.shape[0]
    dim = Coord.shape[1]

    # Make sure Connectivity dtype is integer for indexing
    Connectivity = Connectivity.astype(int)

    error_in_L2 = 0.0
    error_in_H1 = 0.0

    # Get Gauss points and weights
    r, w = GaussPoints(dim, EleType, NGPTS)

    # If r is shape (npts, dim) and w is length npts, iterate that way
    num_gpts = len(w)

    for ele in range(Nele):
        EleNodes = Connectivity[ele, :].astype(int)   # indices
        uCap = U[EleNodes].reshape(-1, 1)             # (nnode,1)
        xCap = Coord[EleNodes, :]                     # (nnode, dim)

        for gpt in range(num_gpts):
            zeta = np.asarray(r[gpt]).reshape(-1, 1)   # (dim,1) or (ndim,1)
            N, DN = ShapeFunctions(EleType, zeta)     # expect N: (nnode,) or (nnode,1); DN: (nnode, dim)

            N = np.asarray(N).reshape(-1, 1)           # (nnode,1)
            DN = np.asarray(DN).reshape(N.shape[0], dim)  # (nnode,dim)

            # Mapping and Jacobian
            # X (physical coordinates of gauss point) is N^T @ xCap -> shape (1,dim)
            X = (N.T @ xCap).ravel()                   # (dim,)
            J = DN.T @ xCap                            # (dim,dim)
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(f"Non-positive Jacobian determinant {detJ} at element {ele}, gpt {gpt}")
            invJ = np.linalg.inv(J)

            # B matrix: gradient of shape functions w.r.t physical coords
            # DN.T: (dim, nnode), invJ: (dim,dim) -> invJ @ DN.T: (dim, nnode), transpose -> (nnode, dim)
            B = (invJ @ DN.T).T   # (nnode, dim)

            # Interpolated u and gradient
            u_val = (N.T @ uCap).item()
            gradu = (B.T @ uCap).ravel()   # (dim,)

            # Exact
            uExact = float(exact_solution(*X))
            duExact = np.asarray(exact_gradient(*X)).ravel()

            # Quadrature weight
            weight = w[gpt]

            error_in_L2 += weight * (u_val - uExact) ** 2 * detJ
            delGrad = gradu - duExact[:dim]
            error_in_H1 += weight * (delGrad @ delGrad) * detJ

    return float(np.sqrt(error_in_L2)), float(np.sqrt(error_in_H1))
