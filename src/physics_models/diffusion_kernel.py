"""
Finite Element Diffusion Kernel
Functions: Diffusion matrix, Volumetric source, Local stiffness matrix
"""

# Importing libraries

import numpy as np
from typing import Tuple,  Dict, Any
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import meshio


from src.fem.gauss_quadrature import GaussPoints  
from src.fem.shape_functions import ShapeFunctions

# Diffusion matrix

def Get_DMat(diffusivity_function: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """
    Construct and return the diffusivity matrix/value at a given spatial point.

    Parameters
    ----------
    diffusivity_function : Dict[str, Any]
        Dictionary describing the diffusivity. Must contain a key "type" that
        specifies how to build D. Other required keys depend on the type:
        - "scalar"           : no extra keys, returns 1.0
        - "user scalar"      : needs "d0" (float or callable)
        - "isotropic"        : needs "d0" (float or callable), returns d0 * I
        - "diagonal"         : needs "d1", "d2" (float or callable)
        - "rotation"         : needs "theta", "d1", "d2" (float or callable);
                               builds D = R diag(d1, d2) R^T
        - "general"          : needs "D11", "D12", "D21", "D22" (float or callable)
        For callable values, the callable is evaluated at x.
    x : np.ndarray
        Spatial point at which to evaluate the diffusivity parameters. Passed
        to any callables in `diffusivity_function`.

    Returns
    -------
    np.ndarray
        Diffusivity as either a scalar (0D) or a 2x2 matrix, depending on the type.

    Raises
    ------
    KeyError
        If a required parameter for the chosen type is missing.
    ValueError
        If the provided type is not supported.
    """
    # Normalize the type string
    dtype = str(diffusivity_function.get("type")).lower()

    def get_value(key: str) -> float:
        """
        Fetch a parameter from the dict; if it's callable, evaluate at x.
        Raises if the key is missing.
        """
        val = diffusivity_function.get(key)
        if callable(val):
            return float(val(x))
        elif val is not None:
            return float(val)
        else:
            raise KeyError(f"Missing required parameter '{key}' for type '{dtype}'")

    # Handle each supported type
    if dtype == "scalar":
        return 1.0

    elif dtype == "user scalar":
        d0 = get_value("d0")
        return d0
    
    elif dtype == "isotropic":
        d0 = get_value("d0")
        return d0 * np.eye(2, dtype=float)
    
    elif dtype == "diagonal":
        d1 = get_value("d1")
        d2 = get_value("d2")
        return np.diag([d1, d2]).astype(float)

    elif dtype == "rotation":
        theta = get_value("theta")
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=float)
        d1 = get_value("d1")
        d2 = get_value("d2")
        Ddiag = np.diag([d1, d2]).astype(float)
        return R @ Ddiag @ R.T

    elif dtype == "general":
        D11 = get_value("D11")
        D12 = get_value("D12")
        D21 = get_value("D21")
        D22 = get_value("D22")
        return np.array([[D11, D12],
                         [D21, D22]], dtype=float)

    else:
        raise ValueError(f"Unsupported type: {dtype}.")


# Load type
def Get_VolumetricSource(load_type: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    """
    Compute the volumetric source term at a given spatial point.

    Parameters
    ----------
    load_type : Dict[str, Any]
        Dictionary describing the source term. Must contain a key "type"
        specifying how to compute the value. Supported types:
        - "constant" or "user scalar": needs "f0" (float or callable)
        - "linear": needs "a" (array-like or callable) and "b" (float or callable),
                    computes f(x) = a·x + b
        - "custom": needs "func" (callable or constant), directly evaluated at x
    x : np.ndarray
        Spatial point at which to evaluate the volumetric source term.
        Passed to any callable parameters in `load_type`.

    Returns
    -------
    np.ndarray
        Scalar volumetric source value evaluated at the given spatial point.

    Raises
    ------
    KeyError
        If a required parameter for the given type is missing.
    ValueError
        If the specified load type is unsupported or dimensions mismatch.
    """
    dtype = str(load_type.get("type")).lower()

    def get_value(key: str) -> float:
        """
        Fetch parameter from the dictionary; if callable, evaluate at x.
        """
        val = load_type.get(key)
        if callable(val):
            return float(val(x))
        elif val is not None:
            return float(val)
        else:
            raise KeyError(f"Missing required parameter '{key}' for source type '{dtype}'")

    # Source types
    if dtype in ("constant", "user scalar"):
        # Constant volumetric source
        f0 = get_value("f0")
        return f0

    elif dtype == "linear":
        # Linear function of position: f(x) = a·x + b
        a = get_value("a")
        b = get_value("b")
        a = np.array(a, dtype=float).flatten()
        x = np.array(x, dtype=float).flatten()

        if a.size != x.size:
            raise ValueError(f"'a' and 'x' must have same dimensions (got {a.size} and {x.size}).")

        return float(np.dot(a, x) + b)

    elif dtype == "custom":
        # User-specified function or constant
        func = load_type.get("func")
        if callable(func):
            return float(func(x))
        elif func is not None:
            return float(func)
        else:
            raise KeyError(f"Missing required parameter 'func' for custom source type.")
        
    elif dtype == "function":
        # just a nicer name than "custom"
        f = load_type.get("f")
        if f is None:
            raise KeyError("Missing 'f' for function source.")
        return float(f(x))

    else:
        raise ValueError(f"Unsupported load type: '{dtype}'")
    

    # Calculate Local Matrix

def CalculateLocalMatrix(diffusivity_function: Dict[str, Any], 
                             dofs_per_node: int, 
                             EleNodes: np.ndarray,
                             EleType: str,
                             load_type: Dict[str, Any],
                             r: np.ndarray,
                             w: np.ndarray,
                             xcap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assemble the element-level stiffness (diffusion) matrix and load vector
    for a single finite element using Gaussian quadrature.

    Parameters
    ----------
    diffusivity_function : Dict[str, Any]
        Description of the diffusivity; passed to `Get_DMat` to obtain D(x).
    dofs_per_node : int
        Number of degrees of freedom per node (1 for scalar problems).
    EleNodes : np.ndarray
        Node indices of the current element (not directly used here but kept
        for interface completeness).
    EleType : str
        Element type identifier, passed to `ShapeFunctions` (e.g. "L2", "Q4").
    load_type : Dict[str, Any]
        Description of the volumetric source; passed to `Get_VolumetricSource`.
    r : np.ndarray
        Gauss point locations in the reference element (1D, 2D, or 3D array).
    w : np.ndarray
        Corresponding Gauss weights.
    xCap : np.ndarray
        Nodal coordinates of the element in physical space, shape (n_nodes, dim).

    Returns
    -------
    klocal : np.ndarray
        Local element stiffness/diffusion matrix of shape
        (n_nodes * dofs_per_node, n_nodes * dofs_per_node).
    rlocal : np.ndarray
        Local element load vector of shape (n_nodes * dofs_per_node, 1).

    """
        
    #Initialization
    NodesPerEle = len(EleNodes)
    KLocal = np.zeros((NodesPerEle*dofs_per_node, NodesPerEle*dofs_per_node))
    RLocal = np.zeros((NodesPerEle*dofs_per_node, 1))

    # Loop over Gauss Points
    for gpt in range(len(w)):
        zeta = np.array([r[gpt, :]])

    # N and DN from shape functions

        N, DN = ShapeFunctions(EleType, zeta)
 


        x = xcap.T @ N.T 
       
        J = xcap.T @ DN 
       
        B = DN @ np.linalg.inv(J)
        
        detJ = np.linalg.det(J)

        # Diffusivity at current point
        D = Get_DMat(diffusivity_function, x)

        # Local stiffness matrix
        KLocal = KLocal + w[gpt] * (B @ D @ B.T)* detJ
        

        #Source term 
        f = Get_VolumetricSource(load_type, x)

         #Local Load vector
        RLocal = RLocal+ w[gpt] * (N.T * f)* detJ

    return KLocal, RLocal



