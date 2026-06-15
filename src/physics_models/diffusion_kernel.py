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

#Assemble

def Assemble(dofs_per_node: int,
             EleNodes: np.ndarray,
             GlobalID: np.ndarray,
             KLocal: np.ndarray,
             RLocal: np.ndarray,
             K_FF: np.ndarray,
             K_FP: np.ndarray,
             K_PP: np.ndarray,
             R_F: np.ndarray,
             R_P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    

    """
    Assemble an element's local stiffness matrix and load vector into the
    partitioned global system (free–free, free–prescribed, and prescribed–prescribed
    blocks).

    Parameters
    ----------
    dofs_per_node : int
        Number of degrees of freedom per node.
    EleNodes : np.ndarray
        Array of node indices belonging to the current element.
    GlobalID : np.ndarray
        Mapping array of shape (n_nodes, dofs_per_node). Each entry defines
        the global equation number:
        - Positive values → free DOFs (active unknowns)
        - Negative values → prescribed DOFs (Dirichlet conditions)
    Klocal : np.ndarray
        Element stiffness matrix of shape (n_ele_dofs, n_ele_dofs).
    rlocal : np.ndarray
        Element load vector of shape (n_ele_dofs, 1).
    K_FF : np.ndarray
        Global stiffness matrix for free–free DOF interactions.
    K_FP : np.ndarray
        Global coupling matrix between free and prescribed DOFs.
    K_PP : np.ndarray
        Global stiffness matrix for prescribed–prescribed DOF interactions.
    R_F : np.ndarray
        Global right-hand-side vector for free DOFs.
    R_P : np.ndarray
        Global right-hand-side vector for prescribed DOFs.

    Returns
    -------
    K_FF : np.ndarray
        Updated global free–free stiffness matrix.
    K_FP : np.ndarray
        Updated global free–prescribed coupling matrix.
    K_PP : np.ndarray
        Updated global prescribed–prescribed stiffness matrix.
    R_F : np.ndarray
        Updated global right-hand-side vector for free DOFs.
    R_P : np.ndarray
        Updated global right-hand-side vector for prescribed DOFs.

    Notes
    -----
    - Negative entries in `GlobalID` correspond to prescribed DOFs.
    - In standard diffusion problems, only K_FF, K_FP, and R_F are required;
      K_PP and R_P are typically omitted for efficiency.
    - The routine supports extension to coupled or interface formulations
      where K_PP and R_P are needed explicitly.
    """

    Nodes_per_ele = len(EleNodes)

    #Initialization
    v_vector = np.zeros(Nodes_per_ele * dofs_per_node, dtype=int)

    for i in range(dofs_per_node):
        v_vector[i::dofs_per_node] = GlobalID[EleNodes - 1, i]

    # Assemble local matrices into global matrices

    # Assemble stiffness matrix

    for row in range(len(v_vector)):
        rowId = v_vector[row]
        for col in range(len(v_vector)):
            colId = v_vector[col]

            if rowId > 0 and colId > 0:
                K_FF[rowId - 1, colId - 1] += KLocal[row, col]
            elif rowId > 0 and colId < 0:
                K_FP[rowId - 1, -(colId) - 1] += KLocal[row, col]
            elif rowId < 0 and colId < 0:
                K_PP[-(rowId) - 1, -(colId) - 1] += KLocal[row, col]

        # Assemble load vector

        if rowId > 0:
            R_F[rowId - 1, 0] += RLocal[row, 0]
        else:
            R_P[-(rowId) - 1, 0] += RLocal[row, 0] 

    return K_FF, K_FP, K_PP, R_F, R_P


# Global matrices

def CalculateGlobalMatrices(connectivity: np.ndarray,
                            coord: np.ndarray,
                            diffusivity_function: Dict[str, Any],
                            dim: int,
                            dofs_per_node: int,
                            EleType: str,
                            GlobalId: np.ndarray,
                            load_type: Dict[str, Any],
                            NCons: int,
                            Nele: int,
                            NEqns: int,
                            NGPTS: int) -> Tuple[lil_matrix, lil_matrix, np.ndarray]:
    """
    Assemble the global stiffness (diffusion) matrices and global load vector
    by looping over all elements in the finite element mesh.

    Parameters
    ----------
    Conectivity : np.ndarray
        Element connectivity matrix of shape (Nele, Nodes_per_Ele) specifying
        the node indices for each element.
    Coord : np.ndarray
        Global nodal coordinates array of shape (Nnodes, dim).
    diffusivity_function : Dict[str, Any]
        Dictionary defining the diffusivity properties; passed to `Get_DMat`.
    dim : int
        Spatial dimension of the problem (1, 2, or 3).
    dofs_per_node : int
        Number of degrees of freedom per node.
    EleType : str
        Element type identifier (e.g., "L2", "Q4", etc.).
    GlobalID : np.ndarray
        Global degree-of-freedom mapping for each node and DOF.
        Positive entries correspond to free DOFs, negative to prescribed ones.
    load_type : Dict[str, Any]
        Dictionary describing the volumetric source term; passed to `Get_VolumetricSource`.
    NCons : int
        Number of prescribed (constrained) degrees of freedom.
    Nele : int
        Total number of elements in the mesh.
    NEqns : int
        Number of global free equations (size of the free DOF system).
    NGPTS : int
        Number of Gauss integration points per element.

    Returns
    -------
    K_FF : lil_matrix
        Global free–free stiffness matrix assembled from all elements.
    K_FP : lil_matrix
        Global free–prescribed stiffness coupling matrix.
    R_F : np.ndarray
        Global right-hand side vector for free DOFs.

    Notes
    -----
    - Uses `CalculateLocalMatrices` to compute element matrices and
      `Assemble` to add them into the global matrices.
    - Supports both isotropic and anisotropic diffusivity definitions.
    """

    #Initilaization
    K_FF = lil_matrix((NEqns, NEqns), dtype = float)
    K_FP = lil_matrix((NEqns, NCons), dtype = float)
    K_PP = lil_matrix((NCons, NCons), dtype = float)
    R_F = np.zeros((NEqns, 1), dtype = float)
    R_P = np.zeros((NCons, 1), dtype = float)
    
    NodesPerEle = connectivity.shape[1]

    # Get gauss points and weights (r and w) from gauss quadrature

    r, w = GaussPoints(dim, EleType, NGPTS)

    # Loop over all the elements

    for ele in range(Nele):

        # 1 based node numbers from mesh connectivity
        EleNodes_1based = connectivity[ele, :]

        # 0 based node numbers from mesh connectivity
        EleNodes_0based = EleNodes_1based - 1

        xCap = coord[EleNodes_0based, :]

        # local element matrix and vector
        Klocal, rlocal = CalculateLocalMatrix(
            diffusivity_function,
            dofs_per_node,
            EleNodes_1based,
            EleType,
            load_type,
            r,
            w,
            xCap
        )

        # assemble into global matrices
        K_FF, K_FP, K_PP, R_F, R_P = Assemble(
            dofs_per_node,
            EleNodes_1based,
            GlobalId,
            Klocal,
            rlocal,
            K_FF,
            K_FP,
            K_PP,
            R_F,
            R_P
        )

    return (
        K_FF.tocsr(),
        K_FP.tocsr(),
        K_PP.tocsr(),
        R_F,
        R_P,
    )

# Create constraint vector
def Create_ConstraintsVector(Constraints: np.ndarray, Global_ID: np.ndarray) -> np.ndarray:
    """
    Construct the prescribed solution vector U_P.

    Constraints should have rows:
        [node_number, dof_number, prescribed_value]

    Global_ID:
        positive value = free DOF
        negative value = prescribed DOF
    """

    NCons = Constraints.shape[0]
    U_P = np.zeros((NCons, 1), dtype=float)

    for i in range(NCons):
        node = int(Constraints[i, 0]) - 1
        dof = int(Constraints[i, 1]) - 1
        value = float(Constraints[i, 2])


        constraint_id = int(Global_ID[node, dof])



        U_P[abs(constraint_id) - 1, 0] = value

    return U_P


def SolveSystem(K_FF,
                K_FP,
                R_F: np.ndarray,
                Constraints: np.ndarray,
                Global_ID: np.ndarray):

    from scipy.sparse.linalg import spsolve

    U_P = Create_ConstraintsVector(Constraints, Global_ID)

    RHS = R_F - K_FP @ U_P

    U_F = spsolve(K_FF, RHS).reshape(-1, 1)

    U = PostProcessing(Global_ID, U_F, U_P)

    return U, U_F, U_P

# Post processing
def PostProcessing(Global_ID: np.ndarray,
                   U_F: np.ndarray,
                   U_P: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full global solution matrix U.

    Global_ID:
        positive value = index in U_F
        negative value = index in U_P
    """

    NumNodes, dofs_per_node = Global_ID.shape
    U = np.zeros((NumNodes, dofs_per_node), dtype=float)

    U_F = np.asarray(U_F).reshape(-1, 1)
    print(U_F)
    U_P = np.asarray(U_P).reshape(-1, 1)

    for node in range(NumNodes):
        for dof in range(dofs_per_node):
            ID = int(Global_ID[node, dof])

            if ID > 0:
                U[node, dof] = U_F[ID - 1, 0]
            else:
                U[node, dof] = U_P[abs(ID) - 1, 0]

    return U