import numpy as np
from typing import Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from src.fem.create_id import create_id_matrix
from src.physics_models.diffusion_kernel import (
    CalculateGlobalMatrices,
    Create_ConstraintsVector,
    PostProcessing,
)
# from src.physics_models.vtk_output import Write_VTK_Solution


def Driver_Steady_Diffusion(Connectivity: np.ndarray,
                            Constraints: np.ndarray,
                            Coord: np.ndarray,
                            diffusion_function: dict,
                            dim: int,
                            dofs_per_node: int,
                            EleType: str,
                            load_type: dict,
                            NCons: int,
                            Nele: int,
                            NGPTS: int,
                            NumNodes: int):
                            # vtk_filename=None,
                            # vtk_solution_name="concentration",
                            # vtk_extra_point_data=None) -> np.ndarray:
    """
    Run the full steady-state diffusion FEM solver.

    This driver performs the full solution process:

        1. Create the Global_ID matrix.
        2. Create the prescribed solution vector U_P.
        3. Assemble the global FEM matrices.
        4. Solve the reduced system for the free unknowns U_F.
        5. Reconstruct the full nodal solution U.

    The partitioned FEM system is:

        K_FF U_F + K_FP U_P = R_F

    Since U_P is known from the boundary conditions, the unknown free solution
    is computed from:

        K_FF U_F = R_F - K_FP U_P

    Parameters
    ----------
    Connectivity : np.ndarray
        Element connectivity matrix of shape (Nele, NodesPerEle).
        Each row contains the 1-based node numbers for one element.

    Constraints : np.ndarray
        Constraint matrix of shape (NCons, 3).
        Each row is:

            [node_number, dof_number, prescribed_value]

    Coord : np.ndarray
        Nodal coordinate matrix of shape (NumNodes, dim).

    diffusion_function : dict
        Dictionary defining the diffusion tensor.
        Example:

            {"type": "isotropic", "d0": 1.0}

    dim : int
        Spatial dimension. For this project, usually dim = 2.

    dofs_per_node : int
        Number of unknowns per node.
        For steady-state diffusion, this is usually 1.

    EleType : str
        Element type.
        Example: "Q4".

    load_type : dict
        Dictionary defining the volumetric source term.
        Example:

            {"type": "constant", "f0": 1.0}

    NCons : int
        Number of prescribed/constrained degrees of freedom.

    Nele : int
        Number of elements.

    NGPTS : int
        Number of Gauss points per element direction.

    NumNodes : int
        Number of nodes in the mesh.

    Returns
    -------
    U : np.ndarray
        Full nodal solution array of shape (NumNodes, dofs_per_node).


    Notes
    -----
    Global_ID convention:

        positive ID  -> free unknown DOF
        negative ID  -> prescribed DOF

    Example:

        Global_ID = [[-1],
                     [ 1],
                     [-2]]

    means:

        node 1 is prescribed constraint 1
        node 2 is free equation 1
        node 3 is prescribed constraint 2
    """

    print("──────── Simulation status report ────────")

    # ------------------------------------------------------------
    # Step 1: Create Global_ID matrix
    # ------------------------------------------------------------
    # Global_ID tells the code which DOFs are free and which are prescribed.
    #
    # Example:
    #   positive value -> free unknown equation
    #   negative value -> prescribed boundary value
    # ------------------------------------------------------------
    Global_ID, NEqns = create_id_matrix(
        Constraints,
        dofs_per_node,
        NumNodes
    )

    print(f"[1/6] Global_ID created.")
   

    # ------------------------------------------------------------
    # Step 2: Create prescribed vector U_P
    # ------------------------------------------------------------
    # U_P stores the known boundary values from Constraints.
    #
    # Example:
    #   Constraints = [[1, 1, 0.0],
    #                  [3, 1, 5.0]]
    #
    # means:
    #   node 1 has prescribed value 0.0
    #   node 3 has prescribed value 5.0
    # ------------------------------------------------------------
    U_P = Create_ConstraintsVector(
        Constraints,
        Global_ID
    )

    print("[2/6] Prescribed vector U_P created.")

    # ------------------------------------------------------------
    # Step 3: Assemble global matrices
    # ------------------------------------------------------------
    # This loops over every element, calculates local element matrices,
    # and adds them into the global system.
    #
    # The important equation is:
    #
    #   K_FF U_F + K_FP U_P = R_F
    #
    # ------------------------------------------------------------
    K_FF, K_FP, K_PP, R_F, R_P = CalculateGlobalMatrices(
        Connectivity,
        Coord,
        diffusion_function,
        dim,
        dofs_per_node,
        EleType,
        Global_ID,
        load_type,
        NCons,
        Nele,
        NEqns,
        NGPTS
    )

    print("[3/6] Global matrices assembled.")
    

    # ------------------------------------------------------------
    # Step 4: Solve the reduced system
    # ------------------------------------------------------------
    # Starting equation:
    #
    #   K_FF U_F + K_FP U_P = R_F
    #
    # Move known prescribed term to the right side:
    #
    #   K_FF U_F = R_F - K_FP U_P
    #
    # Then solve for U_F.
    # ------------------------------------------------------------

    # Make sure K_FF is in sparse CSR format for spsolve.
    try:
        K_FF_csr = K_FF.tocsr()
    except AttributeError:
        K_FF_csr = csr_matrix(K_FF)

    # Build right-hand side.
    RHS = R_F - K_FP @ U_P

    # spsolve expects a 1D right-hand side.
    RHS = np.asarray(RHS).reshape(-1)

    # Solve for free unknown values.
    U_F = spsolve(K_FF_csr, RHS).reshape(-1, 1)

    print("[4/6] Linear system solved.")
    

    # ------------------------------------------------------------
    # Step 5: Reconstruct full solution
    # ------------------------------------------------------------
    # U_F only contains unknown/free values.
    # U_P only contains prescribed values.
    #
    # PostProcessing combines them into one full nodal solution U.
    # ------------------------------------------------------------
    U = PostProcessing(
        Global_ID,
        U_F,
        U_P
    )

    print("[5/6] Postprocessing done.")
    # if vtk_filename is not None:
    #     Write_VTK_Solution(
    #         Coord,
    #         Connectivity,
    #         U,
    #         vtk_filename,
    #         EleType=EleType,
    #         solution_name=vtk_solution_name,
    #         extra_point_data=vtk_extra_point_data,
    #     )
    #     print("[6/7] VTK output written.")
    # else:
    #     print("[6/6] Simulation complete.")


    print("[6/6] FEM simulation complete.")

    return U