"""
Finite Element Framework - create ID matrix

"""

# Opening Rituals
import numpy as np
from typing import Tuple

def create_id_matrix(constraints: np.ndarray, dofs_per_nodes: int, NumNodes: int) -> tuple[np.ndarray, int]:
    """
    Build the GlobalId matrix that maps each node DOF to either:
      +eqn number (free DOF) or
      -constraint number (Dirichlet DOF, numbered by input order, starting at 1).

    Parameters
    ----------
    constraints : iterable of [node, dof, value]
        Node and dof are 1-based indices. Value is unused here but carried for BC data.
    dofs_per_nodes : int
        Degrees of freedom per node. Must be >= 1.
    NumNodes : int
        Total number of nodes. Must be >= 1.

    Returns
    -------
    Global_ID : (NumNodes, dofs_per_nodes) int array
        Positive entries are equation numbers 1..eqn_num (free DOFs).
        Negative entries are -k where k is the constraint number in the input list (1..NCons).
    eqn_num : int
        Count of free equations.
    """

    Global_ID = np.zeros((NumNodes, dofs_per_nodes), dtype=int)
    NEqns = 0

    Ncons = constraints.shape[0]  # number of constraint rows

    # Apply constraints (Dirichlet BCs)
    for i in range(Ncons):
        node = int(constraints[i, 0])
        dof = int(constraints[i, 1])
        Global_ID[node - 1, dof - 1] = -(i + 1)

    # Assign equation numbers to free DOFs
    for i in range(NumNodes):
        for j in range(dofs_per_nodes):
            if Global_ID[i, j] < 0:
                continue
            NEqns += 1
            Global_ID[i, j] = NEqns

    return Global_ID, NEqns