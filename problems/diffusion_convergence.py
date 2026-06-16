"""
Manufactured-Solution Verification of the FEM Diffusion Solver
===============================================================

Governing equation:
    -div(D grad(c)) = f    in Omega

Domain:
    Omega = [0, 1] x [0, 1]

Diffusion tensor:
    D = I

Manufactured solution:
    c_exact(x, y) = x^2 + y^2 + xy

Source term:
    f = -4

Boundary conditions:
    c = c_exact on the entire boundary

Finite element formulations:
    Q4: Four-node quadrilateral elements
    T3: Three-node triangular elements

Expected convergence rates:
    L2 error:         O(h^2)
    H1 seminorm error: O(h)

The program:
    1. Solves the manufactured diffusion problem.
    2. Calculates L2 and H1 seminorm errors.
    3. Computes observed convergence rates.
    4. Compares Q4 and T3 using one convergence plot.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.meshes.Q4 import generate_unit_square_q4_mesh
from src.meshes.T3 import generate_unit_square_t3_mesh

from src.physics_models.diffusion_driver import Driver_Steady_Diffusion
from src.error_calculation import Calculate_Error


# ============================================================
# 1. Analytical solution
# ============================================================

def exact_solution(x, y=None):
    """
    Manufactured analytical solution:

        c(x, y) = x^2 + y^2 + xy

    Accepts either:
        exact_solution(x, y)
    or:
        exact_solution([x, y])
    """

    if y is None:
        x, y = np.asarray(x, dtype=float).reshape(2)

    return x**2 + y**2 + x*y


def exact_gradient(x, y=None):
    """
    Gradient of the manufactured solution:

        dc/dx = 2x + y
        dc/dy = 2y + x
    """

    if y is None:
        x, y = np.asarray(x, dtype=float).reshape(2)

    return np.array(
        [
            2.0*x + y,
            2.0*y + x,
        ],
        dtype=float,
    )


# ============================================================
# 2. Dirichlet boundary conditions
# ============================================================

def create_unit_square_constraints(Coord, tol=1e-12):
    """
    Apply the exact analytical solution on the entire
    boundary of the unit square.

    Returns
    -------
    Constraints : np.ndarray
        Rows are:
        [node_number, dof_number, prescribed_value]
    """

    constraints = []

    for node_id, (x, y) in enumerate(Coord, start=1):

        on_boundary = (
            np.isclose(x, 0.0, atol=tol)
            or np.isclose(x, 1.0, atol=tol)
            or np.isclose(y, 0.0, atol=tol)
            or np.isclose(y, 1.0, atol=tol)
        )

        if on_boundary:

            prescribed_value = exact_solution(x, y)

            constraints.append(
                [
                    node_id,
                    1,
                    prescribed_value,
                ]
            )

    return np.asarray(
        constraints,
        dtype=float,
    )


# ============================================================
# 3. FEM simulation
# ============================================================

def solve_diffusion_problem(n, element_type="Q4"):
    """
    Solve the manufactured diffusion problem using
    either Q4 or T3 finite elements.

    Parameters
    ----------
    n : int
        Number of subdivisions along each coordinate
        direction.

    element_type : str
        "Q4" or "T3".

    Returns
    -------
    Coord : np.ndarray
        Nodal coordinates.

    Connectivity : np.ndarray
        Element connectivity (1-based indexing).

    U : np.ndarray
        Computed nodal concentration.
    """

    element_type = element_type.upper()

    # --------------------------------------------------------
    # Select mesh and assembly quadrature
    # --------------------------------------------------------

    if element_type == "Q4":

        Coord, Connectivity = generate_unit_square_q4_mesh(
            n
        )

        NGPTS = 2

    elif element_type == "T3":

        Coord, Connectivity = generate_unit_square_t3_mesh(
            n
        )

        NGPTS = 1

    else:

        raise ValueError(
            "Supported element types are 'Q4' and 'T3'."
        )

    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------

    Constraints = create_unit_square_constraints(
        Coord
    )

    # --------------------------------------------------------
    # Isotropic diffusion: D = I
    # --------------------------------------------------------

    diffusion_function = {
        "type": "isotropic",
        "d0": 1.0,
    }

    # --------------------------------------------------------
    # Manufactured source: f = -4
    #
    # -Laplace(x^2 + y^2 + xy) = -4
    # --------------------------------------------------------

    load_type = {
        "type": "constant",
        "f0": -4.0,
    }

    # --------------------------------------------------------
    # FEM solver
    # --------------------------------------------------------

    U = Driver_Steady_Diffusion(
        Connectivity=Connectivity,
        Constraints=Constraints,
        Coord=Coord,
        diffusion_function=diffusion_function,
        dim=2,
        dofs_per_node=1,
        EleType=element_type,
        load_type=load_type,
        NCons=Constraints.shape[0],
        Nele=Connectivity.shape[0],
        NGPTS=NGPTS,
        NumNodes=Coord.shape[0],
    )

    return Coord, Connectivity, U


# ============================================================
# 4. Mesh-refinement and convergence study
# ============================================================

def run_refinement_study(
    mesh_sizes=(4, 8, 16, 32),
    element_type="Q4",
):
    """
    Perform an analytical convergence study.

    For each mesh:
        1. Solve the FEM problem.
        2. Calculate L2 error.
        3. Calculate H1 seminorm error.
        4. Calculate observed convergence rates.

    Returns
    -------
    h_values : np.ndarray
        Structured-grid spacings.

    L2_errors : np.ndarray
        L2 errors.

    H1_errors : np.ndarray
        H1 seminorm errors.
    """

    element_type = element_type.upper()

    # --------------------------------------------------------
    # Select error-integration quadrature
    # --------------------------------------------------------

    if element_type == "Q4":

        error_NGPTS = 3

    elif element_type == "T3":

        error_NGPTS = 7

    else:

        raise ValueError(
            "Supported element types are 'Q4' and 'T3'."
        )

    h_values = []
    L2_errors = []
    H1_errors = []

    print()
    print("=" * 65)
    print(
        f"MANUFACTURED SOLUTION — {element_type} CONVERGENCE STUDY"
    )
    print("=" * 65)

    # --------------------------------------------------------
    # Solve successive mesh resolutions
    # --------------------------------------------------------

    for n in mesh_sizes:

        print()
        print("-" * 65)
        print(f"Mesh refinement: n = {n}")
        print("-" * 65)

        Coord, Connectivity, U = solve_diffusion_problem(
            n,
            element_type=element_type,
        )

        # ----------------------------------------------------
        # Grid spacing
        # ----------------------------------------------------

        h = 1.0 / n

        # ----------------------------------------------------
        # Calculate analytical errors
        # ----------------------------------------------------

        L2_error, H1_error = Calculate_Error(
            Connectivity=Connectivity,
            Coord=Coord,
            EleType=element_type,
            NGPTS=error_NGPTS,
            U=U,
            exact_solution=exact_solution,
            exact_gradient=exact_gradient,
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        h_values.append(h)
        L2_errors.append(float(L2_error))
        H1_errors.append(float(H1_error))

        # ----------------------------------------------------
        # Print mesh information
        # ----------------------------------------------------

        print(f"Element type: {element_type}")

        print(
            f"Nodes per element: {Connectivity.shape[1]}"
        )

        print(
            f"Number of nodes: {Coord.shape[0]}"
        )

        print(
            f"Number of elements: {Connectivity.shape[0]}"
        )

        print(f"Mesh size h: {h:.6f}")

        print(
            f"L2 error: {L2_error:.8e}"
        )

        print(
            f"H1 seminorm error: {H1_error:.8e}"
        )

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    h_values = np.asarray(
        h_values,
        dtype=float,
    )

    L2_errors = np.asarray(
        L2_errors,
        dtype=float,
    )

    H1_errors = np.asarray(
        H1_errors,
        dtype=float,
    )

    # --------------------------------------------------------
    # Print refinement summary
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print(
        f"REFINEMENT SUMMARY — {element_type}"
    )
    print("=" * 65)

    print(
        f"{'n':>6} "
        f"{'h':>12} "
        f"{'L2 error':>18} "
        f"{'H1 error':>18}"
    )

    print("-" * 65)

    for i, n in enumerate(mesh_sizes):

        print(
            f"{n:6d} "
            f"{h_values[i]:12.6f} "
            f"{L2_errors[i]:18.8e} "
            f"{H1_errors[i]:18.8e}"
        )

    # --------------------------------------------------------
    # Calculate observed convergence rates
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print(
        f"OBSERVED CONVERGENCE RATES — {element_type}"
    )
    print("=" * 65)

    for i in range(1, len(mesh_sizes)):

        h_coarse = h_values[i - 1]
        h_fine = h_values[i]

        L2_coarse = L2_errors[i - 1]
        L2_fine = L2_errors[i]

        H1_coarse = H1_errors[i - 1]
        H1_fine = H1_errors[i]

        L2_rate = (
            np.log(L2_coarse / L2_fine)
            / np.log(h_coarse / h_fine)
        )

        H1_rate = (
            np.log(H1_coarse / H1_fine)
            / np.log(h_coarse / h_fine)
        )

        print(
            f"n = {mesh_sizes[i-1]:2d} -> "
            f"{mesh_sizes[i]:2d}: "
            f"L2 rate = {L2_rate:.4f}, "
            f"H1 rate = {H1_rate:.4f}"
        )

    return (
        h_values,
        L2_errors,
        H1_errors,
    )


# ============================================================
# 5. Combined Q4 versus T3 convergence plot
# ============================================================

def plot_q4_t3_convergence(
    q4_results,
    t3_results,
):
    """
    Plot the L2 and H1 seminorm errors for Q4
    and T3 elements on the same log-log figure.

    No theoretical reference lines are included.
    """

    (
        h_q4,
        L2_q4,
        H1_q4,
    ) = q4_results

    (
        h_t3,
        L2_t3,
        H1_t3,
    ) = t3_results

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # --------------------------------------------------------
    # Q4 L2 error
    # --------------------------------------------------------

    ax.loglog(
        h_q4,
        L2_q4,
        "o-",
        linewidth=2,
        markersize=7,
        label=r"Q4 — $L^2$ error",
    )

    # --------------------------------------------------------
    # T3 L2 error
    # --------------------------------------------------------

    ax.loglog(
        h_t3,
        L2_t3,
        "s--",
        linewidth=2,
        markersize=7,
        label=r"T3 — $L^2$ error",
    )

    # --------------------------------------------------------
    # Q4 H1 seminorm error
    # --------------------------------------------------------

    ax.loglog(
        h_q4,
        H1_q4,
        "^-",
        linewidth=2,
        markersize=7,
        label=r"Q4 — $H^1$ seminorm",
    )

    # --------------------------------------------------------
    # T3 H1 seminorm error
    # --------------------------------------------------------

    ax.loglog(
        h_t3,
        H1_t3,
        "d--",
        linewidth=2,
        markersize=7,
        label=r"T3 — $H^1$ seminorm",
    )

    # --------------------------------------------------------
    # Figure formatting
    # --------------------------------------------------------

    ax.set_xlabel(
        "Mesh size, h"
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_title(
        "Manufactured Solution — Q4 vs T3 Convergence"
    )

    ax.grid(
        True,
        which="both",
        linestyle=":",
        alpha=0.5,
    )

    ax.legend()

    # Show finer meshes toward the right
    ax.invert_xaxis()

    plt.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    Path("figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        "figures/diffusion_convergence_q4_t3.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved: {filename}")


# ============================================================
# 6. Main execution
# ============================================================

def main():
    """
    Run Q4 and T3 analytical convergence studies
    and generate their combined comparison plot.
    """

    Path("figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh_sizes = (
        4,
        8,
        16,
        32,
    )

    # --------------------------------------------------------
    # Q4 convergence
    # --------------------------------------------------------

    q4_results = run_refinement_study(
        mesh_sizes=mesh_sizes,
        element_type="Q4",
    )

    # --------------------------------------------------------
    # T3 convergence
    # --------------------------------------------------------

    t3_results = run_refinement_study(
        mesh_sizes=mesh_sizes,
        element_type="T3",
    )

    # --------------------------------------------------------
    # Combined convergence comparison
    # --------------------------------------------------------

    plot_q4_t3_convergence(
        q4_results,
        t3_results,
    )


if __name__ == "__main__":
    main()