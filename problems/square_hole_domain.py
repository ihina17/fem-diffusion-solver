"""
Steady-State Anisotropic Diffusion on a Square-Hole Domain
==========================================================

Governing equation:
    -div(D grad(c)) = 0

Domain:
    Omega = [0, 1] x [0, 1]
            minus [4/9, 5/9] x [4/9, 5/9]

Boundary conditions:
    c = 0 on the outer square boundary
    c = 1 on the square-hole boundary

Diffusion tensor:
    D = R(theta) @ diag(d1, d2) @ R(theta).T

    theta = pi/6
    d1 = 10000
    d2 = 0

Finite element formulations:
    Q4: Four-node quadrilateral elements
    T3: Three-node triangular elements

Numerical investigations:
    1. Mesh visualization
    2. Concentration distribution
    3. Negative nodal concentrations
    4. Mesh-refinement analysis
    5. Q4 versus T3 comparison

Note:
    The diffusion tensor is positive semidefinite because d2 = 0.
    Numerical non-negativity is investigated without assuming
    the standard uniformly elliptic maximum principle applies.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

from src.meshes.Q4 import generate_square_hole_q4_mesh
from src.meshes.T3 import generate_square_hole_t3_mesh

from src.physics_models.diffusion_driver import Driver_Steady_Diffusion


# ============================================================
# 1. Boundary conditions
# ============================================================

def create_square_hole_constraints(Coord, tol=1e-12):
    """
    Apply Dirichlet boundary conditions.

    Outer square:
        c = 0

    Square-hole boundary:
        c = 1

    Hole:
        [4/9, 5/9] x [4/9, 5/9]

    Returns
    -------
    Constraints : np.ndarray
        Rows have format:
        [node_number, dof_number, prescribed_value]
    """

    constraints = []

    hole_min = 4.0 / 9.0
    hole_max = 5.0 / 9.0

    for node_id, (x, y) in enumerate(Coord, start=1):

        # ----------------------------------------------------
        # Outer boundary: c = 0
        # ----------------------------------------------------

        on_outer_boundary = (
            np.isclose(x, 0.0, atol=tol)
            or np.isclose(x, 1.0, atol=tol)
            or np.isclose(y, 0.0, atol=tol)
            or np.isclose(y, 1.0, atol=tol)
        )

        if on_outer_boundary:

            constraints.append([
                node_id,
                1,
                0.0,
            ])

            continue

        # ----------------------------------------------------
        # Square-hole boundary: c = 1
        # ----------------------------------------------------

        on_hole_vertical = (
            (
                np.isclose(x, hole_min, atol=tol)
                or np.isclose(x, hole_max, atol=tol)
            )
            and hole_min - tol <= y <= hole_max + tol
        )

        on_hole_horizontal = (
            (
                np.isclose(y, hole_min, atol=tol)
                or np.isclose(y, hole_max, atol=tol)
            )
            and hole_min - tol <= x <= hole_max + tol
        )

        if on_hole_vertical or on_hole_horizontal:

            constraints.append([
                node_id,
                1,
                1.0,
            ])

    return np.array(
        constraints,
        dtype=float,
    )


# ============================================================
# 2. FEM simulation
# ============================================================

def solve_square_hole_problem(n, element_type="Q4"):
    """
    Solve the anisotropic diffusion problem on the
    unit square with a square hole.

    Parameters
    ----------
    n : int
        Number of subdivisions along each outer side.
        Must be divisible by 9.

    element_type : str
        "Q4" or "T3".

    Returns
    -------
    Coord : np.ndarray
        Nodal coordinates.

    Connectivity : np.ndarray
        Element connectivity (1-based indexing).

    U : np.ndarray
        FEM nodal concentration field.
    """

    element_type = element_type.upper()

    # --------------------------------------------------------
    # Select element type and Gaussian quadrature
    # --------------------------------------------------------

    if element_type == "Q4":

        Coord, Connectivity = generate_square_hole_q4_mesh(
            n=n
        )

        NGPTS = 2

    elif element_type == "T3":

        Coord, Connectivity = generate_square_hole_t3_mesh(
            n=n
        )

        NGPTS = 1

    else:

        raise ValueError(
            "Supported element types are 'Q4' and 'T3'."
        )

    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------

    Constraints = create_square_hole_constraints(
        Coord
    )

    # --------------------------------------------------------
    # Anisotropic diffusion tensor
    # --------------------------------------------------------

    diffusion_function = {
        "type": "rotation",
        "theta": np.pi / 6.0,
        "d1": 10000.0,
        "d2": 0.0,
    }

    # --------------------------------------------------------
    # Zero volumetric source
    # --------------------------------------------------------

    load_type = {
        "type": "constant",
        "f0": 0.0,
    }

    # --------------------------------------------------------
    # FEM solution
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
# 3. Visualization: Finite element mesh
# ============================================================

def plot_mesh(Coord, Connectivity):
    """
    Plot the actual Q4 or T3 finite element mesh.
    """

    nodes_per_element = Connectivity.shape[1]

    if nodes_per_element == 4:
        element_type = "Q4"

    elif nodes_per_element == 3:
        element_type = "T3"

    else:
        raise ValueError(
            "Only Q4 and T3 elements are supported."
        )

    # --------------------------------------------------------
    # Construct element boundaries
    # --------------------------------------------------------

    element_edges = []

    for element in Connectivity:

        nodes = Coord[element - 1]

        # Close each quadrilateral or triangle
        edges = np.vstack([
            nodes,
            nodes[0],
        ])

        element_edges.append(edges)

    # --------------------------------------------------------
    # Plot mesh
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 7))

    mesh_lines = LineCollection(
        element_edges,
        colors="black",
        linewidths=0.5,
    )

    ax.add_collection(mesh_lines)

    ax.scatter(
        Coord[:, 0],
        Coord[:, 1],
        s=5,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect("equal")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.set_title(
        f"Square-Hole Domain — {element_type} Mesh"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    Path("figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"figures/square_hole_mesh_{element_type.lower()}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# 4. Visualization: Concentration field
# ============================================================

def plot_concentration(Coord, Connectivity, U):
    """
    Plot the concentration field for Q4 or T3 elements.

    Negative nodal concentrations are marked with black dots.

    Both formulations use the same concentration color scale.
    """

    # --------------------------------------------------------
    # Identify element type
    # --------------------------------------------------------

    nodes_per_element = Connectivity.shape[1]

    if nodes_per_element == 4:

        element_type = "Q4"

        # Divide Q4 elements into triangles for visualization
        triangles = []

        for element in Connectivity:

            n1, n2, n3, n4 = element - 1

            triangles.append([n1, n2, n3])
            triangles.append([n1, n3, n4])

        triangles = np.array(
            triangles,
            dtype=int,
        )

    elif nodes_per_element == 3:

        element_type = "T3"

        # T3 connectivity is already triangular
        triangles = Connectivity - 1

    else:

        raise ValueError(
            "Only Q4 and T3 elements are supported."
        )

    U = np.asarray(U).reshape(-1)

    # --------------------------------------------------------
    # Concentration contour
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 7))

    # Fixed color scale for Q4-T3 comparison
    levels = np.linspace(
        -0.05,
        1.0,
        43,
    )

    contour = ax.tricontourf(
        Coord[:, 0],
        Coord[:, 1],
        triangles,
        U,
        levels=levels,
        cmap="coolwarm",
        extend="both",
    )

    fig.colorbar(
        contour,
        ax=ax,
        label="Concentration, c",
    )

    # --------------------------------------------------------
    # Zero-concentration contour
    # --------------------------------------------------------

    if np.min(U) < 0.0 < np.max(U):

        ax.tricontour(
            Coord[:, 0],
            Coord[:, 1],
            triangles,
            U,
            levels=[0.0],
            colors="black",
            linewidths=1.0,
        )

    # --------------------------------------------------------
    # Mark negative nodal concentrations
    # --------------------------------------------------------

    negative_mask = U < -1e-10

    if np.any(negative_mask):

        ax.scatter(
            Coord[negative_mask, 0],
            Coord[negative_mask, 1],
            s=10,
            c="black",
            label="Negative nodes",
        )

        ax.legend(
            loc="upper right",
        )

    # --------------------------------------------------------
    # Figure formatting
    # --------------------------------------------------------

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect("equal")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_title(
        f"Anisotropic Diffusion — Square-Hole Domain ({element_type})"
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    Path("figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"figures/square_hole_concentration_{element_type.lower()}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# 5. Mesh-refinement study
# ============================================================

def run_refinement_study(
    mesh_sizes=(18, 36, 54, 72),
    element_type="Q4",
    visualize_n=None,
):
    """
    Investigate numerical non-negativity under mesh refinement.

    For each mesh:
        1. Solve the diffusion problem.
        2. Calculate minimum concentration.
        3. Calculate maximum concentration.
        4. Count negative nodal concentrations.
        5. Calculate percentage of negative nodes.

    Parameters
    ----------
    mesh_sizes : tuple
        Mesh resolutions.

    element_type : str
        "Q4" or "T3".

    visualize_n : int or None
        If provided, also save the mesh and concentration
        figures for this resolution.

    Returns
    -------
    h_values : np.ndarray
        Structured-grid spacings.

    minimum_concentrations : np.ndarray
        Minimum nodal concentration for each mesh.

    negative_node_counts : np.ndarray
        Number of negative nodes for each mesh.

    negative_node_percentages : np.ndarray
        Percentage of nodes with negative concentration.
    """

    element_type = element_type.upper()

    h_values = []

    minimum_concentrations = []
    maximum_concentrations = []

    negative_node_counts = []
    negative_node_percentages = []

    print()
    print("=" * 72)
    print(
        f"SQUARE-HOLE DOMAIN — {element_type} MESH REFINEMENT"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Solve successive meshes
    # --------------------------------------------------------

    for n in mesh_sizes:

        print()
        print("-" * 72)
        print(f"Mesh refinement: n = {n}, Element = {element_type}")
        print("-" * 72)

        # FEM solution
        Coord, Connectivity, U = solve_square_hole_problem(
            n,
            element_type=element_type,
        )

        U = np.asarray(U).reshape(-1)

        # ----------------------------------------------------
        # Save selected mesh and concentration figures
        # ----------------------------------------------------

        if visualize_n is not None and n == visualize_n:

            plot_mesh(
                Coord,
                Connectivity,
            )

            plot_concentration(
                Coord,
                Connectivity,
                U,
            )

        # ----------------------------------------------------
        # Mesh size
        # ----------------------------------------------------

        h = 1.0 / n

        # ----------------------------------------------------
        # Concentration statistics
        # ----------------------------------------------------

        min_c = float(np.min(U))
        max_c = float(np.max(U))

        negative_nodes = int(
            np.count_nonzero(U < -1e-10)
        )

        negative_percentage = (
            100.0
            * negative_nodes
            / Coord.shape[0]
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        h_values.append(h)

        minimum_concentrations.append(min_c)
        maximum_concentrations.append(max_c)

        negative_node_counts.append(
            negative_nodes
        )

        negative_node_percentages.append(
            negative_percentage
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(f"h = {h:.8f}")

        print(f"Nodes = {Coord.shape[0]}")
        print(f"Elements = {Connectivity.shape[0]}")

        print(f"Minimum concentration = {min_c:.8f}")
        print(f"Maximum concentration = {max_c:.8f}")

        print(f"Negative nodes = {negative_nodes}")

        print(
            f"Negative nodes (%) = "
            f"{negative_percentage:.2f}"
        )

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    h_values = np.array(
        h_values,
        dtype=float,
    )

    minimum_concentrations = np.array(
        minimum_concentrations,
        dtype=float,
    )

    maximum_concentrations = np.array(
        maximum_concentrations,
        dtype=float,
    )

    negative_node_counts = np.array(
        negative_node_counts,
        dtype=int,
    )

    negative_node_percentages = np.array(
        negative_node_percentages,
        dtype=float,
    )

    # --------------------------------------------------------
    # Print refinement summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(f"REFINEMENT SUMMARY — {element_type}")
    print("=" * 72)

    print(
        f"{'n':>6} "
        f"{'h':>12} "
        f"{'min(c)':>14} "
        f"{'max(c)':>14} "
        f"{'neg nodes':>12} "
        f"{'neg %':>10}"
    )

    print("-" * 72)

    for i, n in enumerate(mesh_sizes):

        print(
            f"{n:6d} "
            f"{h_values[i]:12.6f} "
            f"{minimum_concentrations[i]:14.8f} "
            f"{maximum_concentrations[i]:14.8f} "
            f"{negative_node_counts[i]:12d} "
            f"{negative_node_percentages[i]:10.2f}"
        )

    # --------------------------------------------------------
    # Plot minimum concentration versus mesh size
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.semilogx(
        h_values,
        minimum_concentrations,
        "o-",
        linewidth=2,
        markersize=7,
        label=element_type,
    )

    # Reference: minimum prescribed boundary concentration
    ax.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Prescribed boundary minimum",
    )

    ax.set_xlabel("Grid spacing, h")

    ax.set_ylabel(
        "Minimum nodal concentration"
    )

    ax.set_title(
        f"Minimum Concentration Under Mesh Refinement ({element_type})"
    )

    ax.grid(
        True,
        which="both",
        linestyle=":",
        alpha=0.6,
    )

    ax.legend()

    # Finer meshes toward the right
    ax.invert_xaxis()

    plt.tight_layout()

    # --------------------------------------------------------
    # Save refinement figure
    # --------------------------------------------------------

    Path("figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"figures/square_hole_refinement_{element_type.lower()}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved: {filename}")

    return (
        h_values,
        minimum_concentrations,
        negative_node_counts,
        negative_node_percentages,
    )


# ============================================================
# 6. Q4 versus T3: Minimum concentration comparison
# ============================================================

def plot_minimum_concentration_comparison(
    h_q4,
    minimum_q4,
    h_t3,
    minimum_t3,
):
    """
    Compare minimum nodal concentrations under mesh
    refinement for Q4 and T3 elements.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # --------------------------------------------------------
    # Q4 results
    # --------------------------------------------------------

    ax.semilogx(
        h_q4,
        minimum_q4,
        "o-",
        linewidth=2,
        markersize=7,
        label="Q4",
    )

    # --------------------------------------------------------
    # T3 results
    # --------------------------------------------------------

    ax.semilogx(
        h_t3,
        minimum_t3,
        "s-",
        linewidth=2,
        markersize=7,
        label="T3",
    )

    # --------------------------------------------------------
    # Prescribed boundary minimum
    # --------------------------------------------------------

    ax.axhline(
        y=0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Prescribed boundary minimum",
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlabel("Grid spacing, h")

    ax.set_ylabel(
        "Minimum nodal concentration"
    )

    ax.set_title(
        "Minimum Concentration — Q4 vs T3"
    )

    ax.grid(
        True,
        which="both",
        linestyle=":",
        alpha=0.6,
    )

    ax.legend()

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
        "figures/square_hole_minimum_concentration_comparison.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# 7. Q4 versus T3: Negative-node percentage comparison
# ============================================================

def plot_negative_percentage_comparison(
    h_q4,
    negative_percent_q4,
    h_t3,
    negative_percent_t3,
):
    """
    Compare the percentage of negative nodal concentrations
    under mesh refinement for Q4 and T3 elements.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # --------------------------------------------------------
    # Q4 results
    # --------------------------------------------------------

    ax.semilogx(
        h_q4,
        negative_percent_q4,
        "o-",
        linewidth=2,
        markersize=7,
        label="Q4",
    )

    # --------------------------------------------------------
    # T3 results
    # --------------------------------------------------------

    ax.semilogx(
        h_t3,
        negative_percent_t3,
        "s-",
        linewidth=2,
        markersize=7,
        label="T3",
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xlabel("Grid spacing, h")

    ax.set_ylabel(
        "Nodes with negative concentration (%)"
    )

    ax.set_title(
        "Negative Nodal Concentrations — Q4 vs T3"
    )

    ax.grid(
        True,
        which="both",
        linestyle=":",
        alpha=0.6,
    )

    ax.legend()

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
        "figures/square_hole_negative_percentage_comparison.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# 8. Main execution
# ============================================================

def main():
    """
    Run the complete square-hole diffusion analysis
    using Q4 and T3 elements.
    """

    Path("figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh_sizes = (18, 36, 54, 72)

    # --------------------------------------------------------
    # Part 1: Q4 analysis
    # --------------------------------------------------------

    (
        h_q4,
        minimum_q4,
        negative_counts_q4,
        negative_percent_q4,
    ) = run_refinement_study(
        mesh_sizes=mesh_sizes,
        element_type="Q4",
        visualize_n=18,
    )

    # --------------------------------------------------------
    # Part 2: T3 analysis
    # --------------------------------------------------------

    (
        h_t3,
        minimum_t3,
        negative_counts_t3,
        negative_percent_t3,
    ) = run_refinement_study(
        mesh_sizes=mesh_sizes,
        element_type="T3",
        visualize_n=18,
    )

    # --------------------------------------------------------
    # Part 3: Compare minimum concentrations
    # --------------------------------------------------------

    plot_minimum_concentration_comparison(
        h_q4,
        minimum_q4,
        h_t3,
        minimum_t3,
    )

    # --------------------------------------------------------
    # Part 4: Compare negative-node percentages
    # --------------------------------------------------------

    plot_negative_percentage_comparison(
        h_q4,
        negative_percent_q4,
        h_t3,
        negative_percent_t3,
    )


if __name__ == "__main__":
    main()