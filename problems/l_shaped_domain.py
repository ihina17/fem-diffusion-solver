"""
Steady-State Diffusion on an L-Shaped Domain
============================================

Governing equation:
    -Laplace(c) = 1  in Omega

Boundary conditions:
    c = 0 on the entire boundary

Domain:
    Omega = [0, 2] x [0, 2] minus [1, 2] x [1, 2]

Re-entrant corner:
    A = (1, 1)

Finite element formulations:
    Q4: Four-node quadrilateral elements
    T3: Three-node triangular elements

Numerical investigations:
    1. Concentration distribution
    2. Concentration-gradient magnitude
    3. Mesh refinement and gradient growth
    4. Q4 versus T3 comparison
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection, PolyCollection

from src.fem.shape_functions import ShapeFunctions
from src.meshes.Q4 import generate_l_shaped_q4_mesh
from src.meshes.T3 import generate_l_shaped_t3_mesh
from src.physics_models.diffusion_driver import Driver_Steady_Diffusion


# ============================================================
# 1. Boundary conditions
# ============================================================

def create_l_shape_constraints(Coord, tol=1e-12):
    """
    Apply homogeneous Dirichlet boundary conditions
    on the entire boundary of the L-shaped domain.

    Returns
    -------
    Constraints : np.ndarray
        Array containing [node, dof, prescribed_value].
    """

    constraints = []

    for node_id, (x, y) in enumerate(Coord, start=1):

        # Outer boundary
        on_left = np.isclose(x, 0.0, atol=tol)
        on_bottom = np.isclose(y, 0.0, atol=tol)

        on_top_left = (
            np.isclose(y, 2.0, atol=tol)
            and x <= 1.0 + tol
        )

        on_right_bottom = (
            np.isclose(x, 2.0, atol=tol)
            and y <= 1.0 + tol
        )

        # Inner boundary of the removed quadrant
        on_inner_vertical = (
            np.isclose(x, 1.0, atol=tol)
            and y >= 1.0 - tol
        )

        on_inner_horizontal = (
            np.isclose(y, 1.0, atol=tol)
            and x >= 1.0 - tol
        )

        if (
            on_left
            or on_bottom
            or on_top_left
            or on_right_bottom
            or on_inner_vertical
            or on_inner_horizontal
        ):
            constraints.append([node_id, 1, 0.0])

    return np.array(constraints, dtype=float)


# ============================================================
# 2. FEM simulation
# ============================================================

def solve_l_shaped_problem(n, element_type="Q4"):
    """
    Solve the steady-state diffusion equation:

        -Laplace(c) = 1 in Omega
         c = 0 on the boundary

    Parameters
    ----------
    n : int
        Number of subdivisions per unit length.

    element_type : str
        "Q4" or "T3".

    Returns
    -------
    Coord : np.ndarray
        Nodal coordinates.

    Connectivity : np.ndarray
        Element connectivity (1-based).

    U : np.ndarray
        FEM nodal concentration field.
    """

    element_type = element_type.upper()

    # --------------------------------------------------------
    # Select element type and Gaussian quadrature
    # --------------------------------------------------------

    if element_type == "Q4":

        Coord, Connectivity = generate_l_shaped_q4_mesh(n=n)
        NGPTS = 2

    elif element_type == "T3":

        Coord, Connectivity = generate_l_shaped_t3_mesh(n=n)
        NGPTS = 1

    else:

        raise ValueError(
            "Supported element types are 'Q4' and 'T3'."
        )

    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------

    Constraints = create_l_shape_constraints(Coord)

    # --------------------------------------------------------
    # Isotropic diffusivity: D = I
    # --------------------------------------------------------

    diffusion_function = {
        "type": "isotropic",
        "d0": 1.0,
    }

    # --------------------------------------------------------
    # Constant volumetric source: f = 1
    # --------------------------------------------------------

    load_type = {
        "type": "constant",
        "f0": 1.0,
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
# 3. Numerical analysis: concentration gradients
# ============================================================

def compute_element_gradient_magnitudes(Coord, Connectivity, U):
    """
    Calculate the concentration-gradient magnitude
    at the center of each Q4 or T3 element.

    For Q4 elements:
        Evaluate the gradient at the natural-coordinate center.

    For T3 elements:
        The gradient is constant within each triangle.

    Returns
    -------
    element_centers : np.ndarray
        Coordinates of the element centers.

    grad_magnitudes : np.ndarray
        Concentration-gradient magnitudes.
    """

    element_centers = []
    grad_magnitudes = []

    # --------------------------------------------------------
    # Identify element type
    # --------------------------------------------------------

    nodes_per_element = Connectivity.shape[1]

    if nodes_per_element == 4:

        element_type = "Q4"
        zeta = np.array([0.0, 0.0])

    elif nodes_per_element == 3:

        element_type = "T3"
        zeta = np.array([1.0 / 3.0, 1.0 / 3.0])

    else:

        raise ValueError(
            "Only Q4 and T3 elements are supported."
        )

    # --------------------------------------------------------
    # Shape functions at element center
    # --------------------------------------------------------

    N, DN = ShapeFunctions(element_type, zeta)

    N = np.asarray(N, dtype=float).reshape(1, -1)
    DN = np.asarray(DN, dtype=float)

    U = np.asarray(U).reshape(-1)

    # --------------------------------------------------------
    # Loop over elements
    # --------------------------------------------------------

    for element in Connectivity:

        # Convert 1-based to 0-based indexing
        ids = element - 1

        xCap = Coord[ids, :]
        uCap = U[ids].reshape(-1, 1)

        # Physical coordinates of element center
        x_center = (N @ xCap).reshape(-1)

        # Jacobian
        J = xCap.T @ DN

        detJ = np.linalg.det(J)

        if detJ <= 0.0:

            raise ValueError(
                f"Non-positive Jacobian determinant: {detJ}"
            )

        # Physical shape-function gradients
        B = DN @ np.linalg.inv(J)

        # Concentration gradient
        grad_c = (B.T @ uCap).reshape(-1)

        # Gradient magnitude
        grad_mag = np.linalg.norm(grad_c)

        element_centers.append(x_center)
        grad_magnitudes.append(grad_mag)

    return (
        np.array(element_centers),
        np.array(grad_magnitudes),
    )


# ============================================================
# 4. Visualization: Concentration field
# ============================================================

def plot_concentration(Coord, Connectivity, U, show_mesh=False):
    """
    Plot the concentration field for Q4 or T3 elements.

    Q4 elements are subdivided into triangles only for
    visualization. The FEM solution remains Q4.
    """

    # --------------------------------------------------------
    # Identify element type and construct plotting triangles
    # --------------------------------------------------------

    nodes_per_element = Connectivity.shape[1]

    if nodes_per_element == 4:

        element_type = "Q4"

        triangles = []

        for element in Connectivity:

            n1, n2, n3, n4 = element - 1

            triangles.append([n1, n2, n3])
            triangles.append([n1, n3, n4])

        triangles = np.array(triangles, dtype=int)

    elif nodes_per_element == 3:

        element_type = "T3"

        triangles = Connectivity - 1

    else:

        raise ValueError(
            "Only Q4 and T3 elements are supported."
        )

    # --------------------------------------------------------
    # Concentration contour
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 7))

    contour = ax.tricontourf(
        Coord[:, 0],
        Coord[:, 1],
        triangles,
        np.asarray(U).reshape(-1),
        levels=30,
        cmap="viridis",
    )

    fig.colorbar(
        contour,
        ax=ax,
        label="Concentration, c",
    )

    # --------------------------------------------------------
    # Optional mesh overlay
    # --------------------------------------------------------

    if show_mesh:

        element_edges = []

        for element in Connectivity:

            nodes = Coord[element - 1]

            edges = np.vstack([nodes, nodes[0]])

            element_edges.append(edges)

        mesh_lines = LineCollection(
            element_edges,
            colors="black",
            linewidths=0.3,
            alpha=0.3,
        )

        ax.add_collection(mesh_lines)

    # --------------------------------------------------------
    # Re-entrant corner
    # --------------------------------------------------------

    ax.plot(
        1.0,
        1.0,
        "ro",
        markersize=5,
    )

    ax.text(
        1.03,
        1.03,
        "A",
        fontsize=11,
    )

    # --------------------------------------------------------
    # Figure formatting
    # --------------------------------------------------------

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 2.0)

    ax.set_aspect("equal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(
        f"Concentration Field — L-Shaped Domain ({element_type})"
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
        f"figures/l_shape_concentration_{element_type.lower()}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# 5. Visualization: Concentration-gradient magnitude
# ============================================================

def plot_gradient(Coord, Connectivity, U, show_mesh=False):
    """
    Visualize the concentration-gradient magnitude
    using the actual Q4 or T3 element geometry.

    Each element is colored according to its
    computed gradient magnitude at the element center.
    """

    # --------------------------------------------------------
    # Identify element type
    # --------------------------------------------------------

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
    # Calculate gradients
    # --------------------------------------------------------

    element_centers, grad_magnitudes = (
        compute_element_gradient_magnitudes(
            Coord,
            Connectivity,
            U,
        )
    )

    # --------------------------------------------------------
    # Construct element polygons
    # --------------------------------------------------------

    polygons = []

    for element in Connectivity:

        ids = element - 1

        polygons.append(Coord[ids, :])

    # --------------------------------------------------------
    # Gradient field
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 7))

    collection = PolyCollection(
        polygons,
        array=np.asarray(grad_magnitudes),
        cmap="inferno",
        edgecolors="none",
    )

    ax.add_collection(collection)

    fig.colorbar(
        collection,
        ax=ax,
        label=r"Gradient magnitude, $|\nabla c|$",
    )

    # --------------------------------------------------------
    # Optional mesh overlay
    # --------------------------------------------------------

    if show_mesh:

        element_edges = []

        for element in Connectivity:

            nodes = Coord[element - 1]

            edges = np.vstack([nodes, nodes[0]])

            element_edges.append(edges)

        mesh_lines = LineCollection(
            element_edges,
            colors="black",
            linewidths=0.3,
            alpha=0.35,
        )

        ax.add_collection(mesh_lines)

    # --------------------------------------------------------
    # Re-entrant corner
    # --------------------------------------------------------

    ax.plot(
        1.0,
        1.0,
        "co",
        markersize=6,
    )

    ax.text(
        1.03,
        1.03,
        "A",
        fontsize=11,
    )

    # --------------------------------------------------------
    # Figure formatting
    # --------------------------------------------------------

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 2.0)

    ax.set_aspect("equal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_title(
        f"Concentration Gradient — L-Shaped Domain ({element_type})"
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
        f"figures/l_shape_gradient_{element_type.lower()}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# 6. Mesh-refinement study
# ============================================================

def run_refinement_study(
    mesh_sizes=(4, 8, 16, 32),
    element_type="Q4",
):
    """
    Investigate concentration-gradient growth
    under uniform mesh refinement.

    Supports both Q4 and T3 elements.

    For each mesh:
        1. Solve the diffusion equation.
        2. Calculate maximum concentration.
        3. Calculate maximum element-center gradient.
        4. Locate the maximum gradient.

    Returns
    -------
    h_values : np.ndarray
        Underlying structured-grid spacings.

    max_concentrations : np.ndarray
        Maximum nodal concentration for each mesh.

    max_gradients : np.ndarray
        Maximum element-center gradient magnitude.
    """

    element_type = element_type.upper()

    h_values = []
    max_concentrations = []
    max_gradients = []

    print()
    print("=" * 60)
    print(f"L-SHAPED DOMAIN — {element_type} MESH REFINEMENT")
    print("=" * 60)

    # --------------------------------------------------------
    # Solve successive meshes
    # --------------------------------------------------------

    for n in mesh_sizes:

        print()
        print("-" * 60)
        print(f"Mesh refinement: n = {n}, Element = {element_type}")
        print("-" * 60)

        # FEM solution
        Coord, Connectivity, U = solve_l_shaped_problem(
            n,
            element_type=element_type,
        )

        # Structured-grid spacing
        h = 1.0 / n

        # Maximum concentration
        max_c = np.max(U)

        # Element-center gradient magnitudes
        centers, grad_magnitudes = (
            compute_element_gradient_magnitudes(
                Coord,
                Connectivity,
                U,
            )
        )

        # Maximum gradient and location
        max_idx = np.argmax(grad_magnitudes)

        max_grad = grad_magnitudes[max_idx]
        max_grad_location = centers[max_idx]

        # Store results
        h_values.append(h)
        max_concentrations.append(max_c)
        max_gradients.append(max_grad)

        # Print results
        print(f"h = {h:.6f}")
        print(f"Nodes = {Coord.shape[0]}")
        print(f"Elements = {Connectivity.shape[0]}")
        print(f"Maximum concentration = {max_c:.8f}")
        print(f"Maximum |grad c| = {max_grad:.8f}")
        print(f"Maximum gradient location = {max_grad_location}")

    # --------------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------------

    h_values = np.array(h_values)
    max_concentrations = np.array(max_concentrations)
    max_gradients = np.array(max_gradients)

    # --------------------------------------------------------
    # Plot gradient growth with mesh refinement
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(
        h_values,
        max_gradients,
        "o-",
        linewidth=2,
        markersize=7,
    )

    ax.set_xlabel("Grid spacing, h")

    ax.set_ylabel(
        r"Maximum element-center $|\nabla c|$"
    )

    ax.set_title(
        f"Gradient Growth Near Re-Entrant Corner ({element_type})"
    )

    ax.grid(
        True,
        which="both",
        linestyle=":",
        alpha=0.6,
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save refinement figure
    # --------------------------------------------------------

    Path("figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"figures/l_shape_gradient_refinement_{element_type.lower()}.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {filename}")

    # --------------------------------------------------------
    # Print refinement summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(f"REFINEMENT SUMMARY — {element_type}")
    print("=" * 60)

    print(
        f"{'n':>6} "
        f"{'h':>12} "
        f"{'max(c)':>15} "
        f"{'max|grad c|':>15}"
    )

    print("-" * 60)

    for i, n in enumerate(mesh_sizes):

        print(
            f"{n:6d} "
            f"{h_values[i]:12.6f} "
            f"{max_concentrations[i]:15.8f} "
            f"{max_gradients[i]:15.8f}"
        )

    return (
        h_values,
        max_concentrations,
        max_gradients,
    )


# ============================================================
# 7. Q4 versus T3 comparison
# ============================================================

def plot_refinement_comparison(
    h_q4,
    gradients_q4,
    h_t3,
    gradients_t3,
):
    """
    Compare the maximum concentration-gradient magnitude
    under mesh refinement for Q4 and T3 elements.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # --------------------------------------------------------
    # Q4 results
    # --------------------------------------------------------

    ax.loglog(
        h_q4,
        gradients_q4,
        "o-",
        linewidth=2,
        markersize=7,
        label="Q4",
    )

    # --------------------------------------------------------
    # T3 results
    # --------------------------------------------------------

    ax.loglog(
        h_t3,
        gradients_t3,
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
        r"Maximum element-center $|\nabla c|$"
    )

    ax.set_title(
        "Gradient Growth Near L-Shaped Re-Entrant Corner"
    )

    ax.grid(
        True,
        which="both",
        linestyle=":",
        alpha=0.6,
    )

    ax.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    Path("figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        "figures/l_shape_gradient_refinement_comparison.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {filename}")


# ============================================================
# 8. Main execution
# ============================================================

def main():
    """
    Run the complete L-shaped diffusion analysis
    using Q4 and T3 elements.
    """

    Path("figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Part 1: Concentration and gradient distributions
    # --------------------------------------------------------

    n = 16

    for element_type in ("Q4", "T3"):

        print()
        print("=" * 60)
        print(f"L-SHAPED DIFFUSION — {element_type}")
        print("=" * 60)

        # Solve FEM problem
        Coord, Connectivity, U = solve_l_shaped_problem(
            n,
            element_type=element_type,
        )

        Constraints = create_l_shape_constraints(Coord)

        print("\nNumber of nodes:", Coord.shape[0])
        print("Number of elements:", Connectivity.shape[0])
        print("Number of constrained nodes:", Constraints.shape[0])

        print("\nMinimum concentration:", np.min(U))
        print("Maximum concentration:", np.max(U))

        # Concentration field
        plot_concentration(
            Coord,
            Connectivity,
            U,
            show_mesh=True,
        )

        # Calculate gradient statistics
        centers, gradients = (
            compute_element_gradient_magnitudes(
                Coord,
                Connectivity,
                U,
            )
        )

        print(
            "\nMaximum gradient magnitude:",
            np.max(gradients),
        )

        # Gradient field
        plot_gradient(
            Coord,
            Connectivity,
            U,
            show_mesh=False,
        )

    # --------------------------------------------------------
    # Part 2: Q4 mesh refinement
    # --------------------------------------------------------

    h_q4, concentrations_q4, gradients_q4 = (
        run_refinement_study(
            mesh_sizes=(4, 8, 16, 32),
            element_type="Q4",
        )
    )

    # --------------------------------------------------------
    # Part 3: T3 mesh refinement
    # --------------------------------------------------------

    h_t3, concentrations_t3, gradients_t3 = (
        run_refinement_study(
            mesh_sizes=(4, 8, 16, 32),
            element_type="T3",
        )
    )

    # --------------------------------------------------------
    # Part 4: Combined Q4 versus T3 comparison
    # --------------------------------------------------------

    plot_refinement_comparison(
        h_q4,
        gradients_q4,
        h_t3,
        gradients_t3,
    )


if __name__ == "__main__":
    main()