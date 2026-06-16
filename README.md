# Finite Element Solver for Steady-State Diffusion

**Author:** Ihina Mahajan
**Affiliation:** Materials Science and Engineering, University of Houston
---

A Python implementation of the finite element method (FEM) for solving two-dimensional steady-state diffusion problems using four-node quadrilateral (Q4) and three-node triangular (T3) elements.

The project includes a modular FEM framework, verification against an analytical solution, and numerical investigations of diffusion on geometries with re-entrant corners and strongly anisotropic material properties.


## 1. Overview

The solver considers the steady-state diffusion equation

\[
-\nabla \cdot \left(\mathbf{D}\nabla c\right)=f
\qquad \text{in } \Omega,
\]

subject to prescribed Dirichlet boundary conditions

\[
c=\bar{c}
\qquad \text{on } \Gamma_D,
\]

where:

- \(c\) is the concentration field.
- \(\mathbf{D}\) is the diffusion tensor.
- \(f\) is the volumetric source.
- \(\Omega\) is the computational domain.
- \(\Gamma_D\) is the Dirichlet boundary.

The finite element formulation uses the standard Galerkin method to assemble and solve the discrete system.

### Features

- Q4 and T3 finite element formulations.
- Structured mesh generation for multiple geometries.
- Shape-function evaluation and numerical quadrature.
- Element-level stiffness matrix and load-vector calculations.
- Global matrix assembly and application of Dirichlet boundary conditions.
- Isotropic and anisotropic diffusion models.
- Numerical error calculation in the \(L^2\) norm and \(H^1\) seminorm.
- Mesh-refinement and convergence studies.
- Visualization of concentration fields, gradients, and numerical results.

## 2. Numerical Verification

### Manufactured-solution convergence study

The solver is verified using an analytical solution on the unit square,

\[
\Omega=[0,1]\times[0,1].
\]

The manufactured solution is

\[
c(x,y)=x^2+y^2+xy.
\]

For an isotropic diffusion tensor \(\mathbf{D}=\mathbf{I}\), the corresponding source term is

\[
f=-4.
\]

The exact solution is prescribed on the entire boundary.

Both Q4 and T3 formulations are evaluated using a sequence of uniformly refined meshes.

The numerical errors are calculated using

\[
\|c-c_h\|_{L^2(\Omega)}
=
\left(
\int_\Omega (c-c_h)^2\,d\Omega
\right)^{1/2}
\]

and

\[
|c-c_h|_{H^1(\Omega)}
=
\left(
\int_\Omega
\|\nabla c-\nabla c_h\|^2\,d\Omega
\right)^{1/2}.
\]

### Convergence results

Both formulations recover the expected convergence orders:

| Element type | \(L^2\) convergence rate | \(H^1\) convergence rate |
|---|---:|---:|
| Q4 | 2.0000 | 1.0000 |
| T3 | 2.0000 | 1.0000 |

The measured errors decrease according to

\[
\|c-c_h\|_{L^2}=O(h^2),
\qquad
|c-c_h|_{H^1}=O(h).
\]

![Q4 and T3 convergence comparison](figures/diffusion_convergence_q4_t3.png)

**Figure 1.** Convergence of the \(L^2\) error and \(H^1\) seminorm error under uniform mesh refinement. Both element formulations recover the expected convergence rates.

## 3. Numerical Applications

### 3.1. Diffusion on an L-shaped domain

The first application examines the steady-state diffusion equation

\[
-\nabla^2 c=1
\]

on an L-shaped domain, with homogeneous Dirichlet boundary conditions.

The geometry contains a re-entrant corner at \((1,1)\), where the solution gradient exhibits singular behavior.

Both Q4 and T3 discretizations are used to investigate:

- The concentration distribution.
- The spatial distribution of the concentration-gradient magnitude.
- The growth of the maximum computed gradient under mesh refinement.

The numerical results show increasing maximum element-gradient magnitudes as the mesh is refined near the re-entrant corner.

![L-shaped domain: Q4 concentration](figures/l_shape_concentration_q4.png)

![L-shaped domain: T3 concentration](figures/l_shape_concentration_t3.png)

**Figure 2.** Concentration distributions obtained using Q4 and T3 elements.

![L-shaped domain gradient comparison](figures/l_shape_gradient_refinement_comparison.png)

**Figure 3.** Maximum sampled concentration-gradient magnitude under mesh refinement for Q4 and T3 elements.

The increasing gradient magnitude is consistent with the expected singular behavior near the re-entrant corner.

### 3.2. Anisotropic diffusion on a square-hole domain

The second application investigates diffusion through a unit square containing a smaller square hole.

The boundary conditions are

\[
c=0 \quad \text{on the outer boundary},
\]

\[
c=1 \quad \text{on the square-hole boundary}.
\]

The diffusion tensor is defined by

\[
\mathbf{D}
=
\mathbf{R}(\theta)
\begin{bmatrix}
d_1 & 0\\
0 & d_2
\end{bmatrix}
\mathbf{R}(\theta)^T,
\]

where

\[
d_1=10^4,\qquad
d_2=0,\qquad
\theta=\frac{\pi}{6}.
\]

The source term is zero.

This configuration produces strongly directional diffusion. The diffusion tensor is positive semidefinite because one principal diffusivity is zero.

#### Concentration fields

![Square-hole domain: Q4 concentration](figures/square_hole_concentration_q4.png)

![Square-hole domain: T3 concentration](figures/square_hole_concentration_t3.png)

**Figure 4.** Concentration fields computed using Q4 and T3 elements. Black markers identify nodes with negative computed concentrations.

Both discretizations exhibit negative nodal concentrations despite the nonnegative prescribed boundary values.

#### Mesh-refinement analysis

A mesh-refinement study is performed to investigate the behavior of the minimum nodal concentration.

![Minimum concentration comparison](figures/square_hole_minimum_concentration_comparison.png)

**Figure 5.** Minimum nodal concentration versus structured-grid spacing for Q4 and T3 elements.

The fraction of nodes with negative concentrations is also examined.

![Negative nodal concentration comparison](figures/square_hole_negative_percentage_comparison.png)

**Figure 6.** Percentage of nodes exhibiting negative concentrations under mesh refinement.

Negative nodal concentrations persist across the tested mesh resolutions for both formulations.

Because the diffusion tensor is degenerate, the standard uniformly elliptic assumptions do not apply. The results are therefore presented as observations of numerical undershoot for the specified model and discretizations, rather than as a general convergence or maximum-principle result.

## 4. Project Structure

```text
fem-diffusion-solver/
│
├── src/
│   │
│   ├── fem/
│   │   ├── gauss_quadrature.py
│   │   └── shape_functions.py
│   │
│   ├── meshes/
│   │   ├── Q4.py
│   │   └── T3.py
│   │
│   ├── physics_models/
│   │   ├── diffusion_driver.py
│   │   └── diffusion_kernel.py
│   │
│   └── error_calculation.py
│
├── problems/
│   ├── diffusion_convergence.py
│   ├── l_shaped_domain.py
│   └── square_hole_domain.py
│
├── figures/
│   └── ...
│
└── README.md
```

The code is organized into three main components:

**FEM framework:** Implements shape functions, numerical quadrature, and the numerical routines used by the finite element formulation.

**Physics models:** Defines the diffusion problem and performs element-level calculations, global assembly, and solution of the resulting algebraic system.

**Problem examples:** Specify computational geometries, material properties, boundary conditions, numerical experiments, and visualization routines.

## 5. Installation

The project requires Python and the following packages:

- NumPy
- SciPy
- Matplotlib

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/fem-diffusion-solver.git
cd fem-diffusion-solver
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install numpy scipy matplotlib
```

## 6. Running the Examples

Run the scripts from the project root directory.

### Analytical convergence study

```bash
python -m problems.diffusion_convergence
```

Runs the Q4 and T3 manufactured-solution studies, prints the observed convergence rates, and generates the combined convergence figure.

### L-shaped domain

```bash
python -m problems.l_shaped_domain
```

Generates Q4 and T3 concentration fields, gradient visualizations, and mesh-refinement results.

### Square-hole domain

```bash
python -m problems.square_hole_domain
```

Solves the anisotropic diffusion problem using Q4 and T3 elements and generates concentration and mesh-refinement comparisons.

The mesh resolution and element formulation can also be selected programmatically through the problem-specific solver functions.

For example:

```python
from problems.square_hole_domain import solve_square_hole_problem

Coord, Connectivity, U = solve_square_hole_problem(
    n=18,
    element_type="T3",
)
```

## 7. Numerical Implementation

The formulation uses the standard finite element approximation

\[
c_h(\mathbf{x})
=
\sum_{a=1}^{n_{\mathrm{en}}}
N_a(\mathbf{x})c_a,
\]

where \(N_a\) are the element shape functions and \(c_a\) are the nodal unknowns.

For the diffusion equation, the element stiffness matrix is

\[
\mathbf{K}^{(e)}
=
\int_{\Omega_e}
\mathbf{B}^{T}\mathbf{D}\mathbf{B}\,d\Omega,
\]

and the element load vector is

\[
\mathbf{F}^{(e)}
=
\int_{\Omega_e}
\mathbf{N}^{T}f\,d\Omega.
\]

These quantities are evaluated numerically using Gaussian quadrature and assembled into the global system.

The current examples use:

| Element | Nodes per element | Assembly quadrature |
|---|---:|---|
| Q4 | 4 | \(2\times2\) Gauss points |
| T3 | 3 | One centroid point |

Higher-order quadrature is used when integrating the squared errors in the manufactured-solution convergence study.

## 8. Scope and Limitations

This project focuses on two-dimensional steady-state diffusion with prescribed Dirichlet boundary conditions.

The examples demonstrate numerical verification, geometric effects, and the behavior of standard Galerkin discretizations under strong anisotropy.

The project does not currently implement positivity-preserving constraints, adaptive mesh refinement, or time-dependent diffusion.