# Finite Element Solver for Steady-State Diffusion

**Author:** Ihina Mahajan

A modular finite element method (FEM) solver written in Python for two-dimensional steady-state diffusion problems.

The code supports both four-node quadrilateral (Q4) and three-node triangular (T3) elements and includes analytical verification, mesh-refinement studies, isotropic and anisotropic diffusion, and benchmark problems with nontrivial numerical behavior.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ihina17/fem-diffusion-solver.git
cd fem-diffusion-solver
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

The main dependencies are:

- NumPy
- SciPy
- Matplotlib

---

## Project Structure

```text
fem-diffusion-solver/
│
├── src/
│   ├── fem/
│   │   ├── create_id.py
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
├── tests/
│   └── test_error_calculation.py
│
├── figures/
│
├── requirements.txt
├── .gitignore
└── README.md
```

### `src/fem/`

Contains the core finite element framework.

- `create_id.py` — global equation and constraint numbering.
- `gauss_quadrature.py` — numerical integration rules for supported element types.
- `shape_functions.py` — finite element shape functions and their derivatives.

### `src/meshes/`

Contains mesh-generation routines.

- `Q4.py` — quadrilateral meshes for the unit square, L-shaped domain, and square-hole domain.
- `T3.py` — triangular meshes generated from the corresponding Q4 meshes.

### `src/physics_models/`

Contains the steady-state diffusion implementation.

- `diffusion_driver.py` — manages the overall FEM solution procedure.
- `diffusion_kernel.py` — calculates element matrices, load vectors, and performs global assembly.

### `src/error_calculation.py`

Calculates numerical errors using the analytical solution and gradient, including the L2 norm and H1 seminorm.

### `problems/`

Contains the numerical examples used to verify and demonstrate the solver.

- `diffusion_convergence.py` — analytical convergence study using Q4 and T3 elements.
- `l_shaped_domain.py` — diffusion on a domain with a re-entrant corner.
- `square_hole_domain.py` — strongly anisotropic diffusion on a square domain containing a square hole.

### `tests/`

Contains automated tests for numerical routines.

### `figures/`

Contains figures generated from the numerical examples and mesh-refinement studies.

---

## Features

- Q4 quadrilateral elements
- T3 triangular elements
- Gaussian quadrature
- Shape-function evaluation
- Global matrix assembly
- Dirichlet boundary conditions
- Isotropic diffusion
- Anisotropic diffusion
- Analytical error calculation
- Mesh-refinement studies
- L2 and H1 convergence analysis
- Concentration-field visualization
- Gradient analysis

---

## Numerical Verification

The FEM implementation is verified using a manufactured analytical solution on the unit square.

Uniform mesh refinement is performed using both Q4 and T3 elements.

The observed convergence rates are:

| Element | L2 rate | H1 rate |
|---|---:|---:|
| Q4 | 2.0000 | 1.0000 |
| T3 | 2.0000 | 1.0000 |

Both formulations recover the expected second-order convergence in the L2 norm and first-order convergence in the H1 seminorm.

![Q4 and T3 convergence](figures/diffusion_convergence_q4_t3.png)

---

## L-Shaped Domain

The first benchmark considers steady-state diffusion on an L-shaped domain with homogeneous boundary conditions.

The geometry contains a re-entrant corner, which produces a singular concentration gradient. Q4 and T3 elements are compared under uniform mesh refinement.

### Concentration field

![L-shaped Q4 concentration](figures/l_shape_concentration_q4.png)

### Gradient refinement

![L-shaped gradient refinement](figures/l_shape_gradient_refinement_comparison.png)

The maximum sampled concentration gradient increases as the mesh is refined, consistent with singular behavior near the re-entrant corner.

Additional Q4 and T3 concentration and gradient plots are available in the `figures/` directory.

---

## Strongly Anisotropic Diffusion

The second benchmark considers a square domain containing a square hole with strongly directional anisotropic diffusion.

The inner boundary is prescribed a concentration of 1, while the outer boundary is prescribed a concentration of 0.

Both Q4 and T3 discretizations produce an elongated concentration field along the preferred diffusion direction.

### Q4 concentration field

![Square-hole Q4 concentration](figures/square_hole_concentration_q4.png)

### Q4 vs T3 under mesh refinement

![Minimum concentration comparison](figures/square_hole_minimum_concentration_comparison.png)

The standard Galerkin solutions exhibit negative nodal concentrations for both element formulations. The numerical undershoot persists over the mesh resolutions investigated.

![Negative concentration percentage](figures/square_hole_negative_percentage_comparison.png)

Additional mesh and concentration plots are available in the `figures/` directory.

---

## Tests

The repository includes automated numerical tests for the error-calculation routines.

Current test suite:

```text
29 tests passed
```

---

## Scope

This project currently focuses on two-dimensional steady-state diffusion with prescribed Dirichlet boundary conditions.

Possible future extensions include:

- higher-order finite elements
- adaptive mesh refinement
- transient diffusion
- mixed boundary conditions
- positivity-preserving formulations