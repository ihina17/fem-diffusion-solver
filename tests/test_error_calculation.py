import numpy as np

from src.error_calculation import Calculate_Error


def test_linear_field_q4():
    """
    Verify that the error calculation reproduces
    a linear analytical solution exactly.
    """

    # --------------------------------------------------------
    # 1. Single Q4 element
    # --------------------------------------------------------

    Coord = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.5, 1.0],
        [0.5, 1.0],
    ])

    # 1-based node numbering
    Connectivity = np.array([
        [1, 2, 3, 4],
    ])

    # --------------------------------------------------------
    # 2. Exact analytical solution
    #
    # c(x,y) = 1 + 2x - 3y
    # --------------------------------------------------------

    def exact_solution(x, y):
        return 1.0 + 2.0*x - 3.0*y

    def exact_gradient(x, y):
        return np.array([2.0, -3.0])

    # --------------------------------------------------------
    # 3. Assign exact nodal values
    # --------------------------------------------------------

    U = (
        1.0
        + 2.0 * Coord[:, 0]
        - 3.0 * Coord[:, 1]
    )

    # --------------------------------------------------------
    # 4. Calculate errors
    # --------------------------------------------------------

    L2_error, H1_error = Calculate_Error(
        Connectivity=Connectivity,
        Coord=Coord,
        EleType="Q4",
        NGPTS=2,
        U=U,
        exact_solution=exact_solution,
        exact_gradient=exact_gradient,
    )

    print("\nL2 error:", L2_error)
    print("H1 seminorm error:", H1_error)

    # --------------------------------------------------------
    # 5. Verify results
    # --------------------------------------------------------

    assert L2_error < 1e-12
    assert H1_error < 1e-12