import numpy as np

def generate_l_shaped_q4_mesh(n):
    """
    Generate a structured Q4 mesh for the L-shaped domain.

    Domain:

        Omega = [0,2] x [0,2]
                minus
                [1,2] x [1,2]

    The re-entrant corner is:

        A = (1,1)

    Parameters
    ----------
    n : int
        Number of elements per unit length.

        For example:
            n = 2  -> element size h = 0.5
            n = 4  -> element size h = 0.25
            n = 8  -> element size h = 0.125

    Returns
    -------
    Coord : np.ndarray
        Nodal coordinates.

    Connectivity : np.ndarray
        Q4 connectivity using 1-based node numbering.
    """

    if n < 1:
        raise ValueError("n must be at least 1.")

    # --------------------------------------------------
    # Full 2 x 2 background grid
    # --------------------------------------------------

    h = 1.0 / n

    x_values = np.linspace(
        0.0,
        2.0,
        2 * n + 1,
    )

    y_values = np.linspace(
        0.0,
        2.0,
        2 * n + 1,
    )

    # --------------------------------------------------
    # Store only nodes actually used by L-shaped elements
    # --------------------------------------------------

    node_map = {}
    coordinates = []
    connectivity = []

    def get_node_id(x, y):
        """
        Return a 1-based node number for coordinate (x,y).
        Create the node if it does not already exist.
        """

        key = (
            round(float(x), 12),
            round(float(y), 12),
        )

        if key not in node_map:

            node_id = len(coordinates) + 1

            node_map[key] = node_id

            coordinates.append([
                float(x),
                float(y),
            ])

        return node_map[key]

    # --------------------------------------------------
    # Generate elements
    # --------------------------------------------------

    for j in range(2 * n):

        y0 = y_values[j]
        y1 = y_values[j + 1]

        for i in range(2 * n):

            x0 = x_values[i]
            x1 = x_values[i + 1]

            # ------------------------------------------
            # Remove upper-right 1 x 1 quadrant
            # ------------------------------------------

            if x0 >= 1.0 and y0 >= 1.0:
                continue

            # Q4 local ordering:
            #
            #   4 ----- 3
            #   |       |
            #   |       |
            #   1 ----- 2

            n1 = get_node_id(x0, y0)
            n2 = get_node_id(x1, y0)
            n3 = get_node_id(x1, y1)
            n4 = get_node_id(x0, y1)

            connectivity.append([
                n1,
                n2,
                n3,
                n4,
            ])

    Coord = np.array(
        coordinates,
        dtype=float,
    )

    Connectivity = np.array(
        connectivity,
        dtype=int,
    )

    return Coord, Connectivity


import numpy as np


def generate_square_hole_q4_mesh(n):
    """
    Generate a structured Q4 mesh for a unit square
    containing a square hole.

    Domain:
        [0, 1] x [0, 1]

    Square hole:
        [4/9, 5/9] x [4/9, 5/9]

    Parameters
    ----------
    n : int
        Number of elements along each side.
        Must be divisible by 9.
    """

    if n < 9:
        raise ValueError("n must be at least 9.")

    if n % 9 != 0:
        raise ValueError(
            "n must be divisible by 9 so the square hole "
            "aligns with the mesh."
        )

    x_values = np.linspace(
        0.0,
        1.0,
        n + 1,
    )

    y_values = np.linspace(
        0.0,
        1.0,
        n + 1,
    )

    hole_min = 4.0 / 9.0
    hole_max = 5.0 / 9.0

    node_map = {}
    coordinates = []
    connectivity = []

    def get_node_id(x, y):

        key = (
            round(float(x), 12),
            round(float(y), 12),
        )

        if key not in node_map:

            node_id = len(coordinates) + 1

            node_map[key] = node_id

            coordinates.append([
                float(x),
                float(y),
            ])

        return node_map[key]

    for j in range(n):

        y0 = y_values[j]
        y1 = y_values[j + 1]

        for i in range(n):

            x0 = x_values[i]
            x1 = x_values[i + 1]

            # Remove elements inside the square hole
            inside_hole = (
                x0 >= hole_min
                and x1 <= hole_max
                and y0 >= hole_min
                and y1 <= hole_max
            )

            if inside_hole:
                continue

            n1 = get_node_id(x0, y0)
            n2 = get_node_id(x1, y0)
            n3 = get_node_id(x1, y1)
            n4 = get_node_id(x0, y1)

            connectivity.append([
                n1,
                n2,
                n3,
                n4,
            ])

    return (
        np.array(
            coordinates,
            dtype=float,
        ),
        np.array(
            connectivity,
            dtype=int,
        ),
    )


def generate_unit_square_q4_mesh(n):
    """
    Generate a structured Q4 mesh on the unit square.

    Domain:
        0 <= x <= 1
        0 <= y <= 1

    Parameters
    ----------
    n : int
        Number of elements along each coordinate direction.

    Returns
    -------
    Coord : np.ndarray
        Nodal coordinates.

    Connectivity : np.ndarray
        Q4 element connectivity using 1-based node numbering.
    """

    if not isinstance(n, int):
        raise TypeError("n must be an integer.")

    if n < 1:
        raise ValueError("n must be at least 1.")

    # --------------------------------------------------------
    # Nodal coordinates
    # --------------------------------------------------------

    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)

    Coord = []

    for j in range(n + 1):
        for i in range(n + 1):
            Coord.append([
                x[i],
                y[j],
            ])

    Coord = np.array(
        Coord,
        dtype=float,
    )

    # --------------------------------------------------------
    # Q4 connectivity
    #
    # Node ordering:
    #
    #   4 ----- 3
    #   |       |
    #   |       |
    #   1 ----- 2
    #
    # --------------------------------------------------------

    Connectivity = []

    for j in range(n):
        for i in range(n):

            n1 = j * (n + 1) + i + 1
            n2 = n1 + 1
            n4 = n1 + (n + 1)
            n3 = n4 + 1

            Connectivity.append([
                n1,
                n2,
                n3,
                n4,
            ])

    Connectivity = np.array(
        Connectivity,
        dtype=int,
    )

    return Coord, Connectivity