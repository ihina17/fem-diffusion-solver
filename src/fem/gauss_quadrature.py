"""
Finite Element Framework - Gauss Quadrature
Master Function: GaussPoints
Sub Functions: Gauss_1D, Gauss_2D, Gauss_3D
"""


#Importing libraries
import pandas as pd
import numpy as np
from numpy.polynomial.legendre import leggauss

#Gauss Points

def GaussPoints(dim: int, EleType: str, NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gauss quadrature points and weights for 1D, 2D or 3D elements.

    Parameters
    ----------
    dim : int
        Dimension of the element(1, 2 or 3)
    EleType : str
        Element type (e.g., "Q4" for quadilateral, "b8" for 3D brick)
        Only required for 2D and 3D elements
    NGPTS : int
        Number of Gauss quadrature points per direction
    
    Returns
    --------
    r: np.ndarray
        Array of Gauss point coordinates
    w: np.ndarray
        Array of corresponding Gauss weights.add()
    
    Raises
    -------
    TypeError
        If 'dim' is not an integer.
    ValueError
        If 'dim' is not either 1, 2 or 3.
    
    Notes
    ------
    - For 'dim = 1, calls Gauss_1D.
    - For 'dim = 2', calls Gauss_2D with the specified element type.
    - For 'dim = 3', calls Gauss_3D with the specified element type.
    - The element type must match the dimensionality.
    """

    # Validating dim variable
    if not isinstance(dim, int):
        raise TypeError("Dimension must be an integer.")
    if dim < 1 or dim > 3:
        raise ValueError("Dimension must be 1, 2, or 3.")
    
    # Assigning functions based on dimension

    if dim == 1:
        r, w = Gauss_1D(NGPTS)
    
    elif dim == 2:
        r, w = Gauss_2D(EleType, NGPTS)

    elif dim == 3:
        r, w = Gauss_3D(EleType, NGPTS)
    
    return r, w


# Gauss Points for 1D elements
def Gauss_1D(NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gauss–Legendre quadrature points and weights for 1D elements 

    Parameters
    ----------
    NGPTS : int
        Number of Gauss quadrature points.

    Returns
    ----------
    r : np.ndarray
        Gauss points.
    w : np.ndarray
        Corresponding Gauss weights.

    Raises
    ----------
    TypeError
        If `NGPTS` is not an integer.
    ValueError
        If `NGPTS` is less than 1.

    Notes
    ----------
    - Uses NumPy’s `leggauss` function to compute the points and weights.
    """
    
    # Validating NGPTS
    if not isinstance(NGPTS, int):
        raise TypeError("Number of Gauss points must be an integer.")
    
    # Using legendre polynomial in-built function
    return leggauss(NGPTS)


# Gauss Points for 2D elements
def Gauss_2D(EleType: str, NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gauss quadrature points and weights for 2D elements.

    Parameters
    ----------
    EleType : str
        Element type. Supported:
        "q4" : 2D quadrilateral
        "t3" : Triangular element
    NGPTS : int
        Number of Gauss quadrature points.

    Returns
    ----------
    r : np.ndarray
        Gauss points.
    w : np.ndarray
        Corresponding Gauss weights.

    Raises
    ----------
    TypeError
        If `NGPTS` is not an integer or `EleType` is not a string.
    ValueError
        If `EleType` is unsupported, or `NGPTS` is unsupported for "t3".

    Notes
    ----------
    - "q4" uses NumPy’s `leggauss` in each direction.
    - "t3" NGPTS are 1, 3, 4, 7, 9, 12, 13.
    """
    
    # Validating NGPTS
    if not isinstance(NGPTS, int):
        raise TypeError("Number of Gauss points must be an integer.")
    if not isinstance(EleType, str):
        raise TypeError("EleType must be a string.")

    # Converting Element type to lowercase
    EleType = EleType.lower()

    # Validating Element Type
    allowed = {"q4", "t3"}
    if EleType not in allowed:
        raise ValueError(f"Bad element type: {EleType!r}. Supported: {sorted(allowed)}")
    
    # Square element
    if EleType == 'q4':
        r1d, w1d = leggauss(NGPTS)
        rx = np.tile(r1d, NGPTS)
        ry = np.repeat(r1d, NGPTS)

        w = np.tile(w1d, NGPTS) * np.repeat(w1d, NGPTS)

        return (np.column_stack((rx, ry)), w)

    # Three node right-angled triangle element values hard coded
    elif EleType == 't3':

        if NGPTS == 1:
            r = np.array([[1.0/3.0, 1.0/3.0]])
            w = np.array([0.5])

        elif NGPTS == 3:
            r = 0.5 * np.array([[1.0, 1.0],[1.0, 0.0],[0.0, 1.0]])
            w = (1.0/6.0) * np.ones(3)

        elif NGPTS == 4:
            r = np.array([
                    [1/3, 1/3],
                    [0.6, 0.2],
                    [0.2, 0.6],
                    [0.2, 0.2],
                ])
            w = 0.5 * np.array([
                    -0.5625,
                    0.520833333333333,
                    0.520833333333333,
                    0.520833333333333
                ])

        elif NGPTS == 7:
            r = np.array([
                [0.333333333333333, 0.333333333333333],
                [0.797426985353087, 0.101286507323456],
                [0.101286507323456, 0.797426985353087],
                [0.101286507323456, 0.101286507323456],
                [0.470142064105115, 0.059715871789770],
                [0.059715871789770, 0.470142064105115],
                [0.470142064105115, 0.470142064105115],
            ])
            w = 0.5 * np.array([
                0.225000000000000,
                0.125939180544827,
                0.125939180544827,
                0.125939180544827,
                0.132394152788506,
                0.132394152788506,
                0.132394152788506
            ])

        elif NGPTS == 9:
            r = np.array([
                [0.124949503233232, 0.437525248383384],
                [0.437525248383384, 0.124949503233232],
                [0.437525248383384, 0.437525248383384],
                [0.797112651860071, 0.165409927389841],
                [0.797112651860071, 0.037477420750088],
                [0.165409927389841, 0.797112651860071],
                [0.165409927389841, 0.037477420750088],
                [0.037477420750088, 0.797112651860071],
                [0.037477420750088, 0.165409927389841],
            ])
            w = 0.5 * np.array([
                0.205950504760887,
                0.205950504760887,
                0.205950504760887,
                0.063691414286223,
                0.063691414286223,
                0.063691414286223,
                0.063691414286223,
                0.063691414286223,
                0.063691414286223
            ])

        elif NGPTS == 12:
                r = np.array([
                    [0.873821971016996, 0.063089014491502],
                    [0.063089014491502, 0.873821971016996],
                    [0.063089014491502, 0.063089014491502],
                    [0.501426509658179, 0.249286745170910],
                    [0.249286745170910, 0.501426509658179],
                    [0.249286745170910, 0.249286745170910],
                    [0.636502499121399, 0.310352451033785],
                    [0.636502499121399, 0.053145049844816],
                    [0.310352451033785, 0.636502499121399],
                    [0.310352451033785, 0.053145049844816],
                    [0.053145049844816, 0.636502499121399],
                    [0.053145049844816, 0.310352451033785],
                ])
                w = 0.5 * np.array([
                    0.050844906370207,
                    0.050844906370207,
                    0.050844906370207,
                    0.116786275726379,
                    0.116786275726379,
                    0.116786275726379,
                    0.082851075618374,
                    0.082851075618374,
                    0.082851075618374,
                    0.082851075618374,
                    0.082851075618374,
                    0.082851075618374
                ])

        elif NGPTS == 13:
            r = np.array([
                [0.333333333333333, 0.333333333333333],
                [0.479308067841923, 0.260345966079038],
                [0.260345966079038, 0.479308067841923],
                [0.260345966079038, 0.260345966079038],
                [0.869739794195568, 0.065130102902216],
                [0.065130102902216, 0.869739794195568],
                [0.065130102902216, 0.065130102902216],
                [0.638444188569809, 0.312865496004875],
                [0.638444188569809, 0.086903154253160],
                [0.312865496004875, 0.638444188569809],
                [0.312865496004875, 0.086903154253160],
                [0.086903154253160, 0.638444188569809],
                [0.086903154253160, 0.312865496004875],
            ])
            w = 0.5 * np.array([
                -0.149570044467670,
                0.175615257433204,
                0.175615257433204,
                0.175615257433204,
                0.053347235608839,
                0.053347235608839,
                0.053347235608839,
                0.077113760890257,
                0.077113760890257,
                0.077113760890257,
                0.077113760890257,
                0.077113760890257,
                0.077113760890257
            ])

        else:
            raise ValueError(f"Unsupported NGPTS={NGPTS} for T3.")

    
        return r, w

# Gauss Points for 3D elements
def Gauss_3D(EleType: str, NGPTS: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Gauss quadrature points and weights for 3D elements.

    Parameters
    ----------
    EleType : str
        Element type. Supported:
        "b8" : 3D cube
        "tet4" : tetrahedral
    NGPTS : int
        Number of Gauss quadrature points.

    Returns
    ----------
    r : np.ndarray
        Gauss points.
    w : np.ndarray
        Corresponding Gauss weights.

    Raises
    ----------
    TypeError
        If `NGPTS` is not an integer or `EleType` is not a string.
    ValueError
        If `EleType` is unsupported, or `NGPTS` is unsupported for "t3".

    Notes
    ----------
    - "b8" uses NumPy’s `leggauss` in each direction.
    - "tet4" NGPTS are 1, 4, 5, 11, 15.
    """

    # Validating NGPTS
    if not isinstance(NGPTS, int):
        raise TypeError("Number of Gauss points must be an integer.")
    if not isinstance(EleType, str):
        raise TypeError("EleType must be a string.")
    
    EleType = EleType.lower()
    
    # Validating element type
    allowed = {"b8", "tet4"}
    if EleType not in allowed:
        raise ValueError(f"Bad element type: {EleType!r}. Supported: {sorted(allowed)}")

    # Brick element
    if EleType == 'b8':
        r1d, w1d = leggauss(NGPTS)
        rx = np.tile(r1d, NGPTS * NGPTS)
        ry = np.repeat(r1d,  NGPTS * NGPTS)
        rz = np.tile(np.repeat(r1d, NGPTS), NGPTS)

        w = np.tile(w1d, NGPTS * NGPTS) * np.repeat(w1d, NGPTS * NGPTS) * np.tile(np.repeat(w1d, NGPTS), NGPTS)

        return (np.column_stack((rx, ry, rz)), w)

    # Four node right-angled tetrahedron element
    elif EleType == 'tet4':

        if NGPTS == 1:
            r = np.array([[1.0/4.0, 1.0/4.0, 1.0/4.0]])
            w = np.array([1.0/6.0])

        elif NGPTS == 4:
            p1 = 0.5854101966249638
            p2 = 0.1381966011250105
            r = np.array([
                [p1, p2, p2],
                [p2, p1, p2],
                [p2, p2, p1],
                [p1, p1, p1], 
            ], dtype=float)
            w = (1.0 / (6.0 * 4.0)) * np.array([1.0, 1.0, 1.0, 1.0])

        elif NGPTS == 5:
            r = np.array([
                (1.0/4.0) * np.array([1.0, 1.0, 1.0]),
                [1.0/2.0, 1.0/6.0, 1.0/6.0],
                [1.0/6.0, 1.0/2.0, 1.0/6.0],
                [1.0/6.0, 1.0/6.0, 1.0/2.0],
                [1.0/6.0, 1.0/6.0, 1.0/6.0],
            ])
            w = (1.0/6.0) * np.array([
                -4.0/5.0,
                9.0/20.0,
                9.0/20.0,
                9.0/20.0,
                9.0/20.0
            ])

        elif NGPTS == 11:
            p1 = 0.250000000000000
            p2 = 0.785714285714286
            p3 = 0.071428571428571
            p4 = 0.399403576166799
            p5 = 0.100596423833201

            r = np.array([
                [p1, p1, p1],
                [p2, p3, p3],
                [p3, p2, p3],
                [p3, p3, p2],
                [p3, p3, p3],
                [p4, p5, p5],
                [p5, p4, p5],
                [p5, p5, p4],
                [p5, p4, p4],
                [p4, p5, p4],
                [p4, p4, p5],
            ])

            q1 = -0.013155555555556
            q2 =  0.007622222222222
            q3 =  0.024888888888889
            w = np.array([q1, q2, q2, q2, q2, q3, q3, q3, q3, q3, q3])

        elif NGPTS == 15:
            p1 = 0.250000000000000
            p2 = 0.000000000000000
            p3 = 0.333333333333333
            p4 = 0.727272727272727
            p5 = 0.090909090909091
            p6 = 0.066550153573664
            p7 = 0.433449846426336

            r = np.array([
                [p1, p1, p1],
                [p2, p3, p3],
                [p3, p2, p3],
                [p3, p3, p2],
                [p3, p3, p3],
                [p4, p5, p5],
                [p5, p4, p5],
                [p5, p5, p4],
                [p5, p5, p5],
                [p6, p7, p7],
                [p7, p6, p7],
                [p7, p7, p6],
                [p7, p6, p6],
                [p6, p7, p6],
                [p6, p6, p7],
            ])

            q1 = 0.030283678097089
            q2 = 0.006026785714286
            q3 = 0.011645249086029
            q4 = 0.010949141561386
            w = np.array([q1, q2, q2, q2, q2, q3, q3, q3, q3, q4, q4, q4, q4, q4, q4])

        else:
            raise ValueError(f"Unsupported NGPTS={NGPTS} for TET4.")

    
        return r, w
