import numpy as np

def apply_rotation(strings: np.ndarray, rotation_angle: float) -> np.ndarray:
    """Apply rotation to a set of string positions around (0, 0)

    Parameters:
    -----------
    strings: np.array, shape N x 2
        Set of N string positions centered around (0, 0)
    rotation_angle: float, radian
        Rotation angle, counter-clockwise

    Returns:
    --------
    rot_strings: np.array, shape N x 2
        Rotated set of string positions
    """
    if np.shape(strings)[-1] != 2:
        raise ValueError(
            f"strings needs to have shape (N, 2) but has shape {np.shape(strings)}."
        )
    rotation_matrix = np.array(
        [
            np.array([np.cos(rotation_angle), -np.sin(rotation_angle)]),
            np.array([np.sin(rotation_angle), np.cos(rotation_angle)]),
        ]
    )
    return rotation_matrix.dot(strings.T).T


def apply_displacement(strings: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """Apply displacement to set of string positions.
    Note that the displacement must be applied only *after* the rotation!

    Parameters:
    -----------
    displacement: np.array, shape 2
        Displacement of set of strings wrt. origin

    Returns:
    --------
    displaced strings: np.array, shape N x 2
    """
    if np.shape(displacement) != (2,):
        raise ValueError(
            f"displacement needs to have shape (2,) but has shape {np.shape(displacement)}."
        )
    return displacement + strings


def apply_transformations(
    strings: np.ndarray, displacement: np.ndarray, rotation_angle: float
) -> np.ndarray:
    """Apply rotation and displacement to baseline set of strings centered around (0, 0).

    Parameters:
    -----------
    strings: np.array, shape N x 2
        Set of N string positions centered around (0, 0).
    displacement: np.array, shape 2
        Displacement of set of strings wrt. origin
    rotation_angle: float, radian
        Rotation angle wrt. x axis, counter-clockwise

    Returns:
    --------
    moved_strings: np.array, shape N x 2
        rotated and displaced strings
    """
    if np.shape(strings)[-1] != 2:
        raise ValueError(
            f"strings needs to have shape (N, 2) but has shape {np.shape(strings)}."
        )
    moved_strings = apply_rotation(strings, rotation_angle)
    moved_strings = apply_displacement(moved_strings, displacement)
    return moved_strings

def create_symmetry(N_symmetry: int, starting_angle=0., N_partial=0) -> np.ndarray:
    """Create a set of strings around (0,0) that is symmetric under N_symmetry rotations.
    
    Parameters:
    -----------
    N_symmetry: int
        Number of rotational symmetry steps, i.e. 5 yields a pentagram or 6 yields a hexagon

    Optional parameters:
    --------------------
    starting_angle = 0.: float, radian
        Angle of first string wrt. x axis, counter-clockwise, default = 0
    N_partial = 0: int, < N_symmetry
        if > 0, only make partial of symmetric shape
        if 0: full shape (default)

    Returns:
    --------
    n_gon: np.ndarray, shape (N_symmetry x 2)
        String positions of n-gon
    
    """
    if N_partial>=N_symmetry:
        raise ValueError(f"N_partial ({N_partial}) is larger than N_symmetry ({N_symmetry}). ")
    sym_angle = np.deg2rad(360/N_symmetry)
    stepper = np.arange(N_symmetry) if N_partial==0 else np.arange(N_partial)
    all_angles = starting_angle + sym_angle * stepper
    n_gon = np.array([np.cos(all_angles), np.sin(all_angles)]).T
    return n_gon
