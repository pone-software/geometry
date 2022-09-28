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
