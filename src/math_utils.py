# src/utils.py

def normalize(v):
    """
    Normalize a 3D vector.

    Args:
        v (list): A list of three elements representing the vector.

    Returns:
        list: The normalized vector. If the input vector has zero norm, the original vector is returned.
    """
    norm = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
    return [v[i] / norm for i in range(3)] if norm != 0 else v

def midpoint(v1, v2):
    """
    Calculate the midpoint of two 3D vectors.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.

    Returns:
        list: The midpoint vector.
    """
    return [(v1[i] + v2[i]) / 2 for i in range(3)]


def norm(vec):
    """
    Calculate the norm (magnitude) of a 3D vector.

    Args:
        vec (list): A list of three elements representing the vector.

    Returns:
        float: The norm of the vector.
    """
    return (vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])**0.5

def dot_product(v1, v2):
    """
    Calculate the dot product of two 3D vectors.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.

    Returns:
        float: The dot product of the two vectors.
    """
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]

def cross_product(v1, v2):
    """
    Calculate the cross product of two 3D vectors.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.

    Returns:
        list: The cross product vector.
    """
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]

def is_same_direction(v1, v2, v3):
    """
    Check if the cross product of v1 and v2 is in the same direction as v3.

    Args:
        v1 (list): The first vector.
        v2 (list): The second vector.
        v3 (list): The third vector to compare against.

    Returns:
        bool: True if the cross product of v1 and v2 is in the same direction as v3, False otherwise.
    """
    # Calculate the cross product of v1 and v2
    cross_prod = cross_product(v1, v2)

    # Normalize both the cross product and v3 to compare directions
    cross_prod_normalized = normalize(cross_prod)
    v3_normalized = normalize(v3)

    # Calculate the dot product of the normalized vectors
    dot_prod = dot_product(cross_prod_normalized, v3_normalized)

    # Debug the dot product value
    # logger.debug(f"Dot product : {dot_prod}")

    # Check if the dot product is close to 1 (same direction)
    return abs(dot_prod - 1) < 1e-6