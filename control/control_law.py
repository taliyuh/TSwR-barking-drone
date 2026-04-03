import numpy as np

# math helper functions
def ortho(v1, v2):
    """
    docstring for ortho
    
    :param v1: first vector, should be a list or array of length 2
    :param v2: second vector, should be a list or array of length 2

    this function orthogonalises v2 with respect to v1 and returns the result
    math:
    f(v1, v2) = v2 - (v1 . v2) * v1
    """

    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    # orthogonalise v2 with respect to v1:
    return v2 - np.dot(v1, v2) * v1

def normalise_ortho(v1, v2):
    """
    this function orthogonalises v2 with respect to v1 and normalises the result
    math:
    f(v1, v2) = (v2 - (v1 . v2) * v1) / ||v2 - (v1 . v2) * v1||
    """

    ortho_v = ortho(v1, v2)
    norm_val = np.linalg.norm(ortho_v)
    if norm_val == 0:
        return np.zeros(2)
    return ortho_v / norm_val  
    
def check_direction(v1, v2):
    """
    this function checks if the two vectors point in the same direction
    math:
    return 1 if (v1 . v2) > 0 and -1 if (v1 . v2) <= 0
    """

    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    if np.dot(v1, v2) > 0:
        return 1
    else:
        return -1
    
def closest_point_vector(drone_position, vertex1, vertex):
    """
    Docstring for closest_point_vector
    
    :param drone_position: xy position of the drone, should be a list or array of length 2
    :param vertex1: first vertex of the edge, should be a list or array of length 2
    :param vertex: second vertex of the edge, should be a list or array of length 2
    
    find the shortest vector from the drone position to the edge defined by vertex1 and vertex
    
    """

    q_vector = vertex - vertex1
    p_vector = drone_position - vertex1

    # projection scalar t
    t = np.dot(p_vector, q_vector) / np.dot(q_vector, q_vector)

    if t < 0:
        closest_point = vertex1
        return -p_vector
    elif t > 1:
        closest_point = vertex
        return q_vector - p_vector
    else:
        return (t * q_vector) - p_vector
    
def fly_to_edge(drone, vertex1, vertex):
    """
    Docstring for fly_to_edge
    
    :param drone: instance of the Drone class
    :param vertex1: first vertex of the edge, should be a list or array of length 2
    :param vertex: second vertex of the edge, should be a list or array of length 2
    
    drone should fly towards the edge defined by vertex1 and vertex, using the control law defined in the article
    """

    # b(t) - pulls drone towards a vertex
    b = closest_point_vector(drone.d, vertex1, vertex)

    if np.linalg.norm(b) < 0.001:
        drone.u = np.zeros(2)
        drone.v = 0
        return drone.u, drone.v
    
    drone.u = drone.u_max * check_direction(drone.a, b) * normalise_ortho(drone.a, b)
    drone.v = drone.v_max * check_direction(drone.a, b)

    return drone.u, drone.v

def fly_on_edge(drone, vertex1, vertex, target_point):
    """
    Docstring for fly_on_edge

    :param target_point: point on the edge that the drone should fly towards, should be a list or array of length 2

    drone should fly along the edge defined by vertex1 and vertex towards the target_point, using the control law defined in the article
    """

    b = closest_point_vector(drone.d, vertex1, vertex)
    
    # o*(t) - vector from drone position to the target point
    o_star = target_point - drone.d

    # master vector:
    b_star = b + o_star

    if np.linalg.norm(b_star) < 0.001:
        drone.u = np.zeros(2)
        drone.v = 0
        return drone.u, drone.v
    
    drone.u = drone.u_max * check_direction(drone.a, b_star) * normalise_ortho(drone.a, b_star)
    drone.v = drone.v_max * check_direction(drone.a, b_star)

    return drone.u, drone.v