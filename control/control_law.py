import numpy as np
# math helper functions
# contents:
# cloesest_point_vector - fnds the shortes vector from the drone to a given edge
# fly_to_edge - simple controller that steers the drone towards the edge
# fly_on_edge = controller that keeps the drone on the edge and moves it towards a target point along that edge

def ortho(v1, v2):
    """
    Orthogonalises v2 with respect to v1.
    
    :param v1: first vector, should be a list or array of length 2
    :param v2: second vector, should be a list or array of length 2
    :return: orthogonalised vector

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
    Orthogonalises v2 with respect to v1 and normalises the result.

    :return: normalised orthogonalised vector

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
    Checks if v1 and v2 point in the same direction.

    :return: 1 if v1 and v2 point in the same direction, -1 otherwise

    math:
    return 1 if (v1 . v2) > 0 and -1 if (v1 . v2) <= 0
    """

    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)

    if np.dot(v1, v2) > 0:
        return 1
    else:
        return -1
    
def closest_point_vector(drone_position, vertex1, vertex2):
    """
    Finds the shortest vector from the drone position to the edge.
    
    :param drone_position: xy position of the drone, should be a list or array of length 2
    :param vertex1: first vertex of the edge, should be a list or array of length 2
    :param vertex2: second vertex of the edge, should be a list or array of length 2
    :return: vector from the drone position to the closest point on the edge defined by vertex1 and vertex2

    find the shortest vector from the drone position to the edge defined by vertex1 and vertex2
    
    """
    drone_position = np.array(drone_position, dtype=float)
    vertex1 = np.array(vertex1, dtype=float)
    vertex2 = np.array(vertex2, dtype=float)

    q_vector = vertex2 - vertex1
    p_vector = drone_position - vertex1

    # projection scalar t
    t = np.dot(p_vector, q_vector) / np.dot(q_vector, q_vector)

    # if 0 < t < 1, the closest point is on the edge
    if t < 0:
        return -p_vector
    # if t > 1, the closest point is vertex1
    elif t > 1:
        return q_vector - p_vector
    # if t < 0, the closest point is vertex2
    else:
        return (t * q_vector) - p_vector
    
def fly_to_edge(drone, vertex1, vertex2):
    """
    Calculates control inputs to fly the drone towards the edge.
    
    :param drone: instance of the Drone class
    :param vertex1: first vertex of the edge, should be a list or array of length 2
    :param vertex2: second vertex of the edge, should be a list or array of length 2
    :return drone.u: control input (acceleration)
    :return drone.v: velocity

    drone should fly towards the edge defined by vertex1 and vertex2, using the control law defined in the article
    """

    # b(t) - pulls drone towards a vertex
    b = closest_point_vector(drone.d, vertex1, vertex2)

    # if drone is very close to the edge, do not apply fly_to_edge input
    if np.linalg.norm(b) < 0.001:
        drone.u = np.zeros(2)
        drone.v = 0
        return drone.u, drone.v
    
    # unit vector from drone to closest edge point
    b_normalised = b / np.linalg.norm(b)
    # perpendicular part of desired drection relative to current direction
    # it's length tells how much drone should turn
    ortho_component = ortho(drone.a, b_normalised)
    # 0 = perfectly aligned with the edge, 1 = perpendicular to the edge, so we can use it as a steering magnitude  
    steering_magnitude = np.linalg.norm(ortho_component)

    # check if there is a need to steer
    if steering_magnitude > 1e-9:
        steer_direction = ortho_component / steering_magnitude
        drone.u = drone.u_max * steering_magnitude * check_direction(drone.a, b) * steer_direction
    else:
        drone.u = np.zeros(2)

    # alignment is 1 when heading directly towards edge, 0 when perpendicular
    # move faster when alignment is better
    alignment = np.dot(drone.a, b_normalised)
    drone.v = drone.v_max * alignment

    return drone.u, drone.v

def fly_on_edge(drone, vertex1, vertex2, target_point, k_edge=1.0, k_target=1.0, v_scale=1.0):
    """
    Calculates control inputs to fly the drone along the edge towards the target_point.

    :param target_point: point on the edge that the drone should fly towards, should be a list or array of length 2
    :param k_edge: gain for attraction to the edge
    :param k_target: gain for attraction to the target
    :param v_scale: velocity multiplier
    :return drone.u: control input (acceleration)
    :return drone.v: velocity

    drone should fly along the edge defined by vertex1 and vertex2 towards the target_point, using the control law defined in the article
    """

    b = closest_point_vector(drone.d, vertex1, vertex2)
    
    # o*(t) - vector from drone position to the target point
    o_star = target_point - drone.d

    # master vector
    # weghted sum of two desires:
    # b that attracts the drone to the edge
    # o_star that attracts the drone to the target point along the edge
    b_star = (k_edge * b) + (k_target * o_star)

    # avoid jitter
    if np.linalg.norm(b_star) < 0.001:
        drone.u = np.zeros(2)
        drone.v = 0
        return drone.u, drone.v
    
    # rest similar to fly_to_edge, but with b_star instead of b
    b_star_normalised = b_star / np.linalg.norm(b_star)
    ortho_component = ortho(drone.a, b_star_normalised)
    steering_magnitude = np.linalg.norm(ortho_component)

    if steering_magnitude > 1e-9:
        steer_direction = ortho_component / steering_magnitude
        drone.u = drone.u_max * steering_magnitude * check_direction(drone.a, b_star) * steer_direction
    else:
        drone.u = np.zeros(2)

    alignment = np.dot(drone.a, b_star_normalised)
    drone.v = drone.v_max * alignment * v_scale

    return drone.u, drone.v