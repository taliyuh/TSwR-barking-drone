import numpy as np
import itertools
# simplify the 2d problem to 1d
# contents:
# point_to_polygon_distance - project any 2d point onto the polygon and get its 1d coords
# polygon_to_line - convert all drones and target points to 1d
# calculate_travel_dist - how far drone has to fly to reach the target
# check_collision - verigy the drones won't cross each other
# send_drones_wherever - find the best allocation of drones to targets

def point_to_polygon_distance(point, vertices):
    """
    Calculates distance from the start of the polygon line to the given point.
    
    :param point: 2d position of the point
    :param vertices: list of 2d positions of vertices of the polygon
    :return: 1d distance along the polygon perimeter

    calculates distance from the start of the polygon line to the given point
    """

    # avoid type error
    point = np.array(point, dtype=float)
    accumulated_distance = 0.0

    # find the edge that is closest to the given point
    min_dist = np.inf
    closest_1d_pos = 0.0

    for i in range(len(vertices)):
        # iterate through edges of the polygon
        vertex1 = np.array(vertices[i], dtype=float)
        vertex2 = np.array(vertices[(i + 1) % len(vertices)], dtype=float)

        # calculate distance between two vertices
        edge_length = np.linalg.norm(vertex2 - vertex1)

        # project point onto edge
        edge_dir = vertex2 - vertex1
        if edge_length > 0:
            t = np.dot(point - vertex1, edge_dir) / (edge_length ** 2)
            t_clamped = max(0.0, min(1.0, t))
            proj = vertex1 + t_clamped * edge_dir
        else:
            proj = vertex1
            t_clamped = 0.0

        # pick the edge with smallest distance to (closest edge)
        dist_to_edge = np.linalg.norm(point - proj)

        # check if dist(v1, p1) + dist(v2, p1) == dist(v1, v2)
        # by checking which edge has the minimum orthogonal distance
        if dist_to_edge < min_dist:
            min_dist = dist_to_edge
            # if yes, then the point is on the current edge
            # add length of previous edges and distance from v1 to the projected point to get the 1d position
            distance_v1_p1 = t_clamped * edge_length
            closest_1d_pos = accumulated_distance + distance_v1_p1
        
        # if no, add the edge length and continue to the next edge
        accumulated_distance += edge_length

    return closest_1d_pos

def polygon_to_line(vertices, targets, drones):
    """
    Projects drone and target positions onto a 1D line representing the polygon perimeter.
    
    :param vertices: list of 2d positions of vertices of the polygon
    :param targets: list of 2d positions of target points
    :param drones: list of drone objects
    :return drone_positions: 1d list of positions of drones
    :return target_positions: 1d list of target positions
    :return polygon_length: the length of the polygon
    """

    drone_positions = []
    target_positions = []

    # calculate the length of the polygon
    polygon_length = 0
    for i in range(len(vertices)):
        vertex1 = vertices[i]
        vertex2 = vertices[(i + 1) % len(vertices)]

        polygon_length += np.linalg.norm(np.array(vertex2) - np.array(vertex1))
     
    # calculate position of each drone
    for drone in range(len(drones)):
        drone_positions.append(point_to_polygon_distance(drones[drone].d, vertices))

    # and position of each target
    for target in range(len(targets)):
        target_positions.append(point_to_polygon_distance(targets[target], vertices))

    # position of 1st drone
    offset = drone_positions[0]

    # treat position of 1st drone as origin, offset all other positions
    drone_positions = [((drone_position - offset) % polygon_length) for drone_position in drone_positions]
    target_positions = [((target_position - offset) % polygon_length) for target_position in target_positions]

    return drone_positions, target_positions, polygon_length

def calculate_travel_dist(drone_position, target_position, direction, polygon_length):
    """
    Calculates distance to travel from drone_position to target_position in the given direction.
    
    :param drone_position: position of the drone on the line, should be a scalar
    :param target_position: position of the target on the line, should be a scalar
    :param direction: direction of travel, should be 0 (left) or 1 (right)
    :param polygon_length: length of the polygon, should be a scalar
    :return: distance to travel from drone_position to target_position in the given direction

    calculates distance to travel from drone_position to target_position in the given direction
    """

    # check if drone crossed the target, differentiate direction
    # wrapping is to ensure that drone can pass the origins
    if target_position > drone_position and direction == 0:
        wrap_left = 1
    else:
        wrap_left = 0

    if target_position < drone_position and direction == 1:
        wrap_right = 1
    else:
        wrap_right = 0

    # calculate travel distance based on direction
    if direction == 0:
        # unwrapped target coordinate
        z_star = target_position - wrap_left * polygon_length
        # e.g. if drone is at 0.1, target is at 0.9, and direction is left, then wrap_left is 1, so z_star is -0.1, and travel distance is 0.2
        travel_distance = drone_position - z_star

    elif direction == 1:
        z_star = target_position + wrap_right * polygon_length
        travel_distance = z_star - drone_position

    return travel_distance, z_star

def check_collision(z_star_list, polygon_length):
    """
    Checks if there is a collision between drones based on their z_star values.
    
    :param z_star_list: list of z_star values for each drone, should be a list of scalars
    :param polygon_length: length of the polygon, should be a scalar
    :return: True if there is a collision, False otherwise

    checks if there is a collision between drones based on their z_star values
    """

    # the drones are sorted
    # check if their unwrapped positions are also sorted, if not, there is a collision
    # check first and last drone
    if not (0 < (z_star_list[-1] - z_star_list[0]) < polygon_length):
        return True
    
    # check other drones
    for j in range(len(z_star_list) - 1):
        if z_star_list[j] >= z_star_list[j + 1]:
            return True
        
    return False

def send_drones_wherever(drone_positions, target_positions, polygon_length):
    """
    Finds the best allocation of targets to drones and optimal flying directions.
    
    :param drone_positions: list of positions of drones on the line, should be a list of scalars
    :param target_positions: list of positions of targets on the line, should be a list of scalars
    :param polygon_length: length of the polygon, should be a scalar
    :return: list of travel distances for each drone to reach its target

    sends drones to their targets without checking for collisions
    """

    # we must sort the drone positions to ensure that check_collision receives correctly ordered positions,
    # as its validity relies on the cyclic ordering of elements mathematically.
    sorted_indices = np.argsort(drone_positions)
    sorted_drone_positions = [drone_positions[i] for i in sorted_indices]

    # generate list of all possible allocations of targets to drones and all possible flying directions
    possible_targets = itertools.permutations(target_positions)
    possible_flying_directions = list(itertools.product([0, 1], repeat=len(drone_positions)))
    best_allocation = None
    best_gamma = np.inf
    best_directions = None
    
    # iterate through all possible allocations
    for target_allocation in possible_targets:
        for flying_directions in possible_flying_directions:

            current_gammas = []
            current_z_stars = []

            # calculate travel distance for each drone based on the current allocation and flying direction
            for drone in range(len(sorted_drone_positions)):
                travel_distance, z_star = calculate_travel_dist(sorted_drone_positions[drone], target_allocation[drone], flying_directions[drone], polygon_length)
                
                current_gammas.append(travel_distance)
                current_z_stars.append(z_star)

            # ditch if collision occurs
            if check_collision(current_z_stars, polygon_length):
                continue
            
            # check if the current allocation is better than the best one found so far
            if max(current_gammas) < best_gamma:
                best_gamma = max(current_gammas)
                best_allocation = target_allocation
                best_directions = flying_directions

    if best_allocation is not None:
        # map the assignment back to the original unstructured list of drones
        final_allocation = [None] * len(drone_positions)
        final_directions = [None] * len(drone_positions)
        for sorted_idx, original_idx in enumerate(sorted_indices):
            final_allocation[original_idx] = best_allocation[sorted_idx]
            final_directions[original_idx] = best_directions[sorted_idx]
            
        return best_gamma, tuple(final_allocation), tuple(final_directions)

    return best_gamma, best_allocation, best_directions