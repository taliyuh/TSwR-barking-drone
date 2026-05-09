import numpy as np
from drone import Drone
from swarm_control import polygon_to_line, point_to_polygon_distance, send_drones_wherever
from control_law import fly_on_edge, closest_point_vector

class SwarmManager:
    def __init__(self, number_of_drones, initial_positions, v_max, u_max):
        """
        Initialises the swarm manager and its drones.
        
        :param number_of_drones: amount of drones in the swarm
        :param initial_positions: list of starting positions of the drones
        :param v_max: maximal speed of the drones
        :param u_max: maximal control input (acceleration) of the drones
        """
        self.drones = []
        
        # initialise the drones
        for i in range(number_of_drones):
            drone = Drone(initial_positions[i], [1.0, 0.0], v_max, u_max)
            self.drones.append(drone)


    def update_swarm(self, dt, vertices, target_points, gains=(1.0, 1.0, 1.0)):
        """
        Updates the physical state of the whole swarm.
        
        :param dt: time step for the update
        :param vertices: list of 2d positions of vertices of the polygon
        :param target_points: list of 2d positions of target points
        :param gains: tuple of (k_edge, k_target, v_scale) for control law
        """

        # find best allocation of the drones
        drone_1d, target_1d, polygon_length = polygon_to_line(vertices, target_points, self.drones)        
        best_gamma, self.best_allocation, self.best_directions = send_drones_wherever(drone_1d, target_1d, polygon_length)
        
        if self.best_allocation is not None:
            assigned_targets = self.best_allocation
            assigned_directions = self.best_directions
        else: 
            return
        
        k_edge, k_target, v_scale = gains

        for i, drone in enumerate(self.drones):
            
            # find edge closest to the drone
            min_dist = np.inf
            closest_v1 = None
            closest_v2 = None

            for j in range(len(vertices)):
                v1 = vertices[j]
                v2 = vertices[(j+1) % len(vertices)]

                b_vector = closest_point_vector(drone.d, v1, v2)
                distance = np.linalg.norm(b_vector)
                
                if distance < min_dist:
                    min_dist = distance
                    closest_v1 = v1
                    closest_v2 = v2
            
            assigned_target = assigned_targets[i]
            assigned_direction = assigned_directions[i]
            target_2d = None

            for k, t in enumerate(target_1d):
                if np.isclose(t, assigned_target):
                    target_2d = target_points[k]
                    break

            dist_v1 = np.linalg.norm(np.array(closest_v1) - np.array(target_2d))
            dist_v2 = np.linalg.norm(np.array(closest_v2) - np.array(target_2d))
            edge_length = np.linalg.norm(np.array(closest_v1) - np.array(closest_v2))
            
            if np.isclose(dist_v1 + dist_v2, edge_length, atol=1e-5):
                intermediate_target = target_2d
            else:
                if assigned_direction == 1:
                    intermediate_target = closest_v2
                else:
                    intermediate_target = closest_v1
            
            u, v = fly_on_edge(drone, closest_v1, closest_v2, intermediate_target, 
                               k_edge=k_edge, k_target=k_target, v_scale=v_scale)

            drone.update_state(dt, u, v)
    
    def get_swarm_status(self):
        """
        Retrieves the current status of the swarm.
        
        :return positions_list: list of current xy positions of all drones
        :return heading_vectors_list: list of current heading vectors of all drones
        """
        positions_list = []
        heading_vectors_list = []

        for drone in self.drones:
            positions_list.append(drone.d)
            heading_vectors_list.append(drone.a)

        return positions_list, heading_vectors_list
    
    def update_driving(self, dt, patrol_points, gains=(1.0, 1.0, 1.0)):
        """
        Executes the side-to-side sweeping motion for the driving phase.
        patrol_points is a list of length (number_of_drones + 1).
        :param gains: tuple of (k_edge, k_target, v_scale) for control law
        """
        # Initialize patrol directions if they don't exist yet (1 for forward, -1 for backward)
        if not hasattr(self, 'patrol_directions'):
            self.patrol_directions = [1] * len(self.drones)

        k_edge, k_target, v_scale = gains

        for i, drone in enumerate(self.drones):
            p1 = patrol_points[i]
            p2 = patrol_points[i + 1]
            
            # Determine current target based on patrol direction
            target = p2 if self.patrol_directions[i] == 1 else p1
            
            # Check if drone reached the end of its patrol segment
            dist_to_target = np.linalg.norm(drone.d - target)
            if dist_to_target < 0.5:
                self.patrol_directions[i] *= -1  # Swap direction
                target = p1 if self.patrol_directions[i] == 1 else p2

            # Use the sliding mode edge logic to slide along the segment
            u, v = fly_on_edge(drone, p1, p2, target,
                               k_edge=k_edge, k_target=k_target, v_scale=v_scale)
            drone.update_state(dt, u, v)