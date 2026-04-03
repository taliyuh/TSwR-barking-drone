import numpy as np

class Drone:
    def __init__(self, initial_position, initial_heading, v_max, u_max):
        """
        Initialises the physical model of the drone.
        
        :param initial_position: starting position of the drone, should be a list or array of length 2
        :param initial_heading: starting heading direction of the drone, should be a list or array of length 2
        :param v_max: maximal speed of the drone
        :param u_max: maximal control input (acceleration) of the drone

        this function initialises the physical model of the drone
        """

        # using symbolic convention from the article
        # xy position vector
        self.d = np.array(initial_position, dtype=float)

        # normalisig heading vector
        a_not_normal = np.array(initial_heading, dtype=float)
        try:
            self.a = a_not_normal / np.linalg.norm(a_not_normal)
        except ZeroDivisionError:
            raise ValueError("initial heading vector cannot be zero")
        
        self.v = 0.0
        self.v_max = v_max
        self.u = np.zeros(2)
        self.u_max = u_max

    def update_state(self, dt, u, v):
        """
        Updates the state of the drone based on the control input and velocity.
        
        :param dt: time step for the update
        :param u: control input (acceleration) to be applied, should be a scalar
        :param v: velocity to be applied, should be a scalar
        :return: None

        this function updates the state of the drone based on the control input and velocity
        """

        # clip speed v to [0, v_max]:
        self.v = np.clip(v, -self.v_max, self.v_max)

        # clip control input u to [-u_max, u_max]:
        u_norm = np.linalg.norm(u)
        if u_norm > self.u_max:
            u = (u / u_norm) * self.u_max

        # orthogonalise the control input to the heading direction
        u = u - np.dot(u, self.a) * self.a

        # euler integration to update the state
        # derivative of heading:
        a_dot = u
        
        # derivative of position:
        d_dot = self.v * self.a

        # update state variables based on the time step
        self.a = self.a + a_dot * dt
        self.d = self.d + d_dot * dt

        # normalise the heading vector after the update
        heading_norm = np.linalg.norm(self.a)
        if heading_norm == 0:
            raise ValueError("Heading vector cannot be zero")
        self.a = self.a / heading_norm