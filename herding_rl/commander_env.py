import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_swarm_sim import (
    SimConfig, HerdState, AnimalProfile, ANIMAL_PROFILES,
    update_herd, compute_centroid, herd_radius, find_furthest_animal_from_centroid,
    compute_observations,
)
from control.swarm_manager import SwarmManager
from herding_rl.config import EnvConfig, RewardConfig, SimConfigRL


class HerdingCommanderEnv(gym.Env):
    """
    Gymnasium environment for Hierarchical Drone Herding.
    RL Commander outputs waypoints, SMC Layer tracks them.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_cfg: EnvConfig = None, reward_cfg: RewardConfig = None, sim_cfg: SimConfigRL = None):
        super().__init__()
        self.env_cfg = env_cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.sim_cfg = sim_cfg or SimConfigRL()

        self.profile = ANIMAL_PROFILES["sheep"]

        # Action space: 4 drones * 2 (x, y) = 8 continuous values in [-1, 1]
        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(self.env_cfg.n_drones * 2,),
            dtype=np.float32
        )

        # Observation space: 19 features (normalized)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(19,),
            dtype=np.float32
        )

        # Pre-build the SimConfig used for update_herd (avoid re-creating each sub-step)
        self._sim_cfg_for_herd = self._build_sim_config()

        self.herd = None
        self.swarm_manager = None
        self.steps_count = 0
        self.prev_dist_to_goal = 0.0
        self.prev_radius = 0.0
        self.prev_drone_positions = None

    # ------------------------------------------------------------------
    # Curriculum — called from CurriculumCallback via env_method
    # ------------------------------------------------------------------
    def _build_sim_config(self):
        """Build a SimConfig from current env/sim configs."""
        return SimConfig(
            n_animals=self.env_cfg.n_animals,
            n_drones=self.env_cfg.n_drones,
            dt=self.env_cfg.dt,
            world_min=self.sim_cfg.world_min,
            world_max=self.sim_cfg.world_max,
            drone_influence_radius=self.sim_cfg.drone_influence_radius,
            accel_threshold=self.sim_cfg.accel_threshold,
            velocity_damping=self.sim_cfg.velocity_damping,
            drone_vision_radius=self.sim_cfg.drone_vision_radius,
        )

    def set_curriculum(self, n_animals, goal, success_radius):
        """Update difficulty parameters (called between episodes)."""
        self.env_cfg.n_animals = n_animals
        self.sim_cfg.goal_position = goal
        self.sim_cfg.success_radius = success_radius
        # Rebuild the cached SimConfig
        self._sim_cfg_for_herd = self._build_sim_config()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_obs(self):
        positions = self.herd.positions
        centroid = compute_centroid(positions)
        radius = herd_radius(positions, centroid)
        mean_vel = np.mean(self.herd.velocities, axis=0)

        goal = np.array(self.sim_cfg.goal_position)
        dist_to_goal = np.linalg.norm(centroid - goal)
        vec_to_goal = (goal - centroid) / (dist_to_goal + 1e-6)

        # Drone positions relative to centroid
        drone_pos, _ = self.swarm_manager.get_swarm_status()
        drone_feats = []
        for p in drone_pos:
            rel_p = (np.asarray(p) - centroid) / self.env_cfg.world_size
            drone_feats.extend(rel_p)

        # Fraction of animals near goal
        dists_to_goal = np.linalg.norm(positions - goal, axis=1)
        in_rc = float(np.mean(dists_to_goal <= self.sim_cfg.success_radius))
        in_2rc = float(np.mean(dists_to_goal <= (self.sim_cfg.success_radius * 2)))

        obs = np.array([
            (centroid[0] - goal[0]) / self.env_cfg.world_size,
            (centroid[1] - goal[1]) / self.env_cfg.world_size,
            radius / self.env_cfg.world_size,
            mean_vel[0] / self.profile.max_speed,
            mean_vel[1] / self.profile.max_speed,
            *drone_feats,                                      # 8 values
            vec_to_goal[0],
            vec_to_goal[1],
            dist_to_goal / (self.env_cfg.world_size * 2),
            in_2rc,
            in_rc,
            1.0 - (self.steps_count / self.env_cfg.max_decisions),
        ], dtype=np.float32)

        return obs

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        cfg = self.sim_cfg
        n = self.env_cfg.n_animals

        positions = np.column_stack([
            self.np_random.uniform(cfg.world_min + cfg.spawn_margin,
                                   cfg.world_max - cfg.spawn_margin, n),
            self.np_random.uniform(cfg.world_min + cfg.spawn_margin,
                                   cfg.world_max - cfg.spawn_margin, n),
        ])
        velocities = self.np_random.standard_normal((n, 2)) * 0.1
        panic_timers = np.zeros(n, dtype=int)
        panic_directions = np.zeros((n, 2), dtype=float)
        self.herd = HerdState(
            positions=positions,
            velocities=velocities,
            panic_timers=panic_timers,
            panic_directions=panic_directions,
        )

        # Drones start in a circle around the world boundary
        initial_positions = []
        for i in range(self.env_cfg.n_drones):
            angle = i * (2 * np.pi / self.env_cfg.n_drones)
            r = cfg.world_max * 1.1
            initial_positions.append([r * np.cos(angle), r * np.sin(angle)])

        self.swarm_manager = SwarmManager(
            number_of_drones=self.env_cfg.n_drones,
            initial_positions=initial_positions,
            v_max=cfg.drone_v_max,
            u_max=cfg.drone_u_max,
        )

        self.steps_count = 0
        centroid = compute_centroid(self.herd.positions)
        self.prev_dist_to_goal = float(np.linalg.norm(
            centroid - np.array(cfg.goal_position)))
        self.prev_radius = float(herd_radius(self.herd.positions, centroid))
        self.prev_drone_positions = [np.array(p) for p in initial_positions]

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        # Decode action → 4 waypoints using Polar Coordinates around the herd
        centroid = compute_centroid(self.herd.positions)
        herd_rad = herd_radius(self.herd.positions, centroid)
        
        actions = action.reshape(self.env_cfg.n_drones, 2)
        waypoints = []
        
        for i in range(self.env_cfg.n_drones):
            # Action 0: Angle from -pi to pi
            theta = actions[i, 0] * np.pi 
            
            # Action 1: Radial offset.
            # An output of 0.0 places the drone exactly at (herd_radius + drone_influence_radius).
            # The agent can adjust this radius in/out by up to 10 meters.
            r = herd_rad + self.sim_cfg.drone_influence_radius + (actions[i, 1] * 10.0)
            
            # Hard lower limit to prevent diving completely into the center
            r = max(2.0, r)
            
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            waypoints.append([centroid[0] + dx, centroid[1] + dy])
            
        waypoints = np.array(waypoints)

        # Run SMC sub-steps
        for _ in range(self.env_cfg.decision_interval):
            self.swarm_manager.track_waypoints(self.env_cfg.dt, waypoints)
            self.herd = update_herd(
                self.herd, self.swarm_manager.drones,
                self._sim_cfg_for_herd, self.profile
            )

        # Apply partial-observability estimation when enabled
        if self.sim_cfg.use_partial_observability:
            observed_pos, observed_vel, _ = compute_observations(
                self.herd, self.swarm_manager.drones,
                self._sim_cfg_for_herd, self.profile,
            )
            self.herd.positions = observed_pos
            self.herd.velocities = observed_vel

        self.steps_count += 1

        # ---- Compute reward ----
        centroid = compute_centroid(self.herd.positions)
        curr_radius = float(herd_radius(self.herd.positions, centroid))
        goal = np.array(self.sim_cfg.goal_position)
        curr_dist_to_goal = float(np.linalg.norm(centroid - goal))

        # Dense: progress toward goal
        r_progress = self.reward_cfg.w_progress * (
            self.prev_dist_to_goal - curr_dist_to_goal)

        # Dense: herd compaction
        r_compact = self.reward_cfg.w_compactness * (
            self.prev_radius - curr_radius)

        # Constant time penalty
        r_time = self.reward_cfg.w_time

        # Drone energy penalty (sum of squared displacements)
        drone_pos, _ = self.swarm_manager.get_swarm_status()
        energy = sum(
            np.sum((np.asarray(drone_pos[i]) - self.prev_drone_positions[i]) ** 2)
            for i in range(self.env_cfg.n_drones)
        )
        r_energy = self.reward_cfg.w_energy * energy

        # Success check
        dists_to_goal = np.linalg.norm(self.herd.positions - goal, axis=1)
        success = bool(np.all(dists_to_goal <= self.sim_cfg.success_radius))
        r_success = self.reward_cfg.w_success if success else 0.0

        # Partial success bonus (fraction of animals within Rc)
        frac_in_goal = float(np.mean(dists_to_goal <= self.sim_cfg.success_radius))
        r_partial = 2.0 * frac_in_goal  # small bonus that scales with proximity

        reward = r_progress + r_compact + r_time + r_energy + r_success + r_partial
        reward = float(np.clip(reward, -15.0, 50.0))

        # Update state for next step
        self.prev_dist_to_goal = curr_dist_to_goal
        self.prev_radius = curr_radius
        self.prev_drone_positions = [np.array(p) for p in drone_pos]

        terminated = success
        truncated = self.steps_count >= self.env_cfg.max_decisions

        info = {
            "is_success": success,
            "dist_to_goal": curr_dist_to_goal,
            "herd_radius": curr_radius,
            "frac_in_goal": frac_in_goal,
            "r_progress": r_progress,
            "r_compact": r_compact,
        }

        return self._get_obs(), reward, terminated, truncated, info
