import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_swarm_sim import (
    SimConfig, HerdState, AnimalProfile, ANIMAL_PROFILES,
    update_herd, compute_centroid, herd_radius,
    compute_convex_hull, compute_extended_hull, generate_target_points, generate_driving_arc,
    compute_observations,
)
from control.swarm_manager import SwarmManager
from herding_rl.config import EnvConfig, RewardConfig, SimConfigRL

class HerdingGainTunerEnv(gym.Env):
    """
    Gymnasium environment for Approach B: Gain Tuning.
    RL tuner outputs k_edge, k_target, and v_scale for the existing SMC heuristic.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_cfg: EnvConfig = None, reward_cfg: RewardConfig = None, sim_cfg: SimConfigRL = None):
        super().__init__()
        self.env_cfg = env_cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.sim_cfg = sim_cfg or SimConfigRL()

        self.profile = ANIMAL_PROFILES["sheep"]

        # Action space: 3 continuous values for [k_edge, k_target, v_scale] mapped from [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation space: 19 features (normalized) + phase + time + 3 gains = 23
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)

        self._sim_cfg_for_herd = self._build_sim_config()

        self.herd = None
        self.swarm_manager = None
        self.steps_count = 0
        self.prev_dist_to_goal = 0.0
        self.prev_radius = 0.0
        self.prev_drone_positions = None
        self.current_gains = np.array([1.0, 1.0, 1.0]) # Default gains
        
        # Gathering vs driving state
        self.gathering_threshold_radius = 6.0
        self.gathering_phase = True

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

    def _get_obs(self):
        positions = self.herd.positions
        centroid = compute_centroid(positions)
        radius = herd_radius(positions, centroid)
        mean_vel = np.mean(self.herd.velocities, axis=0)

        goal = np.array(self.sim_cfg.goal_position)
        dist_to_goal = np.linalg.norm(centroid - goal)
        vec_to_goal = (goal - centroid) / (dist_to_goal + 1e-6)

        drone_pos, _ = self.swarm_manager.get_swarm_status()
        drone_feats = []
        for p in drone_pos:
            rel_p = (np.asarray(p) - centroid) / self.env_cfg.world_size
            drone_feats.extend(rel_p)

        dists_to_goal = np.linalg.norm(positions - goal, axis=1)
        in_rc = float(np.mean(dists_to_goal <= self.sim_cfg.success_radius))
        in_2rc = float(np.mean(dists_to_goal <= (self.sim_cfg.success_radius * 2)))

        obs = np.array([
            (centroid[0] - goal[0]) / self.env_cfg.world_size,
            (centroid[1] - goal[1]) / self.env_cfg.world_size,
            radius / self.env_cfg.world_size,
            mean_vel[0] / self.profile.max_speed,
            mean_vel[1] / self.profile.max_speed,
            *drone_feats,
            vec_to_goal[0],
            vec_to_goal[1],
            dist_to_goal / (self.env_cfg.world_size * 2),
            in_2rc,
            in_rc,
            1.0 if self.gathering_phase else -1.0, # Phase indicator
            1.0 - (self.steps_count / self.env_cfg.max_decisions),
            self.current_gains[0] / 5.0, # Normalized k_edge
            self.current_gains[1] / 5.0, # Normalized k_target
            self.current_gains[2] / 2.0  # Normalized v_scale
        ], dtype=np.float32)

        return obs

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
        self.prev_dist_to_goal = float(np.linalg.norm(centroid - np.array(cfg.goal_position)))
        self.prev_radius = float(herd_radius(self.herd.positions, centroid))
        self.prev_drone_positions = [np.array(p) for p in initial_positions]
        self.gathering_phase = True
        self.current_gains = np.array([1.0, 1.0, 1.0])

        return self._get_obs(), {}

    def step(self, action):
        # Decode action to gains
        k_edge = np.interp(action[0], [-1, 1], [0.1, 5.0])
        k_target = np.interp(action[1], [-1, 1], [0.1, 5.0])
        v_scale = np.interp(action[2], [-1, 1], [0.1, 2.0])
        
        self.current_gains = np.array([k_edge, k_target, v_scale])
        gains_tuple = tuple(self.current_gains)

        # Run SMC sub-steps with the heuristic strategy
        for _ in range(self.env_cfg.decision_interval):
            centroid = compute_centroid(self.herd.positions)
            radius = herd_radius(self.herd.positions, centroid)
            goal = np.array(self.sim_cfg.goal_position)
            
            if self.gathering_phase and radius < self.gathering_threshold_radius:
                self.gathering_phase = False
            elif not self.gathering_phase and radius > self.gathering_threshold_radius * 1.5:
                self.gathering_phase = True

            if self.gathering_phase:
                hull_points = compute_convex_hull(self.herd.positions)
                ext_hull = compute_extended_hull(hull_points, self.sim_cfg.extended_hull_margin)
                
                if ext_hull is not None and not ext_hull.is_empty:
                    coords = list(ext_hull.exterior.coords)
                    vertices = [list(c) for c in coords[:-1]]
                    target_points = generate_target_points(ext_hull, self.env_cfg.n_drones)
                    
                    if len(vertices) >= 3 and len(target_points) > 0:
                        self.swarm_manager.update_swarm(self.env_cfg.dt, vertices, target_points, gains=gains_tuple)
            else:
                patrol_radius = radius + self.sim_cfg.extended_hull_margin
                patrol_points = generate_driving_arc(centroid, goal, patrol_radius, self.env_cfg.n_drones)
                
                if len(patrol_points) == self.env_cfg.n_drones + 1:
                    self.swarm_manager.update_driving(self.env_cfg.dt, patrol_points, gains=gains_tuple)

            self.herd = update_herd(self.herd, self.swarm_manager.drones, self._sim_cfg_for_herd, self.profile)

        # Apply partial-observability estimation when enabled
        if self.sim_cfg.use_partial_observability:
            observed_pos, observed_vel, _ = compute_observations(
                self.herd, self.swarm_manager.drones,
                self._sim_cfg_for_herd, self.profile,
            )
            self.herd.positions = observed_pos
            self.herd.velocities = observed_vel

        self.steps_count += 1

        centroid = compute_centroid(self.herd.positions)
        curr_radius = float(herd_radius(self.herd.positions, centroid))
        goal = np.array(self.sim_cfg.goal_position)
        curr_dist_to_goal = float(np.linalg.norm(centroid - goal))

        # Dense: progress toward goal
        r_progress = self.reward_cfg.w_progress * (self.prev_dist_to_goal - curr_dist_to_goal)

        # Dense: herd compaction
        r_compact = self.reward_cfg.w_compactness * (self.prev_radius - curr_radius)

        # Constant time penalty
        r_time = self.reward_cfg.w_time

        # Success check
        dists_to_goal = np.linalg.norm(self.herd.positions - goal, axis=1)
        success = bool(np.all(dists_to_goal <= self.sim_cfg.success_radius))
        r_success = self.reward_cfg.w_success if success else 0.0

        # Partial success bonus
        frac_in_goal = float(np.mean(dists_to_goal <= self.sim_cfg.success_radius))
        r_partial = 2.0 * frac_in_goal

        reward = r_progress + r_compact + r_time + r_success + r_partial
        reward = float(np.clip(reward, -15.0, 50.0))

        self.prev_dist_to_goal = curr_dist_to_goal
        self.prev_radius = curr_radius
        
        drone_pos, _ = self.swarm_manager.get_swarm_status()
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
