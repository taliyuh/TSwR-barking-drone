from dataclasses import dataclass, field
import torch

@dataclass
class SimConfigRL:
    n_animals: int = 40
    n_drones: int = 4
    dt: float = 0.1
    world_min: float = -25.0
    world_max: float = 25.0
    spawn_margin: float = 5.0
    drone_influence_radius: float = 8.0
    drone_v_max: float = 4.0
    drone_u_max: float = 3.0
    extended_hull_margin: float = 3.0
    goal_position: tuple = (18.0, 18.0)
    success_radius: float = 3.0
    accel_threshold: float = 0.15
    velocity_damping: float = 0.95
    drone_vision_radius: float = 10.0
    use_partial_observability: bool = False

@dataclass
class EnvConfig:
    n_animals: int = 40
    n_drones: int = 4
    dt: float = 0.1
    decision_interval: int = 10      # 1 RL step = 10 sim steps (1 second)
    max_decisions: int = 200         # episode ends after 200 decisions (200s sim time)
    world_size: float = 25.0         # scaling for normalization
    
@dataclass
class RewardConfig:
    w_progress: float = 10.0         # reward for reducing centroid-to-goal distance
    w_compactness: float = 5.0       # reward for reducing herd radius
    w_success: float = 500.0         # terminal bonus
    w_time: float = -0.5             # per-step penalty to encourage speed
    w_energy: float = 0.0            # temporarily disabled to avoid drowning out progress reward
    w_collision: float = -2.0        # penalize drone-drone proximity

@dataclass
class TrainConfig:
    total_timesteps: int = 2_000_000
    n_envs: int = 8
    learning_rate: float = 3e-4
    batch_size: int = 256            # PPO batch size
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    net_arch: list = field(default_factory=lambda: [256, 256])
    checkpoint_freq: int = 100_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "models/commander_ppo"
    log_path: str = "logs/commander_ppo"

@dataclass
class CurriculumConfig:
    # Phase 1: Easy (20 animals, close goal)
    phase1_steps: int = 500_000
    phase1_animals: int = 20
    phase1_goal: tuple = (10.0, 10.0)
    phase1_radius: float = 6.0
    
    # Phase 2: Medium (30 animals, mid goal)
    phase2_steps: int = 1_000_000
    phase2_animals: int = 30
    phase2_goal: tuple = (14.0, 14.0)
    phase2_radius: float = 4.5
