import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, LineString
import sys
import os

# Append the control module so we can import it
sys.path.append(os.path.join(os.path.dirname(__file__), 'control'))
from swarm_manager import SwarmManager

# =========================
# CONFIG
# =========================

@dataclass
class SimConfig:
    n_animals: int = 100
    n_drones: int = 4
    dt: float = 0.1
    world_min: float = -20.0
    world_max: float = 20.0
    spawn_margin: float = 5.0

    accel_threshold: float = 0.15
    velocity_damping: float = 0.95

    drone_influence_radius: float = 8.0
    drone_v_max: float = 4.0
    drone_u_max: float = 3.0
    drone_vision_radius: float = 10.0

    # Geometry
    extended_hull_margin: float = 3.0
    
    # Driving Phase Parameters
    goal_position: tuple = (18.0, 18.0)
    gathering_threshold_radius: float = 7.0

    # Visual
    xlim: tuple = (-25, 25)
    ylim: tuple = (-25, 25)


@dataclass
class AnimalProfile:
    name: str

    # Social distances
    neighbor_radius: float
    separation_radius: float

    # Behaviour weights
    cohesion_weight: float
    separation_weight: float
    alignment_weight: float
    drone_repulsion_weight: float
    noise_weight: float

    # Motion
    max_speed: float

    panic_probability: float
    panic_duration_min: int
    panic_duration_max: int
    panic_strength: float


ANIMAL_PROFILES = {
    "sheep": AnimalProfile(
        name="Sheep",
        neighbor_radius=4.0/2,
        separation_radius=1.5/2,
        cohesion_weight=0.6/2,
        separation_weight=1.5/2,
        alignment_weight=0.4/2,
        drone_repulsion_weight=3.0/2,
        noise_weight=0.15/2,
        max_speed=1.2/2,
        panic_probability=0.012/2,
        panic_duration_min=8,
        panic_duration_max=18,
        panic_strength=1.8/2,
    ),
    "goats": AnimalProfile(
        name="Goats",
        neighbor_radius=3.5/2,
        separation_radius=1.2/2,
        cohesion_weight=0.4/2,
        separation_weight=1.8/2,
        alignment_weight=0.3/2,
        drone_repulsion_weight=2.2/2,
        noise_weight=0.25/2,
        max_speed=1.4/2,
        panic_probability=0.018/2,
        panic_duration_min=6,
        panic_duration_max=14,
        panic_strength=2.1/2,
    ),
    "cows": AnimalProfile(
        name="Cows",
        neighbor_radius=5.0/2,
        separation_radius=2.0/2,
        cohesion_weight=0.8/2,
        separation_weight=1.2/2,
        alignment_weight=0.5/2,
        drone_repulsion_weight=1.8/2,
        noise_weight=0.08/2,
        max_speed=0.9/2,
        panic_probability=0.006/2,
        panic_duration_min=10,
        panic_duration_max=24,
        panic_strength=1.4/2,
    ),
}

# =========================
# DATA STRUCTURES
# =========================

@dataclass
class HerdState:
    positions: np.ndarray   # shape (N, 2)
    velocities: np.ndarray  # shape (N, 2)
    panic_timers: np.ndarray
    panic_directions: np.ndarray

# =========================
# GEOMETRY HELPERS
# =========================

def compute_centroid(positions: np.ndarray) -> np.ndarray:
    return np.mean(positions, axis=0)


def compute_convex_hull(positions: np.ndarray):
    if len(positions) < 3:
        return positions
    hull = ConvexHull(positions)
    return positions[hull.vertices]


def compute_extended_hull(hull_points: np.ndarray, margin: float):
    if len(hull_points) < 3:
        return None
    poly = Polygon(hull_points)
    if not poly.is_valid:
        poly = poly.convex_hull
    return poly.buffer(margin, join_style=2)  # 2 = sharper corners


def polygon_to_xy(poly: Polygon):
    if poly is None:
        return None, None
    x, y = poly.exterior.xy
    return np.array(x), np.array(y)


def herd_radius(positions: np.ndarray, centroid: np.ndarray) -> float:
    dists = np.linalg.norm(positions - centroid, axis=1)
    if len(dists) == 0: return 0.0
    return np.max(dists)


def find_furthest_animal_from_centroid(positions: np.ndarray, centroid: np.ndarray):
    dists = np.linalg.norm(positions - centroid, axis=1)
    if len(dists) == 0: return -1, np.zeros(2), 0.0
    idx = np.argmax(dists)
    return idx, positions[idx], dists[idx]

def generate_target_points(ext_hull: Polygon, n_points: int):
    """Generates evenly spaced points along the polygon perimeter."""
    if ext_hull is None or n_points <= 0 or ext_hull.is_empty:
        return []
    
    perimeter = ext_hull.length
    distances = np.linspace(0, perimeter, n_points, endpoint=False)
    
    exterior = LineString(ext_hull.exterior.coords)
    points = [exterior.interpolate(distance) for distance in distances]
    return [np.array([p.x, p.y]) for p in points]

def generate_driving_arc(centroid: np.ndarray, goal: np.ndarray, radius: float, n_drones: int):
    """Generates points along a rear semicircle relative to the goal."""
    vec_to_goal = goal - centroid
    norm = np.linalg.norm(vec_to_goal)
    if norm == 0:
        return []
    dir_to_goal = vec_to_goal / norm
    
    angle_to_goal = np.arctan2(dir_to_goal[1], dir_to_goal[0])
    
    # The rear arc is exactly opposite to the goal direction
    rear_angle = angle_to_goal + np.pi
    start_angle = rear_angle - np.pi / 2.0
    end_angle = rear_angle + np.pi / 2.0
    
    # We need n_drones + 1 points to create n_drones segments
    angles = np.linspace(start_angle, end_angle, n_drones + 1)
    points = []
    for a in angles:
        points.append(centroid + radius * np.array([np.cos(a), np.sin(a)]))
    return points

# =========================
# FLOCKING / HERD DYNAMICS
# =========================


def _clamp_to_bounds(positions: np.ndarray, velocities: np.ndarray, world_min: float, world_max: float, restitution: float = -0.6):
    """Apply boundary bounce to positions and velocities. Modifies arrays in-place."""
    for dim in range(2):
        low_hit = positions[:, dim] < world_min
        high_hit = positions[:, dim] > world_max

        positions[low_hit, dim] = world_min
        velocities[low_hit, dim] *= restitution

        positions[high_hit, dim] = world_max
        velocities[high_hit, dim] *= restitution


def compute_observations(state: HerdState, swarm_drones: list, cfg: SimConfig, profile: AnimalProfile,
                         prev_positions=None, prev_velocities=None):
    positions = state.positions
    velocities = state.velocities
    N = len(positions)

    visible_mask = np.zeros(N, dtype=bool)

    for drone in swarm_drones:
        dists = np.linalg.norm(positions - drone.d, axis=1)
        visible_mask |= (dists < cfg.drone_vision_radius)

    if prev_positions is None:
        observed_positions = positions.copy()
        observed_velocities = velocities.copy()
    else:
        observed_positions = prev_positions.copy()
        observed_velocities = prev_velocities.copy()

    new_positions = observed_positions.copy()
    new_velocities = observed_velocities.copy()

    for i in range(N):
        if visible_mask[i]:
            new_positions[i] = positions[i]
            new_velocities[i] = velocities[i]
        else:
            accel = deterministic_update(
                observed_positions[i],
                observed_velocities[i],
                swarm_drones,
                observed_positions,
                observed_velocities,
                profile,
                cfg
            )

            if np.linalg.norm(accel) > cfg.accel_threshold:
                v_est = observed_velocities[i] + cfg.dt * accel
            else:
                v_est = cfg.velocity_damping * observed_velocities[i]

            # Use vectorized limit_speed for a single vector
            v_norm = np.linalg.norm(v_est)
            if v_norm > profile.max_speed:
                v_est = (v_est / v_norm) * profile.max_speed

            p_est = observed_positions[i] + cfg.dt * v_est

            new_positions[i] = p_est
            new_velocities[i] = v_est

    observed_positions = new_positions
    observed_velocities = new_velocities

    _clamp_to_bounds(observed_positions, observed_velocities, cfg.world_min, cfg.world_max)

    return observed_positions, observed_velocities, visible_mask


def deterministic_update(p_i, v_i, swarm_drones, positions, velocities, profile, cfg):
    rel = positions - p_i
    dists = np.linalg.norm(rel, axis=1)

    neighbor_mask = (dists > 1e-8) & (dists < profile.neighbor_radius)
    sep_mask = (dists > 1e-8) & (dists < profile.separation_radius)

    # Cohesion
    cohesion = np.zeros(2)
    if np.any(neighbor_mask):
        local_center = np.mean(positions[neighbor_mask], axis=0)
        cohesion = local_center - p_i

    # Alignment
    alignment = np.zeros(2)
    if np.any(neighbor_mask):
        local_velocity = np.mean(velocities[neighbor_mask], axis=0)
        alignment = local_velocity - v_i

    # Separation
    separation = np.zeros(2)
    if np.any(sep_mask):
        close_rel = rel[sep_mask]
        close_dists = dists[sep_mask][:, None]
        separation = -np.sum(close_rel / (close_dists ** 2 + 1e-6), axis=0)

    # Drone repulsion from ALL drones
    drone_repulsion = np.zeros(2)
    for drone in swarm_drones:
        drone_vec = drone.d - p_i
        drone_dist = np.linalg.norm(drone_vec)

        if 1e-8 < drone_dist < cfg.drone_influence_radius:
            away = -drone_vec / (drone_dist + 1e-6)
            strength = (1.0 - drone_dist / cfg.drone_influence_radius)
            drone_repulsion += away * strength

    accel = (
            profile.cohesion_weight * cohesion
            + profile.alignment_weight * alignment
            + profile.separation_weight * separation
            + profile.drone_repulsion_weight * drone_repulsion
    )

    return accel

def limit_speed(v: np.ndarray, max_speed: float) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    too_fast = norms > max_speed
    safe_norms = np.where(norms == 0, 1.0, norms)
    v_limited = v.copy()
    v_limited[too_fast[:, 0]] = (
        v[too_fast[:, 0]] / safe_norms[too_fast[:, 0]]
    ) * max_speed
    return v_limited


def update_herd(state: HerdState, swarm_drones: list, cfg: SimConfig, profile: AnimalProfile) -> HerdState:
    positions = state.positions.copy()
    velocities = state.velocities.copy()
    panic_timers = state.panic_timers.copy()
    panic_directions = state.panic_directions.copy()
    N = len(positions)

    new_velocities = velocities.copy()

    for i in range(N):
        p_i = positions[i]
        v_i = velocities[i]

        # Relative vectors to all others
        rel = positions - p_i
        dists = np.linalg.norm(rel, axis=1)

        # Exclude self
        neighbor_mask = (dists > 1e-8) & (dists < profile.neighbor_radius)
        sep_mask = (dists > 1e-8) & (dists < profile.separation_radius)

        # Cohesion
        cohesion = np.zeros(2)
        if np.any(neighbor_mask):
            local_center = np.mean(positions[neighbor_mask], axis=0)
            cohesion = local_center - p_i

        # Alignment
        alignment = np.zeros(2)
        if np.any(neighbor_mask):
            local_velocity = np.mean(velocities[neighbor_mask], axis=0)
            alignment = local_velocity - v_i

        # Separation
        separation = np.zeros(2)
        if np.any(sep_mask):
            close_rel = rel[sep_mask]
            close_dists = dists[sep_mask][:, None]
            separation = -np.sum(close_rel / (close_dists**2 + 1e-6), axis=0)

        # Drone repulsion from ALL drones
        drone_repulsion = np.zeros(2)
        for drone in swarm_drones:
            drone_vec = drone.d - p_i
            drone_dist = np.linalg.norm(drone_vec)

            if 1e-8 < drone_dist < cfg.drone_influence_radius:
                away = -drone_vec / (drone_dist + 1e-6)
                strength = (1.0 - drone_dist / cfg.drone_influence_radius)
                drone_repulsion += away * strength

        base_accel = (
                profile.cohesion_weight * cohesion
                + profile.alignment_weight * alignment
                + profile.separation_weight * separation
                + profile.drone_repulsion_weight * drone_repulsion
        )

        noise = profile.noise_weight * np.random.randn(2)

        if panic_timers[i] == 0:
            if np.random.rand() < profile.panic_probability:
                centroid = compute_centroid(positions)
                dir_vec = positions[i] - centroid
                panic_directions[i] = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)

                panic_timers[i] = np.random.randint(profile.panic_duration_min, profile.panic_duration_max + 1)

        panic_force = np.zeros(2)
        if panic_timers[i] > 0:
            panic_force = profile.panic_strength * panic_directions[i]
            panic_timers[i] -= 1

        if np.linalg.norm(base_accel) > cfg.accel_threshold:
            accel = base_accel + noise + panic_force
            new_velocities[i] = v_i + cfg.dt * accel
        else:
            new_velocities[i] = cfg.velocity_damping * v_i

    new_velocities = limit_speed(new_velocities, profile.max_speed)
    new_positions = positions + cfg.dt * new_velocities

    # Simple boundary handling (bounce)
    _clamp_to_bounds(new_positions, new_velocities, cfg.world_min, cfg.world_max)

    return HerdState(new_positions, new_velocities, panic_timers, panic_directions)


# =========================
# SIMULATION
# =========================

class Simulation:
    def __init__(self, cfg: SimConfig, profile: AnimalProfile):
        self.cfg = cfg
        self.profile = profile

        positions = np.column_stack([
            np.random.uniform(cfg.world_min + cfg.spawn_margin, cfg.world_max - cfg.spawn_margin, cfg.n_animals),
            np.random.uniform(cfg.world_min + cfg.spawn_margin, cfg.world_max - cfg.spawn_margin, cfg.n_animals),
        ])
        velocities = np.random.randn(cfg.n_animals, 2) * 0.3

        panic_timers = np.zeros(cfg.n_animals, dtype=int)
        panic_directions = np.zeros((cfg.n_animals, 2), dtype=float)

        self.herd = HerdState(positions=positions, velocities=velocities, panic_timers=panic_timers,
                              panic_directions=panic_directions)
        
        self.goal = np.array(self.cfg.goal_position)
        self.is_driving = False

        self.observed_positions = positions
        self.observed_velocities = velocities
        self.visible_mask = np.zeros(len(positions), dtype=bool)

        # Initialize Swarm Manager
        initial_positions = []
        for i in range(cfg.n_drones):
            angle = i * (2 * np.pi / cfg.n_drones)
            r = cfg.world_max * 1.2
            initial_positions.append([r * np.cos(angle), r * np.sin(angle)])
            
        self.swarm_manager = SwarmManager(
            number_of_drones=cfg.n_drones,
            initial_positions=initial_positions,
            v_max=cfg.drone_v_max,
            u_max=cfg.drone_u_max
        )

        self.time = 0.0

    def step(self):
        geom = self.get_geometry()
        ext_hull = geom["extended_hull"]
        radius = geom["radius"]
        centroid = geom["centroid"]
        
        # Check for Phase Transition
        if not self.is_driving and radius < self.cfg.gathering_threshold_radius:
            print(f"Phase Transition at t={self.time:.1f}s: Gathering complete. Commencing Driving Phase.")
            self.is_driving = True

        if not self.is_driving:
            # GATHERING PHASE
            if ext_hull is not None and not ext_hull.is_empty:
                coords = list(ext_hull.exterior.coords)
                if len(coords) > 0 and coords[0] == coords[-1]:
                    vertices = [list(c) for c in coords[:-1]]
                else:
                    vertices = [list(c) for c in coords]

                target_points = generate_target_points(ext_hull, self.cfg.n_drones)
                
                if len(vertices) >= 3 and len(target_points) > 0:
                    self.swarm_manager.update_swarm(self.cfg.dt, vertices, target_points)
        else:
            # DRIVING PHASE
            patrol_radius = radius + self.cfg.extended_hull_margin
            patrol_points = generate_driving_arc(centroid, self.goal, patrol_radius, self.cfg.n_drones)
            
            if len(patrol_points) == self.cfg.n_drones + 1:
                self.swarm_manager.update_driving(self.cfg.dt, patrol_points)

        # Update animal positions considering drone fields
        self.herd = update_herd(self.herd, self.swarm_manager.drones, self.cfg, self.profile)
        self.time += self.cfg.dt
        self.observed_positions, self.observed_velocities, self.visible_mask = compute_observations(
            self.herd,
            self.swarm_manager.drones,
            self.cfg,
            self.profile,
            self.observed_positions,
            self.observed_velocities
        )

    def get_geometry(self):
        positions = self.herd.positions
        centroid = compute_centroid(positions)
        hull_points = compute_convex_hull(positions)
        ext_hull = compute_extended_hull(hull_points, self.cfg.extended_hull_margin)
        radius = herd_radius(positions, centroid)
        furthest_idx, furthest_pos, furthest_dist = find_furthest_animal_from_centroid(
            positions, centroid
        )
        
        # Calculate driving arc for visualization if in driving phase
        driving_arc = []
        if self.is_driving:
            patrol_radius = radius + self.cfg.extended_hull_margin
            driving_arc = generate_driving_arc(centroid, self.goal, patrol_radius, self.cfg.n_drones)

        return {
            "centroid": centroid,
            "hull_points": hull_points,
            "extended_hull": ext_hull,
            "radius": radius,
            "furthest_idx": furthest_idx,
            "furthest_pos": furthest_pos,
            "furthest_dist": furthest_dist,
            "driving_arc": driving_arc
        }


# =========================
# VISUALIZATION
# =========================

def draw_simulation(ax, sim: Simulation):
    ax.clear()

    geom = sim.get_geometry()
    positions = sim.herd.positions
    observed_positions = sim.observed_positions
    centroid = geom["centroid"]
    hull_points = geom["hull_points"]
    ext_hull = geom["extended_hull"]
    furthest_pos = geom["furthest_pos"]
    driving_arc = geom.get("driving_arc", [])

    # Draw Goal Location
    ax.scatter(sim.goal[0], sim.goal[1], s=200, marker="s", color='limegreen', label="Goal/Sheepfold")

    # Animals
    ax.scatter(positions[:, 0], positions[:, 1], s=30, label="Animals", color='tab:blue')

    # Centroid
    ax.scatter(centroid[0], centroid[1], s=80, marker="*", label="Centroid", color='tab:orange')

    if not sim.is_driving:
        # Furthest animal
        ax.scatter(furthest_pos[0], furthest_pos[1], s=60, marker="o", label="Furthest", color='tab:red')

        # Convex hull
        if len(hull_points) >= 3:
            hull_closed = np.vstack([hull_points, hull_points[0]])
            ax.plot(hull_closed[:, 0], hull_closed[:, 1], linewidth=2, label="Convex Hull", color='tab:orange')

        # Extended hull
        if ext_hull is not None and not ext_hull.is_empty:
            x_ext, y_ext = polygon_to_xy(ext_hull)
            ax.plot(x_ext, y_ext, linestyle="--", linewidth=2, label="Extended Hull", color='tab:green')
            
        # Draw target points
        target_points = generate_target_points(ext_hull, sim.cfg.n_drones)
        if target_points:
            tp = np.array(target_points)
            ax.scatter(tp[:, 0], tp[:, 1], s=50, marker="D", label="Targets", color='purple')
            
    else:
        # Draw the driving arc patrol path
        if len(driving_arc) > 0:
            arc_arr = np.array(driving_arc)
            ax.plot(arc_arr[:, 0], arc_arr[:, 1], linestyle="-", linewidth=3, color='tab:green', label="Patrol Arc")
            ax.scatter(arc_arr[:, 0], arc_arr[:, 1], s=50, marker="D", label="Patrol Bounds", color='purple')
        
    # Draw Drones
    drone_positions, drone_headings = sim.swarm_manager.get_swarm_status()
    if drone_positions:
        dp = np.array(drone_positions)
        dh = np.array(drone_headings)
        ax.scatter(dp[:, 0], dp[:, 1], s=100, marker="X", label="Drones", color='black')
        ax.quiver(dp[:, 0], dp[:, 1], dh[:, 0], dh[:, 1], color='black', scale=20, width=0.005)

    for drone in sim.swarm_manager.drones:
        x, y = drone.d

        # Vision radius (większy, "sensor")
        vision_circle = plt.Circle(
            (x, y),
            sim.cfg.drone_vision_radius,
            color='blue',
            fill=False,
            linestyle='--',
            alpha=0.3
        )
        ax.add_patch(vision_circle)

        # Influence radius (mniejszy, "force")
        influence_circle = plt.Circle(
            (x, y),
            sim.cfg.drone_influence_radius,
            color='red',
            fill=False,
            linestyle='-',
            alpha=0.4
        )
        ax.add_patch(influence_circle)

    visible = sim.visible_mask

    ax.scatter(
        observed_positions[visible, 0],
        observed_positions[visible, 1],
        color=(0.0, 0.8, 0.3),
        label="Visible",
        s=20,
        marker="x"
    )

    ax.scatter(
        observed_positions[~visible, 0],
        observed_positions[~visible, 1],
        color="red",
        label="Estimated",
        s=20,
        marker="x"
    )

    ax.set_xlim(*sim.cfg.xlim)
    ax.set_ylim(*sim.cfg.ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    phase = "DRIVING" if sim.is_driving else "GATHERING"
    ax.set_title(
        f"Phase: {phase} | t={sim.time:.1f} | herd radius={geom['radius']:.2f}"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))


# =========================
# MAIN
# =========================

def main():
    cfg = SimConfig()
    selected_profile = "sheep"
    profile = ANIMAL_PROFILES[selected_profile]
    sim = Simulation(cfg, profile)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))

    # Adjust subplot layout to accommodate the legend outside the plot
    plt.subplots_adjust(right=0.75)

    try:
        for i in range(2500):
            sim.step()
            draw_simulation(ax, sim)
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("Simulation interrupted.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
