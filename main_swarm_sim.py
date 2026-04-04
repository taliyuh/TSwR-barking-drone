import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point, LineString
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
    n_animals: int = 40
    n_drones: int = 4
    dt: float = 0.1
    world_min: float = -20.0
    world_max: float = 20.0
    spawn_margin: float = 5.0

    drone_influence_radius: float = 8.0
    drone_v_max: float = 4.0
    drone_u_max: float = 3.0

    # Geometry
    extended_hull_margin: float = 3.0

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


ANIMAL_PROFILES = {
    "sheep": AnimalProfile(
        name="Sheep",
        neighbor_radius=4.0,
        separation_radius=1.5,
        cohesion_weight=0.6,
        separation_weight=1.5,
        alignment_weight=0.4,
        drone_repulsion_weight=3.0,
        noise_weight=0.15,
        max_speed=1.2,
    ),
    "goats": AnimalProfile(
        name="Goats",
        neighbor_radius=3.5,
        separation_radius=1.2,
        cohesion_weight=0.4,
        separation_weight=1.8,
        alignment_weight=0.3,
        drone_repulsion_weight=2.2,
        noise_weight=0.25,
        max_speed=1.4,
    ),
    "cows": AnimalProfile(
        name="Cows",
        neighbor_radius=5.0,
        separation_radius=2.0,
        cohesion_weight=0.8,
        separation_weight=1.2,
        alignment_weight=0.5,
        drone_repulsion_weight=1.8,
        noise_weight=0.08,
        max_speed=0.9,
    ),
}

# =========================
# DATA STRUCTURES
# =========================


@dataclass
class HerdState:
    positions: np.ndarray   # shape (N, 2)
    velocities: np.ndarray  # shape (N, 2)

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


# =========================
# FLOCKING / HERD DYNAMICS
# =========================

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

        # Noise
        noise = np.random.randn(2)

        accel = (
            profile.cohesion_weight * cohesion
            + profile.alignment_weight * alignment
            + profile.separation_weight * separation
            + profile.drone_repulsion_weight * drone_repulsion
            + profile.noise_weight * noise
        )

        new_velocities[i] = v_i + cfg.dt * accel

    new_velocities = limit_speed(new_velocities, profile.max_speed)
    new_positions = positions + cfg.dt * new_velocities

    # Simple boundary handling (bounce)
    for dim in range(2):
        low = cfg.world_min
        high = cfg.world_max

        low_hit = new_positions[:, dim] < low
        high_hit = new_positions[:, dim] > high

        new_positions[low_hit, dim] = low
        new_velocities[low_hit, dim] *= -0.6

        new_positions[high_hit, dim] = high
        new_velocities[high_hit, dim] *= -0.6

    return HerdState(new_positions, new_velocities)


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

        self.herd = HerdState(positions=positions, velocities=velocities)

        # Initialize Swarm Manager
        # Place drones initially at corners of the world
        initial_positions = []
        for i in range(cfg.n_drones):
            angle = i * (2 * np.pi / cfg.n_drones)
            r = cfg.world_max * 1.2 # Start slightly outside
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

        if ext_hull is not None and not ext_hull.is_empty:
            # We need vertices in standard format without the duplicate last coordinate generated by Shapely
            coords = list(ext_hull.exterior.coords)
            if len(coords) > 0 and coords[0] == coords[-1]:
                vertices = [list(c) for c in coords[:-1]]
            else:
                vertices = [list(c) for c in coords]

            target_points = generate_target_points(ext_hull, self.cfg.n_drones)
            
            # Swarm manager updates all drone positions
            if len(vertices) >= 3 and len(target_points) > 0:
                self.swarm_manager.update_swarm(self.cfg.dt, vertices, target_points)

        # Update animal positions considering drone fields
        self.herd = update_herd(self.herd, self.swarm_manager.drones, self.cfg, self.profile)
        self.time += self.cfg.dt

    def get_geometry(self):
        positions = self.herd.positions
        centroid = compute_centroid(positions)
        hull_points = compute_convex_hull(positions)
        ext_hull = compute_extended_hull(hull_points, self.cfg.extended_hull_margin)
        radius = herd_radius(positions, centroid)
        furthest_idx, furthest_pos, furthest_dist = find_furthest_animal_from_centroid(
            positions, centroid
        )

        return {
            "centroid": centroid,
            "hull_points": hull_points,
            "extended_hull": ext_hull,
            "radius": radius,
            "furthest_idx": furthest_idx,
            "furthest_pos": furthest_pos,
            "furthest_dist": furthest_dist,
        }


# =========================
# VISUALIZATION
# =========================

def draw_simulation(ax, sim: Simulation):
    ax.clear()

    geom = sim.get_geometry()
    positions = sim.herd.positions
    centroid = geom["centroid"]
    hull_points = geom["hull_points"]
    ext_hull = geom["extended_hull"]
    furthest_pos = geom["furthest_pos"]

    # Animals
    ax.scatter(positions[:, 0], positions[:, 1], s=30, label="Animals", color='tab:blue')

    # Centroid
    ax.scatter(centroid[0], centroid[1], s=80, marker="*", label="Centroid", color='tab:orange')

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
        
    # Draw Drones
    drone_positions, drone_headings = sim.swarm_manager.get_swarm_status()
    if drone_positions:
        dp = np.array(drone_positions)
        dh = np.array(drone_headings)
        ax.scatter(dp[:, 0], dp[:, 1], s=100, marker="X", label="Drones", color='black')
        ax.quiver(dp[:, 0], dp[:, 1], dh[:, 0], dh[:, 1], color='black', scale=20, width=0.005)

    # Draw target points
    target_points = generate_target_points(ext_hull, sim.cfg.n_drones)
    if target_points:
        tp = np.array(target_points)
        ax.scatter(tp[:, 0], tp[:, 1], s=50, marker="D", label="Targets", color='purple')

    ax.set_xlim(*sim.cfg.xlim)
    ax.set_ylim(*sim.cfg.ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.set_title(
        f"t={sim.time:.1f} | herd radius={geom['radius']:.2f} | furthest={geom['furthest_dist']:.2f}"
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))


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
    plt.subplots_adjust(right=0.8)

    try:
        for i in range(1000):
            sim.step()
            draw_simulation(ax, sim)
            plt.pause(0.01)
    except KeyboardInterrupt:
        print("Simulation interrupted.")

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
