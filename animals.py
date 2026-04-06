import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point


# =========================
# CONFIG
# =========================

@dataclass
class SimConfig:
    n_animals: int = 100
    dt: float = 0.1
    world_min: float = -20.0
    world_max: float = 20.0
    spawn_margin: float = 2.0

    drone_influence_radius: float = 6.0

    # Geometry
    extended_hull_margin: float = 2.0

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
    accel_threshold: float
    velocity_damping: float

    # Panic
    panic_probability: float
    panic_duration_min: int
    panic_duration_max: int
    panic_strength: float


ANIMAL_PROFILES = {
    "sheep": AnimalProfile(
        name="Sheep",
        neighbor_radius=2.0,
        separation_radius=0.75,
        cohesion_weight=0.3,
        separation_weight=0.75,
        alignment_weight=0.2,
        drone_repulsion_weight=1.5,
        noise_weight=0.1,
        max_speed=0.6,
        accel_threshold=0.08,
        velocity_damping=0.90,
        panic_probability=0.012,
        panic_duration_min=8,
        panic_duration_max=18,
        panic_strength=0.9,
    ),
    "goats": AnimalProfile(
        name="Goats",
        neighbor_radius=1.75,
        separation_radius=0.6,
        cohesion_weight=0.2,
        separation_weight=0.9,
        alignment_weight=0.15,
        drone_repulsion_weight=1.1,
        noise_weight=0.2,
        max_speed=0.7,
        accel_threshold=0.04,
        velocity_damping=0.82,
        panic_probability=0.018,
        panic_duration_min=6,
        panic_duration_max=14,
        panic_strength=1.1,
    ),
    "cows": AnimalProfile(
        name="Cows",
        neighbor_radius=2.5,
        separation_radius=1.0,
        cohesion_weight=0.4,
        separation_weight=0.6,
        alignment_weight=0.25,
        drone_repulsion_weight=0.9,
        noise_weight=0.04,
        max_speed=0.45,
        accel_threshold=0.14,
        velocity_damping=0.95,
        panic_probability=0.006,
        panic_duration_min=10,
        panic_duration_max=24,
        panic_strength=0.7,
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


class Drone:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)

    def set_position(self, position):
        self.position = np.array(position, dtype=float)

# =========================
# GEOMETRY HELPERS
# =========================


def random_point_on_polygon_boundary(poly: np.ndarray) -> np.ndarray:
    n = len(poly)
    i = np.random.randint(n)
    p1 = poly[i]
    p2 = poly[(i + 1) % n]
    t = np.random.rand()
    return (1 - t) * p1 + t * p2


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
    return np.max(dists)


def find_furthest_animal_from_centroid(positions: np.ndarray, centroid: np.ndarray):
    dists = np.linalg.norm(positions - centroid, axis=1)
    idx = np.argmax(dists)
    return idx, positions[idx], dists[idx]


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


def update_herd(state: HerdState, drone: Drone, cfg: SimConfig, profile: AnimalProfile) -> HerdState:
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

        # Drone repulsion
        drone_vec = drone.position - p_i
        drone_dist = np.linalg.norm(drone_vec)

        drone_repulsion = np.zeros(2)
        if 1e-8 < drone_dist < cfg.drone_influence_radius:
            away = -drone_vec / (drone_dist + 1e-6)
            strength = (1.0 - drone_dist / cfg.drone_influence_radius)
            drone_repulsion = away * strength

        base_accel = (
                profile.cohesion_weight * cohesion
                + profile.alignment_weight * alignment
                + profile.separation_weight * separation
                + profile.drone_repulsion_weight * drone_repulsion
        )

        noise = profile.noise_weight * np.random.randn(2)

        if panic_timers[i] == 0:
            if np.random.rand() < profile.panic_probability * cfg.dt:
                centroid = compute_centroid(positions)
                dir_vec = positions[i] - centroid
                panic_directions[i] = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)

                panic_timers[i] = np.random.randint(profile.panic_duration_min, profile.panic_duration_max + 1)

        panic_force = np.zeros(2)
        if panic_timers[i] > 0:
            panic_force = profile.panic_strength * panic_directions[i]
            panic_timers[i] -= 1

        total_accel = base_accel + noise + panic_force
        if np.linalg.norm(total_accel) > profile.accel_threshold:
            new_velocities[i] = v_i + cfg.dt * total_accel
        else:
            new_velocities[i] = profile.velocity_damping * v_i

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
        self.drone = Drone(position=np.array([8.0, 8.0]))

        self.time = 0.0

    def step(self):
        self.herd = update_herd(self.herd, self.drone, self.cfg, self.profile)
        self.time += self.cfg.dt

    def move_drone(self):
        # Temporary simple motion so you can test reactions
        # r = 10.0
        # omega = 0.15
        # x = r * np.cos(omega * self.time)
        # y = r * np.sin(omega * self.time)
        geom = self.get_geometry()
        ext_hull = geom["extended_hull"]

        if ext_hull is None:
            return

        ext_hull_points = np.array(ext_hull.exterior.coords[:-1])

        target = random_point_on_polygon_boundary(ext_hull_points)

        self.drone.set_position(target)

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
    drone_pos = sim.drone.position
    centroid = geom["centroid"]
    hull_points = geom["hull_points"]
    ext_hull = geom["extended_hull"]
    furthest_pos = geom["furthest_pos"]

    # Animals
    ax.scatter(positions[:, 0], positions[:, 1], s=30, label="Animals")

    # Drone
    ax.scatter(drone_pos[0], drone_pos[1], s=80, marker="x", label="Drone")

    # Centroid
    ax.scatter(centroid[0], centroid[1], s=80, marker="*", label="Centroid")

    # Furthest animal
    ax.scatter(furthest_pos[0], furthest_pos[1], s=60, marker="o", label="Furthest")

    # Convex hull
    if len(hull_points) >= 3:
        hull_closed = np.vstack([hull_points, hull_points[0]])
        ax.plot(hull_closed[:, 0], hull_closed[:, 1], linewidth=2, label="Convex Hull")

    # Extended hull
    if ext_hull is not None:
        x_ext, y_ext = polygon_to_xy(ext_hull)
        ax.plot(x_ext, y_ext, linestyle="--", linewidth=2, label="Extended Hull")

    ax.set_xlim(*sim.cfg.xlim)
    ax.set_ylim(*sim.cfg.ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.set_title(
        f"t={sim.time:.1f} | herd radius={geom['radius']:.2f} | furthest={geom['furthest_dist']:.2f}"
    )
    ax.legend(loc="upper right")


# =========================
# MAIN
# =========================

def main():
    cfg = SimConfig()
    selected_profile = "sheep"
    profile = ANIMAL_PROFILES[selected_profile]
    sim = Simulation(cfg, profile)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(1000):
        if i % 10 == 0:
            sim.move_drone()   # temporary test motion
        sim.step()
        draw_simulation(ax, sim)
        plt.pause(0.03)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
