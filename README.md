# Barking Drone

[Video link](https://youtu.be/P8s404sTP94)

The project group consists of Mikołaj Lipiński and Bartłomiej Hryniewski.

## Overview

This repository implements a multi-drone livestock herding simulation inspired by
[Robotic Herding of Farm Animals Using a Network of Barking Aerial Drones](https://www.mdpi.com/2504-446X/6/2/29).

The current system simulates:
- Reynolds-style herd behavior with species-specific profiles (`sheep`, `goats`, `cows`)
- Geometric herd analysis (centroid, convex hull, buffered/extended hull, herd radius)
- A two-phase swarm strategy:
  1. **Gathering phase**: drones spread around the buffered hull and compress dispersed animals
  2. **Driving phase**: once compact enough, drones patrol a rear arc and push the herd toward a goal

## Implemented Functionality

### Herd / Animal Simulation
- Agent-based herd dynamics with cohesion, alignment, separation, noise, and drone repulsion.
- Configurable species profiles via `ANIMAL_PROFILES` in `main_swarm_sim.py`.
- Speed limiting and boundary handling inside a configurable world box.

### Swarm Control
- Multi-drone manager (`control/swarm_manager.py`) handling per-drone control updates.
- Polygon-to-1D mapping (`control/swarm_control.py`) for assignment over hull perimeter.
- Target allocation and direction selection with collision-order checks.
- Sliding-mode-inspired edge-following control law (`control/control_law.py`).

### Two-Phase Herding Logic
- **Gathering**:
  - Build convex hull and extended hull around herd.
  - Generate evenly spaced target points on the extended hull.
  - Assign drones to targets and move along closest edges.
- **Driving**:
  - Triggered when herd radius drops below `gathering_threshold_radius`.
  - Generate rear semicircle patrol bounds behind herd relative to goal.
  - Drones perform segment sweep motion to drive herd toward target.

### Visualization
- Real-time matplotlib animation of animals, drones, centroid, hull geometry, and phase-dependent guides.
- Goal marker and phase/radius status in plot title.

## Repository Structure

- `main_swarm_sim.py` – main simulation loop, geometry, herd dynamics, phase switching, visualization.
- `control/drone.py` – drone state model and dynamics update.
- `control/control_law.py` – steering and edge-following control laws.
- `control/swarm_control.py` – polygon mapping, assignment logic, travel distance/collision checks.
- `control/swarm_manager.py` – swarm-level update logic for gathering and driving phases.
- `control/test_pilot.py` – standalone visual pilot/swarm control script.
- `animals.py` – older/alternate single-drone simulation variant.

## Installation

```bash
pip install -r requirements.txt
```

If Qt backend issues occur on Linux, install:

```bash
sudo apt install libxcb-cursor0
```

## Running

Run the full two-phase simulation:

```bash
python main_swarm_sim.py
```

Run the pilot/control visual test script:

```bash
python control/test_pilot.py
```

## Main Configuration

Tune simulation behavior in `SimConfig` (`main_swarm_sim.py`), including:
- `n_animals`, `n_drones`
- `drone_influence_radius`, `drone_v_max`, `drone_u_max`
- `extended_hull_margin`
- `goal_position`
- `gathering_threshold_radius`

Switch animal type by changing `selected_profile` in `main_swarm_sim.py`.
