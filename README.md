# Barking Drone

The project group consists of Mikołaj Lipiński and Bartłomiej Hryniewski.

## Overview

This repository implements a drone-assisted livestock herding simulation inspired by
[Robotic Herding of Farm Animals Using a Network of Barking Aerial Drones](https://www.mdpi.com/2504-446X/6/2/29).

The current codebase includes:
- Agent-based herd dynamics with species profiles (`sheep`, `goats`, `cows`)
- Multi-drone geometric control using convex/extended hull logic
- Two-phase herding (gathering → driving)
- Reinforcement learning environments and PPO training/evaluation pipeline (`herding_rl/`)

## Implemented Core Functionality

### Herd and Geometry
- Reynolds-style behavior terms (cohesion, separation, alignment, drone repulsion, noise).
- Herd geometry utilities: centroid, convex hull, extended hull, herd radius.
- World boundary handling and speed limiting.

### Swarm Control
- Swarm manager for multi-drone updates.
- Polygon-to-1D perimeter mapping for assignment.
- Target allocation with direction search and collision-order checks.
- Sliding-mode-inspired control law for moving to/following edges.

### Two-Phase Herding
- **Gathering phase:** drones spread around the extended hull and compress herd dispersion.
- **Driving phase:** triggered when herd radius crosses threshold; drones sweep along a rear arc toward the goal.

### Visualization
- Real-time matplotlib simulation view of animals, drones, hulls, targets, phase, and goal.

## Reinforcement Learning (RL)

RL components live under `herding_rl/` and use the simulation/control stack as the plant.

### RL Environments
- `herding_rl/gains_env.py` (`HerdingGainTunerEnv`)
  - Action: continuous tuning of heuristic gains (`k_edge`, `k_target`, `v_scale`)
  - Uses gathering/driving logic and SMC-based swarm control beneath the RL policy
  - This is the environment currently used by training (`herding_rl/train.py`)

- `herding_rl/commander_env.py` (`HerdingCommanderEnv`)
  - Hierarchical setup where RL outputs waypoint-like commands and lower-level control tracks them

### Training and Evaluation
- `herding_rl/train.py`
  - PPO training via Stable-Baselines3
  - Supports checkpointing, curriculum phases, vectorized envs, and normalization stats
- `herding_rl/evaluate.py`
  - Deterministic rollout evaluation, success metrics, optional video export

### RL Config
- `herding_rl/config.py` defines:
  - `SimConfigRL`, `EnvConfig`, `RewardConfig`, `TrainConfig`, `CurriculumConfig`
  - Includes optional partial observability setting (`use_partial_observability`)

### RL Integration Test Script
- `test_rl_integration.py`
  - Script that checks env reset/step shapes, PPO forward pass, short training loop, and partial observability mode

## Repository Structure

- `main_swarm_sim.py` – main simulation, phase switching, geometry, visualization
- `animals.py` – alternate/legacy single-drone-focused simulation variant
- `control/drone.py` – drone dynamics model
- `control/control_law.py` – steering/edge-following control law
- `control/swarm_control.py` – assignment + polygon mapping logic
- `control/swarm_manager.py` – swarm coordination for gathering/driving
- `control/test_pilot.py` – visual pilot script for swarm behavior
- `herding_rl/` – RL envs, training, evaluation, and config

## Installation

Base simulation dependencies:

```bash
pip install -r requirements.txt
```

For RL training/evaluation, also install:

```bash
pip install torch gymnasium stable-baselines3
```

If Qt backend issues occur on Linux:

```bash
sudo apt install libxcb-cursor0
```

## Running

Run the simulation:

```bash
python main_swarm_sim.py
```

Run the pilot visualization script:

```bash
python control/test_pilot.py
```

Run RL training:

```bash
python -m herding_rl.train
```

Run RL evaluation:

```bash
python -m herding_rl.evaluate --model <path-to-model.zip> [--stats <vec-normalize.pkl>]
```

Run RL integration checks:

```bash
python test_rl_integration.py
```

## Notes

- `requirements.txt` currently contains core simulation dependencies; RL dependencies are listed above separately.
- The active simulation profile in `main_swarm_sim.py` is selected via `selected_profile`.
