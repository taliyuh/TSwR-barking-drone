# Barking Drone

The project group consists of Mikołaj Lipiński and Bartłomiej Hryniewski.


## Description

The goal of this project is to create a control algorithm for a swarm of drones for the purpose of livestock herding. The project is based on the paper [Robotic Herding of Farm Animals Using a Network of Barking Aerial Drones](https://www.mdpi.com/2504-446X/6/2/29). Our contribution to the project will include application of appropriate reinforced learning algorithm, either for modelling or estimating animal herd model, or generating drones trajectory.

## Goals

1. **Milestone**: Implementation of the animal agent model (behavioural rules) and the integration of a convex hull algorithm to dynamically define the herd boundary and drone trajectory.

2. **Project Completion**: A fully autonomous, two-phase control system capable of herd aggregation (grouping) and trajectory tracking to guide the herd to a predefined target location.

## System

Cow behavior will be modeled using Reynolds' rules of flocking behavior. The drones will operate in two modes: 
1. **Gathering mode:** Drones will navigate alongside a complex polygon surrounding the dispersed animals to push the furthest outlier animals towards the center of the herd.
2. **Driving mode:** The polygon simplifies to a circle in order to push the aggregated herd to the target location.

The core control algorithm will be Sliding Mode Control (SMC), utilizing geometric switching logic. 

## Inputs and Outputs

The inputs to the plant (simulation) will consist of:

- Linear drone velocity $v(t)$, where $v(t) \in [0, V_{max}]$
- Angular steering command $u(t)$, where $|u(t)| \le U_{max}$

The outputs of the plant (which serve as feedback to the controller) will be:

- Drone Cartesian coordinates $d(t) = [x(t), y(t)]$
- Drone heading direction vector $a(t)$
- 2D cow positions (specifically the vertices of the herd's convex hull)
- Centroid of the herd $C_o$

---

## Usage

### Setup

```bash
cd tswr_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Launch the standard simulation

Runs the full SMC herding simulation with hand-tuned gains ($k_{edge}=1$, $k_{target}=1$, $v_{scale}=1$) and live Matplotlib visualization:

```bash
python3 main_swarm_sim.py
```

- 100 sheep, 4 drones, gathering → driving phases
- Goal at $(15, 15)$, success radius $5$
- Simulation runs up to 2500 steps (250 s simulated time, ~15 s wall-clock)
- Closing the plot window terminates the simulation early

### 2. Train the RL gain tuner

Trains a PPO policy to learn the optimal $k_{edge}$ gain for the SMC controller.
$k_{target}$ and $v_{scale}$ are fixed at $1.0$ — only the ratio $k_{edge}/k_{target}$
matters in the normalized control law.

```bash
# Full training (2M timesteps, ~2–3 hours)
python3 herding_rl/train.py

# Quick test run (10k timesteps, ~1 minute)
python3 herding_rl/train.py --timesteps 10000

# Resume from checkpoint
python3 herding_rl/train.py --resume models/gains_ppo/commander_ppo_100000_steps
```

**Curriculum:** The environment difficulty auto-increases:
- **Phase 1** (0–500k steps): 20 animals
- **Phase 2** (500k–1M steps): 30 animals
- **Phase 3** (1M+ steps): 40 animals

Goal position $(15, 15)$ and success radius $5$ stay **fixed** across all phases.

Monitor progress with TensorBoard:
```bash
tensorboard --logdir logs/gains_ppo/
```
Watch the `rollout/k_edge` scalar to see the learned gain evolve over time.

### 3. Evaluate a trained model

Run deterministic rollouts and see success statistics. The first episode is recorded as a video:

```bash
python3 herding_rl/evaluate.py \
  --model models/gains_ppo/commander_ppo_final.zip \
  --stats models/gains_ppo/vec_normalize_final.pkl \
  --episodes 5
```

Output shows per-episode success/failure, steps taken, reward, and final distance to goal.
Videos are saved to `videos/`.

To skip video generation: add `--no-video`.


## Project Structure

```
tswr_project/
├── main_swarm_sim.py          # Standard simulation entry point
├── animals.py                 # Herd dynamics & animal profiles
├── requirements.txt
├── control/
│   ├── control_law.py         # SMC: fly_on_edge() — b* = k_edge*b + k_target*o*
│   ├── drone.py               # Drone state: position d, heading a, velocity v
│   ├── swarm_control.py       # 1D projection & drone-to-target allocation
│   ├── swarm_manager.py       # Orchestrates drones in gathering/driving modes
│   └── test_pilot.py
├── herding_rl/
│   ├── config.py              # All RL hyperparameters, curriculum, sim configs
│   ├── gains_env.py           # Gymnasium env — learns k_edge via PPO
│   ├── train.py               # PPO training loop with curriculum callback
│   └── evaluate.py            # Deterministic rollouts + video generation
├── models/gains_ppo/          # Saved PPO checkpoints
├── logs/gains_ppo/            # TensorBoard logs
└── videos/                    # Evaluation recordings
```

