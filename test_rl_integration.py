#!/usr/bin/env python3
"""
Integration test script for RL environment changes (animals-model compatibility).
Tests:
  1. env.reset() → obs shape matches observation_space (both envs)
  2. env.step(action) → reward, done, info shapes (gains_env — the one used in training)
  3. Model forward pass (PPO) accepts new observation tensor
  4. 2 training iterations to catch runtime errors
  5. Partial observability mode
"""
import sys
import os
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from herding_rl.commander_env import HerdingCommanderEnv
from herding_rl.gains_env import HerdingGainTunerEnv
from herding_rl.config import SimConfigRL

# ============================================================
# Test 1: Shape check — HerdingCommanderEnv (reset only)
# ============================================================
print("=" * 60)
print("TEST 1: HerdingCommanderEnv — reset shape check")
print("=" * 60)

env = HerdingCommanderEnv()
obs, info = env.reset()
print(f"  obs shape       : {obs.shape}")
print(f"  expected shape  : {env.observation_space.shape}")
assert obs.shape == env.observation_space.shape, \
    f"Obs shape mismatch: {obs.shape} vs {env.observation_space.shape}"
print(f"  ✓ obs shape OK")
print(f"  ✓ HerdingCommanderEnv reset PASSED\n")

# NOTE: step() calls self.swarm_manager.track_waypoints() which does not exist
# in SwarmManager. This is a pre-existing bug in commander_env.py, not caused
# by our animals-model integration changes. The gains_env is the one used in
# training (see train.py: from herding_rl.gains_env import HerdingGainTunerEnv).

# ============================================================
# Test 2: Shape check — HerdingGainTunerEnv (reset & step)
# ============================================================
print("=" * 60)
print("TEST 2: HerdingGainTunerEnv — reset & step shape check")
print("=" * 60)

env2 = HerdingGainTunerEnv()
obs, info = env2.reset()
print(f"  obs shape       : {obs.shape}")
print(f"  expected shape  : {env2.observation_space.shape}")
assert obs.shape == env2.observation_space.shape, \
    f"Obs shape mismatch: {obs.shape} vs {env2.observation_space.shape}"
print(f"  ✓ obs shape OK")

action = env2.action_space.sample()
obs, reward, terminated, truncated, info = env2.step(action)
print(f"  reward          : {reward:.4f}")
print(f"  terminated      : {terminated}")
print(f"  truncated       : {truncated}")
print(f"  info keys       : {list(info.keys())}")
assert obs.shape == env2.observation_space.shape, \
    f"Step obs shape mismatch: {obs.shape} vs {env2.observation_space.shape}"
print(f"  ✓ step obs shape OK")
print(f"  ✓ HerdingGainTunerEnv PASSED\n")

# ============================================================
# Test 3: Test with use_partial_observability=True
# ============================================================
print("=" * 60)
print("TEST 3: Partial observability mode (use_partial_observability=True)")
print("=" * 60)

sim_cfg_po = SimConfigRL(use_partial_observability=True)
env_po = HerdingGainTunerEnv(sim_cfg=sim_cfg_po)
obs, info = env_po.reset()
print(f"  obs shape       : {obs.shape}")
assert obs.shape == env_po.observation_space.shape, \
    f"PO obs shape mismatch: {obs.shape} vs {env_po.observation_space.shape}"

action = env_po.action_space.sample()
obs, reward, terminated, truncated, info = env_po.step(action)
print(f"  reward          : {reward:.4f}")
assert obs.shape == env_po.observation_space.shape, \
    f"PO step obs shape mismatch: {obs.shape} vs {env_po.observation_space.shape}"
print(f"  ✓ Partial observability PASSED\n")

# ============================================================
# Test 4: Model forward pass (PPO)
# ============================================================
print("=" * 60)
print("TEST 4: PPO model forward pass")
print("=" * 60)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return HerdingGainTunerEnv()

vec_env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=3,
    verbose=0,
    device="cpu",
)

# Forward pass: get action from policy
obs = vec_env.reset()
print(f"  vec_env obs shape: {obs.shape}")  # (1, 23)
action, _states = model.predict(obs, deterministic=True)
print(f"  predicted action : {action.shape}")  # (1, 3)
assert action.shape == (1, env2.action_space.shape[0]), \
    f"Action shape mismatch: {action.shape} vs (1, {env2.action_space.shape[0]})"
print(f"  ✓ PPO forward pass OK\n")

# ============================================================
# Test 5: 2 training iterations
# ============================================================
print("=" * 60)
print("TEST 5: 2 training iterations (mini-batch)")
print("=" * 60)

# Train for 2 gradient steps
model.learn(total_timesteps=64, reset_num_timesteps=True)
print(f"  ✓ 64 timesteps of training completed without error")

model.learn(total_timesteps=64, reset_num_timesteps=False)
print(f"  ✓ Another 64 timesteps completed without error")
print(f"  ✓ Training loop PASSED\n")

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print("""
  ✓ HerdingCommanderEnv reset (obs shape OK)
  ✓ HerdingGainTunerEnv reset/step (obs shape, reward, done OK)
  ✓ Partial observability mode (use_partial_observability=True)
  ✓ PPO forward pass accepts new observation tensor
  ✓ 2 training iterations (128 timesteps total)

NOTE: HerdingCommanderEnv.step() has a pre-existing bug:
  'SwarmManager' object has no attribute 'track_waypoints'
  This is NOT caused by the animals-model integration.
  The gains_env (HerdingGainTunerEnv) is the one used in training.
""")
