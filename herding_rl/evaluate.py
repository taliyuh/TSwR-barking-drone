# irrelevant, helper script just to see the rl results

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys

# Allow running as `python herding_rl/evaluate.py` from the project root.
if __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from herding_rl.gains_env import HerdingGainTunerEnv

def evaluate(model_path, stats_path=None, n_episodes=3, render=True, save_video=True):
    """Run deterministic rollouts and report success statistics with optional video generation."""
    
    env = DummyVecEnv([lambda: HerdingGainTunerEnv()])

    if stats_path and os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False

    if model_path.endswith(".zip"):
        model_path = model_path[:-4]

    print(f"Loading model from {model_path}")
    model = PPO.load(model_path, env=env)

    success_count = 0
    total_dist = 0.0
    total_reward = 0.0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        step = 0
        
        frames_data = [] # Store data for animation

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            step += 1
            
            # Extract underlying env to access physical state for rendering
            base_env = env.envs[0]
            if save_video:
                drone_pos, _ = base_env.swarm_manager.get_swarm_status()
                frames_data.append({
                    'animals': base_env.herd.positions.copy(),
                    'drones': np.array(drone_pos).copy(),
                    'goal': np.array(base_env.sim_cfg.goal_position).copy(),
                    'radius': base_env.sim_cfg.success_radius,
                    'step': step,
                    'reward': ep_reward
                })

            if render and step % 20 == 0:
                d = info[0].get("dist_to_goal", -1)
                r = info[0].get("herd_radius", -1)
                f = info[0].get("frac_in_goal", 0)
                print(f"  step {step:3d} | dist={d:.2f} | radius={r:.2f} | frac_in_goal={f:.2%}")

        ep_info = info[0]
        is_success = ep_info.get("is_success", False)
        if is_success:
            success_count += 1
        total_dist += ep_info.get("dist_to_goal", 0)
        total_reward += ep_reward

        tag = "✓ SUCCESS" if is_success else "✗ FAIL"
        print(f"Episode {ep+1:2d}: {tag}  steps={step}  reward={ep_reward:.1f}  dist={ep_info.get('dist_to_goal', -1):.2f}")
        
        if save_video and ep == 0:
            print("Generating video for Episode 1...")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            
            goal_circle = plt.Circle(frames_data[0]['goal'], frames_data[0]['radius'], color='g', alpha=0.2)
            ax.add_patch(goal_circle)
            
            animal_scatter = ax.scatter([], [], c='blue', s=10, label='Animals')
            drone_scatter = ax.scatter([], [], c='red', s=40, marker='^', label='Drones')
            goal_scatter = ax.scatter([frames_data[0]['goal'][0]], [frames_data[0]['goal'][1]], c='green', s=100, marker='*', label='Goal')
            
            title = ax.set_title("")
            ax.legend()
            
            def update(frame_idx):
                data = frames_data[frame_idx]
                animal_scatter.set_offsets(data['animals'])
                drone_scatter.set_offsets(data['drones'])
                title.set_text(f"Step: {data['step']} | Reward: {data['reward']:.1f}")
                return animal_scatter, drone_scatter, title

            ani = animation.FuncAnimation(fig, update, frames=len(frames_data), interval=50, blit=True)
            os.makedirs("videos", exist_ok=True)
            video_path = f"videos/eval_ep{ep+1}.mp4"
            try:
                ani.save(video_path, writer='ffmpeg', fps=20)
                print(f"Saved video to {video_path}")
            except Exception as e:
                print(f"Failed to save video (is ffmpeg installed?): {e}")
            plt.close(fig)

    print(f"\n{'='*50}")
    print(f"Results over {n_episodes} episodes:")
    print(f"  Success rate : {success_count}/{n_episodes} ({100*success_count/n_episodes:.0f}%)")
    print(f"  Avg reward   : {total_reward/n_episodes:.1f}")
    print(f"  Avg final dist: {total_dist/n_episodes:.2f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Herding Commander")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--stats", type=str, default=None, help="Path to VecNormalize .pkl")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--no-video", action="store_true", help="Disable video generation")
    args = parser.parse_args()

    evaluate(args.model, args.stats, args.episodes, save_video=not args.no_video)
