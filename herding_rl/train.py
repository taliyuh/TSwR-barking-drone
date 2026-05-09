import os
import sys
import signal
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from herding_rl.gains_env import HerdingGainTunerEnv
from herding_rl.config import EnvConfig, RewardConfig, TrainConfig, CurriculumConfig, SimConfigRL


# ---------------------------------------------------------------------------
# Curriculum Callback — adjusts environment difficulty during training
# ---------------------------------------------------------------------------
class CurriculumCallback(BaseCallback):
    """
    Adjusts environment difficulty based on total training timesteps.
    Uses env_method to call a setter on each sub-environment.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cfg = CurriculumConfig()
        self._last_phase = -1

    def _on_step(self) -> bool:
        steps = self.num_timesteps

        if steps < self.cfg.phase1_steps:
            phase = 1
            n_animals = self.cfg.phase1_animals
            goal = self.cfg.phase1_goal
            radius = self.cfg.phase1_radius
        elif steps < self.cfg.phase2_steps:
            phase = 2
            n_animals = self.cfg.phase2_animals
            goal = self.cfg.phase2_goal
            radius = self.cfg.phase2_radius
        else:
            phase = 3
            n_animals = 40
            goal = (18.0, 18.0)
            radius = 3.0

        # Only update when phase changes to avoid overhead
        if phase != self._last_phase:
            self._last_phase = phase
            if self.verbose:
                print(f"\n>>> Curriculum: entering Phase {phase} "
                      f"(animals={n_animals}, goal={goal}, Rc={radius})")
            try:
                self.training_env.env_method("set_curriculum", n_animals, goal, radius)
            except Exception:
                pass  # eval env may not support this
        return True


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def make_env(rank, seed=0):
    def _init():
        env = HerdingGainTunerEnv()
        env = Monitor(env)
        return env
    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(resume_path=None, timesteps=None):
    train_cfg = TrainConfig()
    if timesteps:
        train_cfg.total_timesteps = timesteps

    os.makedirs(train_cfg.save_path, exist_ok=True)
    os.makedirs(train_cfg.log_path, exist_ok=True)

    # Create vectorized environment
    if train_cfg.n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(train_cfg.n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # Observation / reward normalization (load if resuming)
    stats_path = os.path.join(train_cfg.save_path, "vec_normalize.pkl")
    if resume_path and os.path.exists(stats_path):
        print(f"Loading VecNormalize stats from {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Strip .zip if user passed it (SB3 appends it automatically)
    if resume_path and resume_path.endswith(".zip"):
        resume_path = resume_path[:-4]

    if resume_path:
        print(f"Resuming training from {resume_path}")
        model = PPO.load(resume_path, env=env, device=train_cfg.device)
    else:
        print("Starting fresh training")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=train_cfg.learning_rate,
            n_steps=2048,                         # rollout buffer per env
            batch_size=train_cfg.batch_size,
            n_epochs=train_cfg.n_epochs,
            gamma=train_cfg.gamma,
            gae_lambda=train_cfg.gae_lambda,
            clip_range=train_cfg.clip_range,
            ent_coef=train_cfg.ent_coef,
            policy_kwargs=dict(net_arch=train_cfg.net_arch),
            tensorboard_log=train_cfg.log_path,
            verbose=1,
            device=train_cfg.device,
        )

    # ---- Callbacks ----
    checkpoint_cb = CheckpointCallback(
        save_freq=max(train_cfg.checkpoint_freq // train_cfg.n_envs, 1),
        save_path=train_cfg.save_path,
        name_prefix="commander_ppo",
    )

    curriculum_cb = CurriculumCallback(verbose=1)

    class SaveNormCallback(BaseCallback):
        """Save VecNormalize stats alongside each checkpoint."""
        def __init__(self, path, freq, verbose=0):
            super().__init__(verbose)
            self._path = path
            self._freq = freq
        def _on_step(self) -> bool:
            if self.n_calls % self._freq == 0:
                vec_env = self.model.get_vec_normalize_env()
                if vec_env is not None:
                    vec_env.save(os.path.join(self._path, "vec_normalize.pkl"))
            return True

    save_norm_cb = SaveNormCallback(
        train_cfg.save_path,
        freq=max(train_cfg.checkpoint_freq // train_cfg.n_envs, 1),
    )

    callbacks = CallbackList([checkpoint_cb, curriculum_cb, save_norm_cb])

    # ---- Graceful Ctrl+C: save model before exiting ----
    def _signal_handler(sig, frame):
        print("\n\n>>> Ctrl+C received — saving model before exit...")
        model.save(os.path.join(train_cfg.save_path, "commander_ppo_interrupted"))
        env.save(os.path.join(train_cfg.save_path, "vec_normalize.pkl"))
        print(f">>> Saved to {train_cfg.save_path}/commander_ppo_interrupted.zip")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    # ---- Train ----
    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume_path is None),
    )

    # Save final model
    model.save(os.path.join(train_cfg.save_path, "commander_ppo_final"))
    env.save(os.path.join(train_cfg.save_path, "vec_normalize_final.pkl"))
    print(f"Training complete. Model saved to {train_cfg.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Herding Commander with PPO")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model checkpoint to resume from (with or without .zip)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total timesteps to train")
    args = parser.parse_args()

    train(resume_path=args.resume, timesteps=args.timesteps)
