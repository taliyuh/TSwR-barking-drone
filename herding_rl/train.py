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
import os
import sys

if __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from herding_rl.gains_env import HerdingGainTunerEnv
from herding_rl.config import EnvConfig, RewardConfig, TrainConfig, CurriculumConfig, SimConfigRL

# callback for automatically increasing difficulty as training progresses
class CurriculumCallback(BaseCallback):
    """
    Adjusts environment difficulty based on total training timesteps.
    Only varies animal count — goal position and success radius stay fixed
    (read from SimConfigRL, never change between phases).
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
        else:
            phase = 2
            n_animals = self.cfg.phase2_animals

        if phase != self._last_phase:
            self._last_phase = phase
            if self.verbose:
                print(f"\n>>> Curriculum: entering Phase {phase} "
                      f"(animals={n_animals})")
            try:
                self.training_env.env_method("set_curriculum", n_animals)
            except Exception:
                pass
        return True

# when called, return fresh environment instance
# avoids sharing memory and ensures proper seeding across subprocesses
def make_env(rank, seed=0):
    def _init():
        env = HerdingGainTunerEnv()
        env = Monitor(env)
        return env
    return _init

def train(resume_path=None, timesteps=None):

    # load the training hyperparameters from config
    # when user passes --timesteps, it overrides the config value for total_timesteps
    train_cfg = TrainConfig()
    if timesteps:
        train_cfg.total_timesteps = timesteps

    os.makedirs(train_cfg.save_path, exist_ok=True)
    os.makedirs(train_cfg.log_path, exist_ok=True)

    # create multiple parallel copies of the environment, eaach running independent simulation
    # agent sees many different scenarious at once, speeding up learning
    if train_cfg.n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(train_cfg.n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # normalse observations and rewards, and save stats to imaginal disk for resuming later
    stats_path = os.path.join(train_cfg.save_path, "vec_normalize.pkl")
    if resume_path and os.path.exists(stats_path):
        print(f"Loading VecNormalize stats from {stats_path}")
        env = VecNormalize.load(stats_path, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # ether start a fresh agent or resume from a checkpoint
    if resume_path and resume_path.endswith(".zip"):
        resume_path = resume_path[:-4]

    if resume_path:
        print(f"Resuming training from {resume_path}")
        model = PPO.load(resume_path, env=env, device=train_cfg.device)
    else:
        print("Starting fresh training")
        model = PPO(
            "MlpPolicy", # simple feed-forward neural network
            env,
            learning_rate=train_cfg.learning_rate,
            n_steps=2048, # number of steps to run for each environment per update
            batch_size=train_cfg.batch_size, # number of samples per gradient update
            n_epochs=train_cfg.n_epochs,
            gamma=train_cfg.gamma, # discount factor
            gae_lambda=train_cfg.gae_lambda, # gae lambda for advantage estimation
            clip_range=train_cfg.clip_range, # how much the policy can change per update
            ent_coef=train_cfg.ent_coef, # entropy coefficient
            policy_kwargs=dict(net_arch=train_cfg.net_arch), # nn size
            tensorboard_log=train_cfg.log_path, # directory for logging
            verbose=1,
            device=train_cfg.device,
        )

    # saves full model sometimes
    checkpoint_cb = CheckpointCallback(
        save_freq=max(train_cfg.checkpoint_freq // train_cfg.n_envs, 1),
        save_path=train_cfg.save_path,
        name_prefix="commander_ppo",
    )

    # adjust difficulty as training progresses
    curriculum_cb = CurriculumCallback(verbose=1)

    # log k_edge to TensorBoard so you can watch the learned gain evolve
    class GainLoggingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self._k_edge_buffer = []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            for info in infos:
                if isinstance(info, dict) and "k_edge" in info:
                    self._k_edge_buffer.append(info["k_edge"])
            # flush buffer every ~1000 env steps
            if len(self._k_edge_buffer) >= 1000:
                mean_k = sum(self._k_edge_buffer) / len(self._k_edge_buffer)
                self.logger.record("rollout/k_edge", mean_k)
                self._k_edge_buffer.clear()
            return True

    gain_logging_cb = GainLoggingCallback()

    # save statstics alongside the checkpoint to resume normalisation
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

    callbacks = CallbackList([checkpoint_cb, curriculum_cb, gain_logging_cb, save_norm_cb])

    # not lose the entire progress when ragequiting
    def _signal_handler(sig, frame):
        print("\n\n>>> Ctrl+C received — saving model before exit...")
        model.save(os.path.join(train_cfg.save_path, "commander_ppo_interrupted"))
        env.save(os.path.join(train_cfg.save_path, "vec_normalize.pkl"))
        print(f">>> Saved to {train_cfg.save_path}/commander_ppo_interrupted.zip")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    # training loop
    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=(resume_path is None),
    )

    # save final model
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
