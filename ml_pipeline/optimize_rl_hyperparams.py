"""
Optuna Hyperparameter Optimization for PPO Agent

This script uses Optuna to find the best hyperparameters for the PPO agent
to address the early peaking problem and improve long-term learning.
"""

import sys
sys.path.insert(0, 'src')

import os
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from rl.environment import FundingArbitrageEnv


class EntropyAnnealingCallback(BaseCallback):
    """
    Custom callback to anneal entropy coefficient during training.
    """
    def __init__(self, initial_ent: float, final_ent: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_timesteps = total_timesteps

    def _on_rollout_end(self) -> None:
        """Update entropy coefficient at the end of each rollout."""
        progress = self.num_timesteps / self.total_timesteps
        new_ent_coef = self.initial_ent - progress * (self.initial_ent - self.final_ent)
        new_ent_coef = max(new_ent_coef, self.final_ent)
        self.model.ent_coef = new_ent_coef
        self.logger.record("train/ent_coef", new_ent_coef)

    def _on_step(self) -> bool:
        return True


class TrialEvalCallback(BaseCallback):
    """
    Callback for Optuna trials to report intermediate values and prune trials.
    """
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=5000):
        super().__init__()
        self.trial = trial
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the model
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()  # VecEnv returns just obs, not (obs, info)
                done = False
                episode_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = self.eval_env.step(action)  # VecEnv returns 4 values
                    episode_reward += rewards[0]  # Extract reward for single env
                    done = dones[0]  # Extract done for single env
                episode_rewards.append(episode_reward)

            mean_reward = np.mean(episode_rewards)

            # Report intermediate value
            self.trial.report(mean_reward, self.eval_idx)
            self.eval_idx += 1

            # Check if trial should be pruned
            if self.trial.should_prune():
                self.is_pruned = True
                return False

        return True


def create_env(data_path: str, price_history_path: str = None, seed: int = None, **reward_kwargs):
    """Create and wrap the environment."""
    env = FundingArbitrageEnv(
        data_path=data_path,
        price_history_path=price_history_path,
        initial_capital=10000.0,
        episode_length_days=3,
        max_positions=10,
        max_opportunities_per_hour=10,
        **reward_kwargs
    )
    env = Monitor(env)
    return env


def make_env(data_path: str, price_history_path: str = None, seed: int = None, rank: int = 0, **reward_kwargs):
    """Factory function for parallel environment creation."""
    def _init():
        env = create_env(data_path, price_history_path, seed=seed + rank if seed else None, **reward_kwargs)
        return env
    return _init


def objective(trial: optuna.Trial, data_path: str, eval_data_path: str, price_history_path: str, tune_rewards: bool = True):
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        data_path: Path to training data
        eval_data_path: Path to evaluation data
        price_history_path: Path to price history
        tune_rewards: Whether to tune reward shaping parameters

    Returns:
        Mean evaluation reward
    """

    # Sample PPO hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 5, 20)
    gamma = trial.suggest_float("gamma", 0.99, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.05, 0.3)
    ent_coef_initial = trial.suggest_float("ent_coef_initial", 0.005, 0.05)
    ent_coef_final = trial.suggest_float("ent_coef_final", 0.001, 0.01)

    # Reward shaping hyperparameters
    reward_kwargs = {}
    if tune_rewards:
        reward_kwargs['pnl_reward_scale'] = trial.suggest_float("reward_pnl_scale", 0.01, 0.5, log=True)
        reward_kwargs['hold_bonus'] = trial.suggest_float("reward_hold_bonus", 0.0, 0.3)
        reward_kwargs['quality_entry_bonus'] = trial.suggest_float("reward_quality_bonus", 0.0, 1.0)
        reward_kwargs['quality_entry_penalty'] = trial.suggest_float("reward_quality_penalty", -1.0, 0.0)

    # Network architecture
    net_arch_choice = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_choice == "small":
        net_arch = [128, 128]
    elif net_arch_choice == "medium":
        net_arch = [256, 256]
    else:  # large
        net_arch = [512, 512]

    # Ensure ent_coef_final < ent_coef_initial
    if ent_coef_final >= ent_coef_initial:
        ent_coef_final = ent_coef_initial * 0.25

    # Create environments
    n_envs = 4  # Reduced for faster trials
    train_env = SubprocVecEnv([make_env(data_path, price_history_path, 42, i, **reward_kwargs) for i in range(n_envs)])
    eval_env = DummyVecEnv([lambda: create_env(eval_data_path, price_history_path, seed=999, **reward_kwargs)])

    try:
        # Create model
        policy_kwargs = dict(
            net_arch=dict(pi=net_arch, vf=net_arch),
            normalize_images=False
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef_initial,
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            device="cpu",
            verbose=0,
            seed=42
        )

        # Create callbacks
        entropy_callback = EntropyAnnealingCallback(
            initial_ent=ent_coef_initial,
            final_ent=ent_coef_final,
            total_timesteps=100000  # Updated for better convergence
        )

        trial_eval_callback = TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=3,  # Fewer episodes for speed
            eval_freq=5000
        )

        # Train
        model.learn(
            total_timesteps=100000,  # 100K timesteps per trial
            callback=[entropy_callback, trial_eval_callback],
            progress_bar=False
        )

        # Final evaluation
        episode_rewards = []
        for _ in range(5):
            obs = eval_env.reset()  # VecEnv returns just obs, not (obs, info)
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_env.step(action)  # VecEnv returns 4 values
                episode_reward += rewards[0]  # Extract reward for single env
                done = dones[0]  # Extract done for single env
            episode_rewards.append(episode_reward)

        mean_reward = np.mean(episode_rewards)

    except Exception as e:
        import traceback
        print(f"Trial failed with error: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        mean_reward = -np.inf
    finally:
        train_env.close()
        eval_env.close()

    return mean_reward


def optimize_hyperparameters(
    data_path: str = 'data/rl_train.csv',
    eval_data_path: str = 'data/rl_test.csv',
    price_history_path: str = 'data/price_history',
    n_trials: int = 50,
    study_name: str = None
):
    """
    Run Optuna hyperparameter optimization.

    Args:
        data_path: Path to training data
        eval_data_path: Path to evaluation data
        price_history_path: Path to price history
        n_trials: Number of trials to run
        study_name: Name for the study (auto-generated if None)
    """

    if study_name is None:
        study_name = f"ppo_optim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Training data: {data_path}")
    print(f"Evaluation data: {eval_data_path}")
    print()

    # Create study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    # Optimize
    # Reward tuning is now enabled - tunes PPO hyperparameters AND reward shaping
    study.optimize(
        lambda trial: objective(trial, data_path, eval_data_path, price_history_path, tune_rewards=True),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (mean reward): {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    # Save study
    study_dir = "models/rl_optuna"
    os.makedirs(study_dir, exist_ok=True)
    study_path = os.path.join(study_dir, f"{study_name}.pkl")

    import joblib
    joblib.dump(study, study_path)
    print(f"\nStudy saved to: {study_path}")

    # Save best params to text file
    params_path = os.path.join(study_dir, f"{study_name}_best_params.txt")
    with open(params_path, 'w') as f:
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best value: {study.best_value:.2f}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"Best parameters saved to: {params_path}")

    return study


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Optimize PPO hyperparameters with Optuna')
    parser.add_argument('--data-path', type=str, default='data/rl_train.csv',
                        help='Path to training data')
    parser.add_argument('--eval-data-path', type=str, default='data/rl_test.csv',
                        help='Path to evaluation data')
    parser.add_argument('--price-history-path', type=str, default='data/price_history',
                        help='Path to price history')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Name for the Optuna study')

    args = parser.parse_args()

    study = optimize_hyperparameters(
        data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        price_history_path=args.price_history_path,
        n_trials=args.n_trials,
        study_name=args.study_name
    )
