#!/usr/bin/env python3
"""
Population-Based Training for Execution Mode (1 Opportunity + 1 Execution)

This script trains agents using the execution-based architecture:
- Observation: 36 dims (14 portfolio+execution + 22 opportunity features)
  - Portfolio base: 4 dims (capital_ratio, utilization, pnl, drawdown)
  - Execution features: 10 dims (net_pnl, hours, net_funding_ratio, net_funding_rate,
                                 current_spread, entry_spread, value_ratio, funding_efficiency,
                                 long_pnl, short_pnl)
  - Opportunity: 22 features
- Actions: 3 (HOLD, ENTER, EXIT)
- Network: [64, 32] (simplified for faster training)
- Key improvements:
  - Model sees both long AND short positions of execution
  - CAPITAL-INDEPENDENT: Fixed $1k position size + percentage-based rewards
"""

import argparse
import os
import sys
import multiprocessing as mp
from datetime import datetime
import json
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from models.rl.core.environment import FundingArbitrageEnv


class SimpleAgent:
    """Single agent in the PBT population (simple mode)."""

    def __init__(self, agent_id, hyperparams, train_data, eval_data,
                 price_history, feature_scaler, save_dir):
        self.agent_id = agent_id
        self.hyperparams = hyperparams
        self.train_data = train_data
        self.eval_data = eval_data
        self.price_history = price_history
        self.feature_scaler = feature_scaler
        self.iteration_count = 0  # Track training iterations for probability sampling
        self.save_dir = save_dir
        self.best_reward = -np.inf

    def train(self, timesteps_per_iteration):
        """Train for one PBT iteration."""
        # Create environments with SIMPLE MODE enabled (silent mode)
        train_env = DummyVecEnv([lambda: Monitor(FundingArbitrageEnv(
            data_path=self.train_data,
            price_history_path=self.price_history,
            feature_scaler_path=self.feature_scaler,
            initial_capital=10000.0,
            episode_length_days=3,
            simple_mode=True,  # Enable 1-opp, 1-pos architecture
            use_full_range_episodes=False,
            fixed_position_size_usd=1000.0,  # CAPITAL-INDEPENDENT: Fixed $1k per execution
            quality_entry_bonus=0.0,
            quality_entry_penalty=0.0,
            verbose=False,  # Silent mode - no logs
        ))])

        eval_env = Monitor(FundingArbitrageEnv(
            data_path=self.eval_data,
            price_history_path=self.price_history,
            feature_scaler_path=self.feature_scaler,
            initial_capital=10000.0,
            simple_mode=True,  # Enable 1-opp, 1-pos architecture
            use_full_range_episodes=True,  # Full range for evaluation
            fixed_position_size_usd=1000.0,  # CAPITAL-INDEPENDENT: Fixed $1k per execution
            quality_entry_bonus=0.0,
            quality_entry_penalty=0.0,
            verbose=False,  # Silent mode - no logs
        ))

        # Create model with network architecture from hyperparameters (PBT-tunable)
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=self.hyperparams['learning_rate'],
            gamma=self.hyperparams['gamma'],
            gae_lambda=self.hyperparams['gae_lambda'],
            ent_coef=self.hyperparams['ent_coef'],
            clip_range=self.hyperparams['clip_range'],
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=self.hyperparams['net_arch']),  # PBT-tunable architecture
            verbose=0,
        )

        # Load weights from checkpoint if it exists AND architecture matches
        model_path = os.path.join(self.save_dir, f'agent_{self.agent_id}_model.zip')
        hyperparam_path = os.path.join(self.save_dir, f'agent_{self.agent_id}_hyperparams.json')

        # Only load if architecture hasn't changed
        should_load = False
        if os.path.exists(model_path) and os.path.exists(hyperparam_path):
            with open(hyperparam_path, 'r') as f:
                saved_hyperparams = json.load(f)
            # Check if net_arch matches
            if saved_hyperparams.get('net_arch') == self.hyperparams['net_arch']:
                should_load = True

        if should_load:
            saved_model = PPO.load(model_path)
            model.policy.load_state_dict(saved_model.policy.state_dict())
            del saved_model

        # Train
        model.learn(total_timesteps=timesteps_per_iteration, reset_num_timesteps=False)

        # Evaluate
        obs = eval_env.reset()[0]
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        # Extract metrics
        pnl_pct = info.get('episode_pnl_pct', 0.0)
        win_rate = info.get('episode_win_rate', 0.0)
        trades = info.get('episode_trades_count', 0)

        # Increment iteration counter
        self.iteration_count += 1

        # Sample action probabilities to monitor learning (every iteration for now)
        if True:  # Show every iteration to monitor learning
            sample_obs = []
            eval_env_sample = eval_env.reset()[0]
            for _ in range(3):  # Sample 3 random states
                sample_obs.append(eval_env_sample)
                # Take a random action to get next state
                action = eval_env.action_space.sample()
                eval_env_sample, _, terminated, truncated, _ = eval_env.step(action)
                if terminated or truncated:
                    eval_env_sample = eval_env.reset()[0]

            # Get action probabilities for samples
            import torch
            with torch.no_grad():
                sample_probs = []
                for obs in sample_obs:
                    obs_tensor = torch.as_tensor(obs).unsqueeze(0).float()
                    action_dist = model.policy.get_distribution(obs_tensor)
                    probs = action_dist.distribution.probs.cpu().numpy()[0]
                    sample_probs.append(probs)

            # Calculate average probabilities across samples
            avg_probs = sum(sample_probs) / len(sample_probs)
            # Store in results for logging
            self.sample_probs = {
                'enter': float(avg_probs[0]),
                'exit': float(avg_probs[1]),
                'hold': float(avg_probs[2])
            }

        # Save model and hyperparameters
        model.save(model_path)
        hyperparam_path = os.path.join(self.save_dir, f'agent_{self.agent_id}_hyperparams.json')
        with open(hyperparam_path, 'w') as f:
            json.dump(self.hyperparams, f, indent=2)

        train_env.close()
        eval_env.close()

        result = {
            'agent_id': self.agent_id,
            'episode_reward': episode_reward,
            'pnl_pct': pnl_pct,
            'win_rate': win_rate,
            'trades': trades,
            'hyperparams': self.hyperparams.copy()
        }

        # Add sample probabilities if available
        if hasattr(self, 'sample_probs'):
            result['sample_probs'] = self.sample_probs

        return result


def train_agent_worker(args):
    """Worker function for multiprocessing."""
    agent, timesteps = args
    try:
        return agent.train(timesteps)
    except Exception as e:
        import traceback
        print(f"Agent {agent.agent_id} failed: {e}")
        print(traceback.format_exc())
        return None


def sample_hyperparameters():
    """Sample random hyperparameters for PBT initialization."""
    # Fixed architecture [64, 32] (simplified for better learning)
    FIXED_NET_ARCH = [64, 32]

    return {
        'learning_rate': np.random.uniform(1e-5, 1e-4),
        'gamma': np.random.uniform(0.95, 0.99),
        'gae_lambda': np.random.uniform(0.85, 0.95),
        'ent_coef': np.random.uniform(0.001, 0.05),  # Lower entropy for more decisive actions
        'clip_range': np.random.uniform(0.15, 0.3),
        'net_arch': FIXED_NET_ARCH,  # Fixed architecture
    }


def perturb_hyperparameters(hyperparams):
    """Perturb hyperparameters for PBT exploration."""
    perturbed = hyperparams.copy()

    # Randomly perturb each parameter
    for key in hyperparams:
        if key == 'net_arch':
            # Keep architecture fixed (NAS disabled for now)
            pass
        else:
            # For continuous parameters, perturb by ±20%
            if np.random.rand() < 0.3:  # 30% chance to perturb each param
                factor = np.random.choice([0.8, 1.2])  # Shrink or grow by 20%
                perturbed[key] = hyperparams[key] * factor

                # Clip to valid ranges
                if key == 'learning_rate':
                    perturbed[key] = np.clip(perturbed[key], 1e-5, 1e-3)
                elif key == 'gamma':
                    perturbed[key] = np.clip(perturbed[key], 0.9, 0.99)
                elif key == 'gae_lambda':
                    perturbed[key] = np.clip(perturbed[key], 0.8, 0.98)
                elif key == 'ent_coef':
                    perturbed[key] = np.clip(perturbed[key], 0.001, 0.05)
                elif key == 'clip_range':
                    perturbed[key] = np.clip(perturbed[key], 0.1, 0.4)

    return perturbed


def exploit_and_explore(agents, results):
    """PBT exploit-and-explore: copy best agents and perturb."""
    # Sort agents by P&L performance
    sorted_results = sorted(results, key=lambda x: x['pnl_pct'], reverse=True)

    print(f"   Top 3 Agents:")
    for i, result in enumerate(sorted_results[:3]):
        print(f"      #{i+1} Agent {result['agent_id']}: P&L={result['pnl_pct']:+.2f}%, WR={result['win_rate']:.1f}%, Trades={result['trades']}")

    # Bottom 25% copies from top 25%
    n_agents = len(agents)
    n_exploit = max(1, n_agents // 4)

    exploited = 0
    for i in range(n_exploit):
        bottom_agent_id = sorted_results[-(i+1)]['agent_id']
        top_agent_id = sorted_results[i]['agent_id']

        if bottom_agent_id != top_agent_id:
            # Copy model weights and hyperparameters
            bottom_model_path = os.path.join(agents[bottom_agent_id].save_dir,
                                            f'agent_{bottom_agent_id}_model.zip')
            top_model_path = os.path.join(agents[top_agent_id].save_dir,
                                         f'agent_{top_agent_id}_model.zip')
            bottom_hyperparam_path = os.path.join(agents[bottom_agent_id].save_dir,
                                                 f'agent_{bottom_agent_id}_hyperparams.json')
            top_hyperparam_path = os.path.join(agents[top_agent_id].save_dir,
                                              f'agent_{top_agent_id}_hyperparams.json')

            if os.path.exists(top_model_path):
                import shutil
                shutil.copy(top_model_path, bottom_model_path)

                # Also copy hyperparameters to keep them in sync with model
                if os.path.exists(top_hyperparam_path):
                    shutil.copy(top_hyperparam_path, bottom_hyperparam_path)

                # Perturb hyperparameters (after copying to ensure consistency)
                agents[bottom_agent_id].hyperparams = perturb_hyperparameters(
                    agents[top_agent_id].hyperparams
                )

                # If architecture changed after perturbation, delete the model file
                # so it trains from scratch with the new architecture
                if agents[bottom_agent_id].hyperparams['net_arch'] != agents[top_agent_id].hyperparams['net_arch']:
                    if os.path.exists(bottom_model_path):
                        os.remove(bottom_model_path)

                exploited += 1

    if exploited > 0:
        print(f"   ⚡ Exploited: {exploited} agents copied from top performers")

    return agents


def main():
    parser = argparse.ArgumentParser(description='PBT Training for Simple Mode')
    parser.add_argument('--population', type=int, default=8, help='Population size')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total timesteps')
    parser.add_argument('--perturbation-interval', type=int, default=10000,
                       help='Timesteps between PBT perturbations')
    parser.add_argument('--train-data', type=str, default='data/rl_train.csv')
    parser.add_argument('--eval-data', type=str, default='data/rl_test.csv')
    parser.add_argument('--price-history', type=str, default='data/price_history')
    parser.add_argument('--feature-scaler', type=str, default='trained_models/rl/feature_scaler.pkl')
    parser.add_argument('--save-dir', type=str, default='models/simple_mode_pbt')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: population size)')
    args = parser.parse_args()

    print("="*80)
    print("POPULATION-BASED TRAINING - SIMPLE MODE (Fixed Architecture)")
    print("="*80)
    print(f"Architecture: 1 opportunity + 1 execution (36 dims: 14 portfolio+execution + 22 opportunity, 3 actions)")
    print(f"Network: [64, 32, 16] (fixed)")
    print(f"Population: {args.population} agents")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Perturbation interval: {args.perturbation_interval:,}")
    print()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, f"pbt_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Initialize population
    print("🔧 Initializing population...")
    agents = []
    for i in range(args.population):
        hyperparams = sample_hyperparameters()
        agent = SimpleAgent(
            agent_id=i,
            hyperparams=hyperparams,
            train_data=args.train_data,
            eval_data=args.eval_data,
            price_history=args.price_history,
            feature_scaler=args.feature_scaler,
            save_dir=run_dir
        )
        agents.append(agent)
        net_arch_str = '-'.join(map(str, hyperparams['net_arch']))
        print(f"  Agent {i}: net_arch={net_arch_str}, LR={hyperparams['learning_rate']:.2e}, "
              f"gamma={hyperparams['gamma']:.3f}, ent={hyperparams['ent_coef']:.3f}")

    # PBT training loop
    n_iterations = args.timesteps // args.perturbation_interval
    n_workers = args.workers or args.population

    print(f"\n🚀 Starting PBT training ({n_iterations} iterations)...")
    print(f"   Workers: {n_workers}")
    print()

    # Track global best across all iterations
    global_best = None
    all_iteration_results = []

    for iteration in range(n_iterations):
        print(f"\n📈 Iteration {iteration+1}/{n_iterations} - Training {args.population} agents ({args.perturbation_interval:,} steps each)...")

        # Train all agents in parallel
        with mp.Pool(n_workers) as pool:
            results = pool.map(train_agent_worker, [(agent, args.perturbation_interval) for agent in agents])

        # Filter out failed agents
        results = [r for r in results if r is not None]

        if len(results) < args.population:
            print(f"  ⚠️  Warning: {args.population - len(results)} agents failed")

        # Track iteration results
        all_iteration_results.extend(results)

        # Update global best if this iteration has better model
        iteration_best = max(results, key=lambda x: x['pnl_pct'])
        if global_best is None or iteration_best['pnl_pct'] > global_best['pnl_pct']:
            global_best = iteration_best.copy()
            global_best['iteration'] = iteration + 1

            # Save global best model immediately
            import shutil
            src_path = os.path.join(run_dir, f"agent_{iteration_best['agent_id']}_model.zip")
            dst_path = os.path.join(run_dir, 'best_global_model.zip')
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                net_arch_str = '-'.join(map(str, iteration_best['hyperparams']['net_arch']))
                print(f"   🏆 NEW GLOBAL BEST! Agent #{iteration_best['agent_id']} (net_arch={net_arch_str}) - P&L={iteration_best['pnl_pct']:+.2f}%, WR={iteration_best['win_rate']:.1f}%, Trades={iteration_best['trades']}")

        # Show sample probabilities (every iteration to monitor learning)
        if any('sample_probs' in r for r in results):
            print(f"\n   📊 Sample Action Probabilities:")
            for result in sorted(results, key=lambda x: x['pnl_pct'], reverse=True)[:3]:
                if 'sample_probs' in result:
                    probs = result['sample_probs']
                    print(f"      Agent {result['agent_id']}: ENTER={probs['enter']*100:.1f}% | EXIT={probs['exit']*100:.1f}% | HOLD={probs['hold']*100:.1f}%")

        # Exploit and explore
        if iteration < n_iterations - 1:  # Don't perturb on last iteration
            agents = exploit_and_explore(agents, results)

    # Final evaluation and summary
    print("="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)

    # Find best agent from final iteration
    final_results = sorted(results, key=lambda x: x['pnl_pct'], reverse=True)
    final_best = final_results[0]

    print(f"\n🏆 Global Best Agent (across all {n_iterations} iterations):")
    print(f"   Agent #{global_best['agent_id']} from iteration {global_best['iteration']}")
    print(f"   P&L: {global_best['pnl_pct']:+.2f}%")
    print(f"   Win Rate: {global_best['win_rate']:.1f}%")
    print(f"   Trades: {global_best['trades']}")
    print(f"\n   Hyperparameters:")
    for key, value in global_best['hyperparams'].items():
        if key == 'net_arch':
            net_arch_str = '-'.join(map(str, value))
            print(f"     {key}: {net_arch_str}")
        elif key == 'learning_rate':
            print(f"     {key}: {value:.2e}")
        else:
            print(f"     {key}: {value:.4f}")

    print(f"\n📊 Final Iteration Best (for comparison):")
    print(f"   Agent #{final_best['agent_id']}")
    print(f"   P&L: {final_best['pnl_pct']:+.2f}%")
    print(f"   Win Rate: {final_best['win_rate']:.1f}%")
    print(f"   Trades: {final_best['trades']}")

    # Save results summary
    summary_path = os.path.join(run_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'global_best': global_best,
            'final_iteration_best': final_best,
            'all_final_agents': final_results,
            'config': vars(args)
        }, f, indent=2)

    global_net_arch_str = '-'.join(map(str, global_best['hyperparams']['net_arch']))
    print(f"\n📁 Models saved to: {run_dir}")
    print(f"   🌟 Global best model: best_global_model.zip")
    print(f"      Architecture: {global_net_arch_str}, P&L={global_best['pnl_pct']:+.2f}%, WR={global_best['win_rate']:.1f}%")
    print(f"   Final best model: agent_{final_best['agent_id']}_model.zip (P&L={final_best['pnl_pct']:+.2f}%)")
    print(f"   Summary: training_summary.json")
    print()


if __name__ == '__main__':
    main()
