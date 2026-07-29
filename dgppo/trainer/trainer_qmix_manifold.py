"""Trainer for QMIX with Manifold Safety Controller

Implements off-policy training for QMIX with:
- Replay buffer for experience storage
- Manifold/ATACOM low-level safety controller
- Hierarchical rollout collection
- ε-greedy exploration with decay
"""

import wandb
import os
import numpy as np
import jax
import jax.random as jr
import functools as ft
import jax.numpy as jnp

from time import time
from tqdm import tqdm

from .data import Rollout
from .buffer import ReplayBuffer
from .utils import test_rollout_subgoal_manifold, rollout_hierarchical_manifold
from ..env import MultiAgentEnv
from ..algo.informarl_qmix import InforMARL_QMIX


class TrainerQMixManifold:
    """Trainer for QMIX with Manifold safety controller

    Differences from on-policy trainer:
    - Uses replay buffer for off-policy learning
    - Multiple gradient steps per environment step
    - ε-greedy exploration with decay
    - Target network updates
    """

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: InforMARL_QMIX,
            gamma: float,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
            # QMIX-specific params
            buffer_size: int = 100000,
            batch_size: int = 32,
            learning_starts: int = 1000,  # Start learning after N steps
            train_freq: int = 1,  # Train every N steps
            gradient_steps: int = 1,  # Gradient steps per train call
            save_log: bool = True
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.gamma = gamma
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed

        if TrainerQMixManifold._check_params(params):
            self.params = params

        # QMIX-specific
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps

        # Make directories
        if save_log:
            os.makedirs(log_dir, exist_ok=True)
            self.model_dir = os.path.join(log_dir, 'models')
            os.makedirs(self.model_dir, exist_ok=True)

        wandb.login()
        wandb.init(
            name=params['run_name'],
            project='dgppo',
            group=env.__class__.__name__,
            dir=self.log_dir
        )

        self.save_log = save_log
        self.steps = params['training_steps']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']

        self.update_steps = 0
        self.total_env_steps = 0
        self.key = jax.random.PRNGKey(seed)

        # Initialize replay buffer
        self.buffer = ReplayBuffer(max_size=buffer_size)

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_steps' in params, 'training_steps not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be >= 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        return True

    def collect_rollouts(
            self,
            n_rollouts: int = 1,
            reach_thresh: float = 0.01
    ) -> Rollout:
        """Collect rollouts with Manifold safety and add to replay buffer

        Args:
            n_rollouts: Number of parallel rollouts to collect
            reach_thresh: Distance threshold for subgoal reaching

        Returns:
            rollouts: Collected rollout data
        """
        init_rnn_state = self.algo.init_rnn_state
        subgoal_interval = self.algo.subgoal_interval

        def collect_single(key):
            return rollout_hierarchical_manifold(
                self.env,
                ft.partial(self.algo.step, params=self.algo.params, add_noise=True),
                init_rnn_state,
                key,
                subgoal_interval=subgoal_interval,
                reach_thresh=reach_thresh,
            )

        # Generate keys and collect
        collect_key, self.key = jr.split(self.key)
        collect_keys = jr.split(collect_key, n_rollouts)

        rollouts = jax.vmap(collect_single)(collect_keys)

        # Add to buffer
        self.buffer.add(rollouts)

        # Update total environment steps
        self.total_env_steps += n_rollouts * rollouts.dones.shape[1] * subgoal_interval

        return rollouts

    def train(self):
        """Main training loop for QMIX"""
        start_time = time()

        init_rnn_state = self.algo.init_rnn_state
        subgoal_interval = self.algo.subgoal_interval

        # === Test function (greedy evaluation) ===
        def test_fn_single(params, key, reach_thresh):
            act_fn = ft.partial(self.algo.act, params=params)
            return test_rollout_subgoal_manifold(
                self.env_test,
                act_fn,
                init_rnn_state,
                key,
                subgoal_interval=subgoal_interval,
                filter_high_level=True,
                reach_thresh=reach_thresh,
            )

        def test_fn(params, keys, reach_thresh):
            return jax.vmap(ft.partial(test_fn_single, params, reach_thresh=reach_thresh))(keys)

        test_fn = jax.jit(test_fn)

        # Setup test keys
        test_key = jr.PRNGKey(self.seed + 1000)
        assert self.n_env_test <= 1_000, 'n_env_test must be <= 1_000'
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]

        reach_thresh_schedule_fn = getattr(
            self.algo,
            'reach_thresh_schedule_fn',
            lambda x: 0.01
        )

        pbar = tqdm(total=self.steps, ncols=100)

        for step in range(0, self.steps + 1):
            # === Evaluation ===
            if step % self.eval_interval == 0:
                eval_info = {}
                reach_thresh = reach_thresh_schedule_fn(step)

                test_rollouts: Rollout = test_fn(
                    self.algo.train_state.params,
                    test_keys,
                    reach_thresh
                )

                # Environment reward statistics
                total_reward = test_rollouts.rewards.sum(axis=-1)
                reward_mean = np.mean(total_reward)
                reward_final = np.mean(test_rollouts.rewards[:, -1])

                # Sparse rewards statistics
                total_sparse_reward = test_rollouts.sparse_rewards.sum(axis=-1)
                sparse_reward_mean = np.mean(total_sparse_reward)
                sparse_reward_final = np.mean(test_rollouts.sparse_rewards[:, -1])

                # Safety metrics
                cost = jnp.maximum(test_rollouts.costs, 0.0).max(axis=-1).max(axis=-1).sum(axis=-1).mean()
                unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1).max(axis=-2) >= 1e-6)

                final_dist2goal = np.mean(test_rollouts.dist2goal[:, -1])

                eval_info.update({
                    "eval/reward": reward_mean,
                    "eval/reward_final": reward_final,
                    "eval/sparse_reward": sparse_reward_mean,
                    "eval/sparse_reward_final": sparse_reward_final,
                    "eval/cost": cost,
                    "eval/unsafe_frac": unsafe_frac,
                    "eval/final_dist2goal": final_dist2goal,
                    "eval/reach_thresh": reach_thresh,
                    "buffer/size": self.buffer.length,
                    "train/total_env_steps": self.total_env_steps,
                })

                time_since_start = time() - start_time

                eval_verbose = (
                    f'step: {step:3}, time: {time_since_start:5.0f}s, '
                    f'reward: {reward_mean:9.4f}, sparse: {sparse_reward_mean:9.4f}, '
                    f'cost: {cost:8.4f}, unsafe: {unsafe_frac:6.2f}, '
                    f'dist: {final_dist2goal:6.4f}, eps: {self.algo.epsilon:.3f}, '
                    f'buf: {self.buffer.length}'
                )
                tqdm.write(eval_verbose)

                wandb.log(eval_info, step=self.update_steps)

            # === Save model ===
            if self.save_log and step % self.save_interval == 0:
                self.algo.save(self.model_dir, step)

            # === Collect rollouts ===
            reach_thresh = reach_thresh_schedule_fn(step)
            rollouts = self.collect_rollouts(
                n_rollouts=self.n_env_train,
                reach_thresh=reach_thresh
            )

            # === Update algorithm (if enough data) ===
            if self.buffer.length >= self.learning_starts and step % self.train_freq == 0:
                update_info = {}

                for grad_step in range(self.gradient_steps):
                    # Sample batch from buffer
                    sample_key, self.key = jr.split(self.key)
                    batch = self.buffer.sample(self.batch_size, sample_key)

                    # Update Q-network
                    self.algo.train_state, step_info = self.algo.update(
                        self.algo.train_state,
                        batch
                    )

                    # Accumulate metrics
                    for k, v in step_info.items():
                        if k not in update_info:
                            update_info[k] = []
                        update_info[k].append(v)

                # Average metrics over gradient steps
                update_info = {k: np.mean(v) for k, v in update_info.items()}

                # Update target network
                if self.update_steps % self.algo.target_update_interval == 0:
                    self.algo.update_target_network()

                # Decay epsilon
                self.algo.decay_epsilon()

                # Log training metrics
                wandb.log(update_info, step=self.update_steps)
                self.update_steps += 1

            pbar.update(1)

        pbar.close()
        wandb.finish()
