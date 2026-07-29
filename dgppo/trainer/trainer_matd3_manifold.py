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
from .buffer_flashbax import FlashbaxReplayBuffer
from .utils import test_rollout_subgoal_manifold
from ..env import MultiAgentEnv
from ..algo.informarl_matd3 import InforMARL_MATD3
from ..trainer.utils import rollout_hierarchical_manifold as rollout_fn


class TrainerMATD3Manifold:
    """Trainer for MATD3 with hierarchical RL, manifold safety, and replay buffer

    Key differences from TrainerMATD3:
    - Uses rollout_hierarchical_manifold (ATACOM safety)
    - Initializes and tracks slack variables s_all
    - Uses manifold-specific test rollout
    """

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: InforMARL_MATD3,
            gamma: float,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
            save_log: bool = True,
            # MATD3-specific parameters
            buffer_size: int = 100000,
            min_buffer_size: int = 1000,
            updates_per_step: int = 1,
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.gamma = gamma
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed

        if TrainerMATD3Manifold._check_params(params):
            self.params = params

        # Make directory for models
        if save_log:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            self.model_dir = os.path.join(log_dir, 'models')
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

        wandb.login()
        wandb.init(name=params['run_name'], project='dgppo', group=env.__class__.__name__, dir=self.log_dir)

        self.save_log = save_log

        self.steps = params['training_steps']
        self.eval_interval = params['eval_interval']
        self.eval_epi = params['eval_epi']
        self.save_interval = params['save_interval']

        # MATD3-specific
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.updates_per_step = updates_per_step

        # Initialize Flashbax replay buffer (JAX-native, GPU-only)
        self.replay_buffer = FlashbaxReplayBuffer(
            max_length=buffer_size,
            min_length=min_buffer_size,
            sample_batch_size=algo.batch_size,
            add_batch_size=n_env_train,  # Number of envs per rollout
        )

        self.update_steps = 0
        self.key = jax.random.PRNGKey(seed)

        # Set up rollout function for data collection (manifold version)
        subgoal_interval = self.algo.subgoal_interval
        init_rnn_state = self.algo.init_actor_rnn_state

        def rollout_fn_single_(cur_params, cur_key, reach_thresh):
            return rollout_fn(
                self.env,
                ft.partial(self.algo.step, params=cur_params, add_noise=True),  # Exploration noise
                init_rnn_state,
                cur_key,
                subgoal_interval=subgoal_interval,
                reach_thresh=reach_thresh,
            )

        def rollout_fn_(cur_params, cur_keys, reach_thresh):
            return jax.vmap(ft.partial(rollout_fn_single_, cur_params, reach_thresh=reach_thresh))(cur_keys)

        # Re-enable JIT for rollout (10-30x speedup!)
        # Use smaller n_env_train (64-128) to avoid OOM
        # JIT is critical for performance - without it, training is extremely slow
        self.rollout_fn = jax.jit(rollout_fn_)

        # Reach threshold schedule
        self.reach_thresh_schedule_fn = getattr(algo, 'reach_thresh_schedule_fn',
                                                lambda x: float(env.params.get("dist2goal", 0.01)))

    @staticmethod
    def _check_params(params: dict) -> bool:
        assert 'run_name' in params, 'run_name not found in params'
        assert 'training_steps' in params, 'training_steps not found in params'
        assert 'eval_interval' in params, 'eval_interval not found in params'
        assert params['eval_interval'] > 0, 'eval_interval must be positive'
        assert 'eval_epi' in params, 'eval_epi not found in params'
        assert params['eval_epi'] >= 1, 'eval_epi must be greater than or equal to 1'
        assert 'save_interval' in params, 'save_interval not found in params'
        assert params['save_interval'] > 0, 'save_interval must be positive'
        return True

    def collect_rollouts(self, step: int) -> Rollout:
        """Collect rollouts using current policy with exploration noise (manifold version)"""
        reach_thresh = self.reach_thresh_schedule_fn(step)
        key_x0, self.key = jax.random.split(self.key)
        key_x0 = jax.random.split(key_x0, self.n_env_train)

        rollouts = self.rollout_fn(self.algo.params, key_x0, reach_thresh)
        return rollouts

    def train(self):
        # Record start time
        start_time = time()

        # Set up test function (manifold version)
        init_rnn_state = self.algo.init_actor_rnn_state
        subgoal_interval = self.algo.subgoal_interval

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

        # Start training
        test_key = jr.PRNGKey(self.seed)
        assert self.n_env_test <= 1_000, 'n_env_test must be less than or equal to 1_000'
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]

        pbar = tqdm(total=self.steps, ncols=100, desc="Training MATD3-Manifold")

        for step in range(0, self.steps + 1):
            # ===== Evaluation =====
            if step % self.eval_interval == 0:
                eval_info = {}
                reach_thresh = self.reach_thresh_schedule_fn(step)

                test_rollouts: Rollout = test_fn(self.algo.params, test_keys, reach_thresh)

                # Environment reward statistics
                total_reward = test_rollouts.rewards.sum(axis=-1)
                reward_mean = np.mean(total_reward)
                reward_final = np.mean(test_rollouts.rewards[:, -1])

                # Sparse rewards statistics
                total_sparse_reward = test_rollouts.sparse_rewards.sum(axis=-1)
                sparse_reward_mean = np.mean(total_sparse_reward)
                sparse_reward_final = np.mean(test_rollouts.sparse_rewards[:, -1])

                # Safety statistics
                cost = jnp.maximum(test_rollouts.costs, 0.0).max(axis=-1).max(axis=-1).sum(axis=-1).mean()
                unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1).max(axis=-2) >= 1e-6)

                # Final distance to goal
                final_dist2goal = np.mean(test_rollouts.dist2goal[:, -1])

                eval_info = {
                    "eval/reward": reward_mean,
                    "eval/reward_final": reward_final,
                    "eval/sparse_reward": sparse_reward_mean,
                    "eval/sparse_reward_final": sparse_reward_final,
                    "eval/cost": cost,
                    "eval/unsafe_frac": unsafe_frac,
                    "eval/final_dist2goal": final_dist2goal,
                    "eval/reach_thresh": reach_thresh,
                    "buffer/size": self.replay_buffer.length,
                }

                time_since_start = time() - start_time
                eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, '
                                f'reward: {reward_mean:9.4f}, sparse: {sparse_reward_mean:9.4f}, '
                                f'cost: {cost:8.4f}, unsafe: {unsafe_frac:6.2f}, '
                                f'dist: {final_dist2goal:6.4f}, buffer: {self.replay_buffer.length}')
                tqdm.write(eval_verbose)

                wandb.log(eval_info, step=self.update_steps)

            # ===== Save model =====
            if self.save_log and step % self.save_interval == 0:
                self.algo.save(os.path.join(self.model_dir), step)

            # ===== Collect rollouts =====
            t0 = time()
            rollouts = self.collect_rollouts(step)
            jax.block_until_ready(rollouts)
            t_rollout = time() - t0

            # ===== Add to replay buffer (Flashbax - GPU-native) =====
            t0 = time()
            self.replay_buffer.add(rollouts)
            t_buffer_add = time() - t0

            # ===== Update algorithm (off-policy) =====
            t_update_total = 0.0
            if self.replay_buffer.can_sample():
                t0 = time()
                # Perform multiple gradient updates per environment step
                for _ in range(self.updates_per_step):
                    # Sample minibatch from replay buffer (requires PRNG key)
                    sample_key, self.key = jr.split(self.key)
                    batch = self.replay_buffer.sample(sample_key)
                    # Update networks
                    update_info = self.algo.update(batch, step)

                    # Log update info (only for the last update)
                    if _ == self.updates_per_step - 1:
                        wandb.log(update_info, step=self.update_steps)

                # Block once after all updates to measure total time
                jax.block_until_ready(update_info)
                t_update_total = time() - t0

                self.update_steps += 1

            # Log timing every 10 steps
            if step % 10 == 0:
                if self.replay_buffer.can_sample():
                    tqdm.write(f"[TIMING] Rollout: {t_rollout:.3f}s | Buffer add: {t_buffer_add:.3f}s | "
                              f"Update (×{self.updates_per_step}): {t_update_total:.3f}s | "
                              f"Total: {t_rollout+t_buffer_add+t_update_total:.3f}s")
                else:
                    tqdm.write(f"[INIT] Collecting initial data: {self.replay_buffer.length}/{self.min_buffer_size} | "
                              f"Rollout: {t_rollout:.3f}s | Buffer add: {t_buffer_add:.3f}s")

            pbar.update(1)

        pbar.close()
        wandb.finish()
