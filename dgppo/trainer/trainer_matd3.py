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
from .utils import test_rollout_subgoal
from ..env import MultiAgentEnv
from ..algo.informarl_matd3 import InforMARL_MATD3
from ..trainer.utils import rollout_hierarchical as rollout_fn


class TrainerMATD3:
    """Trainer for MATD3 with hierarchical RL and replay buffer

    Key differences from standard trainer:
    - Uses ReplayBuffer for off-policy learning
    - Collects transitions and stores in buffer
    - Samples minibatches from buffer for updates
    - Updates multiple times per collection step
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
            buffer_size: int = 100000,  # Replay buffer size
            min_buffer_size: int = 1000,  # Start training after this many transitions
            updates_per_step: int = 1,  # Number of gradient updates per environment step
    ):
        self.env = env
        self.env_test = env_test
        self.algo = algo
        self.gamma = gamma
        self.n_env_train = n_env_train
        self.n_env_test = n_env_test
        self.log_dir = log_dir
        self.seed = seed

        if TrainerMATD3._check_params(params):
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

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(size=buffer_size)

        self.update_steps = 0
        self.key = jax.random.PRNGKey(seed)

        # Set up rollout function for data collection
        subgoal_interval = self.algo.subgoal_interval
        init_rnn_state = self.algo.init_actor_rnn_state

        def rollout_fn_single_no_cbf_(cur_params, cur_key, reach_thresh):
            return rollout_fn(
                self.env,
                ft.partial(self.algo.step, params=cur_params, add_noise=True),  # Exploration noise
                init_rnn_state,
                cur_key,
                subgoal_interval=subgoal_interval,
                reach_thresh=reach_thresh,
                use_cbf=False
            )

        def rollout_fn_no_cbf_(cur_params, cur_keys, reach_thresh):
            return jax.vmap(ft.partial(rollout_fn_single_no_cbf_, cur_params, reach_thresh=reach_thresh))(cur_keys)

        def rollout_fn_single_with_cbf_(cur_params, cur_key, reach_thresh):
            return rollout_fn(
                self.env,
                ft.partial(self.algo.step, params=cur_params, add_noise=True),
                init_rnn_state,
                cur_key,
                subgoal_interval=subgoal_interval,
                reach_thresh=reach_thresh,
                use_cbf=True
            )

        def rollout_fn_with_cbf_(cur_params, cur_keys, reach_thresh):
            return jax.vmap(ft.partial(rollout_fn_single_with_cbf_, cur_params, reach_thresh=reach_thresh))(cur_keys)

        self.rollout_fn_no_cbf = jax.jit(rollout_fn_no_cbf_)
        self.rollout_fn_with_cbf = jax.jit(rollout_fn_with_cbf_)

        # CBF start step
        self.cbf_start_step = 80

        # Reach threshold schedule
        self.reach_thresh_schedule_fn = getattr(algo, 'reach_thresh_schedule_fn', lambda x: 0.01)

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
        """Collect rollouts using current policy with exploration noise"""
        reach_thresh = self.reach_thresh_schedule_fn(step)
        key_x0, self.key = jax.random.split(self.key)
        key_x0 = jax.random.split(key_x0, self.n_env_train)

        # Choose rollout function based on CBF activation
        if step < self.cbf_start_step:
            rollouts = self.rollout_fn_no_cbf(self.algo.params, key_x0, reach_thresh)
        else:
            rollouts = self.rollout_fn_with_cbf(self.algo.params, key_x0, reach_thresh)

        return rollouts

    def train(self):
        # Record start time
        start_time = time()

        # Set up test function
        init_rnn_state = self.algo.init_actor_rnn_state
        subgoal_interval = self.algo.subgoal_interval

        def test_fn_single(params, key, reach_thresh, use_cbf):
            act_fn = ft.partial(self.algo.act, params=params)
            return test_rollout_subgoal(
                self.env_test,
                act_fn,
                init_rnn_state,
                key,
                subgoal_interval=subgoal_interval,
                filter_high_level=True,
                reach_thresh=reach_thresh,
                use_cbf=use_cbf,
            )

        def test_fn_no_cbf(params, keys, reach_thresh):
            return jax.vmap(ft.partial(test_fn_single, params, reach_thresh=reach_thresh, use_cbf=False))(keys)

        def test_fn_with_cbf(params, keys, reach_thresh):
            return jax.vmap(ft.partial(test_fn_single, params, reach_thresh=reach_thresh, use_cbf=True))(keys)

        test_fn_no_cbf = jax.jit(test_fn_no_cbf)
        test_fn_with_cbf = jax.jit(test_fn_with_cbf)

        # Start training
        test_key = jr.PRNGKey(self.seed)
        assert self.n_env_test <= 1_000, 'n_env_test must be less than or equal to 1_000'
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]

        pbar = tqdm(total=self.steps, ncols=100, desc="Training MATD3")

        for step in range(0, self.steps + 1):
            # ===== Evaluation =====
            if step % self.eval_interval == 0:
                eval_info = {}
                reach_thresh = self.reach_thresh_schedule_fn(step)
                use_cbf_now = step >= self.cbf_start_step

                if use_cbf_now:
                    test_rollouts: Rollout = test_fn_with_cbf(self.algo.params, test_keys, reach_thresh)
                else:
                    test_rollouts: Rollout = test_fn_no_cbf(self.algo.params, test_keys, reach_thresh)

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
            rollouts = self.collect_rollouts(step)

            # ===== Add to replay buffer =====
            self.replay_buffer.append(rollouts)

            # ===== Update algorithm (off-policy) =====
            if self.replay_buffer.length >= self.min_buffer_size:
                # Perform multiple gradient updates per environment step
                for _ in range(self.updates_per_step):
                    # Sample minibatch from replay buffer
                    batch = self.replay_buffer.sample(self.algo.batch_size)

                    # Update networks
                    update_info = self.algo.update(batch, step)

                    # Log update info (only for the last update)
                    if _ == self.updates_per_step - 1:
                        wandb.log(update_info, step=self.update_steps)

                self.update_steps += 1
            else:
                # Still collecting initial data
                tqdm.write(f"Collecting initial data: {self.replay_buffer.length}/{self.min_buffer_size}")

            pbar.update(1)

        pbar.close()
        wandb.finish()
