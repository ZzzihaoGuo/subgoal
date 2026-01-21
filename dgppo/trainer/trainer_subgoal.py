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
from .utils import test_rollout
from .utils import test_rollout_subgoal
from ..env import MultiAgentEnv
from ..algo.base import Algorithm


class Trainer:

    def __init__(
            self,
            env: MultiAgentEnv,
            env_test: MultiAgentEnv,
            algo: Algorithm,
            gamma: float,
            n_env_train: int,
            n_env_test: int,
            log_dir: str,
            seed: int,
            params: dict,
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

        if Trainer._check_params(params):
            self.params = params

        # make dir for the models
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

        self.update_steps = 0
        self.key = jax.random.PRNGKey(seed)

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

    def train(self):
        # record start time
        start_time = time()

        # preprocess the rollout function
        init_rnn_state = self.algo.init_rnn_state

        # get subgoal_interval from algo config if available
        subgoal_interval = getattr(self.algo, 'subgoal_interval', 40)

        def test_fn_single(params, key, reach_thresh, use_cbf):
            act_fn = ft.partial(self.algo.act, params=params)
            return test_rollout_subgoal(
                self.env_test,
                act_fn,
                init_rnn_state,
                key,
                subgoal_interval=subgoal_interval,
                filter_high_level=True,  # 训练统计时只用高层决策点数据
                reach_thresh=reach_thresh,
                use_cbf=use_cbf,  # 与训练时保持一致
            )

        # 创建两个版本的 test_fn，避免 use_cbf 被 trace
        def test_fn_no_cbf(params, keys, reach_thresh):
            return jax.vmap(ft.partial(test_fn_single, params, reach_thresh=reach_thresh, use_cbf=False))(keys)

        def test_fn_with_cbf(params, keys, reach_thresh):
            return jax.vmap(ft.partial(test_fn_single, params, reach_thresh=reach_thresh, use_cbf=True))(keys)

        test_fn_no_cbf = jax.jit(test_fn_no_cbf)
        test_fn_with_cbf = jax.jit(test_fn_with_cbf)

        # start training
        test_key = jr.PRNGKey(self.seed)
        assert self.n_env_test <= 1_000, 'n_env_test must be less than or equal to 1_000'
        test_keys = jr.split(test_key, 1_000)[:self.n_env_test]

        pbar = tqdm(total=self.steps, ncols=80)
        for step in range(0, self.steps + 1):
            # evaluate the algorithm
            if step % self.eval_interval == 0:
                eval_info = {}
                # 获取当前的 reach_thresh
                reach_thresh = getattr(self.algo, 'reach_thresh_schedule_fn', lambda x: 0.1)(step)
                # 根据 step 决定是否使用 CBF（与训练时保持一致）
                cbf_start_step = getattr(self.algo, 'cbf_start_step', 0)
                use_cbf_now = step >= cbf_start_step
                if step % 1000 == 0:  # 每1000步打印一次
                    print(f"[DEBUG] step={step}, cbf_start_step={cbf_start_step}, use_cbf={use_cbf_now}")
                if use_cbf_now:
                    test_rollouts: Rollout = test_fn_with_cbf(self.algo.params, test_keys, reach_thresh)
                else:
                    test_rollouts: Rollout = test_fn_no_cbf(self.algo.params, test_keys, reach_thresh)

                # 环境reward统计
                total_reward = test_rollouts.rewards.sum(axis=-1)
                reward_min, reward_max = total_reward.min(), total_reward.max()
                reward_mean = np.mean(total_reward)
                reward_final = np.mean(test_rollouts.rewards[:, -1])

                # sparse_rewards统计
                total_sparse_reward = test_rollouts.sparse_rewards.sum(axis=-1)
                sparse_reward_mean = np.mean(total_sparse_reward)
                sparse_reward_final = np.mean(test_rollouts.sparse_rewards[:, -1])

                cost = jnp.maximum(test_rollouts.costs, 0.0).max(axis=-1).max(axis=-1).sum(axis=-1).mean()
                unsafe_frac = np.mean(test_rollouts.costs.max(axis=-1).max(axis=-2) >= 1e-6)

                # 计算最终距离：最后一步每个goal到最近agent的平均距离
                final_dist2goal = np.mean(test_rollouts.dist2goal[:, -1])

                eval_info = eval_info | {
                    "eval/reward": reward_mean,
                    "eval/reward_final": reward_final,
                    "eval/sparse_reward": sparse_reward_mean,
                    "eval/sparse_reward_final": sparse_reward_final,
                    "eval/cost": cost,
                    "eval/unsafe_frac": unsafe_frac,
                    "eval/final_dist2goal": final_dist2goal,
                    "eval/reach_thresh": reach_thresh,
                }
                time_since_start = time() - start_time

                eval_verbose = (f'step: {step:3}, time: {time_since_start:5.0f}s, '
                                f'reward: {reward_mean:9.4f}, sparse_reward: {sparse_reward_mean:9.4f}, '
                                f'cost: {cost:8.4f}, unsafe_frac: {unsafe_frac:6.2f}, final_dist: {final_dist2goal:6.4f}, '
                                f'reach_thresh: {reach_thresh:.3f}')
                tqdm.write(eval_verbose)

                wandb.log(eval_info, step=self.update_steps)

            # save the model
            if self.save_log and step % self.save_interval == 0:
                self.algo.save(os.path.join(self.model_dir), step)

            # collect rollouts (传入 step 用于动态调整 reach_thresh)
            key_x0, self.key = jax.random.split(self.key)
            key_x0 = jax.random.split(key_x0, self.n_env_train)
            rollouts = self.algo.collect(self.algo.params, key_x0, step=step)

            # update the algorithm
            update_info = self.algo.update(rollouts, step)
            wandb.log(update_info, step=self.update_steps)
            self.update_steps += 1

            pbar.update(1)
