import jax.numpy as jnp
import jax.tree_util as jtu
import jax
import numpy as np
import socket
import matplotlib.pyplot as plt
import os

from typing import Callable, TYPE_CHECKING
from matplotlib.colors import CenteredNorm

from ..utils.typing import PRNGKey, Array
from .data import Rollout


if TYPE_CHECKING:
    from ..env import MultiAgentEnv
else:
    MultiAgentEnv = None


# ============ Sparse Reward 系数配置（train 和 test 共用）============
# 默认值；每个 env 类可定义同名 class attribute 覆盖（例如 LidarTarget.GOAL_REWARD_COEF = 0.2）。
# GOAL_REWARD_COEF / SUBGOAL_BONUS_COEF / SUBGOAL_SHADOW_COEF / DIST_TO_GOAL_COEF 支持 per-env override。
GOAL_REWARD_COEF = 0.0          # goal_reward 系数

SUBGOAL_BONUS_THRESH = 0.02     # subgoal_bonus 判断阈值
SUBGOAL_BONUS_COEF = 0.001       # subgoal_bonus 系数
DIST_TO_GOAL_COEF = 0.1        # dist_agent_to_goal 系数

SUBGOAL_SHADOW_COEF = 1.00     # subgoal_shadow_cost 系数（生成在障碍物阴影区的惩罚）0, 0.01, 0.1, 1  # 可通过 --subgoal-shadow-coef 覆盖
# ===================================================================


def _resolve_coefs(env):
    """读取 sparse reward 系数；env 类可通过同名 class attribute 覆盖默认值。"""
    return (
        getattr(env, 'GOAL_REWARD_COEF', GOAL_REWARD_COEF),
        getattr(env, 'SUBGOAL_BONUS_COEF', SUBGOAL_BONUS_COEF),
        getattr(env, 'SUBGOAL_SHADOW_COEF', SUBGOAL_SHADOW_COEF),
        getattr(env, 'DIST_TO_GOAL_COEF', DIST_TO_GOAL_COEF),
    )


def _compute_dist2goal(env, goal_pos, agent_pos):
    """根据环境类型计算 dist2goal。
    matched: agent_i 对应 goal_i（LidarTarget 等）
    spread: 每个 goal 找最近 agent（LidarSpread 等）
    """
    mode = getattr(env, 'GOAL_ASSIGNMENT', 'spread')
    if mode == 'spread':
        # 每个 goal 找最近的 agent
        return jnp.linalg.norm(
            jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1
        ).min(axis=1)
    elif mode == 'line':
        # 每个 goal 找最近的 agent
        return jnp.linalg.norm(
            jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1
        ).min(axis=1)
    elif mode == 'target':
        # agent_i 对应 goal_i
        return jnp.linalg.norm(goal_pos - agent_pos, axis=-1)
    else:
        raise ValueError(f"Unknown GOAL_ASSIGNMENT: {mode}")


def rollout_hierarchical(
        env: MultiAgentEnv,
        high_level_actor: Callable,  # [GraphsTuple, RNN_States, PRNGKey] -> [Subgoal, LogPi, RNN_States]
        init_rnn_state: Array,
        key: PRNGKey,
        subgoal_interval: int = 40,  # 每40步生成一个subgoal
        reach_thresh: float = 0.1,  # 判断"到达"的阈值，会随训练逐渐减小
        use_cbf: bool = True,  # 是否使用 CBF 安全控制器
) -> Rollout:
    """
    Hierarchical rollout: 高层每40步生成subgoal，低层每步用u_ref跟踪
    
    Parameters
    ----------
    env: MultiAgentEnv
    high_level_actor: 高层策略，输出subgoal位置 (n_agents, 2)
    init_rnn_state: Array
    key: PRNGKey
    subgoal_interval: int, 多少步生成一次subgoal
    
    Returns
    -------
    rollout: Rollout，但只包含高层决策点的数据 (T//subgoal_interval, ...)
    """
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)
    
    # 初始化：第一个subgoal就是最终目标
    init_subgoal = env.get_agent_goals(init_graph)  # (n_agents, 2)

    goal_reward_coef, subgoal_bonus_coef, subgoal_shadow_coef, dist_to_goal_coef = _resolve_coefs(env)

    def body_(data, inp):
        graph, rnn_state, current_subgoal, step_count = data
        key_ = inp
        
        # === 高层决策：每 subgoal_interval 步生成新的subgoal ===
        should_update = (step_count % subgoal_interval == 0)
        real_goal = env.get_agent_goals(graph)

        def update_subgoal(_):
            new_sg, log_p, new_rnn = high_level_actor(graph, rnn_state, key_)
            return new_sg, log_p, new_rnn

        def keep_subgoal(_):
            return current_subgoal, jnp.zeros((env.num_agents,)), rnn_state

        new_subgoal, log_pi, new_rnn_state = jax.lax.cond(
            should_update,
            update_subgoal,
            keep_subgoal,
            operand=None
        )

        # === 强制最后一个 subgoal 为最终目标 ===
        # 计算还剩多少步
        remaining_steps = env.max_episode_steps - step_count
        is_last_subgoal = remaining_steps <= subgoal_interval

        # 如果是最后一个 subgoal 周期，强制使用最终目标
        #====================================================#
        # new_subgoal = jnp.where(
        #     is_last_subgoal,
        #     real_goal,      # 最后一个周期：使用最终目标
        #     new_subgoal     # 其他：使用策略生成的 subgoal
        # )
        #====================================================#

        # === 低层执行：使用 u_ref 跟踪当前 subgoal ===
        # 判断当前subgoal是否为最终目标
        dist_to_real_goal = jnp.linalg.norm(new_subgoal - real_goal, axis=-1)
        # is_final_goal = (dist_to_real_goal < env.params.get("dist2goal", 0.1) * 2).all()

        # 根据 use_cbf 选择控制器（use_cbf 是编译时常量，不会有运行时开销）
        if use_cbf:
            action = env.safe_u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)
        else:
            action = env.u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)

        # 环境交互
        next_graph, reward, cost, done, info = env.step(graph, action)
        
        # === 稀疏奖励 ===
        # 使用 next_graph (action 之后的状态) 来计算 reward
        agent_states = next_graph.type_states(type_idx=0, n_type=env.num_agents)
        agent_pos = agent_states[:, :env.action_dim]

        # 1. 到达最终目标的奖励 (使用动态阈值)
        goal_pos = real_goal[:, :env.action_dim]
        dist2goal = _compute_dist2goal(env, goal_pos, agent_pos)
        goal_reward = jnp.where(dist2goal < reach_thresh, 0.0, -1.0).mean() * goal_reward_coef
        # goal_reward = jnp.where(dist2goal < reach_thresh, 1.0, 0.0).mean() * goal_reward_coef

        # 2. 到达 subgoal 的奖励 (鼓励生成可达的 subgoal)
        dist2subgoal = jnp.linalg.norm(agent_pos - new_subgoal, axis=-1)
        subgoal_bonus = jnp.where(dist2subgoal < SUBGOAL_BONUS_THRESH, 1, 0.0).mean() * subgoal_bonus_coef

        # 3. agent 距离 goal 的 dense reward
        dist_agent_to_goal = -dist2goal.mean() * dist_to_goal_coef

        # 4. subgoal 阴影区惩罚（subgoal在障碍物后方）
        # 使用当前graph（生成subgoal时的状态）来判断
        shadow_cost = env.get_subgoal_shadow_cost(graph, new_subgoal)  # (n_agents,), -1或0
        subgoal_shadow_penalty = shadow_cost.mean() * subgoal_shadow_coef  # 负值惩罚

        sparse_reward = goal_reward + subgoal_bonus + dist_agent_to_goal + subgoal_shadow_penalty
        
        # === 只在高层决策点保存数据 ===
        # 用一个mask标记哪些timestep需要保存
        save_data = should_update
        
        return (next_graph, new_rnn_state, new_subgoal, step_count + 1), (
            graph,
            new_subgoal,  # 保存subgoal而不是action
            rnn_state,
            reward,  # 环境reward
            cost,
            done,
            log_pi,
            next_graph,
            save_data,  # 额外的标记
            sparse_reward,  # 自定义的稀疏reward
        )

    # 执行rollout
    keys = jax.random.split(key, env.max_episode_steps)
    init_data = (init_graph, init_rnn_state, init_subgoal, 0)

    _, outputs = jax.lax.scan(body_, init_data, keys, length=env.max_episode_steps)

    graphs, subgoals, rnn_states, rewards, costs, dones, log_pis, next_graphs, save_mask, sparse_rewards = outputs

    # === 筛选出高层决策点的数据 ===
    # save_mask: (T,) bool array，标记哪些timestep是高层决策点
    # 我们需要reshape成 (T//subgoal_interval, subgoal_interval) 然后取第一列

    # 高层决策点的索引 (subgoal 生成时刻)
    high_level_indices = jnp.arange(0, env.max_episode_steps, subgoal_interval)

    # sparse_reward 应该取每个 interval 结束时的值 (执行完 subgoal 后的结果)
    # 例如: subgoal_interval=130, 则取 steps 129, 259, ... (或最后一步)
    reward_end_indices = jnp.minimum(high_level_indices + subgoal_interval - 1, env.max_episode_steps - 1)

    rollout_data = Rollout(
        graph=jax.tree.map(lambda x: x[high_level_indices], graphs),
        actions=subgoals[high_level_indices],  # 注意这里是subgoal，不是低层action
        rnn_states=jax.tree.map(lambda x: x[high_level_indices], rnn_states),
        rewards=rewards[high_level_indices],
        costs=costs[high_level_indices],
        dones=dones[high_level_indices],
        log_pis=log_pis[high_level_indices],
        next_graph=jax.tree.map(lambda x: x[reward_end_indices], next_graphs),  # 用 interval 结束时的 next_graph
        sparse_rewards=sparse_rewards[reward_end_indices],  # 用 interval 结束时的 sparse_reward
    )

    return rollout_data

def rollout_hierarchical_manifold(
        env: MultiAgentEnv,
        high_level_actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        subgoal_interval: int = 40,
        reach_thresh: float = 0.1,
) -> Rollout:
    """
    Hierarchical rollout with manifold (ATACOM) safety:
    高层每 subgoal_interval 步生成 subgoal，低层用 manifold 修正的 u_ref 跟踪
    carry 中传递松弛变量 s_all 实现 ATACOM 积分
    """
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    init_subgoal = env.get_agent_goals(init_graph)
    init_s_all = env.manifold_init_slack(init_graph)

    goal_reward_coef, subgoal_bonus_coef, subgoal_shadow_coef, dist_to_goal_coef = _resolve_coefs(env)

    def body_(data, inp):
        graph, rnn_state, current_subgoal, step_count, s_all = data
        key_ = inp

        # === 高层决策 ===
        should_update = (step_count % subgoal_interval == 0)
        real_goal = env.get_agent_goals(graph)

        def update_subgoal(_):
            new_sg, log_p, new_rnn = high_level_actor(graph, rnn_state, key_)
            return new_sg, log_p, new_rnn

        def keep_subgoal(_):
            return current_subgoal, jnp.zeros((env.num_agents,)), rnn_state

        new_subgoal, log_pi, new_rnn_state = jax.lax.cond(
            should_update, update_subgoal, keep_subgoal, operand=None
        )

        remaining_steps = env.max_episode_steps - step_count
        is_last_subgoal = remaining_steps <= subgoal_interval

        # === 低层: LQR → manifold 安全修正 (带松弛变量积分) ===
        nominal_action = env.u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)
        action, _, s_new, _ = env.get_manifold_action(graph, u_ref=nominal_action, s_all=s_all)
        action = env.clip_action(action)

        # 环境交互
        next_graph, reward, cost, done, info = env.step(graph, action)

        # === 稀疏奖励 ===
        agent_states = next_graph.type_states(type_idx=0, n_type=env.num_agents)
        agent_pos = agent_states[:, :env.action_dim]
        goal_pos = real_goal[:, :env.action_dim]
        dist2goal = _compute_dist2goal(env, goal_pos, agent_pos)

        goal_reward = jnp.where(dist2goal < reach_thresh, 0.0, -1.0).mean() * goal_reward_coef
        # goal_reward = jnp.where(dist2goal < reach_thresh, 1.0, 0.0).mean() * goal_reward_coef

        dist2subgoal = jnp.linalg.norm(agent_pos - new_subgoal, axis=-1)
        subgoal_bonus = jnp.where(dist2subgoal < SUBGOAL_BONUS_THRESH, 1, 0.0).mean() * subgoal_bonus_coef
        dist_agent_to_goal = -dist2goal.mean() * dist_to_goal_coef
        shadow_cost = env.get_subgoal_shadow_cost(graph, new_subgoal)
        subgoal_shadow_penalty = shadow_cost.mean() * subgoal_shadow_coef
        sparse_reward = goal_reward + subgoal_bonus + dist_agent_to_goal + subgoal_shadow_penalty

        save_data = should_update

        return (next_graph, new_rnn_state, new_subgoal, step_count + 1, s_new), (
            graph, new_subgoal, rnn_state, reward, cost, done, log_pi,
            next_graph, save_data, sparse_reward,
        )

    keys = jax.random.split(key, env.max_episode_steps)
    init_data = (init_graph, init_rnn_state, init_subgoal, 0, init_s_all)

    _, outputs = jax.lax.scan(body_, init_data, keys, length=env.max_episode_steps)

    graphs, subgoals, rnn_states, rewards, costs, dones, log_pis, next_graphs, save_mask, sparse_rewards = outputs

    # === 筛选出高层决策点的数据 ===
    high_level_indices = jnp.arange(0, env.max_episode_steps, subgoal_interval)
    reward_end_indices = jnp.minimum(high_level_indices + subgoal_interval - 1, env.max_episode_steps - 1)

    rollout_data = Rollout(
        graph=jax.tree.map(lambda x: x[high_level_indices], graphs),
        actions=subgoals[high_level_indices],
        rnn_states=jax.tree.map(lambda x: x[high_level_indices], rnn_states),
        rewards=rewards[high_level_indices],
        costs=costs[high_level_indices],
        dones=dones[high_level_indices],
        log_pis=log_pis[high_level_indices],
        next_graph=jax.tree.map(lambda x: x[reward_end_indices], next_graphs),
        sparse_rewards=sparse_rewards[reward_end_indices],
    )

    return rollout_data


def rollout_hierarchical_manifold_vmas(
        env: MultiAgentEnv,
        high_level_actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        subgoal_interval: int = 40,
        reach_thresh: float = 0.1,
) -> Rollout:
    """
    VMAS 版 hierarchical rollout with manifold:
    与 rollout_hierarchical_manifold 结构相同，但 sparse_reward 直接用环境 reward
    """
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    init_subgoal = env.get_agent_goals(init_graph)
    init_s_all = env.manifold_init_slack(init_graph)

    def body_(data, inp):
        graph, rnn_state, current_subgoal, step_count, s_all = data
        key_ = inp

        # === 高层决策 ===
        should_update = (step_count % subgoal_interval == 0)

        def update_subgoal(_):
            new_sg, log_p, new_rnn = high_level_actor(graph, rnn_state, key_)
            return new_sg, log_p, new_rnn

        def keep_subgoal(_):
            return current_subgoal, jnp.zeros((env.num_agents,)), rnn_state

        new_subgoal, log_pi, new_rnn_state = jax.lax.cond(
            should_update, update_subgoal, keep_subgoal, operand=None
        )

        remaining_steps = env.max_episode_steps - step_count
        is_last_subgoal = remaining_steps <= subgoal_interval

        # === 低层: u_ref → manifold 安全修正 ===
        nominal_action = env.u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)
        action, _, s_new, _ = env.get_manifold_action(graph, u_ref=nominal_action, s_all=s_all)
        action = env.clip_action(action)

        # 环境交互
        next_graph, reward, cost, done, info = env.step(graph, action)

        # === sparse_reward: 直接用 VMAS 环境 reward ===
        sparse_reward = reward

        save_data = should_update

        return (next_graph, new_rnn_state, new_subgoal, step_count + 1, s_new), (
            graph, new_subgoal, rnn_state, reward, cost, done, log_pi,
            next_graph, save_data, sparse_reward,
        )

    keys = jax.random.split(key, env.max_episode_steps)
    init_data = (init_graph, init_rnn_state, init_subgoal, 0, init_s_all)

    _, outputs = jax.lax.scan(body_, init_data, keys, length=env.max_episode_steps)

    graphs, subgoals, rnn_states, rewards, costs, dones, log_pis, next_graphs, save_mask, sparse_rewards = outputs

    high_level_indices = jnp.arange(0, env.max_episode_steps, subgoal_interval)
    reward_end_indices = jnp.minimum(high_level_indices + subgoal_interval - 1, env.max_episode_steps - 1)

    rollout_data = Rollout(
        graph=jax.tree.map(lambda x: x[high_level_indices], graphs),
        actions=subgoals[high_level_indices],
        rnn_states=jax.tree.map(lambda x: x[high_level_indices], rnn_states),
        rewards=rewards[high_level_indices],
        costs=costs[high_level_indices],
        dones=dones[high_level_indices],
        log_pis=log_pis[high_level_indices],
        next_graph=jax.tree.map(lambda x: x[reward_end_indices], next_graphs),
        sparse_rewards=sparse_rewards[reward_end_indices],
    )

    return rollout_data


def rollout(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
) -> Rollout:
    """
    Get a rollout from the environment using the actor.

    Parameters
    ----------
    env: MultiAgentEnv
    actor: Callable, [GraphsTuple, Array, RNN_States, PRNGKey] -> [Action, LogPi, RNN_States]
    init_rnn_state: Array
    key: PRNGKey

    Returns
    -------
    data: Rollout
    """
    key_x0, key_z0, key = jax.random.split(key, 3)
    init_graph = env.reset(key_x0)

    def body(data, key_):
        graph, rnn_state = data
        action, log_pi, new_rnn_state = actor(graph, rnn_state, key_)
        next_graph, reward, cost, done, info = env.step(graph, action)

        return ((next_graph, new_rnn_state),
                (graph, action, rnn_state, reward, cost, done, log_pi, next_graph))

    keys = jax.random.split(key, env.max_episode_steps)
    _, (graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs) = (
        jax.lax.scan(body, (init_graph, init_rnn_state), keys, length=env.max_episode_steps))
    rollout_data = Rollout(graphs, actions, rnn_states, rewards, costs, dones, log_pis, next_graphs)
    return rollout_data


def test_rollout(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        stochastic: bool = False
):
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    def body_(data, key_):
        graph, rnn_state = data
        if not stochastic:
            action, rnn_state = actor(graph, rnn_state)
            # action = env.u_ref(graph)
        else:
            action, rnn_state = actor(graph, rnn_state, key_)
        next_graph, reward, cost, done, info = env.step(graph, action)
        return (next_graph, rnn_state), (graph, action, rnn_state, reward, cost, done, None, next_graph)

    keys = jax.random.split(key, env.max_episode_steps)
    _, (graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs) = (
        jax.lax.scan(body_,
                     (init_graph, init_rnn_state),
                     keys,
                     length=env.max_episode_steps))
    rollout_data = Rollout(graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs)
    return rollout_data

def test_rollout_subgoal(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        stochastic: bool = False,
        subgoal_interval: int = 40,
        filter_high_level: bool = False,  # 是否只返回高层决策点数据
        reach_thresh: float = 0.1,  # 判断"到达"的阈值，与训练时保持一致
        use_cbf: bool = False,  # 是否使用 CBF 安全控制器
):
    """
    测试层级RL的rollout函数

    Parameters
    ----------
    env: MultiAgentEnv
    actor: 高层策略，输出subgoal
    init_rnn_state: Array
    key: PRNGKey
    stochastic: bool, 是否使用随机策略
    subgoal_interval: int, 多少步生成一次subgoal
    filter_high_level: bool, 是否只返回高层决策点数据（用于训练统计），False则返回所有帧（用于视频）
    """
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    # 初始化：第一个subgoal就是最终目标
    init_subgoal = env.get_agent_goals(init_graph)

    goal_reward_coef, subgoal_bonus_coef, subgoal_shadow_coef, dist_to_goal_coef = _resolve_coefs(env)

    def body_(data, inp_data):
        graph, rnn_state, current_subgoal, step_count = data
        key_ = inp_data

        # === 高层决策：每 subgoal_interval 步生成新的subgoal ===
        should_update = (step_count % subgoal_interval == 0)
        real_goal = env.get_agent_goals(graph)

        # 使用 jax.lax.cond 替代 if/else
        def update_subgoal(_):
            if stochastic:
                # 随机策略
                new_sg, rnn = actor(graph, rnn_state, key_)
                return new_sg, rnn
            else:
                # 确定性策略
                new_sg, rnn = actor(graph, rnn_state)
                return new_sg, rnn

        def keep_subgoal(_):
            return current_subgoal, rnn_state

        new_subgoal, new_rnn_state = jax.lax.cond(
            should_update,
            update_subgoal,
            keep_subgoal,
            operand=None
        )

        # === 强制最后一个 subgoal 为最终目标 ===
        remaining_steps = env.max_episode_steps - step_count
        is_last_subgoal = remaining_steps <= subgoal_interval

        ##============================================================##
        # new_subgoal = jnp.where(
        #     is_last_subgoal,
        #     real_goal,      # 最后一个周期：使用最终目标
        #     new_subgoal     # 其他：使用策略生成的 subgoal
        # )
        ##============================================================##

        # === 低层执行：使用 u_ref 跟踪当前 subgoal ===
        dist_to_real_goal = jnp.linalg.norm(new_subgoal - real_goal, axis=-1)
        # is_final_goal = (dist_to_real_goal < env.params.get("dist2goal", 0.1) * 2).all()

        # 根据 use_cbf 选择控制器
        if use_cbf:
            action = env.safe_u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)
        else:
            action = env.u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)

        # 环境交互
        next_graph, reward, cost, done, info = env.step(graph, action)

        # === 计算稀疏奖励（与训练时一致）===
        agent_states = next_graph.type_states(type_idx=0, n_type=env.num_agents)
        goals = real_goal

        agent_pos = agent_states[:, :env.action_dim]
        goal_pos = goals[:, :env.action_dim]
        dist2goal = _compute_dist2goal(env, goal_pos, agent_pos)

        # 1. 到达最终目标的奖励 (使用 reach_thresh)
        goal_reward = jnp.where(dist2goal < reach_thresh, 0.0, -1.0).mean() * goal_reward_coef
        # goal_reward = jnp.where(dist2goal < reach_thresh, 1.0, 0.0).mean() * goal_reward_coef

        # 2. 到达 subgoal 的奖励 (鼓励生成可达的 subgoal)
        dist2subgoal = jnp.linalg.norm(agent_pos - new_subgoal, axis=-1)
        subgoal_bonus = jnp.where(dist2subgoal < SUBGOAL_BONUS_THRESH, 1, 0.0).mean() * subgoal_bonus_coef

        # 3. agent 距离 goal 的 dense reward
        dist_agent_to_goal = -dist2goal.mean() * dist_to_goal_coef

        # 4. subgoal 阴影区惩罚（与训练时一致）
        shadow_cost = env.get_subgoal_shadow_cost(graph, new_subgoal)  # (n_agents,), -1或0
        subgoal_shadow_penalty = shadow_cost.mean() * subgoal_shadow_coef

        sparse_reward = goal_reward + subgoal_bonus + dist_agent_to_goal + subgoal_shadow_penalty

        return (next_graph, new_rnn_state, new_subgoal, step_count + 1), (
            graph,
            new_subgoal,  # 保存subgoal而不是低层action
            rnn_state,
            reward,  # 环境reward
            cost,
            done,
            None,  # log_pi
            next_graph,
            sparse_reward,  # 自定义的稀疏reward
            dist2goal,  # 每个goal到最近agent的距离
        )

    keys = jax.random.split(key, env.max_episode_steps)
    init_data = (init_graph, init_rnn_state, init_subgoal, 0)

    _, (graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs, sparse_rewards, dist2goals) = (
        jax.lax.scan(body_,
                     init_data,
                     keys,
                     length=env.max_episode_steps))

    if filter_high_level:
        # === 筛选出高层决策点的数据（用于训练统计）===
        high_level_indices = jnp.arange(0, env.max_episode_steps, subgoal_interval)
        reward_end_indices = jnp.minimum(high_level_indices + subgoal_interval - 1, env.max_episode_steps - 1)

        rollout_data = Rollout(
            graph=jax.tree.map(lambda x: x[high_level_indices], graphs),
            actions=actions[high_level_indices],
            rnn_states=jax.tree.map(lambda x: x[high_level_indices], actor_rnn_states),
            rewards=rewards[high_level_indices],
            costs=costs[high_level_indices],
            dones=dones[high_level_indices],
            log_pis=None,
            next_graph=jax.tree.map(lambda x: x[reward_end_indices], next_graphs),
            sparse_rewards=sparse_rewards[reward_end_indices],
            dist2goal=dist2goals[reward_end_indices],  # 用 interval 结束时的距离
        )
    else:
        # === 返回所有帧（用于视频渲染）===
        rollout_data = Rollout(
            graph=graphs,
            actions=actions,
            rnn_states=actor_rnn_states,
            rewards=rewards,
            costs=costs,
            dones=dones,
            log_pis=None,
            next_graph=next_graphs,
            sparse_rewards=sparse_rewards,
            dist2goal=dist2goals,
        )

    return rollout_data


def test_rollout_subgoal_manifold(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        stochastic: bool = False,
        subgoal_interval: int = 40,
        filter_high_level: bool = False,
        reach_thresh: float = 0.1,
):
    """测试层级RL的rollout函数 (manifold 安全版本)"""
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    init_subgoal = env.get_agent_goals(init_graph)
    init_s_all = env.manifold_init_slack(init_graph)

    goal_reward_coef, subgoal_bonus_coef, subgoal_shadow_coef, dist_to_goal_coef = _resolve_coefs(env)

    def body_(data, inp_data):
        graph, rnn_state, current_subgoal, step_count, s_all = data
        key_ = inp_data

        should_update = (step_count % subgoal_interval == 0)
        real_goal = env.get_agent_goals(graph)

        def update_subgoal(_):
            if stochastic:
                new_sg, rnn = actor(graph, rnn_state, key_)
                return new_sg, rnn
            else:
                new_sg, rnn = actor(graph, rnn_state)
                return new_sg, rnn

        def keep_subgoal(_):
            return current_subgoal, rnn_state

        new_subgoal, new_rnn_state = jax.lax.cond(
            should_update, update_subgoal, keep_subgoal, operand=None
        )

        remaining_steps = env.max_episode_steps - step_count
        is_last_subgoal = remaining_steps <= subgoal_interval

        # === 低层: LQR → manifold 安全修正 ===
        nominal_action = env.u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)
        action, _, s_new, _ = env.get_manifold_action(graph, u_ref=nominal_action, s_all=s_all)
        action = env.clip_action(action)

        next_graph, reward, cost, done, info = env.step(graph, action)

        # === 稀疏奖励 ===
        agent_states = next_graph.type_states(type_idx=0, n_type=env.num_agents)
        agent_pos = agent_states[:, :env.action_dim]
        goal_pos = real_goal[:, :env.action_dim]
        dist2goal = _compute_dist2goal(env, goal_pos, agent_pos)

        goal_reward = jnp.where(dist2goal < reach_thresh, 0.0, -1.0).mean() * goal_reward_coef
        # goal_reward = jnp.where(dist2goal < reach_thresh, 1.0, 0.0).mean() * goal_reward_coef

        dist2subgoal = jnp.linalg.norm(agent_pos - new_subgoal, axis=-1)
        subgoal_bonus = jnp.where(dist2subgoal < SUBGOAL_BONUS_THRESH, 1, 0.0).mean() * subgoal_bonus_coef
        dist_agent_to_goal = -dist2goal.mean() * dist_to_goal_coef
        shadow_cost = env.get_subgoal_shadow_cost(graph, new_subgoal)
        subgoal_shadow_penalty = shadow_cost.mean() * subgoal_shadow_coef
        sparse_reward = goal_reward + subgoal_bonus + dist_agent_to_goal + subgoal_shadow_penalty

        return (next_graph, new_rnn_state, new_subgoal, step_count + 1, s_new), (
            graph, new_subgoal, rnn_state, reward, cost, done,
            None, next_graph, sparse_reward, dist2goal,
        )

    keys = jax.random.split(key, env.max_episode_steps)
    init_data = (init_graph, init_rnn_state, init_subgoal, 0, init_s_all)

    _, (graphs, actions, actor_rnn_states, rewards, costs, dones,
        log_pis, next_graphs, sparse_rewards, dist2goals) = (
        jax.lax.scan(body_, init_data, keys, length=env.max_episode_steps))

    if filter_high_level:
        high_level_indices = jnp.arange(0, env.max_episode_steps, subgoal_interval)
        reward_end_indices = jnp.minimum(high_level_indices + subgoal_interval - 1, env.max_episode_steps - 1)

        rollout_data = Rollout(
            graph=jax.tree.map(lambda x: x[high_level_indices], graphs),
            actions=actions[high_level_indices],
            rnn_states=jax.tree.map(lambda x: x[high_level_indices], actor_rnn_states),
            rewards=rewards[high_level_indices],
            costs=costs[high_level_indices],
            dones=dones[high_level_indices],
            log_pis=None,
            next_graph=jax.tree.map(lambda x: x[reward_end_indices], next_graphs),
            sparse_rewards=sparse_rewards[reward_end_indices],
            dist2goal=dist2goals[reward_end_indices],
        )
    else:
        rollout_data = Rollout(
            graph=graphs,
            actions=actions,
            rnn_states=actor_rnn_states,
            rewards=rewards,
            costs=costs,
            dones=dones,
            log_pis=None,
            next_graph=next_graphs,
            sparse_rewards=sparse_rewards,
            dist2goal=dist2goals,
        )

    return rollout_data


def test_rollout_subgoal_manifold_vmas(
        env: MultiAgentEnv,
        actor: Callable,
        init_rnn_state: Array,
        key: PRNGKey,
        stochastic: bool = False,
        subgoal_interval: int = 40,
        filter_high_level: bool = False,
        reach_thresh: float = 0.1,
):
    """VMAS 版测试 rollout (manifold 安全, env reward)"""
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    init_subgoal = env.get_agent_goals(init_graph)
    init_s_all = env.manifold_init_slack(init_graph)

    def body_(data, inp_data):
        graph, rnn_state, current_subgoal, step_count, s_all = data
        key_ = inp_data

        should_update = (step_count % subgoal_interval == 0)

        def update_subgoal(_):
            if stochastic:
                new_sg, rnn = actor(graph, rnn_state, key_)
                return new_sg, rnn
            else:
                new_sg, rnn = actor(graph, rnn_state)
                return new_sg, rnn

        def keep_subgoal(_):
            return current_subgoal, rnn_state

        new_subgoal, new_rnn_state = jax.lax.cond(
            should_update, update_subgoal, keep_subgoal, operand=None
        )

        remaining_steps = env.max_episode_steps - step_count
        is_last_subgoal = remaining_steps <= subgoal_interval

        # 低层: u_ref → manifold
        nominal_action = env.u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)
        action, _, s_new, _ = env.get_manifold_action(graph, u_ref=nominal_action, s_all=s_all)
        action = env.clip_action(action)

        next_graph, reward, cost, done, info = env.step(graph, action)

        # sparse_reward = env reward
        sparse_reward = reward
        # dist2goal: 对于 VMAS ReverseTransport, 用 box-to-goal 距离
        env_state = next_graph.env_states
        dist2goal = jnp.linalg.norm(env_state.box_pos - env_state.goal_pos)[None]  # (1,)

        return (next_graph, new_rnn_state, new_subgoal, step_count + 1, s_new), (
            graph, new_subgoal, rnn_state, reward, cost, done,
            None, next_graph, sparse_reward, dist2goal,
        )

    keys = jax.random.split(key, env.max_episode_steps)
    init_data = (init_graph, init_rnn_state, init_subgoal, 0, init_s_all)

    _, (graphs, actions, actor_rnn_states, rewards, costs, dones,
        log_pis, next_graphs, sparse_rewards, dist2goals) = (
        jax.lax.scan(body_, init_data, keys, length=env.max_episode_steps))

    if filter_high_level:
        high_level_indices = jnp.arange(0, env.max_episode_steps, subgoal_interval)
        reward_end_indices = jnp.minimum(high_level_indices + subgoal_interval - 1, env.max_episode_steps - 1)

        rollout_data = Rollout(
            graph=jax.tree.map(lambda x: x[high_level_indices], graphs),
            actions=actions[high_level_indices],
            rnn_states=jax.tree.map(lambda x: x[high_level_indices], actor_rnn_states),
            rewards=rewards[high_level_indices],
            costs=costs[high_level_indices],
            dones=dones[high_level_indices],
            log_pis=None,
            next_graph=jax.tree.map(lambda x: x[reward_end_indices], next_graphs),
            sparse_rewards=sparse_rewards[reward_end_indices],
            dist2goal=dist2goals[reward_end_indices],
        )
    else:
        rollout_data = Rollout(
            graph=graphs,
            actions=actions,
            rnn_states=actor_rnn_states,
            rewards=rewards,
            costs=costs,
            dones=dones,
            log_pis=None,
            next_graph=next_graphs,
            sparse_rewards=sparse_rewards,
            dist2goal=dist2goals,
        )

    return rollout_data


def has_nan(x):
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def has_inf(x):
    return jtu.tree_map(lambda y: jnp.isinf(y).any(), x)


def has_any_inf(x):
    return jnp.array(jtu.tree_flatten(has_inf(x))[0]).any()


def has_any_nan_or_inf(x):
    return has_any_nan(x) | has_any_inf(x)


def compute_norm(grad):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def tree_copy(tree):
    return jtu.tree_map(lambda x: x.copy(), tree)


def jax2np(x):
    return jtu.tree_map(lambda y: np.array(y), x)


def np2jax(x):
    return jtu.tree_map(lambda y: jnp.array(y), x)


def internet(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        print(ex)
        return False


def is_connected():
    return internet()


def centered_norm(vmin: float | list[float], vmax: float | list[float]):
    if isinstance(vmin, list):
        vmin = min(vmin)
    if isinstance(vmax, list):
        vmin = max(vmax)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(0, halfrange)


def plot_rnn_states(rnn_states: Array, name: str, path: str):
    """
    rnn_states: (T, n_layer, n_agent, n_carry, hid_size)
    """
    T, n_layer, n_agent, n_carry, hid_size = rnn_states.shape
    for i_layer in range(n_layer):
        fig, ax = plt.subplots(nrows=n_agent, ncols=n_carry, figsize=(10, 20))
        for i_agent in range(n_agent):
            for i_carry in range(n_carry):
                ax[i_agent, i_carry].plot(rnn_states[:, i_layer, i_agent, i_carry, :])
                ax[i_agent, i_carry].set_title(f'Agent {i_agent}, carry {i_carry}, layer {i_layer}')
                ax[i_agent, i_carry].set_xlabel('Time step')
                ax[i_agent, i_carry].set_ylabel('State value')
        fig.tight_layout()
        plt.savefig(os.path.join(path, f'rnn_states_{name}_layer{i_layer}.png'))
