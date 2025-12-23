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


def rollout_hierarchical(
        env: MultiAgentEnv,
        high_level_actor: Callable,  # [GraphsTuple, RNN_States, PRNGKey] -> [Subgoal, LogPi, RNN_States]
        init_rnn_state: Array,
        key: PRNGKey,
        subgoal_interval: int = 40,  # 每40步生成一个subgoal
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
    init_subgoal = init_graph.type_states(type_idx=1, n_type=env.num_agents)[:, :2]  # (n_agents, 2)
    
    def body_(data, inp):
        graph, rnn_state, current_subgoal, step_count = data
        key_ = inp
        
        # === 高层决策：每 subgoal_interval 步生成新的subgoal ===
        should_update = (step_count % subgoal_interval == 0)
        real_goal = graph.type_states(type_idx=1, n_type=env.num_agents)[:, :2]

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
        new_subgoal = jnp.where(
            is_last_subgoal,
            real_goal,      # 最后一个周期：使用最终目标
            new_subgoal     # 其他：使用策略生成的 subgoal
        )

        # === 低层执行：使用 u_ref 跟踪当前 subgoal ===
        # 判断当前subgoal是否为最终目标
        dist_to_real_goal = jnp.linalg.norm(new_subgoal - real_goal, axis=-1)
        # is_final_goal = (dist_to_real_goal < env.params.get("dist2goal", 0.1) * 2).all()
        
        # 调用u_ref跟踪subgoal
        action = env.u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)
        
        # 环境交互
        next_graph, reward, cost, done, info = env.step(graph, action)
        
        # === 稀疏奖励：只在达到最终目标时给0，其他时候给-1 ===
        agent_pos = next_graph.type_states(type_idx=0, n_type=env.num_agents)[:, :2]
        goal_pos = real_goal
        dist_to_goal = jnp.linalg.norm(agent_pos - goal_pos, axis=-1)
        reached_goal = (dist_to_goal < env.params.get("dist2goal", 0.1)).all()
        sparse_reward = jnp.where(reached_goal, 0.0, -1.0)
        
        # === 只在高层决策点保存数据 ===
        # 用一个mask标记哪些timestep需要保存
        save_data = should_update
        
        return (next_graph, new_rnn_state, new_subgoal, step_count + 1), (
            graph,
            new_subgoal,  # 保存subgoal而不是action
            rnn_state,
            sparse_reward,
            cost,
            done,
            log_pi,
            next_graph,
            save_data  # 额外的标记
        )
    
    # 执行rollout
    keys = jax.random.split(key, env.max_episode_steps)
    init_data = (init_graph, init_rnn_state, init_subgoal, 0)
    
    _, outputs = jax.lax.scan(body_, init_data, keys, length=env.max_episode_steps)
    
    graphs, subgoals, rnn_states, rewards, costs, dones, log_pis, next_graphs, save_mask = outputs
    
    # === 筛选出高层决策点的数据 ===
    # save_mask: (T,) bool array，标记哪些timestep是高层决策点
    # 我们需要reshape成 (T//subgoal_interval, subgoal_interval) 然后取第一列
    
    n_high_level_steps = env.max_episode_steps // subgoal_interval
    
    # 简单方法：直接索引
    high_level_indices = jnp.arange(0, env.max_episode_steps, subgoal_interval)
    
    rollout_data = Rollout(
        graph=jax.tree.map(lambda x: x[high_level_indices], graphs),
        actions=subgoals[high_level_indices],  # 注意这里是subgoal，不是低层action
        rnn_states=jax.tree.map(lambda x: x[high_level_indices], rnn_states),
        rewards=rewards[high_level_indices],
        costs=costs[high_level_indices],
        dones=dones[high_level_indices],
        log_pis=log_pis[high_level_indices],
        next_graph=jax.tree.map(lambda x: x[high_level_indices], next_graphs),
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
        subgoal_interval: int = 40,  # 新增参数
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
    """
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    # 初始化：第一个subgoal就是最终目标
    init_subgoal = init_graph.type_states(type_idx=1, n_type=env.num_agents)[:, :2]

    def body_(data, inp_data):
        graph, rnn_state, current_subgoal, step_count = data
        key_ = inp_data

        # === 高层决策：每 subgoal_interval 步生成新的subgoal ===
        should_update = (step_count % subgoal_interval == 0)
        real_goal = graph.type_states(type_idx=1, n_type=env.num_agents)[:, :2]

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
        new_subgoal = jnp.where(
            is_last_subgoal,
            real_goal,      # 最后一个周期：使用最终目标
            new_subgoal     # 其他：使用策略生成的 subgoal
        )

        # === 低层执行：使用 u_ref 跟踪当前 subgoal ===
        dist_to_real_goal = jnp.linalg.norm(new_subgoal - real_goal, axis=-1)
        # is_final_goal = (dist_to_real_goal < env.params.get("dist2goal", 0.1) * 2).all()

        # 调用u_ref跟踪subgoal
        action = env.u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)

        # 环境交互
        next_graph, reward, cost, done, info = env.step(graph, action)

        # === 计算稀疏奖励（与训练时一致）===
        agent_pos = next_graph.type_states(type_idx=0, n_type=env.num_agents)[:, :2]
        goal_pos = real_goal
        dist_to_goal = jnp.linalg.norm(agent_pos - goal_pos, axis=-1)
        reached_goal = (dist_to_goal < env.params.get("dist2goal", 0.1)).all()
        sparse_reward = jnp.where(reached_goal, 0.0, -1.0)

        return (next_graph, new_rnn_state, new_subgoal, step_count + 1), (
            graph,
            new_subgoal,  # 保存subgoal而不是低层action
            rnn_state,
            sparse_reward,  # 使用稀疏reward而不是环境reward
            cost,
            done,
            None,  # log_pi
            next_graph
        )

    keys = jax.random.split(key, env.max_episode_steps)
    init_data = (init_graph, init_rnn_state, init_subgoal, 0)

    _, (graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs) = (
        jax.lax.scan(body_,
                     init_data,
                     keys,
                     length=env.max_episode_steps))

    # # === 筛选出高层决策点的数据（与训练时一致）===
    # high_level_indices = jnp.arange(0, env.max_episode_steps, subgoal_interval)

    # rollout_data = Rollout(
    #     graphs=graphs,
    #     actions=actions, #=actions[high_level_indices],
    #     rnn_states=jax.tree.map(lambda x: x[high_level_indices], actor_rnn_states),
    #     rewards=rewards[high_level_indices],  # Shape: (128,) -> (3,)
    #     costs=costs[high_level_indices],
    #     dones=dones[high_level_indices],
    #     log_pis=log_pis[high_level_indices] if log_pis is not None else None,
    #     next_graph=jax.tree.map(lambda x: x[high_level_indices], next_graphs),
    # )

    rollout_data = Rollout(graphs, actions, actor_rnn_states, rewards, costs, dones, log_pis, next_graphs)

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
