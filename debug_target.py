"""快速诊断: 在 LinearDrone / CrazyFlie 中, 在起点->goal 直线上均匀放 n 个 subgoal,
agent 接近当前 subgoal 就切换到下一个, 最后一个用 is_final_goal=True 减速停下.
默认只测 nominal policy (u_ref), 不开 manifold."""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dgppo.env import make_env
from dgppo.trainer.data import Rollout


def make_subgoals(init_pos: np.ndarray, goal_pos: np.ndarray, n_subgoals: int) -> np.ndarray:
    """init_pos, goal_pos: (n_agent, dim).  返回 (n_subgoals, n_agent, dim).

    在 init -> goal 的直线上均匀放 n_subgoals 个点, 第 n_subgoals 个就是 goal.
    比例 i/n_subgoals, i=1..n_subgoals.
    """
    fracs = np.linspace(1.0 / n_subgoals, 1.0, n_subgoals)  # (K,)
    diff = goal_pos - init_pos  # (n_agent, dim)
    subgoals = init_pos[None] + fracs[:, None, None] * diff[None]  # (K, n_agent, dim)
    return subgoals


def test_low_level(env_id, n_agents=3, n_obs=4, max_step=128, n_episodes=5,
                   use_manifold=False, dim=2, n_subgoals=4, steps_per_subgoal=None,
                   switch_thresh=0.05):
    """steps_per_subgoal: 若不为 None, 按时间切换 (每 N 步推进 1 个 subgoal); 否则按距离 switch_thresh 切换."""
    env = make_env(env_id=env_id, num_agents=n_agents, num_obs=n_obs, max_step=max_step)

    if use_manifold:
        env.init_manifold(k=5, K=0.5, Kc=30.0, alpha_max=3.0, g_act_thresh=0.02,
                          safety_margin=0.02, n_lookahead=0, w_slack=10.0)
        # warmup
        key = jr.PRNGKey(42)
        graph = env.reset(key)
        s_all = env.manifold_init_slack(graph)
        goal_pos = env.get_agent_goals(graph)
        nominal = env.u_ref(graph, target_pos=goal_pos, is_final_goal=False)
        action, _, s_new, _ = env.get_manifold_action(graph, u_ref=nominal, s_all=s_all)
        jax.block_until_ready(action)
        print("Manifold warmup done.\n")

    mode = f"sps{steps_per_subgoal}" if steps_per_subgoal is not None else f"sg{n_subgoals}"
    tag = ("manifold" if use_manifold else "nominal") + f"_{mode}"

    for epi in range(n_episodes):
        key = jr.PRNGKey(epi)
        graph = env.reset(key)
        goal_pos = np.array(env.get_agent_goals(graph))  # (n_agent, dim)
        if use_manifold:
            s_all = env.manifold_init_slack(graph)

        init_agent_pos = np.array(graph.type_states(type_idx=0, n_type=env.num_agents)[:, :dim])
        init_dist = np.linalg.norm(goal_pos - init_agent_pos, axis=-1)

        # 构造每个 agent 的 subgoal 序列, 每个 agent 单独跟踪当前 subgoal idx
        subgoals = make_subgoals(init_agent_pos, goal_pos, n_subgoals)  # (K, n_agent, dim)
        cur_idx = np.zeros(env.num_agents, dtype=np.int32)

        graphs, actions_list, rewards_list, costs_list, dones_list = [], [], [], [], []
        subgoal_log = []  # 记录每步 target_pos, 方便后面看
        total_cost = 0.0

        for step in range(max_step):
            agent_pos = np.array(graph.type_states(type_idx=0, n_type=env.num_agents)[:, :dim])

            # 选当前 subgoal: 时间模式 (每 steps_per_subgoal 步推进) 或 距离模式 (近到 switch_thresh 推进)
            target_pos = np.zeros_like(agent_pos)
            for a in range(env.num_agents):
                if steps_per_subgoal is not None:
                    cur_idx[a] = min(step // steps_per_subgoal, n_subgoals - 1)
                else:
                    idx = cur_idx[a]
                    tgt = subgoals[idx, a]
                    d = np.linalg.norm(tgt - agent_pos[a])
                    if d < switch_thresh and idx < n_subgoals - 1:
                        cur_idx[a] = idx + 1
                target_pos[a] = subgoals[cur_idx[a], a]

            # 只有所有 agent 都到了最后一个 subgoal 才用 final_goal P-control 减速
            is_final_goal = bool(np.all(cur_idx == n_subgoals - 1))

            target_pos_j = jnp.array(target_pos)
            nominal = env.u_ref(graph, target_pos=target_pos_j, is_final_goal=is_final_goal)

            if use_manifold:
                action, _, s_all, _ = env.get_manifold_action(graph, u_ref=nominal, s_all=s_all)
                action = env.clip_action(action)
            else:
                action = nominal

            graphs.append(graph)
            actions_list.append(action)
            subgoal_log.append(target_pos.copy())

            graph, reward, cost, done, info = env.step(graph, action)
            rewards_list.append(reward)
            costs_list.append(cost)
            dones_list.append(done)
            total_cost += float(jnp.maximum(cost, 0).max())

        final_agent_pos = np.array(graph.type_states(type_idx=0, n_type=env.num_agents)[:, :dim])
        final_dist = np.linalg.norm(goal_pos - final_agent_pos, axis=-1)
        reached = final_dist < 0.05

        print(f"[{tag}] epi {epi}: init_dist={init_dist.round(3)}, final_dist={final_dist.round(3)}, "
              f"reached={reached}, cost={total_cost:.4f}, last_idx={cur_idx.tolist()}")

        # save gif
        stacked_graph = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *graphs)
        stacked_costs = jnp.stack(costs_list, axis=0)

        # render_lidar 把 rollout.actions[t, :, :dim] 当 subgoal 位置画 (见 plot.py:540).
        # 所以我们把 actions 字段塞成 subgoal_log, 同时通过 show_subgoal=True 启用绘制.
        subgoal_arr = np.stack(subgoal_log, axis=0)  # (T, n_agent, dim)
        # 如果 action_dim > dim, 用 0 padding 到 action_dim
        action_dim = int(actions_list[0].shape[-1])
        if subgoal_arr.shape[-1] < action_dim:
            pad = np.zeros((*subgoal_arr.shape[:-1], action_dim - subgoal_arr.shape[-1]))
            subgoal_arr = np.concatenate([subgoal_arr, pad], axis=-1)

        rollout = Rollout(
            graph=stacked_graph,
            actions=jnp.array(subgoal_arr),  # NOTE: 借用 actions 字段传 subgoal 位置给 renderer
            rnn_states=None,
            rewards=jnp.stack(rewards_list, axis=0),
            costs=stacked_costs,
            dones=jnp.stack(dones_list, axis=0),
            log_pis=None,
            next_graph=None,
        )
        Ta_is_unsafe = np.array(stacked_costs.max(axis=-1).max(axis=-1) > 0)

        out_dir = pathlib.Path(f"debug_nominal/{env_id}/{tag}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"epi{epi}.gif"
        env.render_video(rollout, out_path, Ta_is_unsafe=Ta_is_unsafe,
                         viz_opts={}, show_subgoal=True, subgoal_interval=10)
        print(f"  -> saved {out_path}")

    print()


if __name__ == "__main__":
    import sys
    env_id = sys.argv[1] if len(sys.argv) > 1 else "CrazyFlie"
    # 默认: max_step=128, 每 8 步换一个 subgoal => 16 个 subgoal, 时间切换
    max_step = 128
    steps_per_subgoal = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    n_subgoals = max_step // steps_per_subgoal
    use_manifold = (sys.argv[3].lower() == "true") if len(sys.argv) > 3 else False
    print("=" * 60)
    print(f"Test: {env_id} u_ref + {n_subgoals} subgoals "
          f"(time-switch every {steps_per_subgoal} steps, manifold={use_manifold})")
    print("=" * 60)
    test_low_level(env_id, n_agents=3, n_obs=4, max_step=max_step,
                   n_episodes=3, use_manifold=use_manifold, dim=3,
                   n_subgoals=n_subgoals, steps_per_subgoal=steps_per_subgoal)
