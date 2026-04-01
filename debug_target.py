"""快速诊断: 在 LidarBicycleTarget 中，u_ref + manifold 能否把 agent 安全送到 goal，并生成 gif"""
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dgppo.env import make_env
from dgppo.trainer.data import Rollout

def test_low_level(env_id, n_agents=3, n_obs=3, max_step=128, n_episodes=5, use_manifold=False):
    env = make_env(env_id=env_id, num_agents=n_agents, num_obs=n_obs, max_step=max_step)

    if use_manifold:
        env.init_manifold(k=3, K=0.5, Kc=30.0, alpha_max=3.0, g_act_thresh=0.02, safety_margin=0.02, n_lookahead=0, w_slack=10.0)
        # warmup
        key = jr.PRNGKey(42)
        graph = env.reset(key)
        s_all = env.manifold_init_slack(graph)
        goal_pos = graph.type_states(type_idx=1, n_type=env.num_agents)[:, :2]
        nominal = env.u_ref(graph, target_pos=goal_pos, is_final_goal=False)
        action, _, s_new, _ = env.get_manifold_action(graph, u_ref=nominal, s_all=s_all)
        jax.block_until_ready(action)
        print("Manifold warmup done.\n")

    tag = "manifold" if use_manifold else "nominal"

    for epi in range(n_episodes):
        key = jr.PRNGKey(epi)
        graph = env.reset(key)
        goal_pos = graph.type_states(type_idx=1, n_type=env.num_agents)[:, :2]
        if use_manifold:
            s_all = env.manifold_init_slack(graph)

        init_agent_pos = graph.type_states(type_idx=0, n_type=env.num_agents)[:, :2]
        init_dist = np.array(jnp.linalg.norm(goal_pos - init_agent_pos, axis=-1))

        graphs, actions_list, rewards_list, costs_list, dones_list = [], [], [], [], []
        total_cost = 0.0
        for step in range(max_step):
            is_final = (max_step - step) <= 8
            nominal = env.u_ref(graph, target_pos=goal_pos, is_final_goal=is_final)

            if use_manifold:
                action, _, s_all, _ = env.get_manifold_action(graph, u_ref=nominal, s_all=s_all)
                action = env.clip_action(action)
            else:
                action = nominal

            graphs.append(graph)
            actions_list.append(action)

            graph, reward, cost, done, info = env.step(graph, action)
            rewards_list.append(reward)
            costs_list.append(cost)
            dones_list.append(done)
            total_cost += float(jnp.maximum(cost, 0).max())

        final_agent_pos = graph.type_states(type_idx=0, n_type=env.num_agents)[:, :2]
        final_dist = np.array(jnp.linalg.norm(goal_pos - final_agent_pos, axis=-1))
        reached = final_dist < 0.01

        print(f"[{tag}] epi {epi}: init_dist={init_dist.round(3)}, final_dist={final_dist.round(3)}, "
              f"reached={reached}, cost={total_cost:.4f}")

        # save gif
        stacked_graph = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *graphs)
        stacked_costs = jnp.stack(costs_list, axis=0)
        rollout = Rollout(
            graph=stacked_graph,
            actions=jnp.stack(actions_list, axis=0),
            rnn_states=None,
            rewards=jnp.stack(rewards_list, axis=0),
            costs=stacked_costs,
            dones=jnp.stack(dones_list, axis=0),
            log_pis=None,
            next_graph=None,
        )
        Ta_is_unsafe = np.array(stacked_costs.max(axis=-1) > 0)
        out_dir = pathlib.Path(f"debug_nominal/{env_id}/{tag}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"epi{epi}.gif"
        env.render_video(rollout, out_path, Ta_is_unsafe=Ta_is_unsafe, viz_opts={})
        print(f"  -> saved {out_path}")

    print()

if __name__ == "__main__":
    print("=" * 60)
    print("Test: LidarBicycleTarget u_ref + manifold")
    print("=" * 60)
    test_low_level("LidarBicycleTarget", use_manifold=True)
