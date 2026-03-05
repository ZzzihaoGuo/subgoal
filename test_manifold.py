import argparse
import datetime
import functools as ft
import os
import pathlib
import logging

# 抑制 matplotlib 字体警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 抑制 jaxproxqp 的 debug 输出
from loguru import logger
logger.disable("jaxproxqp")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.data import Rollout
from dgppo.trainer.utils import (
    GOAL_REWARD_COEF, SUBGOAL_BONUS_THRESH, SUBGOAL_BONUS_COEF,
    DIST_TO_GOAL_COEF, SUBGOAL_SHADOW_COEF,
)
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.utils import jax_jit_np, jax_vmap
from dgppo.utils.typing import Array


def manifold_rollout(
        env,
        actor,
        init_rnn_state,
        key,
        stochastic=False,
        subgoal_interval=8,
        reach_thresh=0.1,
):
    """自定义 rollout: 在 scan carry 中传递松弛变量 s_all, 实现 ATACOM 积分"""
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    # 初始化 subgoal 和松弛变量
    init_subgoal = init_graph.type_states(type_idx=1, n_type=env.num_agents)[:, :2]
    init_s_all = env.manifold_init_slack(init_graph)

    def body_(data, inp_data):
        graph, rnn_state, current_subgoal, step_count, s_all = data
        key_ = inp_data

        # === 高层决策 ===
        should_update = (step_count % subgoal_interval == 0)
        real_goal = graph.type_states(type_idx=1, n_type=env.num_agents)[:, :2]

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

        # === 低层: LQR → manifold 安全修正 (带松弛变量积分) ===
        nominal_action = env.u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)
        action, _, s_new, dbg = env.get_manifold_action(graph, u_ref=nominal_action, s_all=s_all)
        action = env.clip_action(action)

        # 环境交互
        next_graph, reward, cost, done, info = env.step(graph, action)

        # === 稀疏奖励计算 ===
        agent_states = next_graph.type_states(type_idx=0, n_type=env.num_agents)
        goals = real_goal
        agent_pos = agent_states[:, :2]
        goal_pos = goals[:, :2]
        dist2goal = jnp.linalg.norm(
            jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1
        ).min(axis=1)

        goal_reward = jnp.where(dist2goal < reach_thresh, 0.0, -1.0).mean() * GOAL_REWARD_COEF
        dist2subgoal = jnp.linalg.norm(agent_pos - new_subgoal, axis=-1)
        subgoal_bonus = jnp.where(dist2subgoal < SUBGOAL_BONUS_THRESH, 1, 0.0).mean() * SUBGOAL_BONUS_COEF
        dist_agent_to_goal = -dist2goal.mean() * DIST_TO_GOAL_COEF
        shadow_cost = env.get_subgoal_shadow_cost(graph, new_subgoal)
        subgoal_shadow_penalty = shadow_cost.mean() * SUBGOAL_SHADOW_COEF
        sparse_reward = goal_reward + subgoal_bonus + dist_agent_to_goal + subgoal_shadow_penalty

        return (next_graph, new_rnn_state, new_subgoal, step_count + 1, s_new), (
            graph, new_subgoal, rnn_state, reward, cost, done, None, next_graph,
            sparse_reward, dist2goal, dbg,
        )

    keys = jax.random.split(key, env.max_episode_steps)
    init_data = (init_graph, init_rnn_state, init_subgoal, 0, init_s_all)

    _, (graphs, actions, actor_rnn_states, rewards, costs, dones,
        log_pis, next_graphs, sparse_rewards, dist2goals, debug_infos) = (
        jax.lax.scan(body_, init_data, keys, length=env.max_episode_steps)
    )

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
    return rollout_data, debug_infos


def test(args):
    print(f"> Running test_manifold.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    # load config
    with open(os.path.join(args.path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # create environment
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=config.obs if args.obs is None else args.obs,
        max_step=args.max_step,
        full_observation=args.full_observation,
    )

    # create algorithm
    path = args.path
    model_path = os.path.join(path, "models")
    if args.step is None:
        models = os.listdir(model_path)
        step = max([int(model) for model in models if model.isdigit()])
    else:
        step = args.step
    print("step: ", step)

    use_relative_subgoal = getattr(config, 'use_relative_subgoal', False)
    if args.relative_subgoal is not None:
        use_relative_subgoal = args.relative_subgoal
    max_delta = getattr(config, 'max_delta', None)
    if args.max_delta is not None:
        max_delta = args.max_delta

    algo = make_algo(
        algo=config.algo, env=env,
        node_dim=env.node_dim, edge_dim=env.edge_dim,
        state_dim=env.state_dim, action_dim=env.action_dim,
        n_agents=env.num_agents, cost_weight=config.cost_weight,
        actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers,
        Vh_gnn_layers=config.Vh_gnn_layers if hasattr(config, "Vh_gnn_layers") else 1,
        lr_actor=config.lr_actor, lr_Vl=config.lr_Vl,
        max_grad_norm=2.0, seed=config.seed,
        use_rnn=config.use_rnn, rnn_layers=config.rnn_layers,
        use_lstm=config.use_lstm,
        use_relative_subgoal=use_relative_subgoal, max_delta=max_delta,
    )
    algo.load(model_path, step)
    act_fn = jax.jit(algo.act) if not args.stochastic else jax.jit(
        lambda x, z, rnn_state, key: (algo.step(x, z, rnn_state, key)[0], algo.step(x, z, rnn_state, key)[2])
    )
    init_rnn_state = algo.init_rnn_state

    # === 初始化 manifold (ATACOM v2) ===
    import time
    env.init_manifold(
        k=args.topk,
        K=args.viab_gain,
        Kc=args.err_gain,
        alpha_max=args.alpha_max,
        g_act_thresh=args.g_act_thresh,
        safety_margin=args.safety_margin,
        n_lookahead=args.n_lookahead,
        w_slack=args.w_slack,
    )

    # Warmup JIT
    print("Warming up manifold (JIT compiling)...")
    warmup_key = jr.PRNGKey(9999)
    warmup_graph = env.reset(warmup_key)
    warmup_s = env.manifold_init_slack(warmup_graph)
    target_pos = warmup_graph.type_states(type_idx=1, n_type=env.num_agents)[:, :2]
    nominal = env.u_ref(warmup_graph, target_pos=target_pos, is_final_goal=False)

    start = time.time()
    _ = jax.jit(env.get_manifold_action)(warmup_graph, u_ref=nominal, s_all=warmup_s)
    print(f"Manifold warmup complete ({time.time() - start:.2f}s)")

    # set up keys
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    # 自定义 rollout (带松弛变量积分)
    rollout_fn = ft.partial(
        manifold_rollout, env, act_fn, init_rnn_state,
        stochastic=args.stochastic, subgoal_interval=args.subgoal_interval,
    )
    rollout_fn = jax_jit_np(rollout_fn)

    def unsafe_mask(graph_: GraphsTuple) -> Array:
        cost = env.get_cost(graph_)
        return jnp.any(cost >= 0.0, axis=-1)

    is_unsafe_fn = jax_jit_np(jax_vmap(unsafe_mask))

    # test
    rewards, costs, rollouts, is_unsafes, rates = [], [], [], [], []
    last_rewards, last_dists, is_success_list, success_rates_per_epi = [], [], [], []

    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)
        rollout, debug_infos = rollout_fn(key_x0)
        is_unsafes.append(is_unsafe_fn(rollout.graph))

        epi_reward = rollout.rewards.sum()
        epi_cost = rollout.costs.max()
        last_reward = rollout.rewards[-1]
        last_dist = rollout.dist2goal[-1].mean() if rollout.dist2goal is not None else 0.0
        rewards.append(epi_reward)
        costs.append(epi_cost)
        last_rewards.append(last_reward)
        last_dists.append(last_dist)
        rollouts.append(rollout)
        safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()

        dist_thresh = env.params.get("dist2goal", 0.01)
        if rollout.dist2goal is not None:
            final_dist = rollout.dist2goal[-1]
        else:
            final_states = rollout.graph.states[-1]
            agent_pos = final_states[:env.num_agents, :2]
            goal_pos = final_states[env.num_agents:env.num_agents * 2, :2]
            final_dist = jnp.linalg.norm(
                jnp.expand_dims(goal_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1
            ).min(axis=1)
        agent_reached = np.array(final_dist < dist_thresh)
        is_success_list.append(agent_reached)
        epi_all_success = float(agent_reached.all())
        success_rates_per_epi.append(epi_all_success)

        print(f"epi: {i_epi}, reward: {epi_reward:.7f}, cost: {epi_cost:.7f}, "
              f"last_reward: {last_reward:.7f}, last_dist: {last_dist:.7f}, safe rate: {safe_rate * 100:.7f}%, "
              f"success: {agent_reached.mean() * 100:.1f}% ({agent_reached.sum()}/{len(agent_reached)})")

        # === debug: 碰撞时刻分析 ===
        if safe_rate < 1.0:
            is_unsafe_t = np.array(is_unsafes[-1])  # (T, n_agents) or (T,)
            if is_unsafe_t.ndim == 1:
                unsafe_steps = np.where(is_unsafe_t > 0)[0]
            else:
                unsafe_steps = np.where(is_unsafe_t.max(axis=-1) > 0)[0]
            print(f"  [DEBUG] unsafe steps: {unsafe_steps.tolist()[:10]}{'...' if len(unsafe_steps) > 10 else ''}")
            # 看碰撞前1步的 ATACOM debug info
            dbg = np.array(debug_infos)  # (T, n_agents, 14)
            for t_idx in unsafe_steps[:3]:
                t_prev = max(t_idx - 1, 0)  # 碰撞前一步 (ATACOM在这步计算)
                g_states = rollout.graph.states[t_idx]
                a_pos = g_states[:env.num_agents, :2]
                a_vel = g_states[:env.num_agents, 2:4]
                n_rays = env.params["top_k_rays"]
                n_obs_nodes = n_rays * env.num_agents
                obs_pos = g_states[-n_obs_nodes:, :2]
                obs_pos_per_agent = obs_pos.reshape(env.num_agents, n_rays, 2)
                for ai in range(env.num_agents):
                    obs_dists = np.linalg.norm(np.array(obs_pos_per_agent[ai] - a_pos[ai]), axis=-1)
                    min_obs_dist = obs_dists.min()
                    aa_dists = np.linalg.norm(np.array(a_pos - a_pos[ai]), axis=-1)
                    aa_dists[ai] = 1e6
                    min_aa_dist = aa_dists.min()
                    collision_type = "OBS" if min_obs_dist < min_aa_dist else "AGENT"
                    vel_mag = float(np.linalg.norm(np.array(a_vel[ai])))
                    # ATACOM debug: u_ref, u_opt, a_comp, err, b_proj, g_viab, active, min_dist², k_c
                    d = dbg[t_prev, ai]
                    print(f"    t={t_idx}, agent={ai}: {collision_type}, "
                          f"obs_d={min_obs_dist:.4f}, aa_d={min_aa_dist:.4f}, |v|={vel_mag:.3f}")
                    print(f"      ATACOM@t={t_prev}: u_ref=({d[0]:.3f},{d[1]:.3f}), "
                          f"u_opt=({d[2]:.3f},{d[3]:.3f}), "
                          f"a_comp=({d[4]:.3f},{d[5]:.3f}), "
                          f"err=({d[6]:.3f},{d[7]:.3f}), "
                          f"b_proj=({d[8]:.3f},{d[9]:.3f})")
                    print(f"      g_viab_max={d[10]:.5f}, active_max={d[11]:.3f}, "
                          f"min_dist²={d[12]:.5f} (dist={np.sqrt(d[12]):.4f}), k_c_max={d[13]:.5f}")

        rates.append(np.array(safe_rate))

    is_unsafe = np.max(np.stack(is_unsafes), axis=1)
    safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
    is_success = np.stack(is_success_list)
    success_agent_mean = is_success.mean()
    success_epi_mean = np.mean(success_rates_per_epi)

    print(
        f"reward: {np.mean(rewards):.7f}, min/max reward: {np.min(rewards):.7f}/{np.max(rewards):.7f}, "
        f"cost: {np.mean(costs):.7f}, min/max cost: {np.min(costs):.7f}/{np.max(costs):.7f}, "
        f"last_reward: {np.mean(last_rewards):.7f}, last_dist: {np.mean(last_dists):.7f}, "
        f"safe_rate: {safe_mean * 100:.7f}%, "
        f"success_agent: {success_agent_mean * 100:.7f}%, success_epi: {success_epi_mean * 100:.7f}%"
    )

    # make video
    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos" / f"{step}_manifold"
    videos_dir.mkdir(exist_ok=True, parents=True)
    n_unsafe_videos = 0
    for ii, (rollout, Ta_is_unsafe) in enumerate(zip(rollouts, is_unsafes)):
        safe_rate = rates[ii] * 100
        # 只生成不是 100% safe 的 episode 的视频
        if safe_rate >= 120.0:
            continue
        n_unsafe_videos += 1
        video_name = f"n{num_agents}_epi{ii:02}_reward{rewards[ii]:.7f}_cost{costs[ii]:.7f}_sr{safe_rate:.0f}"
        video_path = videos_dir / f"{stamp_str}_{video_name}.gif"
        env.render_video(rollout, video_path, Ta_is_unsafe, {}, dpi=args.dpi,
                         show_subgoal=True, subgoal_interval=args.subgoal_interval)
    print(f"Generated {n_unsafe_videos} unsafe episode GIFs in {videos_dir}")


def main():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--path", type=str, default="logs/LidarSpread/informarl_subgoal/seed0_212001549_UBBT")

    # manifold (ATACOM) parameters
    parser.add_argument("--topk", type=int, default=3, help="Number of nearest neighbors for manifold")
    parser.add_argument("--viab-gain", type=float, default=0.5, help="Viability constraint gain K (controls activation distance)")
    parser.add_argument("--err-gain", type=float, default=30.0, help="Error correction gain Kc (need Kc*dt<1, dt=0.03→Kc<33)")
    parser.add_argument("--alpha-max", type=float, default=1.0, help="Null space control bound (ATACOM alpha_max)")
    parser.add_argument("--g-act-thresh", type=float, default=0.01, help="Constraint activation threshold")
    parser.add_argument("--safety-margin", type=float, default=0.02, help="Safety margin beyond collision boundary")
    parser.add_argument("--n-lookahead", type=int, default=2, help="Predictive constraint lookahead steps")
    parser.add_argument("--w-slack", type=float, default=10.0, help="Slack weight in pseudo-inverse (higher = prefer action correction)")

    # test parameters
    parser.add_argument("--no-video", action="store_true", default=True)
    parser.add_argument("--epi", type=int, default=1000)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--obs", type=int, default=11)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--subgoal-interval", type=int, default=8)
    parser.add_argument("--relative-subgoal", action="store_true", default=True)
    parser.add_argument("--max-delta", type=float, default=0.2)

    # default
    parser.add_argument("-n", "--num-agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=100)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()
