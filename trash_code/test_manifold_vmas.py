import argparse
import datetime
import functools as ft
import os
import pathlib
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import yaml

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.utils import jax_jit_np, jax_vmap
from dgppo.utils.typing import Array


def vmas_manifold_rollout(
        env,
        actor,
        init_rnn_state,
        key,
        stochastic=False,
        subgoal_interval=8,
):
    """VMAS 版 rollout: manifold 安全修正, carry 中传递松弛变量 s_all"""
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

        # 低层: u_ref → manifold 安全修正
        nominal_action = env.u_ref(graph, target_pos=new_subgoal, is_final_goal=is_last_subgoal)
        action, _, s_new, dbg = env.get_manifold_action(graph, u_ref=nominal_action, s_all=s_all)
        action = env.clip_action(action)

        next_graph, reward, cost, done, info = env.step(graph, action)

        # VMAS metrics: box-to-goal distance
        next_env_state = next_graph.env_states
        box_goal_dist = jnp.linalg.norm(next_env_state.box_pos - next_env_state.goal_pos)

        return (next_graph, new_rnn_state, new_subgoal, step_count + 1, s_new), (
            graph, new_subgoal, rnn_state, reward, cost, done, None, next_graph,
            box_goal_dist, dbg,
        )

    keys = jax.random.split(key, env.max_episode_steps)
    init_data = (init_graph, init_rnn_state, init_subgoal, 0, init_s_all)

    _, (graphs, actions, actor_rnn_states, rewards, costs, dones,
        log_pis, next_graphs, box_goal_dists, debug_infos) = (
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
    )
    return rollout_data, box_goal_dists, debug_infos


def test(args):
    print(f"> Running test_manifold_vmas.py {args}")

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
        max_step=args.max_step if args.max_step is not None else getattr(config, 'max_step', 64),
        full_observation=args.full_observation,
    )

    # load model
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
        n_agents=env.num_agents, cost_weight=getattr(config, 'cost_weight', 0.0),
        actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers,
        Vh_gnn_layers=getattr(config, 'Vh_gnn_layers', 1),
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

    # === 初始化 VMAS manifold ===
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
    print("Warming up VMAS manifold (JIT compiling)...")
    warmup_key = jr.PRNGKey(9999)
    warmup_graph = env.reset(warmup_key)
    warmup_s = env.manifold_init_slack(warmup_graph)
    target_pos = env.get_agent_goals(warmup_graph)
    nominal = env.u_ref(warmup_graph, target_pos=target_pos, is_final_goal=False)

    start = time.time()
    _ = jax.jit(env.get_manifold_action)(warmup_graph, u_ref=nominal, s_all=warmup_s)
    print(f"VMAS manifold warmup complete ({time.time() - start:.2f}s)")

    # set up keys
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    # rollout function
    rollout_fn = ft.partial(
        vmas_manifold_rollout, env, act_fn, init_rnn_state,
        stochastic=args.stochastic, subgoal_interval=args.subgoal_interval,
    )
    rollout_fn = jax_jit_np(rollout_fn)

    # unsafe mask: agent-agent + box-obstacle 都算
    def unsafe_mask(graph_: GraphsTuple) -> Array:
        cost = env.get_cost(graph_)  # (n_agents, 2)
        return jnp.any(cost >= 0.0, axis=-1)  # (n_agents,)

    is_unsafe_fn = jax_jit_np(jax_vmap(unsafe_mask))

    # ====== VMAS 特有指标 ======
    dist2goal_thresh = env.params.get("dist2goal", 0.01)

    rewards_list, costs_list, rates_list = [], [], []
    box_final_dists, is_success_list = [], []
    agent_collision_counts, box_obs_collision_counts = [], []

    for i_epi in range(args.epi):
        key_x0, _ = jr.split(test_keys[i_epi], 2)
        rollout, box_goal_dists, debug_infos = rollout_fn(key_x0)
        is_unsafe = is_unsafe_fn(rollout.graph)  # (T, n_agents)

        epi_reward = rollout.rewards.sum()
        epi_cost = rollout.costs.max()
        safe_rate = 1 - is_unsafe.max(axis=0).mean()

        # VMAS 指标: box 到 goal 的最终距离
        final_box_dist = float(box_goal_dists[-1])
        box_final_dists.append(final_box_dist)

        # 任务成功: box 到达 goal
        task_success = final_box_dist < dist2goal_thresh
        is_success_list.append(task_success)

        # 碰撞分析: agent-agent vs box-obstacle
        costs_arr = np.array(rollout.costs)  # (T, n_agents, 2)
        n_agent_collision_steps = int((costs_arr[:, :, 0] > 0).any(axis=1).sum())
        n_box_obs_collision_steps = int((costs_arr[:, :, 1] > 0).any(axis=1).sum())
        agent_collision_counts.append(n_agent_collision_steps)
        box_obs_collision_counts.append(n_box_obs_collision_steps)

        rewards_list.append(float(epi_reward))
        costs_list.append(float(epi_cost))
        rates_list.append(float(safe_rate))

        print(f"epi: {i_epi:3d}, reward: {epi_reward:9.4f}, cost: {epi_cost:7.4f}, "
              f"box_dist: {final_box_dist:.4f}, success: {'YES' if task_success else 'NO ':>3s}, "
              f"safe_rate: {safe_rate * 100:5.1f}%, "
              f"agent_col: {n_agent_collision_steps:3d}, box_obs_col: {n_box_obs_collision_steps:3d}")

    # ====== 汇总 ======
    rewards_arr = np.array(rewards_list)
    costs_arr_summary = np.array(costs_list)
    rates_arr = np.array(rates_list)
    box_dists_arr = np.array(box_final_dists)
    success_arr = np.array(is_success_list)
    agent_col_arr = np.array(agent_collision_counts)
    box_col_arr = np.array(box_obs_collision_counts)

    print("\n" + "=" * 80)
    print(f"VMAS ReverseTransport Eval Summary ({args.epi} episodes)")
    print("=" * 80)
    print(f"  reward:        {rewards_arr.mean():9.4f} ± {rewards_arr.std():7.4f}  "
          f"(min={rewards_arr.min():.4f}, max={rewards_arr.max():.4f})")
    print(f"  box_final_dist:{box_dists_arr.mean():9.4f} ± {box_dists_arr.std():7.4f}  "
          f"(min={box_dists_arr.min():.4f}, max={box_dists_arr.max():.4f})")
    print(f"  task_success:  {success_arr.mean() * 100:8.2f}%  ({success_arr.sum()}/{len(success_arr)})")
    print(f"  safe_rate:     {rates_arr.mean() * 100:8.2f}%  (每个 agent 全程无碰撞的比例)")
    print(f"  agent_col:     {agent_col_arr.mean():8.2f} steps/epi  (agent-agent 碰撞)")
    print(f"  box_obs_col:   {box_col_arr.mean():8.2f} steps/epi  (box-obstacle 碰撞)")
    print("=" * 80)

    # make video
    if args.no_video:
        return

    videos_dir = pathlib.Path(path) / "videos" / f"{step}_manifold_vmas"
    videos_dir.mkdir(exist_ok=True, parents=True)
    n_videos = 0
    for ii in range(min(args.epi, args.max_videos)):
        key_x0, _ = jr.split(test_keys[ii], 2)
        rollout, _, _ = rollout_fn(key_x0)
        is_unsafe = is_unsafe_fn(rollout.graph)
        safe_rate = rates_list[ii] * 100
        video_name = (f"n{num_agents}_epi{ii:02}_reward{rewards_list[ii]:.4f}"
                      f"_dist{box_final_dists[ii]:.4f}_sr{safe_rate:.0f}")
        video_path = videos_dir / f"{stamp_str}_{video_name}.gif"
        env.render_video(rollout, video_path, is_unsafe, {}, dpi=args.dpi)
        n_videos += 1
    print(f"Generated {n_videos} videos in {videos_dir}")


def main():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--path", type=str, default='logs/VMASReverseTransport/informarl_subgoal/seed0_416171038_XGOW')

    # manifold (ATACOM) parameters
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--viab-gain", type=float, default=0.5)
    parser.add_argument("--err-gain", type=float, default=30.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--g-act-thresh", type=float, default=0.02)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--n-lookahead", type=int, default=0)
    parser.add_argument("--w-slack", type=float, default=10.0)

    # test parameters
    parser.add_argument("--no-video", action="store_true", default=True)
    parser.add_argument("--max-videos", type=int, default=5)
    parser.add_argument("--epi", type=int, default=100)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true", default=False)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--subgoal-interval", type=int, default=8)
    parser.add_argument("--relative-subgoal", action="store_true", default=True)
    parser.add_argument("--max-delta", type=float, default=0.3)

    # default
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=100)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()
