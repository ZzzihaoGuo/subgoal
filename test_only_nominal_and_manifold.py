"""
Test ONLY nominal policy and nominal+manifold (no learned actor).

Per-agent nominal target:
  - GOAL_ASSIGNMENT == 'target' (LidarTarget, LidarBicycleTarget, LinearDrone, CrazyFlie):
        agent_i heads directly to goal_i.
  - GOAL_ASSIGNMENT == 'spread' / 'line' (LidarSpread, LidarLine):
        agent heads to its CLOSEST goal.

Reports three rates over `--epi` episodes:
    safe_rate   — per-agent fraction with no collision over the whole episode
    reach_rate  — per-agent fraction whose final-step distance to (assigned) goal < dist_thresh
    success     — per-agent (safe AND reached)
"""
import argparse
import logging
import os
import pathlib
import time

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
from loguru import logger
logger.disable("jaxproxqp")

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dgppo.env import make_env
from dgppo.trainer.data import Rollout


# ------------------------------------------------------------
# Per-agent target selection
# ------------------------------------------------------------
def per_agent_targets(env, graph):
    """Return (num_agents, pos_dim) target position for each agent."""
    goals = env.get_agent_goals(graph)                              # (num_goals, pos_dim)
    pos_dim = goals.shape[-1]
    agent_pos = graph.type_states(type_idx=0, n_type=env.num_agents)[:, :pos_dim]
    mode = getattr(env, 'GOAL_ASSIGNMENT', 'spread')
    if mode == 'target':
        return goals[:env.num_agents]
    # spread / line: each agent picks its closest goal
    d = jnp.linalg.norm(
        agent_pos[:, None, :] - goals[None, :, :], axis=-1
    )                                                                # (n_agents, n_goals)
    nearest = jnp.argmin(d, axis=-1)
    return goals[nearest]


def final_dist_per_agent(env, final_graph):
    """Distance from each agent to its assigned goal at final step."""
    goals = env.get_agent_goals(final_graph)
    pos_dim = goals.shape[-1]
    agent_pos = final_graph.type_states(type_idx=0, n_type=env.num_agents)[:, :pos_dim]
    mode = getattr(env, 'GOAL_ASSIGNMENT', 'spread')
    if mode == 'target':
        return jnp.linalg.norm(agent_pos - goals[:env.num_agents], axis=-1)
    # spread / line: each agent matched to closest goal at final step
    d = jnp.linalg.norm(
        agent_pos[:, None, :] - goals[None, :, :], axis=-1
    )
    return d.min(axis=-1)


# ------------------------------------------------------------
# Rollout
# ------------------------------------------------------------
def make_rollout(env, use_manifold: bool):
    def rollout(key):
        init_graph = env.reset(key)
        init_s = (env.manifold_init_slack(init_graph)
                  if use_manifold else jnp.zeros(()))

        def body(carry, _):
            graph, s = carry
            tgt = per_agent_targets(env, graph)
            nominal = env.u_ref(graph, target_pos=tgt, is_final_goal=True)
            if use_manifold:
                action, _, s_new, _ = env.get_manifold_action(
                    graph, u_ref=nominal, s_all=s)
            else:
                action, s_new = nominal, s
            action = env.clip_action(action)

            cost = env.get_cost(graph)                          # (n_agents, n_cost)
            unsafe_t = jnp.any(cost >= 0.0, axis=-1)            # (n_agents,)

            next_graph = env.step(graph, action)[0]
            return (next_graph, s_new), unsafe_t

        (final_graph, _), unsafe_seq = jax.lax.scan(
            body, (init_graph, init_s), None, length=env.max_episode_steps
        )
        agent_unsafe_any = unsafe_seq.any(axis=0)               # (n_agents,)
        final_dist = final_dist_per_agent(env, final_graph)     # (n_agents,)
        return agent_unsafe_any, final_dist

    return rollout


def make_rollout_with_data(env, use_manifold: bool):
    """Like make_rollout but also collects the full Rollout for rendering."""
    def rollout(key):
        init_graph = env.reset(key)
        init_s = (env.manifold_init_slack(init_graph)
                  if use_manifold else jnp.zeros(()))

        def body(carry, _):
            graph, s = carry
            tgt = per_agent_targets(env, graph)
            nominal = env.u_ref(graph, target_pos=tgt, is_final_goal=True)
            if use_manifold:
                action, _, s_new, _ = env.get_manifold_action(
                    graph, u_ref=nominal, s_all=s)
            else:
                action, s_new = nominal, s
            action = env.clip_action(action)

            cost = env.get_cost(graph)
            unsafe_t = jnp.any(cost >= 0.0, axis=-1)

            next_graph, reward, cost_step, done, _info = env.step(graph, action)
            return ((next_graph, s_new),
                    (graph, action, reward, cost_step, done, next_graph, unsafe_t))

        (_final_graph, _), out = jax.lax.scan(
            body, (init_graph, init_s), None, length=env.max_episode_steps
        )
        graphs, actions, rewards, costs, dones, next_graphs, unsafe_seq = out
        rollout_data = Rollout(
            graph=graphs,
            actions=actions,
            rnn_states=jnp.zeros((env.max_episode_steps,)),
            rewards=rewards,
            costs=costs,
            dones=dones,
            log_pis=None,
            next_graph=next_graphs,
            sparse_rewards=None,
            dist2goal=None,
        )
        return rollout_data, unsafe_seq

    return rollout


def render_gifs(env, label, use_manifold, n_gifs, seed, out_dir, dpi):
    print(f"\n=== Rendering {n_gifs} GIFs ({label}) -> {out_dir} ===")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_jit = jax.jit(make_rollout_with_data(env, use_manifold))
    keys = jr.split(jr.PRNGKey(seed), n_gifs)

    t0 = time.time()
    rollout_data, unsafe_seq = rollout_jit(keys[0])
    rollout_data.actions.block_until_ready()
    print(f"  warmup ({time.time() - t0:.2f}s)")

    reach_thresh = float(env.params.get("dist2goal", 0.01))
    tag = "manifold" if use_manifold else "nominal"

    for i in range(n_gifs):
        if i > 0:
            rollout_data, unsafe_seq = rollout_jit(keys[i])
        unsafe_np = np.asarray(unsafe_seq)              # (T, n_agents)
        n_safe = int((~unsafe_np.any(axis=0)).sum())

        # final-step reach (works for both 'target' and 'spread/line' modes)
        last_states = np.asarray(rollout_data.next_graph.states[-1])
        pos_dim = np.asarray(env.get_agent_goals(env.reset(keys[i]))).shape[-1]
        last_agents = last_states[:env.num_agents, :pos_dim]
        last_goals = last_states[env.num_agents:env.num_agents + env.num_goals, :pos_dim]
        mode = getattr(env, "GOAL_ASSIGNMENT", "spread")
        if mode == "target":
            final_dist = np.linalg.norm(last_agents - last_goals[:env.num_agents], axis=-1)
        else:
            d = np.linalg.norm(last_agents[:, None, :] - last_goals[None, :, :], axis=-1)
            final_dist = d.min(axis=-1)
        n_reach = int((final_dist < reach_thresh).sum())

        gif_name = (f"{tag}_epi{i:02d}_safe{n_safe}-{env.num_agents}_"
                    f"reach{n_reach}-{env.num_agents}.gif")
        gif_path = out_dir / gif_name
        env.render_video(rollout_data, gif_path, Ta_is_unsafe=unsafe_np,
                         viz_opts={}, dpi=dpi)
        print(f"  [{i+1}/{n_gifs}] safe={n_safe}/{env.num_agents} "
              f"reach={n_reach}/{env.num_agents} -> {gif_path.name}")


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
def evaluate(env, label, use_manifold, n_epi, seed, reach_thresh=None):
    print(f"\n=== {label} ===")
    rollout = make_rollout(env, use_manifold)
    rollout_jit = jax.jit(rollout)

    keys = jr.split(jr.PRNGKey(seed), n_epi)

    t0 = time.time()
    u0, d0 = rollout_jit(keys[0])
    u0.block_until_ready()
    print(f"  warmup ({time.time() - t0:.2f}s)")

    safe_per_epi = [np.asarray(~u0)]
    reach_per_epi = []
    dist_thresh = (float(reach_thresh) if reach_thresh is not None
                   else float(env.params.get("dist2goal", 0.01)))
    print(f"  reach_thresh={dist_thresh}")
    reach_per_epi.append(np.asarray(d0) < dist_thresh)

    log_every = max(n_epi // 20, 1)
    t0 = time.time()
    for i in range(1, n_epi):
        unsafe_any, final_dist = rollout_jit(keys[i])
        safe_per_epi.append(np.asarray(~unsafe_any))
        reach_per_epi.append(np.asarray(final_dist) < dist_thresh)
        if (i + 1) % log_every == 0 or (i + 1) == n_epi:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_epi - i - 1)
            print(f"  [{i+1}/{n_epi}] elapsed={elapsed:.1f}s eta={eta:.1f}s")

    safe_arr = np.stack(safe_per_epi)                            # (n_epi, n_agents) bool
    reach_arr = np.stack(reach_per_epi)
    success_arr = safe_arr & reach_arr

    # per-agent mean over all (epi, agent) entries
    safe_m = float(safe_arr.mean())
    reach_m = float(reach_arr.mean())
    succ_m = float(success_arr.mean())

    print(f"  safe_rate    = {safe_m*100:6.2f}%")
    print(f"  reach_rate   = {reach_m*100:6.2f}%")
    print(f"  success_rate = {succ_m*100:6.2f}%")
    return {
        "label": label,
        "safe": safe_m,
        "reach": reach_m,
        "success": succ_m,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True,
                        help="LidarSpread | LidarTarget | LidarLine | "
                             "LidarBicycleTarget | LinearDrone | CrazyFlie")
    parser.add_argument("-n", "--num-agents", type=int, default=3)
    parser.add_argument("--obs", type=int, default=3)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=1000)
    parser.add_argument("--reach-thresh", type=float, default=None,
                        help="Override env.params['dist2goal'] for success/reach判定")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["nominal", "manifold", "both"])
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--full-observation", action="store_true", default=False)

    # GIF rendering toggle
    parser.add_argument("--render-gifs", type=int, default=0,
                        help="Number of GIFs to render (0 = off).")
    parser.add_argument("--gif-mode", type=str, default="manifold",
                        choices=["nominal", "manifold"],
                        help="Which policy to render.")
    parser.add_argument("--gif-out", type=str, default=None,
                        help="Output dir for GIFs. Default: results_nominal_manifold/<env>_gifs")
    parser.add_argument("--dpi", type=int, default=100)

    # manifold (ATACOM) parameters
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--viab-gain", type=float, default=0.5)
    parser.add_argument("--err-gain", type=float, default=30.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--g-act-thresh", type=float, default=0.02)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--n-lookahead", type=int, default=0)
    parser.add_argument("--w-slack", type=float, default=10.0)

    args = parser.parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        max_step=args.max_step,
        full_observation=args.full_observation,
    )
    print(f"Env: {args.env}, n_agents={args.num_agents}, n_obs={args.obs}, "
          f"max_step={env.max_episode_steps}, "
          f"GOAL_ASSIGNMENT={getattr(env, 'GOAL_ASSIGNMENT', 'spread')}")

    if args.mode in ("manifold", "both"):
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

    results = []
    if args.mode in ("nominal", "both"):
        results.append(evaluate(env, "Nominal only", False, args.epi, args.seed,
                                reach_thresh=args.reach_thresh))
    if args.mode in ("manifold", "both"):
        results.append(evaluate(env, "Nominal + Manifold", True, args.epi, args.seed,
                                reach_thresh=args.reach_thresh))

    print("\n=== Summary ===")
    print(f"{'method':22s} {'safe':>10s} {'reach':>10s} {'success':>10s}")
    for r in results:
        s = f"{r['safe']*100:6.2f}"
        rc = f"{r['reach']*100:6.2f}"
        sc = f"{r['success']*100:6.2f}"
        print(f"{r['label']:22s} {s:>10s} {rc:>10s} {sc:>10s}")

    if args.render_gifs > 0:
        use_manifold = args.gif_mode == "manifold"
        if use_manifold and args.mode == "nominal":
            env.init_manifold(
                k=args.topk, K=args.viab_gain, Kc=args.err_gain,
                alpha_max=args.alpha_max, g_act_thresh=args.g_act_thresh,
                safety_margin=args.safety_margin,
                n_lookahead=args.n_lookahead, w_slack=args.w_slack,
            )
        gif_out = args.gif_out or f"results_nominal_manifold/{args.env}_gifs"
        label = "Nominal + Manifold" if use_manifold else "Nominal only"
        render_gifs(env, label, use_manifold, args.render_gifs,
                    args.seed, gif_out, args.dpi)


if __name__ == "__main__":
    main()
