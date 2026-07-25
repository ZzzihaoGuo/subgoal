"""Multi-episode eval for LidarAnt: safe_rate + success_rate, and render episode 0 to GIF."""
import os, sys, argparse, yaml, pickle
import numpy as np, jax, jax.numpy as jnp, jax.random as jr
sys.path.insert(0, "/home/a5l/zihao1996.a5l/project/subgoal")
from dgppo.env import ENV, LIDAR_ENVS
from dgppo.env.lidar_env.lidar_ant import LidarAnt
ENV['LidarAnt'] = LidarAnt; LIDAR_ENVS.add('LidarAnt')
from dgppo.env import make_env
from dgppo.algo import make_algo

def main(args):
    config = yaml.load(open(os.path.join(args.path, "config.yaml")), Loader=yaml.UnsafeLoader)
    env = make_env(env_id=config.env, num_agents=config.num_agents, num_obs=0,
                   n_rays=config.n_rays, max_step=args.max_step or config.max_step)
    step = args.step or max(int(m) for m in os.listdir(os.path.join(args.path, "models")) if m.isdigit())
    print("checkpoint step:", step)
    algo = make_algo(algo=config.algo, env=env, node_dim=env.node_dim, edge_dim=env.edge_dim,
        state_dim=env.state_dim, action_dim=env.action_dim, n_agents=env.num_agents,
        cost_weight=config.cost_weight, actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers, Vh_gnn_layers=getattr(config,"Vh_gnn_layers",1),
        lr_actor=config.lr_actor, lr_Vl=config.lr_Vl, max_grad_norm=2.0, seed=config.seed,
        use_rnn=config.use_rnn, rnn_layers=config.rnn_layers, use_lstm=config.use_lstm,
        area_size=env.area_size, use_relative_subgoal=getattr(config,"relative_subgoal",True), max_delta=getattr(config,"max_delta",2.5))
    algo.load(os.path.join(args.path, "models"), step)
    env.init_manifold(k=config.topk, K=config.viab_gain, Kc=config.err_gain, alpha_max=config.alpha_max,
        g_act_thresh=config.g_act_thresh, safety_margin=config.safety_margin,
        n_lookahead=config.n_lookahead, w_slack=config.w_slack)
    N = env.num_agents; SI = config.subgoal_interval; reach = float(env.params["dist2goal"])

    def rollout(key):
        g0 = env.reset(key); sg0 = env.get_agent_goals(g0); s0 = env.manifold_init_slack(g0)
        goals = env.get_agent_goals(g0)[:, :2]
        def body(carry, k):
            graph, rnn, sg, cnt, s = carry
            upd = (cnt % SI == 0)
            new_sg, new_rnn = jax.lax.cond(upd, lambda _: algo.act(graph, rnn), lambda _: (sg, rnn), operand=None)
            last = (env.max_episode_steps - cnt) <= SI
            nominal = env.u_ref(graph, target_pos=new_sg, is_final_goal=last)
            action, _, s2, _ = env.get_manifold_action(graph, u_ref=nominal, s_all=s)
            ng, r, c, d, _ = env.step(graph, env.clip_action(action))
            nav = graph.type_states(0, N)[:, :2]
            unsafe = jnp.any(c >= 0.0)                                     # collision this step
            return (ng, new_rnn, new_sg, cnt+1, s2), (nav, new_sg[:, :2], graph.env_states.data.qpos, unsafe)
        init = (g0, algo.init_rnn_state, sg0, 0, s0)
        _, (nav, subgoal, qpos, unsafe) = jax.lax.scan(body, init, jr.split(key, env.max_episode_steps))
        final_d2g = jnp.linalg.norm(nav[-1] - goals, axis=-1)             # (N,)
        return final_d2g, unsafe.any(), nav, subgoal, qpos, goals

    roll = jax.jit(rollout)
    finals, ep_unsafe = [], []
    saved = None
    for e in range(args.epi):
        fd, un, nav, sg, qpos, goals = roll(jr.PRNGKey(1000 + e))
        finals.append(np.array(fd)); ep_unsafe.append(bool(un))
        if e == 0:
            saved = dict(nav=np.array(nav), subgoal=np.array(sg), qpos=np.array(qpos),
                         goals=np.array(goals), car_radius=float(env.params["car_radius"]),
                         area=float(env.area_size), dist2goal=reach)
    finals = np.stack(finals)                                            # (epi, N)
    reached = finals < reach
    safe_rate = 1.0 - np.mean(ep_unsafe)
    success_agent = reached.mean()
    success_epi = reached.all(axis=1).mean()
    print(f"\n=== LidarAnt eval @ step {step} over {args.epi} episodes ===")
    print(f"safe_rate (episode all-safe)     : {100*safe_rate:.1f}%   (collision disabled -> trivially safe)")
    print(f"success_rate (per agent reached) : {100*success_agent:.1f}%  (final d2g < {reach})")
    print(f"success_rate (per episode all)   : {100*success_epi:.1f}%")
    print(f"final d2g: mean {finals.mean():.2f}  median {np.median(finals):.2f}")
    pickle.dump(saved, open(os.path.join(args.path, f"eval_{step}.pkl"), "wb"))
    print("saved episode-0 trajectory for rendering:", os.path.join(args.path, f"eval_{step}.pkl"))
    return step

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True); p.add_argument("--step", type=int, default=None)
    p.add_argument("--max-step", type=int, default=None); p.add_argument("--epi", type=int, default=20)
    main(p.parse_args())
