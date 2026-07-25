"""LidarAnt eval: metrics + GIF from ONE rollout.

Replaces eval_metrics.py + eval_render.py, which duplicated the env/algo setup and
the rollout scan and then wrote the same eval_<step>.pkl from different seeds.

    # metrics over 20 episodes + GIF of episode 0 (needs GPU/MJX)
    python eval.py --path logs/LidarAnt/informarl_subgoal/seed0_xxx --step 1000 --epi 20

    # re-render an existing dump -- pure numpy/matplotlib, no jax, fine on a CPU node
    python eval.py --pkl logs/.../eval_1000.pkl --out foo.gif

The jax/mjx imports live inside rollout_episodes() so that --pkl never touches them.
"""
import os, sys, argparse, yaml, pickle
import numpy as np, mujoco

REPO = "/home/a5l/zihao1996.a5l/project/subgoal"
sys.path.insert(0, REPO)
ANT_XML = os.path.join(REPO, "dgppo/env/lidar_env/ant.xml")


# ----------------------------------------------------------------------------- rollout
def rollout_episodes(path, step=None, max_step=None, epi=20, seed0=1000):
    """Run `epi` episodes of the trained hierarchical policy (+ u_ref + manifold).

    Returns (metrics, traj, step); traj is episode 0, which is also the one rendered,
    so the GIF always shows a trajectory that the metrics actually counted.
    """
    import jax, jax.numpy as jnp, jax.random as jr
    from dgppo.env import ENV, LIDAR_ENVS
    from dgppo.env.lidar_env.lidar_ant import LidarAnt
    ENV['LidarAnt'] = LidarAnt; LIDAR_ENVS.add('LidarAnt')
    from dgppo.env import make_env
    from dgppo.algo import make_algo

    config = yaml.load(open(os.path.join(path, "config.yaml")), Loader=yaml.UnsafeLoader)
    env = make_env(env_id=config.env, num_agents=config.num_agents, num_obs=0,
                   n_rays=config.n_rays, max_step=max_step or config.max_step)
    models = os.path.join(path, "models")
    step = step or max(int(m) for m in os.listdir(models) if m.isdigit())
    print("checkpoint step:", step)

    # NOTE: area_size and max_delta MUST come from the env/config -- make_algo's defaults
    # (1.5 / area_size/4) silently clamp every subgoal and produce bogus results.
    algo = make_algo(algo=config.algo, env=env, node_dim=env.node_dim, edge_dim=env.edge_dim,
        state_dim=env.state_dim, action_dim=env.action_dim, n_agents=env.num_agents,
        cost_weight=config.cost_weight, actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers, Vh_gnn_layers=getattr(config, "Vh_gnn_layers", 1),
        lr_actor=config.lr_actor, lr_Vl=config.lr_Vl, max_grad_norm=2.0, seed=config.seed,
        use_rnn=config.use_rnn, rnn_layers=config.rnn_layers, use_lstm=config.use_lstm,
        area_size=env.area_size, use_relative_subgoal=getattr(config, "relative_subgoal", True),
        max_delta=getattr(config, "max_delta", 2.5))
    algo.load(models, step)
    N = env.num_agents; reach = float(env.params["dist2goal"])
    # Hierarchical algos emit a SUBGOAL from act() and need u_ref + the manifold filter to turn
    # it into an action. Flat algos (dgppo, informarl, ...) emit the action directly and bring
    # their own learned CBF, so no manifold -- and a train.py config has none of its params.
    hierarchical = "subgoal" in str(config.algo)
    if hierarchical:
        env.init_manifold(k=config.topk, K=config.viab_gain, Kc=config.err_gain,
            alpha_max=config.alpha_max, g_act_thresh=config.g_act_thresh,
            safety_margin=config.safety_margin, n_lookahead=config.n_lookahead, w_slack=config.w_slack)
        SI = config.subgoal_interval
    print(f"algo={config.algo}  ({'hierarchical' if hierarchical else 'flat'})")

    def rollout_hier(key):
        g0 = env.reset(key); sg0 = env.get_agent_goals(g0); s0 = env.manifold_init_slack(g0)
        goals = env.get_agent_goals(g0)[:, :2]

        def body(carry, k):
            graph, rnn, sg, cnt, s = carry
            upd = (cnt % SI == 0)                       # high level fires every SI steps
            new_sg, new_rnn = jax.lax.cond(
                upd, lambda _: algo.act(graph, rnn), lambda _: (sg, rnn), operand=None)
            last = (env.max_episode_steps - cnt) <= SI
            nominal = env.u_ref(graph, target_pos=new_sg, is_final_goal=last)
            action, _, s2, _ = env.get_manifold_action(graph, u_ref=nominal, s_all=s)
            ng, r, c, d, _ = env.step(graph, env.clip_action(action))
            log = (graph.type_states(0, N)[:, :2],      # (N,2) nav xy
                   new_sg[:, :2],                       # (N,2) subgoal
                   graph.env_states.data.qpos,          # (N,nq) ant joints, for the 3D skeleton
                   jnp.any(c >= 0.0))                   # collision this step
            return (ng, new_rnn, new_sg, cnt + 1, s2), log

        init = (g0, algo.init_rnn_state, sg0, 0, s0)
        _, (nav, subgoal, qpos, unsafe) = jax.lax.scan(body, init, jr.split(key, env.max_episode_steps))
        return jnp.linalg.norm(nav[-1] - goals, axis=-1), unsafe.any(), nav, subgoal, qpos, goals

    def rollout_flat(key):
        g0 = env.reset(key)
        goals = env.get_agent_goals(g0)[:, :2]

        def body(carry, k):
            graph, rnn = carry
            action, new_rnn = algo.act(graph, rnn)      # the action itself, every env step
            ng, r, c, d, _ = env.step(graph, env.clip_action(action))
            log = (graph.type_states(0, N)[:, :2], graph.env_states.data.qpos, jnp.any(c >= 0.0))
            return (ng, new_rnn), log

        _, (nav, qpos, unsafe) = jax.lax.scan(
            body, (g0, algo.init_rnn_state), jr.split(key, env.max_episode_steps))
        return jnp.linalg.norm(nav[-1] - goals, axis=-1), unsafe.any(), nav, None, qpos, goals

    rollout = rollout_hier if hierarchical else rollout_flat

    roll = jax.jit(rollout)
    finals, ep_unsafe, traj = [], [], None
    for e in range(epi):
        fd, un, nav, sg, qpos, goals = roll(jr.PRNGKey(seed0 + e))
        finals.append(np.array(fd)); ep_unsafe.append(bool(un))
        if e == 0:
            traj = dict(nav=np.array(nav), qpos=np.array(qpos), goals=np.array(goals),
                        subgoal=None if sg is None else np.array(sg),   # flat algos have none
                        car_radius=float(env.params["car_radius"]),
                        area=float(env.area_size), dist2goal=reach)
    finals = np.stack(finals)                                    # (epi, N)
    reached = finals < reach
    metrics = dict(step=step, epi=epi, reach=reach,
                   safe_rate=1.0 - float(np.mean(ep_unsafe)),
                   success_agent=float(reached.mean()),
                   success_epi=float(reached.all(axis=1).mean()),
                   d2g_mean=float(finals.mean()), d2g_median=float(np.median(finals)))
    return metrics, traj, step


def print_metrics(m):
    print(f"\n=== LidarAnt eval @ step {m['step']} over {m['epi']} episodes ===")
    print(f"safe_rate (episode all-safe)     : {100*m['safe_rate']:.1f}%   (collision disabled -> trivially safe)")
    print(f"success_rate (per agent reached) : {100*m['success_agent']:.1f}%  (final d2g < {m['reach']})")
    print(f"success_rate (per episode all)   : {100*m['success_epi']:.1f}%")
    print(f"final d2g: mean {m['d2g_mean']:.2f}  median {m['d2g_median']:.2f}")


# ----------------------------------------------------------------------------- render
def subgoal_change_frames(subgoal):
    """Frames at which the high-level policy emitted a NEW subgoal (it is held constant
    for subgoal_interval steps in between). Used to fade out the subgoal history."""
    d = np.abs(subgoal[1:] - subgoal[:-1]).max(axis=(1, 2))
    return np.concatenate([[0], np.nonzero(d > 1e-6)[0] + 1])


def render(traj, step, path, hist=4, fps=15):
    """traj: the dict from rollout_episodes, or a path to a pickled one. No jax needed."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    D = dict(pickle.load(open(traj, "rb"))) if isinstance(traj, str) else traj
    qpos, nav, subgoal, goals = D["qpos"], D["nav"], D["subgoal"], D["goals"]
    R, AREA = D["car_radius"], D["area"]; T, N, _ = nav.shape
    mj = mujoco.MjModel.from_xml_path(ANT_XML)
    d = mujoco.MjData(mj)
    GT = np.array([mj.geom_type[g] for g in range(mj.ngeom)])
    GS = np.array([mj.geom_size[g] for g in range(mj.ngeom)])
    COL = ['#1f5fa8', '#d9541e', '#2ca25f', '#8e44ad', '#c0392b']
    # flat algos (dgppo, ...) have no subgoal layer -- draw the trajectories without it
    CHG = subgoal_change_frames(subgoal) if subgoal is not None else None
    ZTOP = 1.0                                 # 3D z limit; markers are drawn on the ground (z=0)
    # safety envelope: the ant's leg-reach circle (car_radius), extruded into a vertical
    # cylinder -- agent-agent safe iff centre distance >= 2R, agent-obstacle iff >= R.
    TH = np.linspace(0, 2*np.pi, 61)
    CX, CY = R*np.cos(TH), R*np.sin(TH)
    ZCYL = 0.55                                # cylinder cap height, ~ above the torso
    imgs = []
    for fi in range(0, T, max(1, T//90)):
        # last `hist` issued subgoals, oldest first -> alpha ramps from faint to solid
        h = CHG[CHG <= fi][-hist:] if CHG is not None else []
        alphas = np.linspace(0.15, 1.0, len(h))

        fig = plt.figure(figsize=(9.6, 4.6))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for i in range(N):
            # --- safety cylinder, drawn first so the skeleton overlays it ---
            cx, cy = nav[fi, i, 0] + CX, nav[fi, i, 1] + CY
            ax.plot(cx, cy, np.zeros_like(cx), '-', c=COL[i % 5], lw=1.3, alpha=0.55)
            ax.plot(cx, cy, np.full_like(cx, ZCYL), '-', c=COL[i % 5], lw=0.9, alpha=0.25)
            for k in range(0, len(TH)-1, 15):
                ax.plot([cx[k], cx[k]], [cy[k], cy[k]], [0.0, ZCYL], '-', c=COL[i % 5], lw=0.7, alpha=0.22)

            # --- ant skeleton at its nav position ---
            d.qpos[:] = qpos[fi, i]; mujoco.mj_forward(mj, d)
            off = nav[fi, i] - d.subtree_com[0][:2]
            ztorso = d.subtree_com[0][2]
            for g in range(mj.ngeom):
                if GT[g] == 3:                                            # capsule -> segment
                    hl = GS[g][1]; ez = d.geom_xmat[g].reshape(3, 3)[:, 2]
                    a, b = d.geom_xpos[g] - ez*hl, d.geom_xpos[g] + ez*hl
                    ax.plot([a[0]+off[0], b[0]+off[0]], [a[1]+off[1], b[1]+off[1]], [a[2], b[2]],
                            c=COL[i % 5], lw=2)
                elif GT[g] == 2:                                          # sphere -> torso dot
                    ax.scatter([d.geom_xpos[g][0]+off[0]], [d.geom_xpos[g][1]+off[1]],
                               [d.geom_xpos[g][2]], s=70, c=COL[i % 5])
            # --- travelled path, on the ground ---
            ax.plot(nav[:fi+1, i, 0], nav[:fi+1, i, 1], np.zeros(fi+1), '-', c=COL[i % 5], lw=1.0, alpha=0.45)
            # --- subgoal history, fading in with age ---
            for t, al in zip(h, alphas):
                sx, sy = subgoal[t, i]
                ax.scatter([sx], [sy], [0.0], marker='D', facecolors='none', edgecolors=COL[i % 5],
                           s=42, lw=1.4, alpha=al, depthshade=False)
                ax.plot([sx, sx], [sy, sy], [0.0, 0.16], '-', c=COL[i % 5], lw=0.8, alpha=al*0.7)
            if subgoal is not None:
                ax.plot([nav[fi, i, 0], subgoal[fi, i, 0]], [nav[fi, i, 1], subgoal[fi, i, 1]],
                        [ztorso, 0.0], '--', c=COL[i % 5], lw=1.2, alpha=0.9)
            ax.scatter([goals[i, 0]], [goals[i, 1]], [0.0], marker='*', c=COL[i % 5], s=140,
                       edgecolor='k', lw=0.4, depthshade=False)
        ax.set_xlim(0, AREA); ax.set_ylim(0, AREA); ax.set_zlim(0, ZTOP)
        ax.set_box_aspect([AREA, AREA, ZTOP]); ax.view_init(elev=42, azim=-70)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_title(f"LidarAnt @ step {step}   frame {fi}/{T}", fontsize=10, pad=0)

        ax2 = fig.add_subplot(1, 2, 2)
        for i in range(N):
            ax2.plot(nav[:fi+1, i, 0], nav[:fi+1, i, 1], '-', c=COL[i % 5], lw=1.4, alpha=0.6)
            ax2.add_patch(plt.Circle(nav[fi, i], R, color=COL[i % 5], alpha=0.12))
            for t, al in zip(h, alphas):
                ax2.scatter(*subgoal[t, i], marker='D', facecolor='none', edgecolor=COL[i % 5],
                            s=55, lw=1.6, zorder=6, alpha=al)
            if subgoal is not None:
                ax2.plot([nav[fi, i, 0], subgoal[fi, i, 0]], [nav[fi, i, 1], subgoal[fi, i, 1]],
                         '--', c=COL[i % 5], lw=1.0, alpha=0.9, zorder=4)
            ax2.scatter(*nav[fi, i], c=COL[i % 5], s=55, zorder=6, edgecolor='k', lw=0.5)
            ax2.scatter(*goals[i], marker='*', c=COL[i % 5], s=200, zorder=5, edgecolor='k', lw=0.5)
        d2g = np.linalg.norm(nav[fi] - goals, axis=-1)
        ax2.set_xlim(0, AREA); ax2.set_ylim(0, AREA); ax2.set_aspect('equal'); ax2.grid(alpha=.3)
        legend = "dot=ant  diamond=subgoal (faded=older)  star=goal" if subgoal is not None \
            else "dot=ant  star=goal  (flat policy: no subgoal layer)"
        ax2.set_title(f"{legend}\nmean d2g = {d2g.mean():.2f}", fontsize=10)
        fig.subplots_adjust(left=0.0, right=0.96, bottom=0.06, top=0.90, wspace=0.02)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        imgs.append(img.copy()); plt.close(fig)
    pil = [Image.fromarray(im) for im in imgs]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(1000/fps), loop=0, optimize=True)
    print("wrote", path, len(imgs), "frames", os.path.getsize(path)//1024, "KB")


# ----------------------------------------------------------------------------- cli
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=None, help="training log dir (contains config.yaml + models/)")
    p.add_argument("--step", type=int, default=None, help="checkpoint step; default = latest")
    p.add_argument("--max-step", type=int, default=None)
    p.add_argument("--epi", type=int, default=20)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--pkl", default=None, help="re-render an existing eval_*.pkl (no jax needed)")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--hist", type=int, default=4, help="how many past subgoals to keep, fading out")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.pkl:
        traj = args.pkl
        step = args.step if args.step is not None else int(os.path.basename(args.pkl).split("_")[-1].split(".")[0])
    else:
        assert args.path, "need --path (run the policy) or --pkl (re-render a dump)"
        metrics, traj, step = rollout_episodes(args.path, args.step, args.max_step, args.epi, args.seed0)
        print_metrics(metrics)
        dump = os.path.join(args.path, f"eval_{step}.pkl")
        pickle.dump(traj, open(dump, "wb"))
        with open(os.path.join(args.path, f"eval_{step}.txt"), "w") as f:
            f.write("\n".join(f"{k}: {v}" for k, v in metrics.items()) + "\n")
        print("saved", dump, "and", dump.replace(".pkl", ".txt"))

    if not args.no_render:
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "gifs",
                                       f"ant_eval_step{step}.gif")
        render(traj, step, out, hist=args.hist, fps=args.fps)
