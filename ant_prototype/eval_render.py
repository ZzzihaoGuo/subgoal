"""Load a trained LidarAnt checkpoint, run ONE episode with the learned high-level policy
(+ u_ref + manifold), and render a GIF (3D ant skeleton + top-down nav with subgoals).

Usage: python eval_render.py --path logs/LidarAnt/informarl_subgoal/seed0_..._NAME [--step N]

Re-render only (no JAX / no MJX -- works fine on a CPU-only node, ~seconds):
    python eval_render.py --pkl logs/.../eval_1000.pkl
"""
import os, sys, glob, argparse, yaml, pickle
import numpy as np, mujoco
sys.path.insert(0, "/home/a5l/zihao1996.a5l/project/subgoal")

def collect(args):
    # heavy imports live here so that --pkl re-rendering never touches jax/mjx
    import jax, jax.numpy as jnp, jax.random as jr
    from dgppo.env import ENV, LIDAR_ENVS
    from dgppo.env.lidar_env.lidar_ant import LidarAnt, AntEnvState
    ENV['LidarAnt'] = LidarAnt; LIDAR_ENVS.add('LidarAnt')
    from dgppo.env import make_env
    from dgppo.algo import make_algo

    with open(os.path.join(args.path, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    env = make_env(env_id=config.env, num_agents=config.num_agents, num_obs=0,
                   n_rays=config.n_rays, max_step=args.max_step or config.max_step)
    model_path = os.path.join(args.path, "models")
    step = args.step or max(int(m) for m in os.listdir(model_path) if m.isdigit())
    print("loading step", step)
    algo = make_algo(algo=config.algo, env=env, node_dim=env.node_dim, edge_dim=env.edge_dim,
        state_dim=env.state_dim, action_dim=env.action_dim, n_agents=env.num_agents,
        cost_weight=config.cost_weight, actor_gnn_layers=config.actor_gnn_layers,
        Vl_gnn_layers=config.Vl_gnn_layers, Vh_gnn_layers=getattr(config,"Vh_gnn_layers",1),
        lr_actor=config.lr_actor, lr_Vl=config.lr_Vl, max_grad_norm=2.0, seed=config.seed,
        use_rnn=config.use_rnn, rnn_layers=config.rnn_layers, use_lstm=config.use_lstm,
        area_size=env.area_size, use_relative_subgoal=getattr(config,"relative_subgoal",True), max_delta=getattr(config,"max_delta",2.5))
    algo.load(model_path, step)
    env.init_manifold(k=config.topk, K=config.viab_gain, Kc=config.err_gain, alpha_max=config.alpha_max,
        g_act_thresh=config.g_act_thresh, safety_margin=config.safety_margin,
        n_lookahead=config.n_lookahead, w_slack=config.w_slack)
    N = env.num_agents; SI = config.subgoal_interval

    def rollout(key):
        g0 = env.reset(key); sg0 = env.get_agent_goals(g0); s0 = env.manifold_init_slack(g0)
        def body(carry, k):
            graph, rnn, sg, cnt, s = carry
            upd = (cnt % SI == 0)
            new_sg, new_rnn = jax.lax.cond(upd,
                lambda _: algo.act(graph, rnn), lambda _: (sg, rnn), operand=None)
            last = (env.max_episode_steps - cnt) <= SI
            nominal = env.u_ref(graph, target_pos=new_sg, is_final_goal=last)
            action, _, s2, _ = env.get_manifold_action(graph, u_ref=nominal, s_all=s)
            ng, r, c, d, _ = env.step(graph, env.clip_action(action))
            log = (graph.env_states.data.qpos,                 # (N, nq) ant joints
                   graph.type_states(0, N)[:, :2],             # (N,2) nav xy
                   new_sg[:, :2])                              # (N,2) subgoal
            return (ng, new_rnn, new_sg, cnt+1, s2), log
        init = (g0, algo.init_rnn_state, sg0, 0, s0)
        _, logs = jax.lax.scan(body, init, jr.split(key, env.max_episode_steps))
        return logs, env.get_agent_goals(g0)[:, :2]

    (qpos, nav, subgoal), goals = jax.jit(rollout)(jr.PRNGKey(args.seed))
    out = dict(qpos=np.array(qpos), nav=np.array(nav), subgoal=np.array(subgoal),
               goals=np.array(goals), car_radius=float(env.params["car_radius"]),
               area=float(env.area_size), dist2goal=float(env.params["dist2goal"]))
    pickle.dump(out, open(os.path.join(args.path, f"eval_{step}.pkl"), "wb"))
    d2g = np.linalg.norm(out["nav"] - out["goals"][None], axis=-1)
    print(f"episode: start d2g {np.round(d2g[0],2)} -> end {np.round(d2g[-1],2)}  (reach<{out['dist2goal']})")
    return os.path.join(args.path, f"eval_{step}.pkl"), step

def subgoal_change_frames(subgoal):
    """Frames at which the high-level policy emitted a NEW subgoal (it is held constant
    for subgoal_interval steps in between). Used to fade out the subgoal history."""
    T = subgoal.shape[0]
    d = np.abs(subgoal[1:] - subgoal[:-1]).max(axis=(1, 2))
    return np.concatenate([[0], np.nonzero(d > 1e-6)[0] + 1])


def render(pkl, step, path, hist=4, fps=15):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    D = dict(pickle.load(open(pkl, "rb")))
    qpos, nav, subgoal, goals = D["qpos"], D["nav"], D["subgoal"], D["goals"]
    R, AREA = D["car_radius"], D["area"]; T, N, _ = nav.shape
    mj = mujoco.MjModel.from_xml_path("/home/a5l/zihao1996.a5l/project/subgoal/dgppo/env/lidar_env/ant.xml")
    d = mujoco.MjData(mj)
    GT = np.array([mj.geom_type[g] for g in range(mj.ngeom)]); GS = np.array([mj.geom_size[g] for g in range(mj.ngeom)])
    COL = ['#1f5fa8', '#d9541e', '#2ca25f', '#8e44ad', '#c0392b']
    CHG = subgoal_change_frames(subgoal)       # frames where a new subgoal was issued
    ZTOP = 1.0                                  # 3D z limit; markers are drawn on the ground (z=0)
    # safety envelope: the ant's leg-reach circle (car_radius), extruded into a vertical
    # cylinder -- agent-agent safe iff centre distance >= 2R, agent-obstacle iff >= R.
    TH = np.linspace(0, 2*np.pi, 61)
    CX, CY = R*np.cos(TH), R*np.sin(TH)
    ZCYL = 0.55                                 # cylinder cap height, ~ above the torso
    imgs = []
    for fi in range(0, T, max(1, T//90)):
        # last `hist` issued subgoals, oldest first -> alpha ramps from faint to solid
        h = CHG[CHG <= fi][-hist:]
        alphas = np.linspace(0.15, 1.0, len(h))

        fig = plt.figure(figsize=(9.6, 4.6))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for i in range(N):
            # --- safety cylinder (radius = car_radius) around the ant, drawn first so the
            #     skeleton overlays it: ground ring + cap ring + a few vertical struts ---
            cx, cy = nav[fi,i,0] + CX, nav[fi,i,1] + CY
            ax.plot(cx, cy, np.zeros_like(cx), '-', c=COL[i%5], lw=1.3, alpha=0.55)
            ax.plot(cx, cy, np.full_like(cx, ZCYL), '-', c=COL[i%5], lw=0.9, alpha=0.25)
            for k in range(0, len(TH)-1, 15):
                ax.plot([cx[k],cx[k]], [cy[k],cy[k]], [0.0, ZCYL], '-', c=COL[i%5], lw=0.7, alpha=0.22)

            # --- ant skeleton at its nav position ---
            d.qpos[:] = qpos[fi, i]; mujoco.mj_forward(mj, d)
            off = nav[fi, i] - d.subtree_com[0][:2]
            ztorso = d.subtree_com[0][2]
            for g in range(mj.ngeom):
                if GT[g] == 3:
                    hl = GS[g][1]; a = d.geom_xpos[g] - d.geom_xmat[g].reshape(3,3)[:,2]*hl
                    b = d.geom_xpos[g] + d.geom_xmat[g].reshape(3,3)[:,2]*hl
                    ax.plot([a[0]+off[0],b[0]+off[0]],[a[1]+off[1],b[1]+off[1]],[a[2],b[2]],c=COL[i%5],lw=2)
                elif GT[g] == 2:
                    ax.scatter([d.geom_xpos[g][0]+off[0]],[d.geom_xpos[g][1]+off[1]],[d.geom_xpos[g][2]],s=70,c=COL[i%5])
            # --- travelled path, on the ground ---
            ax.plot(nav[:fi+1,i,0], nav[:fi+1,i,1], np.zeros(fi+1), '-', c=COL[i%5], lw=1.0, alpha=0.45)
            # --- subgoal history, fading in with age; newest also gets the dashed link to the ant ---
            for t, al in zip(h, alphas):
                sx, sy = subgoal[t, i]
                ax.scatter([sx],[sy],[0.0], marker='D', facecolors='none', edgecolors=COL[i%5],
                           s=42, lw=1.4, alpha=al, depthshade=False)
                ax.plot([sx,sx],[sy,sy],[0.0, 0.16], '-', c=COL[i%5], lw=0.8, alpha=al*0.7)   # tiny stalk
            ax.plot([nav[fi,i,0], subgoal[fi,i,0]], [nav[fi,i,1], subgoal[fi,i,1]], [ztorso, 0.0],
                    '--', c=COL[i%5], lw=1.2, alpha=0.9)
            # --- final goal on the ground ---
            ax.scatter([goals[i,0]],[goals[i,1]],[0.0], marker='*', c=COL[i%5], s=140,
                       edgecolor='k', lw=0.4, depthshade=False)
        ax.set_xlim(0,AREA); ax.set_ylim(0,AREA); ax.set_zlim(0,ZTOP)
        ax.set_box_aspect([AREA,AREA,ZTOP]); ax.view_init(elev=42,azim=-70)
        ax.set_xticklabels([]);ax.set_yticklabels([]);ax.set_zticklabels([])
        ax.set_title(f"LidarAnt @ step {step}   frame {fi}/{T}", fontsize=10, pad=0)

        ax2 = fig.add_subplot(1, 2, 2)
        for i in range(N):
            ax2.plot(nav[:fi+1,i,0], nav[:fi+1,i,1], '-', c=COL[i%5], lw=1.4, alpha=0.6)
            ax2.add_patch(plt.Circle(nav[fi,i], R, color=COL[i%5], alpha=0.12))
            # subgoal history, same fade as the 3D panel
            for t, al in zip(h, alphas):
                ax2.scatter(*subgoal[t,i], marker='D', facecolor='none', edgecolor=COL[i%5],
                            s=55, lw=1.6, zorder=6, alpha=al)
            # dashed line agent -> its current subgoal (makes hierarchy legible)
            ax2.plot([nav[fi,i,0], subgoal[fi,i,0]], [nav[fi,i,1], subgoal[fi,i,1]],
                     '--', c=COL[i%5], lw=1.0, alpha=0.9, zorder=4)
            ax2.scatter(*nav[fi,i], c=COL[i%5], s=55, zorder=6, edgecolor='k', lw=0.5)          # agent
            ax2.scatter(*goals[i], marker='*', c=COL[i%5], s=200, zorder=5, edgecolor='k', lw=0.5)  # goal
        d2g = np.linalg.norm(nav[fi]-goals, axis=-1)
        ax2.set_xlim(0,AREA); ax2.set_ylim(0,AREA); ax2.set_aspect('equal'); ax2.grid(alpha=.3)
        ax2.set_title(f"dot=ant  diamond=subgoal (faded=older)  star=goal\nmean d2g = {d2g.mean():.2f}",
                      fontsize=10)
        fig.subplots_adjust(left=0.0, right=0.96, bottom=0.06, top=0.90, wspace=0.02)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(),dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1]+(4,))[...,:3]
        imgs.append(img.copy()); plt.close(fig)
    pil = [Image.fromarray(im) for im in imgs]
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(1000/fps), loop=0, optimize=True)
    print("wrote", path, len(imgs), "frames", os.path.getsize(path)//1024, "KB")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=None); p.add_argument("--step", type=int, default=None)
    p.add_argument("--max-step", type=int, default=None); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pkl", default=None, help="re-render an existing eval_*.pkl (no jax/mjx needed)")
    p.add_argument("--hist", type=int, default=4, help="how many past subgoals to keep, fading out")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.pkl:
        pkl = args.pkl
        step = args.step if args.step is not None else \
            int(os.path.basename(pkl).split("_")[-1].split(".")[0])
    else:
        assert args.path, "need --path (collect) or --pkl (re-render)"
        pkl, step = collect(args)
    out = args.out or os.path.join(os.path.dirname(__file__), "gifs", f"ant_eval_step{step}.gif")
    render(pkl, step, out, hist=args.hist, fps=args.fps)
