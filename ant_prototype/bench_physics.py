"""Speed vs fidelity of the LidarAnt physics settings.

The stock gym ant.xml uses integrator="RK4" (4 dynamics evaluations per timestep) and the
C-MuJoCo accuracy defaults iterations=100 / ls_iterations=50, so ONE control step costs
FS(5) * 4 = 20 full forward-dynamics solves. That is why LidarAnt is ~220x slower per env
step than the bicycle envs. This script measures what the cheaper settings actually buy and
-- more importantly -- whether the open-loop trot, which was CEM-tuned under RK4, survives.

    python ant_prototype/bench_physics.py                 # full sweep
    python ant_prototype/bench_physics.py --batch 2048 --iters 20

RUN THIS BEFORE A LONG TRAINING RUN. If a config prints FAIL, do not train with it: set
LidarAnt.INTEGRATOR / SOLVER_ITER / SOLVER_LS_ITER back to the stock rk4 / 100 / 50.
"""
import argparse, os, sys, time

import numpy as np

REPO = "/home/a5l/zihao1996.a5l/project/subgoal"
sys.path.insert(0, REPO)

import jax
import jax.numpy as jnp

from dgppo.env.lidar_env.lidar_ant import LidarAnt

# (label, integrator, iterations, ls_iterations); the FIRST entry is the fidelity reference.
CONFIGS = [
    ("rk4 100/50  (stock)",   "rk4",          100, 50),
    ("rk4 4/8",               "rk4",            4,  8),
    ("implicitfast 100/50",   "implicitfast", 100, 50),
    ("implicitfast 4/8",      "implicitfast",   4,  8),
    ("implicitfast 1/4",      "implicitfast",   1,  4),
    ("euler 4/8",             "euler",          4,  8),
]


def build(integrator, iters, ls):
    LidarAnt.INTEGRATOR, LidarAnt.SOLVER_ITER, LidarAnt.SOLVER_LS_ITER = integrator, iters, ls
    return LidarAnt(num_agents=1)


def walk(env, T, turn, fwd=1.0):
    """Open-loop gait from a settled stand. Returns (com[T,3], up[T], yaw[T]).
    up = torso z-axis dotted with world z: 1.0 upright, <0 flipped."""
    def body(carry, _):
        dx, t = carry
        dx = env._substeps(dx, turn, fwd, t)
        w, x, y, z = dx.qpos[3], dx.qpos[4], dx.qpos[5], dx.qpos[6]
        yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))   # body heading, as in _nav_of
        return (dx, t + env._dt), (dx.subtree_com[0], 1.0 - 2.0 * (x * x + y * y), yaw)

    dx0 = env._reset_one_ant(jnp.zeros(2), 0.0)
    _, (com, up, yaw) = jax.lax.scan(body, (dx0, 0.0), None, length=T)
    return np.array(com), np.array(up), np.unwrap(np.array(yaw))


def fidelity(env, T):
    """Straight-line cruise + a hard skid-steer turn -- the two behaviours u_ref relies on."""
    com, up, _ = walk(env, T, turn=0.0)
    com_t, up_t, yaw_t = walk(env, T, turn=0.8)
    dur = T * env._dt
    return dict(
        v_fwd=float(com[-1, 0] - com[0, 0]) / dur,          # m/s along the start heading
        drift=float(abs(com[-1, 1] - com[0, 1])),           # lateral wander over the run, m
        z_mean=float(np.mean(com[:, 2])), z_min=float(np.min(com[:, 2])),
        up_min=float(np.min(np.minimum(up, up_t))),         # worst uprightness of either run
        # real body yaw rate, not the net-displacement direction: a curving path makes the
        # latter saturate and it under-reports the turn badly over long runs.
        turn_rate=float(np.degrees(yaw_t[-1] - yaw_t[0]) / dur),
    )


def speed(env, batch, iters):
    f = jax.jit(jax.vmap(lambda dx: env._substeps(dx, 0.0, 1.0, 0.0)))
    dxb = jax.vmap(lambda _: env._make_standing(jnp.zeros(2), 0.0))(jnp.arange(batch))
    t = time.perf_counter(); dxb = f(dxb); jax.block_until_ready(dxb.qpos)
    compile_s = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(iters):
        dxb = f(dxb)
    jax.block_until_ready(dxb.qpos)
    return batch * iters / (time.perf_counter() - t), compile_s


def main(args):
    print(f"backend={jax.default_backend()}  batch={args.batch}  iters={args.iters}  "
          f"walk={args.walk} control steps\n")
    ref, rows = None, []
    for label, integ, it, ls in CONFIGS:
        env = build(integ, it, ls)
        fid = fidelity(env, args.walk)
        sps, comp = speed(env, args.batch, args.iters)
        if ref is None:
            ref = fid
        # the gait was tuned under the reference config; it has to still walk and still turn
        ok = (abs(fid["v_fwd"] - ref["v_fwd"]) <= 0.25 * abs(ref["v_fwd"])
              and abs(fid["turn_rate"] - ref["turn_rate"]) <= 0.25 * abs(ref["turn_rate"])
              and fid["up_min"] > 0.8 and 0.25 < fid["z_mean"] < 0.75)
        rows.append((label, sps, sps / rows[0][1] if rows else 1.0, fid, ok, comp))
        r = rows[-1]
        print(f"{label:22s} {sps:9,.0f} steps/s  {r[2]:5.2f}x   "
              f"v={fid['v_fwd']:.3f} m/s  turn={fid['turn_rate']:+6.1f} deg/s  "
              f"z={fid['z_mean']:.2f}  up_min={fid['up_min']:.2f}  "
              f"{'PASS' if ok else 'FAIL'}   (compile {comp:.0f}s)", flush=True)

    best = max((r for r in rows if r[4]), key=lambda r: r[1], default=None)
    print("\nreference (gait was tuned here):", CONFIGS[0][0])
    if best is None:
        print("no config passed the gait check -- keep the stock rk4 100/50 settings")
    else:
        print(f"fastest PASSing config: {best[0]}  ({best[2]:.2f}x the stock throughput)")
        print("set it in dgppo/env/lidar_env/lidar_ant.py: INTEGRATOR / SOLVER_ITER / SOLVER_LS_ITER")
        v = best[3]["v_fwd"]
        if abs(v - LidarAnt.VMAX) > 0.05 * LidarAnt.VMAX:
            print(f"!! its cruise speed is {v:.3f} m/s but LidarAnt.VMAX = {LidarAnt.VMAX}. "
                  f"VMAX, R_MIN/R_MAX and u_ref's acc scaling are calibrated to VMAX, so either "
                  f"set VMAX = {v:.3f} or re-run the CEM gait search (p05_gait.py) under this "
                  f"config -- otherwise goals stop being reachable within max_step.")
    print("\nNOTE: changing these changes the dynamics, so a checkpoint trained under one "
          "setting must be evaluated under the SAME setting.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1024, help="parallel ants for the speed test")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--walk", type=int, default=200, help="control steps in the gait fidelity test")
    main(p.parse_args())
