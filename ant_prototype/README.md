# Ant + Manifold Prototype

MuJoCo-ant integration prototype for the hierarchical subgoal + manifold safety stack.
**All new files — nothing here modifies the existing `dgppo/` code.**

## What this is

A proof-of-concept that the manifold safety layer (`get_manifold_action`) combines with
MuJoCo-ant dynamics with **zero changes to the manifold solver**, by treating the ant as a
low-level executor under a bicycle-model CoM template:

```
GNN subgoal -> u_ref (bicycle P-ctrl) -> manifold filter -> (v, omega) cmd
            -> gait controller -> ant joint torques -> MuJoCo step
            -> read CoM back as [x, y, theta, v] -> rebuild graph
```

## Key finding (see gifs/)

- Ant walks upright & straight at ~0.22 m/s, robust (open-loop trot from CEM).
- **Turning is limited**: hip joints are ±30° and the gait uses ~26°, so steering saturates.
  Effective heading correction ~±15°. Small course corrections work; sharp turns do not.
- Consequence: manifold+ant reach goals when no hard turn is needed (multi-agent spacing =
  small lateral moves = works). Obstacle avoidance needing a real detour = ant stalls.
  Reliable turning requires a *learned* locomotion policy (deferred: "P1").

## Engine note

- `p05_gait.py`, `p0_bench.py`: use **MJX** (GPU, vmap) — for gait search & throughput.
- `ant_manifold.py`, `ant_mf_multi.py`: use **plain MuJoCo** (CPU, `mj_step`) — single-episode
  GIF prototypes, easy to bridge with the JAX manifold. **NOT usable for training** — the real
  env (`dgppo/env/lidar_env/lidar_ant.py`, not yet written = "P2") must step in MJX so it can be
  `vmap`-ed over thousands of parallel envs.

## Files

| file | what | engine |
|------|------|--------|
| `best_gait.npy` | 7-param open-loop trot [f,hip_amp,ank_mid,ank_amp,kp,kd,phase,turn] | data |
| `ant.xml` | gymnasium ant model (copied, self-contained) | data |
| `p0_bench.py` | MJX throughput benchmark (batch x frame_skip) | MJX |
| `p05_gait.py` | CEM gait search (produces best_gait.npy) | MJX |
| `ant_manifold.py` | single-agent: real u_ref + real manifold + ant, logs rollout | MuJoCo |
| `ant_manifold_render.py` | render single-agent rollout to gif | matplotlib |
| `ant_mf_multi.py` | 3-agent version, same flow | MuJoCo |
| `ant_mf_multi_render.py` | render multi-agent rollout to gif | matplotlib |
| `gifs/` | rendered demos | — |

## Run

```bash
PY=/home/a5l/zihao1996.a5l/ENTER/envs/dgppo/bin/python
$PY ant_manifold.py && $PY ant_manifold_render.py     # single agent
$PY ant_mf_multi.py && $PY ant_mf_multi_render.py     # 3 agents
```

Rendering is matplotlib stick-figure (no working GL backend on this headless node:
egl/osmesa/glfw all fail), not MuJoCo's textured renderer.
