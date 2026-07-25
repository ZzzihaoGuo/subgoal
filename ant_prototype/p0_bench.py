import time, sys
import jax, jax.numpy as jnp, mujoco
from mujoco import mjx

XML = "/home/a5l/zihao1996.a5l/ENTER/envs/SocialJax/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets/ant.xml"
m = mujoco.MjModel.from_xml_path(XML)
print(f"timestep={m.opt.timestep}  nq={m.nq} nv={m.nv} nu={m.nu}  mass={m.body_mass.sum():.4f} kg", flush=True)
mx = mjx.put_model(m)

def ctrl_step(dx, u, frame_skip):
    dx = dx.replace(ctrl=u)
    return jax.lax.fori_loop(0, frame_skip, lambda _, d: mjx.step(mx, d), dx)

def bench(batch, frame_skip, n_iters=20):
    f = jax.jit(jax.vmap(lambda d, u: ctrl_step(d, u, frame_skip)))
    dx = jax.vmap(lambda _: mjx.make_data(mx))(jnp.arange(batch))
    u = jax.random.uniform(jax.random.PRNGKey(0), (batch, mx.nu), minval=-1.0, maxval=1.0)
    t = time.perf_counter(); dx = f(dx, u); jax.block_until_ready(dx.qpos)
    compile_s = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(n_iters): dx = f(dx, u)
    jax.block_until_ready(dx.qpos)
    el = time.perf_counter() - t
    return batch*n_iters/el, compile_s

print(f"\n{'batch':>7} {'skip':>5} {'ctrl steps/s':>14} {'phys steps/s':>14} {'compile(s)':>11}", flush=True)
for fs in (1, 5):
    for b in (1024, 4096, 8192):
        try:
            c, cs = bench(b, fs)
            print(f"{b:>7} {fs:>5} {c:>14,.0f} {c*fs:>14,.0f} {cs:>11.1f}", flush=True)
        except Exception as e:
            print(f"{b:>7} {fs:>5}   FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
print("\nDONE", flush=True)
