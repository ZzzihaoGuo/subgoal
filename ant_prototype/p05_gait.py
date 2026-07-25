import time, numpy as np, jax, jax.numpy as jnp, mujoco
from mujoco import mjx
XML="/home/a5l/zihao1996.a5l/ENTER/envs/SocialJax/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets/ant.xml"
m = mujoco.MjModel.from_xml_path(XML); mx = mjx.put_model(m)
FS, DT = 5, 0.05                                  # frame_skip, control dt
PHI  = jnp.deg2rad(jnp.array([45., 135., 225., 315.]))   # leg 1..4 方位
GRP  = jnp.array([0., 1., 0., 1.])                        # trot: {1,3} vs {2,4}
ASGN = jnp.array([1., -1., -1., 1.])                      # ankle 符号 (leg1,4 正; leg2,3 负)
# ctrl 顺序: hip_4,ankle_4,hip_1,ankle_1,hip_2,ankle_2,hip_3,ankle_3
CTRL_LEG = jnp.array([3,3,0,0,1,1,2,2]); CTRL_ISANK = jnp.array([0,1,0,1,0,1,0,1])
QADR_H = jnp.array([7,9,11,13]); QADR_A = jnp.array([8,10,12,14])
DADR_H = jnp.array([6,8,10,12]); DADR_A = jnp.array([7,9,11,13])

def gait_targets(p, t, psi):
    """p: dict of gait params -> (hip_des(4), ankle_des(4))"""
    th = 2*jnp.pi*p['f']*t + jnp.pi*GRP*p['phase']
    k  = jnp.sin(PHI - psi)                                     # 每条腿对 psi 方向的推进贡献
    hip_des = p['hip_amp']*k*jnp.cos(th) + p['turn']*jnp.cos(th)
    lift    = p['ank_amp']*jnp.maximum(0., jnp.sin(th))         # 摆动相抬脚 = 减小 |ankle|
    ank_des = ASGN*(p['ank_mid'] - lift)
    return hip_des, ank_des

def torques(p, dx, t, psi):
    hip_des, ank_des = gait_targets(p, t, psi)
    q_h, q_a  = dx.qpos[QADR_H], dx.qpos[QADR_A]
    dq_h, dq_a = dx.qvel[DADR_H], dx.qvel[DADR_A]
    tau_h = p['kp']*(hip_des-q_h) - p['kd']*dq_h
    tau_a = p['kp']*(ank_des-q_a) - p['kd']*dq_a
    tau = jnp.where(CTRL_ISANK==1, tau_a[CTRL_LEG], tau_h[CTRL_LEG])
    return jnp.clip(tau, -1., 1.)

def init_data(p):
    dx = mjx.make_data(mx)
    qpos = dx.qpos.at[2].set(0.32)
    qpos = qpos.at[QADR_A].set(ASGN*p['ank_mid'])
    return dx.replace(qpos=qpos)

def rollout(p, psi, n_settle=40, n_run=200):
    dx = init_data(p)
    def step(dx, t):
        dx = dx.replace(ctrl=torques(p, dx, t, psi))
        return jax.lax.fori_loop(0, FS, lambda _, d: mjx.step(mx, d), dx), None
    # settle: 只保持站姿 (f=0 由 t=0 冻结)
    dx, _ = jax.lax.scan(lambda d,_: step(d, 0.0), dx, None, length=n_settle)
    ts = jnp.arange(n_run)*DT
    def run(carry, t):
        dx, com_prev = carry
        dx, _ = step(dx, t)
        com = dx.subtree_com[0]
        vel = (com - com_prev)/DT                        # 质心速度: 位置差分
        up  = 1 - 2*(dx.qpos[4]**2 + dx.qpos[5]**2)      # 躯干 z 轴的世界 z 分量
        return (dx, com), jnp.array([com[0], com[1], com[2], vel[0], vel[1], up])
    (dx, _), tr = jax.lax.scan(run, (dx, dx.subtree_com[0]), ts)
    return tr

def score(p, psi=0.0):
    tr = rollout(p, psi)
    half = tr.shape[0]//2
    v = tr[half:, 3:5]
    v_along = jnp.mean(v[:,0]*jnp.cos(psi) + v[:,1]*jnp.sin(psi))
    z, up = tr[:,2], tr[:,5]
    alive  = jnp.mean((z > 0.18) & (z < 0.40) & (up > 0.8))   # 必须贴地且直立
    z_std  = jnp.std(z)                                        # 惩罚弹跳
    sc = jnp.minimum(v_along, 2.0)*alive - 2.0*(1-alive) - 3.0*z_std
    return sc, v_along, alive, jnp.mean(z), z_std, jnp.mean(up)

KEYS = ['f','hip_amp','ank_mid','ank_amp','kp','kd','phase','turn']
LO = jnp.array([0.5, 0.05, jnp.deg2rad(58), 0.02, 1.0, 0.02, 0.5, 0.0])
HI = jnp.array([4.0, 0.52, jnp.deg2rad(70), 0.40, 20.0, 2.0, 1.5, 0.0])

def to_dict(v): return {k: v[i] for i,k in enumerate(KEYS)}
eval_batch = jax.jit(jax.vmap(lambda v: score(to_dict(v))))

key = jax.random.PRNGKey(0); N = 4096
best_v, best_s = None, -1e9
for gen in range(12):
    key, sk = jax.random.split(key)
    if gen == 0:
        cand = jax.random.uniform(sk, (N, len(KEYS)), minval=LO, maxval=HI)
    else:
        sc = (HI-LO)*0.15*(0.6**gen)
        cand = jnp.clip(best_v + jax.random.normal(sk,(N,len(KEYS)))*sc, LO, HI)
    t0=time.perf_counter(); s, va, al, zh, zs, upm = eval_batch(cand); jax.block_until_ready(s)
    i = int(jnp.argmax(s))
    if float(s[i]) > best_s: best_s, best_v = float(s[i]), cand[i]
    print(f"gen{gen}: score={best_s:+.3f}  v_fwd={float(va[i]):+.3f} m/s  alive={float(al[i]):.2f}  "
          f"z={float(zh[i]):.3f}±{float(zs[i]):.3f}  up={float(upm[i]):.2f}  ({time.perf_counter()-t0:.0f}s)", flush=True)
print("\nBEST PARAMS")
for k,v in zip(KEYS, np.array(best_v)):
    print(f"  {k:9s} {v:.4f}" + (f"  ({np.rad2deg(v):.1f}°)" if 'ank' in k or 'amp' in k else ""))
np.save(f"{__import__('os').path.dirname(__file__)}/best_gait.npy", np.array(best_v))
