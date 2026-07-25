# MJX version of the ant gait+skid-steer step, vmappable. Verify it walks & turns like plain MuJoCo.
import os, time, numpy as np, jax, jax.numpy as jnp, mujoco
from mujoco import mjx
XML=os.path.join(os.path.dirname(__file__),"ant.xml")
m=mujoco.MjModel.from_xml_path(XML); mx=mjx.put_model(m)
PHI=jnp.deg2rad(jnp.array([45.,135.,225.,315.]));GRP=jnp.array([0.,1.,0.,1.]);ASGN=jnp.array([1.,-1.,-1.,1.])
CTRL_LEG=jnp.array([3,3,0,0,1,1,2,2]);CTRL_ISANK=jnp.array([0,1,0,1,0,1,0,1])
QADR_H=jnp.array([7,9,11,13]);QADR_A=jnp.array([8,10,12,14]);DADR_H=jnp.array([6,8,10,12]);DADR_A=jnp.array([7,9,11,13])
GP=np.load(os.path.join(os.path.dirname(__file__),"best_gait.npy")); f,hip_amp,ank_mid,ank_amp,kp,kd,phase,_=[float(x) for x in GP]
K=jnp.sin(PHI);FS,DT=5,0.05;SUBDT=DT/FS;LEFT=jnp.array([1.,1.,-1.,-1.])
def yaw_of(q):
    w,x,y,z=q[3],q[4],q[5],q[6]; return jnp.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def torque(dx,t,turn,fwd):
    th=2*jnp.pi*f*t+jnp.pi*GRP*phase; mult=1.0+turn*(-LEFT); hip_des=fwd*hip_amp*K*mult*jnp.cos(th)
    lift=ank_amp*jnp.maximum(0.,jnp.sin(th)); ank_des=ASGN*(ank_mid-lift)
    q_h,q_a=dx.qpos[QADR_H],dx.qpos[QADR_A]; dq_h,dq_a=dx.qvel[DADR_H],dx.qvel[DADR_A]
    tau_h=kp*(hip_des-q_h)-kd*dq_h; tau_a=kp*(ank_des-q_a)-kd*dq_a
    return jnp.clip(jnp.where(CTRL_ISANK==1, tau_a[CTRL_LEG], tau_h[CTRL_LEG]),-1.,1.)

def ant_step(dx, turn, fwd, t0):
    """One control step = FS substeps, torque recomputed each substep (500Hz). Returns dx_new, nav[x,y,cos,sin,v]."""
    com0=dx.subtree_com[0]
    def body(i, dx):
        t=t0+i*SUBDT
        return mjx.step(mx, dx.replace(ctrl=torque(dx,t,turn,fwd)))
    dx=jax.lax.fori_loop(0, FS, body, dx)
    com=dx.subtree_com[0]; yaw=yaw_of(dx.qpos); vel=(com[:2]-com0[:2])/DT
    nav=jnp.array([com[0],com[1],jnp.cos(yaw),jnp.sin(yaw),jnp.hypot(vel[0],vel[1])])
    return dx, nav

def make_standing():
    dx=mjx.make_data(mx); qpos=dx.qpos.at[2].set(0.32).at[QADR_A].set(ASGN*ank_mid)
    return dx.replace(qpos=qpos)

# settle then run a straight + a turn, single ant
step_j=jax.jit(ant_step)
dx=make_standing()
for i in range(40): dx,_=step_j(dx,0.0,0.0,0.0)   # settle
# straight
tg=0.0; navs=[]
for k in range(120): dx,nav=step_j(dx,0.0,1.0,tg); tg+=DT; navs.append(np.array(nav))
navs=np.array(navs); disp=navs[-1,:2]-navs[0,:2]
print(f"[MJX single] straight: disp=({disp[0]:+.2f},{disp[1]:+.2f}) |{np.hypot(*disp):.2f}|m over 6s, v_end={navs[-1,4]:.3f}")
# turn (turn=+0.8)
dx=make_standing()
for i in range(40): dx,_=step_j(dx,0.0,0.0,0.0)
tg=0.0; yaws=[]
for k in range(120): dx,nav=step_j(dx,0.8,1.0,tg); tg+=DT; yaws.append(np.arctan2(nav[3],nav[2]))
yaws=np.unwrap(np.array(yaws)); print(f"[MJX single] turn=+0.8: yaw_rate={np.rad2deg(yaws[-1]-yaws[0])/6:.1f} deg/s")

# vmap over N=8 ants with different turns, throughput
N=8
vstep=jax.jit(jax.vmap(ant_step, in_axes=(0,0,0,None)))
dxs=jax.vmap(lambda _: make_standing())(jnp.arange(N))
turns=jnp.linspace(-0.8,0.8,N)
for i in range(40): dxs,_=vstep(dxs,turns*0,turns*0,0.0)
t0=time.perf_counter()
tg=0.0
for k in range(60): dxs,navs=vstep(dxs,turns,jnp.ones(N),tg); tg+=DT
jax.block_until_ready(navs); el=time.perf_counter()-t0
print(f"[MJX vmap N={N}] ran 60 ctrl steps in {el:.2f}s = {N*60/el:.0f} ant-ctrl-steps/s")
print(f"  per-ant yaw after: turns={np.round(np.array(turns),2)}")
