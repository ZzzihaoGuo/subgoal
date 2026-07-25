# Does rebuilding Data from qpos/qvel each step (vs carrying full Data) cause instability?
import os, numpy as np, jax, jax.numpy as jnp, mujoco
from mujoco import mjx
XML=os.path.join(os.path.dirname(__file__),"ant.xml"); m=mujoco.MjModel.from_xml_path(XML); mx=mjx.put_model(m)
PHI=jnp.deg2rad(jnp.array([45.,135.,225.,315.]));GRP=jnp.array([0.,1.,0.,1.]);ASGN=jnp.array([1.,-1.,-1.,1.])
CTRL_LEG=jnp.array([3,3,0,0,1,1,2,2]);CTRL_ISANK=jnp.array([0,1,0,1,0,1,0,1])
QADR_H=jnp.array([7,9,11,13]);QADR_A=jnp.array([8,10,12,14]);DADR_H=jnp.array([6,8,10,12]);DADR_A=jnp.array([7,9,11,13])
GP=np.load(os.path.join(os.path.dirname(__file__),"best_gait.npy"));f,hip_amp,ank_mid,ank_amp,kp,kd,phase,_=[float(x) for x in GP]
K=jnp.sin(PHI);FS,DT=5,0.05;SUBDT=DT/FS;LEFT=jnp.array([1.,1.,-1.,-1.])
def torque(dx,t,turn,fwd):
    th=2*jnp.pi*f*t+jnp.pi*GRP*phase; mult=1.0+turn*(-LEFT); hip_des=fwd*hip_amp*K*mult*jnp.cos(th)
    lift=ank_amp*jnp.maximum(0.,jnp.sin(th)); ank_des=ASGN*(ank_mid-lift)
    q_h,q_a=dx.qpos[QADR_H],dx.qpos[QADR_A]; dq_h,dq_a=dx.qvel[DADR_H],dx.qvel[DADR_A]
    return jnp.clip(jnp.where(CTRL_ISANK==1,(kp*(ank_des-q_a)-kd*dq_a)[CTRL_LEG],(kp*(hip_des-q_h)-kd*dq_h)[CTRL_LEG]),-1.,1.)
def std_qpos():
    q=jnp.zeros(mx.nq); return q.at[2].set(0.32).at[QADR_A].set(ASGN*ank_mid).at[3].set(1.0)
# A) rebuild from qpos/qvel each ctrl step (like the env)
def step_rebuild(qpos,qvel,turn,fwd,t0):
    dx=mjx.make_data(mx).replace(qpos=qpos,qvel=qvel); c0=dx.subtree_com[0]
    dx=jax.lax.fori_loop(0,FS,lambda i,d:mjx.step(mx,d.replace(ctrl=torque(d,t0+i*SUBDT,turn,fwd))),dx)
    return dx.qpos,dx.qvel,dx.subtree_com[0], (dx.subtree_com[0][:2]-c0[:2])/DT
sr=jax.jit(step_rebuild)
qpos,qvel=std_qpos(),jnp.zeros(mx.nv); tg=0.
# no settle
for k in range(60):
    qpos,qvel,com,vel=sr(qpos,qvel,0.0,1.0,tg); tg+=DT
    if k in (0,1,5,20,59): print(f" rebuild NO-settle step{k:2d}: z={float(qpos[2]):.2f} |v|={float(jnp.hypot(*vel)):.3f}")
# with settle
qpos,qvel=std_qpos(),jnp.zeros(mx.nv); tg=0.
for k in range(30): qpos,qvel,com,vel=sr(qpos,qvel,0.0,0.0,0.0)   # settle standing
print(" -- after 30 settle: z=%.2f --"%float(qpos[2]))
for k in range(60):
    qpos,qvel,com,vel=sr(qpos,qvel,0.0,1.0,tg); tg+=DT
    if k in (0,5,20,59): print(f" rebuild settled  step{k:2d}: z={float(qpos[2]):.2f} |v|={float(jnp.hypot(*vel)):.3f}")
