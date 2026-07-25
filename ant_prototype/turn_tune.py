# Fast pure-controller test (no manifold): can full-gait + yaw_bias curve to turning goals?
import os, numpy as np, mujoco
XML=os.path.join(os.path.dirname(__file__),"ant.xml"); mj=mujoco.MjModel.from_xml_path(XML)
PHI=np.deg2rad(np.array([45.,135.,225.,315.]));GRP=np.array([0.,1.,0.,1.]);ASGN=np.array([1.,-1.,-1.,1.])
CTRL_LEG=np.array([3,3,0,0,1,1,2,2]);CTRL_ISANK=np.array([0,1,0,1,0,1,0,1])
QADR_H=np.array([7,9,11,13]);QADR_A=np.array([8,10,12,14]);DADR_H=np.array([6,8,10,12]);DADR_A=np.array([7,9,11,13])
GP=np.load(os.path.join(os.path.dirname(__file__),"best_gait.npy"));f,hip_amp,ank_mid,ank_amp,kp,kd,phase,_=[float(x) for x in GP]
K=np.sin(PHI);FS,DT=5,0.05;SUBDT=DT/FS
def yaw_of(q): w,x,y,z=q[3:7]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def torque(d,t,hs,yb):
    th=2*np.pi*f*t+np.pi*GRP*phase; hip_des=hs*hip_amp*K*np.cos(th)+yb
    lift=ank_amp*np.maximum(0.,np.sin(th)); ank_des=ASGN*(ank_mid-lift)
    q_h,q_a=d.qpos[QADR_H],d.qpos[QADR_A]; dq_h,dq_a=d.qvel[DADR_H],d.qvel[DADR_A]
    return np.clip(np.where(CTRL_ISANK==1,(kp*(ank_des-q_a)-kd*dq_a)[CTRL_LEG],(kp*(hip_des-q_h)-kd*dq_h)[CTRL_LEG]),-1.,1.)
def ctrl(err, mode):
    if mode=="A":   # full gait always, proportional yaw_bias
        return 1.0, float(np.clip(0.5*err,-0.35,0.35))
    if mode=="B":   # keep decent gait even turning (0.6), capped yaw
        hs=1.0 if abs(err)<np.deg2rad(30) else 0.6
        return hs, float(np.clip(0.6*err,-0.35,0.35))
def run(goal_ang, mode, n=500):
    d=mujoco.MjData(mj); d.qpos[2]=0.32; d.qpos[QADR_A]=ASGN*ank_mid; mujoco.mj_forward(mj,d)
    for _ in range(40):
        for _ in range(FS): d.ctrl[:]=torque(d,0.,0.,0.); mujoco.mj_step(mj,d)
    off=np.array([0.,0.])-d.subtree_com[0][:2]
    goal=3.0*np.array([np.cos(np.deg2rad(goal_ang)),np.sin(np.deg2rad(goal_ang))])
    tg=0.; best=9.
    for k in range(n):
        com=d.subtree_com[0][:2]+off; th=yaw_of(d.qpos)
        v=goal-com; dist=np.hypot(*v); best=min(best,dist)
        if dist<0.4: return k, dist, best
        err=np.arctan2(np.sin(np.arctan2(v[1],v[0])-th),np.cos(np.arctan2(v[1],v[0])-th))
        hs,yb=ctrl(err,mode)
        for _ in range(FS): d.ctrl[:]=torque(d,tg,hs,yb); mujoco.mj_step(mj,d); tg+=SUBDT
        mujoco.mj_forward(mj,d)
    return n, dist, best
for mode in ("A","B"):
    print(f"--- mode {mode} ---")
    for ang in (0,45,90,135,180):
        k,dist,best=run(ang,mode)
        print(f"  goal {ang:3d}deg: {'REACHED@'+str(k) if dist<0.4 else 'FAIL'}  final_dist={dist:.2f} closest={best:.2f}")
