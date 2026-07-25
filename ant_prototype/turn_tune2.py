import os, numpy as np, mujoco
XML=os.path.join(os.path.dirname(__file__),"ant.xml"); mj=mujoco.MjModel.from_xml_path(XML)
PHI=np.deg2rad(np.array([45.,135.,225.,315.]));GRP=np.array([0.,1.,0.,1.]);ASGN=np.array([1.,-1.,-1.,1.])
CTRL_LEG=np.array([3,3,0,0,1,1,2,2]);CTRL_ISANK=np.array([0,1,0,1,0,1,0,1])
QADR_H=np.array([7,9,11,13]);QADR_A=np.array([8,10,12,14]);DADR_H=np.array([6,8,10,12]);DADR_A=np.array([7,9,11,13])
GP=np.load(os.path.join(os.path.dirname(__file__),"best_gait.npy"));f,hip_amp,ank_mid,ank_amp,kp,kd,phase,_=[float(x) for x in GP]
K=np.sin(PHI);FS,DT=5,0.05;SUBDT=DT/FS;LEFT=np.array([1.,1.,-1.,-1.])
def yaw_of(q): w,x,y,z=q[3:7]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def torque(d,t,turn,fwd):
    th=2*np.pi*f*t+np.pi*GRP*phase; mult=1.0+turn*(-LEFT); hip_des=fwd*hip_amp*K*mult*np.cos(th)
    lift=ank_amp*np.maximum(0.,np.sin(th)); ank_des=ASGN*(ank_mid-lift)
    q_h,q_a=d.qpos[QADR_H],d.qpos[QADR_A]; dq_h,dq_a=d.qvel[DADR_H],d.qvel[DADR_A]
    return np.clip(np.where(CTRL_ISANK==1,(kp*(ank_des-q_a)-kd*dq_a)[CTRL_LEG],(kp*(hip_des-q_h)-kd*dq_h)[CTRL_LEG]),-1.,1.)
def run(goal_ang,n=700):
    d=mujoco.MjData(mj); d.qpos[2]=0.32; d.qpos[QADR_A]=ASGN*ank_mid; mujoco.mj_forward(mj,d)
    for _ in range(40):
        for _ in range(FS): d.ctrl[:]=torque(d,0.,0.,1.); mujoco.mj_step(mj,d)
    off=-d.subtree_com[0][:2].copy(); goal=3.0*np.array([np.cos(np.deg2rad(goal_ang)),np.sin(np.deg2rad(goal_ang))])
    tg=0.; best=9.
    for k in range(n):
        com=d.subtree_com[0][:2]+off; th=yaw_of(d.qpos); v=goal-com; dist=np.hypot(*v); best=min(best,dist)
        if dist<0.4: return k,dist,best
        err=np.arctan2(np.sin(np.arctan2(v[1],v[0])-th),np.cos(np.arctan2(v[1],v[0])-th))
        turn=float(np.clip(1.2*err,-0.8,0.8)); fwd=float(np.clip(np.cos(err),0.4,1.0))  # slow fwd while turning hard
        for _ in range(FS): d.ctrl[:]=torque(d,tg,turn,fwd); mujoco.mj_step(mj,d); tg+=SUBDT
        mujoco.mj_forward(mj,d)
    return n,dist,best
print("skid-steer follower:")
for ang in (0,45,90,135,180,-90):
    k,dist,best=run(ang); print(f"  goal {ang:4d}deg: {'REACHED@'+str(k) if dist<0.4 else 'FAIL'}  final={dist:.2f} closest={best:.2f}")
