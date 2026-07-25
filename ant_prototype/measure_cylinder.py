# Measure the ant's horizontal leg-reach envelope -> cylinder (collision) radius.
import os, numpy as np, mujoco
XML=os.path.join(os.path.dirname(__file__),"ant.xml"); m=mujoco.MjModel.from_xml_path(XML)
GT=np.array([m.geom_type[g] for g in range(m.ngeom)]); GS=np.array([m.geom_size[g] for g in range(m.ngeom)])
GNAME=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) for g in range(m.ngeom)]
PHI=np.deg2rad(np.array([45.,135.,225.,315.]));GRP=np.array([0.,1.,0.,1.]);ASGN=np.array([1.,-1.,-1.,1.])
CTRL_LEG=np.array([3,3,0,0,1,1,2,2]);CTRL_ISANK=np.array([0,1,0,1,0,1,0,1])
QADR_H=np.array([7,9,11,13]);QADR_A=np.array([8,10,12,14]);DADR_H=np.array([6,8,10,12]);DADR_A=np.array([7,9,11,13])
GP=np.load(os.path.join(os.path.dirname(__file__),"best_gait.npy"));f,hip_amp,ank_mid,ank_amp,kp,kd,phase,_=[float(x) for x in GP]
K=np.sin(PHI);FS,DT=5,0.05;SUBDT=DT/FS
def torque(d,t):
    th=2*np.pi*f*t+np.pi*GRP*phase; hip_des=hip_amp*K*np.cos(th)
    lift=ank_amp*np.maximum(0.,np.sin(th)); ank_des=ASGN*(ank_mid-lift)
    q_h,q_a=d.qpos[QADR_H],d.qpos[QADR_A]; dq_h,dq_a=d.qvel[DADR_H],d.qvel[DADR_A]
    return np.clip(np.where(CTRL_ISANK==1,(kp*(ank_des-q_a)-kd*dq_a)[CTRL_LEG],(kp*(hip_des-q_h)-kd*dq_h)[CTRL_LEG]),-1.,1.)
def reach(d):
    """max horizontal distance from CoM to any geom's outer surface (capsule ends + radius)."""
    com=d.subtree_com[0][:2]; mx=0.; who=""
    for g in range(m.ngeom):
        if GNAME[g] in (None,"floor") or GT[g]==0: continue
        xp=d.geom_xpos[g]; rad=GS[g][0]
        if GT[g]==3:  # capsule: two endpoints
            axis=d.geom_xmat[g].reshape(3,3)[:,2]; hl=GS[g][1]
            for e in (xp-axis*hl, xp+axis*hl):
                dd=np.hypot(*(e[:2]-com))+rad
                if dd>mx: mx,who=dd,GNAME[g]
        else:         # sphere torso
            dd=np.hypot(*(xp[:2]-com))+rad
            if dd>mx: mx,who=dd,GNAME[g]
    return mx,who
d=mujoco.MjData(m); d.qpos[2]=0.32; d.qpos[QADR_A]=ASGN*ank_mid; mujoco.mj_forward(m,d)
print(f"initial standing reach = {reach(d)[0]:.3f} m  (limited by {reach(d)[1]})")
for _ in range(40):
    for _ in range(FS): d.ctrl[:]=torque(d,0.); mujoco.mj_step(m,d)
mujoco.mj_forward(m,d)
reaches=[]; tg=0.
for k in range(80):  # ~2 gait cycles
    for _ in range(FS): d.ctrl[:]=torque(d,tg); mujoco.mj_step(m,d); tg+=SUBDT
    mujoco.mj_forward(m,d); reaches.append(reach(d)[0])
reaches=np.array(reaches)
print(f"walking reach over gait cycle: min={reaches.min():.3f}  mean={reaches.mean():.3f}  MAX={reaches.max():.3f} m")
print(f"torso radius (for reference) = {GS[[i for i,n in enumerate(GNAME) if n=='torso_geom'][0]][0]:.3f} m")
print(f"\n=> cylinder (collision) radius candidates:")
print(f"   conservative (max over gait) = {reaches.max():.3f} m")
print(f"   nominal (mean)               = {reaches.mean():.3f} m")
