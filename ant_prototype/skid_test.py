# Skid-steer turning: amplify one side's leg push vs the other. Measure yaw rate.
import os, numpy as np, mujoco
XML=os.path.join(os.path.dirname(__file__),"ant.xml"); mj=mujoco.MjModel.from_xml_path(XML)
PHI=np.deg2rad(np.array([45.,135.,225.,315.]));GRP=np.array([0.,1.,0.,1.]);ASGN=np.array([1.,-1.,-1.,1.])
CTRL_LEG=np.array([3,3,0,0,1,1,2,2]);CTRL_ISANK=np.array([0,1,0,1,0,1,0,1])
QADR_H=np.array([7,9,11,13]);QADR_A=np.array([8,10,12,14]);DADR_H=np.array([6,8,10,12]);DADR_A=np.array([7,9,11,13])
GP=np.load(os.path.join(os.path.dirname(__file__),"best_gait.npy"));f,hip_amp,ank_mid,ank_amp,kp,kd,phase,_=[float(x) for x in GP]
K=np.sin(PHI);FS,DT=5,0.05;SUBDT=DT/FS
def yaw_of(q): w,x,y,z=q[3:7]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
# R: which side each leg is on. legs 1,2 at +y (left), legs 3,4 at -y (right)
LEFT=np.array([1.,1.,-1.,-1.])   # +1 left, -1 right
def torque(d,t,turn,fwd=1.0):
    th=2*np.pi*f*t+np.pi*GRP*phase
    mult=1.0+turn*(-LEFT)                 # turn>0 => amplify LEFT-side legs less / right more => CCW? test both signs
    hip_des=fwd*hip_amp*K*mult*np.cos(th)
    lift=ank_amp*np.maximum(0.,np.sin(th)); ank_des=ASGN*(ank_mid-lift)
    q_h,q_a=d.qpos[QADR_H],d.qpos[QADR_A]; dq_h,dq_a=d.qvel[DADR_H],d.qvel[DADR_A]
    return np.clip(np.where(CTRL_ISANK==1,(kp*(ank_des-q_a)-kd*dq_a)[CTRL_LEG],(kp*(hip_des-q_h)-kd*dq_h)[CTRL_LEG]),-1.,1.)
def run(turn, n=200):
    d=mujoco.MjData(mj); d.qpos[2]=0.32; d.qpos[QADR_A]=ASGN*ank_mid; mujoco.mj_forward(mj,d)
    for _ in range(40):
        for _ in range(FS): d.ctrl[:]=torque(d,0.,0.); mujoco.mj_step(mj,d)
    tg=0.; yaws=[]; c0=d.subtree_com[0][:2].copy()
    for k in range(n):
        for _ in range(FS): d.ctrl[:]=torque(d,tg,turn); mujoco.mj_step(mj,d); tg+=SUBDT
        yaws.append(yaw_of(d.qpos))
    yaws=np.unwrap(yaws); rate=np.rad2deg(yaws[-1]-yaws[0])/(n*DT)
    disp=np.hypot(*(d.subtree_com[0][:2]-c0))
    return rate, disp, d.qpos[2]
print("turn   yaw_rate(deg/s)  disp(m)  torso_z  (n=200 => 10s)")
for turn in (-0.8,-0.4,0.0,0.4,0.8):
    r,disp,z=run(turn); print(f"{turn:+.1f}    {r:+7.1f}         {disp:.2f}    {z:.2f}")
