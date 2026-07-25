# Single agent follows a SEQUENCE of subgoals (waypoints). Answers: can the ant TURN
# to follow subgoals? Flow: subgoal -> u_ref + manifold (safety) -> skid-steer gait -> MuJoCo.
# Turning uses SKID-STEER (differential left/right leg push), which — unlike a uniform hip
# yaw-bias — produces clean monotonic yaw control (±19 deg/s at turn=±0.8) while walking.
import os, sys, numpy as np, mujoco, jax, jax.numpy as jnp, pickle
sys.path.insert(0, "/home/a5l/zihao1996.a5l/project/subgoal")
from dgppo.env.lidar_env.lidar_bicycle_target import LidarBicycleTarget
from dgppo.env.lidar_env.base import LidarEnvState
AREA=6.0
class LidarAnt(LidarBicycleTarget):
    PARAMS={**LidarBicycleTarget.PARAMS,"car_radius":0.3,"comm_radius":3.0,
            "obs_len_range":[0.8,1.2],"n_obs":0,"default_area_size":AREA,
            "dist2goal":0.4,"top_k_rays":8,"n_rays":32,"m":0.1}
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

def main(waypoints, tag):
    env=LidarAnt(num_agents=1,area_size=AREA,max_step=2000,dt=DT)
    env.init_manifold(k=3,K=0.5,Kc=30.0,alpha_max=3.0,g_act_thresh=0.02,safety_margin=0.02,n_lookahead=0,w_slack=10.0)
    man_j=jax.jit(lambda g,u,s: env.get_manifold_action(g,u_ref=u,s_all=s))
    uref_j=jax.jit(lambda g,tp,fin: env.u_ref(g,target_pos=tp,is_final_goal=fin))
    wps=np.array(waypoints); gi=0
    start=wps[0]+np.array([-1.2,0.0])
    agent=jnp.array([[start[0],start[1],1.0,0.0,0.0]])
    d=mujoco.MjData(mj); d.qpos[2]=0.32; d.qpos[QADR_A]=ASGN*ank_mid; mujoco.mj_forward(mj,d)
    for _ in range(40):
        for _ in range(FS): d.ctrl[:]=torque(d,0.,0.,1.); mujoco.mj_step(mj,d)
    off=start-d.subtree_com[0][:2].copy()
    s_all=env.manifold_init_slack(env.get_graph(LidarEnvState(agent,agent,None),None))
    tg=0.; log=[]; reached=[]
    for k in range(1800):
        sg=wps[gi]; goal=jnp.array([[sg[0],sg[1],1.0,0.0,0.0]])
        graph=env.get_graph(LidarEnvState(agent,goal,None),None)
        nominal=uref_j(graph,goal[:,:2],(gi==len(wps)-1))
        safe,relax,s_all,_=man_j(graph,nominal,s_all)
        ag=np.array(agent)[0]; th=np.arctan2(ag[3],ag[2]); v=sg-ag[:2]; dist=float(np.hypot(*v))
        if dist<0.4:
            reached.append(gi)
            if gi<len(wps)-1: gi+=1; continue
            else: print(f"[{tag}] ALL {len(wps)} subgoals reached @step {k}"); break
        err=np.arctan2(np.sin(np.arctan2(v[1],v[0])-th),np.cos(np.arctan2(v[1],v[0])-th))
        turn=float(np.clip(1.2*err,-0.8,0.8)); fwd=float(np.clip(np.cos(err),0.4,1.0))
        cp=d.subtree_com[0].copy()
        for _ in range(FS): d.ctrl[:]=torque(d,tg,turn,fwd); mujoco.mj_step(mj,d); tg+=SUBDT
        mujoco.mj_forward(mj,d); com=d.subtree_com[0].copy(); thn=yaw_of(d.qpos)
        exy=com[:2]+off; vel=(com[:2]-cp[:2])/DT
        agent=jnp.array([[exy[0],exy[1],np.cos(thn),np.sin(thn),float(np.hypot(*vel))]])
        rel=float(np.max(np.abs(np.array(relax)))) if relax is not None else 0.
        log.append((d.geom_xpos.copy(),d.geom_xmat.copy().reshape(-1,3,3),exy.copy(),thn,gi,dist,rel))
    pickle.dump(log,open(os.path.join(os.path.dirname(__file__),f"sgfollow_{tag}.pkl"),"wb"))
    np.save(os.path.join(os.path.dirname(__file__),f"sgfollow_{tag}_wps.npy"),wps)
    print(f"[{tag}] steps={len(log)} reached={sorted(set(reached))} of {len(wps)} subgoals")

if __name__=="__main__":
    square=[[4.5,1.5],[4.5,4.5],[1.5,4.5],[1.5,1.5]]     # 4 subgoals, each ~90-deg left turn
    main(square,"square")
