import os, sys, numpy as np, mujoco, jax, jax.numpy as jnp, pickle
sys.path.insert(0, "/home/a5l/zihao1996.a5l/project/subgoal")
from dgppo.env.lidar_env.lidar_bicycle_target import LidarBicycleTarget
from dgppo.env.lidar_env.base import LidarEnvState
AREA=6.0; N=3
class LidarAnt(LidarBicycleTarget):
    PARAMS={**LidarBicycleTarget.PARAMS,"car_radius":0.3,"comm_radius":3.0,
            "obs_len_range":[0.8,1.2],"n_obs":0,"default_area_size":AREA,
            "dist2goal":0.4,"top_k_rays":8,"n_rays":32,"m":0.1}
XML=os.path.join(os.path.dirname(__file__),"ant.xml")
mj=mujoco.MjModel.from_xml_path(XML)
PHI=np.deg2rad(np.array([45.,135.,225.,315.]));GRP=np.array([0.,1.,0.,1.]);ASGN=np.array([1.,-1.,-1.,1.])
CTRL_LEG=np.array([3,3,0,0,1,1,2,2]);CTRL_ISANK=np.array([0,1,0,1,0,1,0,1])
QADR_H=np.array([7,9,11,13]);QADR_A=np.array([8,10,12,14]);DADR_H=np.array([6,8,10,12]);DADR_A=np.array([7,9,11,13])
GP=np.load(f"{os.path.dirname(__file__)}/best_gait.npy");f,hip_amp,ank_mid,ank_amp,kp,kd,phase,_=[float(x) for x in GP]
K=np.sin(PHI);FS,DT=5,0.05;SUBDT=DT/FS;VMAX=0.22
def yaw_of(q): w,x,y,z=q[3:7]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def torque(d,t,hs,yb):
    th=2*np.pi*f*t+np.pi*GRP*phase; hip_des=hs*hip_amp*K*np.cos(th)+yb
    lift=ank_amp*np.maximum(0.,np.sin(th)); ank_des=ASGN*(ank_mid-lift)
    q_h,q_a=d.qpos[QADR_H],d.qpos[QADR_A]; dq_h,dq_a=d.qvel[DADR_H],d.qvel[DADR_A]
    return np.clip(np.where(CTRL_ISANK==1,(kp*(ank_des-q_a)-kd*dq_a)[CTRL_LEG],(kp*(hip_des-q_h)-kd*dq_h)[CTRL_LEG]),-1.,1.)
def new_ant():
    d=mujoco.MjData(mj); d.qpos[2]=0.32; d.qpos[QADR_A]=ASGN*ank_mid; mujoco.mj_forward(mj,d)
    for _ in range(40):
        for _ in range(FS): d.ctrl[:]=torque(d,0.,0.,0.); mujoco.mj_step(mj,d)
    return d

def main():
    env=LidarAnt(num_agents=N,area_size=AREA,max_step=500,dt=DT)
    env.init_manifold(k=3,K=0.5,Kc=30.0,alpha_max=3.0,g_act_thresh=0.02,safety_margin=0.02,n_lookahead=0,w_slack=10.0)
    starts=np.array([[1.0,2.7],[1.0,3.0],[1.0,3.3]])      # 间距0.3 < 0.6 碰撞距离
    goals =np.array([[5.0,2.4],[5.0,3.0],[5.0,3.6]])       # 散开
    agent=jnp.array([[s[0],s[1],1.0,0.0,0.0] for s in starts])
    goal0=jnp.array([[g[0],g[1],1.0,0.0,0.0] for g in goals])
    def build(a):
        st=LidarEnvState(a,goal0,None); return env.get_graph(st, None)  # n_obs=0 -> no lidar
    graph=build(agent); s_all=env.manifold_init_slack(graph)
    ants=[new_ant() for _ in range(N)]
    offs=[goals[i]*0-starts[i]+ (starts[i]-ants[i].subtree_com[0][:2]) for i in range(N)]  # placeholder
    offs=[starts[i]-ants[i].subtree_com[0][:2].copy() for i in range(N)]
    u_ref_j=jax.jit(lambda g: env.u_ref(g,target_pos=goal0[:,:2],is_final_goal=True))
    man_j=jax.jit(lambda g,u,s: env.get_manifold_action(g,u_ref=u,s_all=s))
    tg=0.0; log=[]
    for k in range(420):
        nominal=u_ref_j(graph); safe,relax,s_all,_=man_j(graph,nominal,s_all); safe=np.array(safe)
        ag=np.array(graph.type_states(0,N)); newa=[]; geoms=[]
        for i in range(N):
            om,acc=float(safe[i,0]),float(safe[i,1]); v=ag[i,4]
            yr=10.0*v*om; vc=float(np.clip(v+10.0*acc*DT,0.0,VMAX))
            hs=float(np.clip(vc/VMAX,0.,1.)); yb=float(np.clip(0.30*yr,-0.35,0.35))
            cp=ants[i].subtree_com[0].copy()
            for _ in range(FS): ants[i].ctrl[:]=torque(ants[i],tg,hs,yb); mujoco.mj_step(mj,ants[i])
            mujoco.mj_forward(mj,ants[i]); com=ants[i].subtree_com[0].copy(); th=yaw_of(ants[i].qpos)
            exy=com[:2]+offs[i]; vel=(com[:2]-cp[:2])/DT
            newa.append([exy[0],exy[1],np.cos(th),np.sin(th),float(np.hypot(*vel))])
            geoms.append((ants[i].geom_xpos.copy(),ants[i].geom_xmat.copy().reshape(-1,3,3),np.array(offs[i])))
        tg+=SUBDT*0+FS*SUBDT
        agent=jnp.array(newa); graph=build(agent)
        dists=[float(np.hypot(*(goals[i]-newa[i][:2]))) for i in range(N)]
        rel=float(np.max(np.abs(np.array(relax)))) if relax is not None else 0.
        log.append((geoms,[a[:2]+[0] and (a[0],a[1]) for a in newa],[float(np.hypot(*(goals[i]-newa[i][:2]))) for i in range(N)],[newa[i][2:4] for i in range(N)],rel))
        if max(dists)<0.4: print("ALL REACHED step",k); break
    pickle.dump(log, open(f"{os.path.dirname(__file__)}/mf_multi_log.pkl","wb"))
    np.save(f"{os.path.dirname(__file__)}/mf_multi_goals.npy", goals)
    ds=np.array([l[2] for l in log])
    print(f"steps={len(log)} final_dists={np.round(ds[-1],2)} max_relax={max(l[4] for l in log):.4f}")
if __name__=="__main__": main()
