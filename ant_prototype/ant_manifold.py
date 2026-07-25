import os, sys, numpy as np, mujoco, jax, jax.numpy as jnp, jax.random as jr
sys.path.insert(0, "/home/a5l/zihao1996.a5l/project/subgoal")
from dgppo.env.lidar_env.lidar_bicycle_target import LidarBicycleTarget
from dgppo.env.lidar_env.base import LidarEnvState
from dgppo.utils.graph import GraphsTuple

AREA=6.0
class LidarAnt(LidarBicycleTarget):
    PARAMS={**LidarBicycleTarget.PARAMS, "car_radius":0.3, "comm_radius":3.0,
            "obs_len_range":[0.8,1.2], "n_obs":1, "default_area_size":AREA,
            "dist2goal":0.4, "top_k_rays":8, "n_rays":32, "m":0.1}

# ---- ant gait (plain mujoco) ----
XML=os.path.join(os.path.dirname(__file__),"ant.xml")
mj=mujoco.MjModel.from_xml_path(XML)
PHI=np.deg2rad(np.array([45.,135.,225.,315.])); GRP=np.array([0.,1.,0.,1.]); ASGN=np.array([1.,-1.,-1.,1.])
CTRL_LEG=np.array([3,3,0,0,1,1,2,2]); CTRL_ISANK=np.array([0,1,0,1,0,1,0,1])
QADR_H=np.array([7,9,11,13]); QADR_A=np.array([8,10,12,14]); DADR_H=np.array([6,8,10,12]); DADR_A=np.array([7,9,11,13])
GP=np.load(f"{os.path.dirname(__file__)}/best_gait.npy"); f,hip_amp,ank_mid,ank_amp,kp,kd,phase,_=[float(x) for x in GP]
K=np.sin(PHI); FS,DT=5,0.05; SUBDT=DT/FS; VMAX=0.22
GT=np.array([mj.geom_type[g] for g in range(mj.ngeom)]); GS=np.array([mj.geom_size[g] for g in range(mj.ngeom)])
def yaw_of(q): w,x,y,z=q[3:7]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def torque(d,t,hs,yb):
    th=2*np.pi*f*t+np.pi*GRP*phase; hip_des=hs*hip_amp*K*np.cos(th)+yb
    lift=ank_amp*np.maximum(0.,np.sin(th)); ank_des=ASGN*(ank_mid-lift)
    q_h,q_a=d.qpos[QADR_H],d.qpos[QADR_A]; dq_h,dq_a=d.qvel[DADR_H],d.qvel[DADR_A]
    return np.clip(np.where(CTRL_ISANK==1,(kp*(ank_des-q_a)-kd*dq_a)[CTRL_LEG],(kp*(hip_des-q_h)-kd*dq_h)[CTRL_LEG]),-1.,1.)

def main():
    env=LidarAnt(num_agents=1, area_size=AREA, max_step=400, dt=DT)
    env.init_manifold(k=3,K=0.5,Kc=30.0,alpha_max=3.0,g_act_thresh=0.02,safety_margin=0.02,n_lookahead=0,w_slack=10.0)
    # ---- hand-built scene: agent (1,3), goal (5,3), obstacle rectangle at (3, 3.5) slightly off-line ----
    agent0=jnp.array([[1.0,3.0,1.0,0.0,0.0]])                       # [x,y,cosθ,sinθ,v], facing +x
    goal0 =jnp.array([[5.0,3.0,1.0,0.0,0.0]])
    obs=env.create_obstacles(jnp.array([[3.0,3.55]]), jnp.array([1.0]), jnp.array([1.0]), jnp.array([0.0]))
    def build_graph(agent):
        st=LidarEnvState(agent, goal0, obs)
        lidar=env.get_lidar_data(agent, obs)
        return env.get_graph(st, lidar)
    graph=build_graph(agent0)
    s_all=env.manifold_init_slack(graph)
    # ---- ant MjData; offset so CoM maps to env (1,3) ----
    d=mujoco.MjData(mj); d.qpos[2]=0.32; d.qpos[QADR_A]=ASGN*ank_mid; mujoco.mj_forward(mj,d)
    for _ in range(40):
        for _ in range(FS): d.ctrl[:]=torque(d,0.,0.,0.); mujoco.mj_step(mj,d)
    com0=d.subtree_com[0].copy(); offset=np.array([1.0,3.0])-com0[:2]
    u_ref_j=jax.jit(lambda g: env.u_ref(g, target_pos=goal0[:,:2], is_final_goal=True))
    man_j =jax.jit(lambda g,u,s: env.get_manifold_action(g, u_ref=u, s_all=s))
    tg=0.0; log=[]
    for k in range(360):
        nominal=u_ref_j(graph)
        safe,relax,s_all,_=man_j(graph,nominal,s_all)
        safe=np.array(safe)[0]; om,acc=float(safe[0]),float(safe[1])
        ag=np.array(graph.type_states(0,1))[0]; v=ag[4]
        yaw_rate_cmd=10.0*v*om; v_cmd=float(np.clip(v+10.0*acc*DT,0.0,VMAX))
        hs=float(np.clip(v_cmd/VMAX,0.0,1.0)); yb=float(np.clip(0.30*yaw_rate_cmd,-0.35,0.35))
        com_prev=d.subtree_com[0].copy()
        for _ in range(FS): d.ctrl[:]=torque(d,tg,hs,yb); mujoco.mj_step(mj,d); tg+=SUBDT
        mujoco.mj_forward(mj,d)
        com=d.subtree_com[0].copy(); th=yaw_of(d.qpos)
        env_xy=com[:2]+offset
        vel=(com[:2]-com_prev[:2])/DT; v_act=float(np.hypot(*vel))
        agent_new=jnp.array([[env_xy[0],env_xy[1],np.cos(th),np.sin(th),v_act]])
        graph=build_graph(agent_new)
        dist=float(np.hypot(*(np.array([5.0,3.0])-env_xy)))
        rel=float(np.max(np.abs(np.array(relax)))) if relax is not None else 0.0
        log.append((d.geom_xpos.copy(), d.geom_xmat.copy().reshape(-1,3,3), env_xy.copy(), th, dist, rel))
        if dist<0.4: print("REACHED at step",k); break
    np.save(f"{os.path.dirname(__file__)}/manifold_log_meta.npy",
            np.array([[l[2][0],l[2][1],l[3],l[4],l[5]] for l in log]))
    import pickle
    with open(f"{os.path.dirname(__file__)}/manifold_geoms.pkl","wb") as fp:
        pickle.dump([(l[0],l[1]) for l in log], fp)
    # obstacle corners for rendering
    oc=np.array(obs.points[0]) if hasattr(obs,'points') else None
    np.save(f"{os.path.dirname(__file__)}/manifold_obs.npy", oc)
    meta=np.array([[l[2][0],l[2][1],l[3],l[4],l[5]] for l in log])
    print(f"steps={len(log)} final_dist={meta[-1,3]:.2f} max_relax={meta[:,4].max():.4f} mean_relax={meta[:,4].mean():.4f}")
    print(f"path x[{meta[:,0].min():.2f},{meta[:,0].max():.2f}] y[{meta[:,1].min():.2f},{meta[:,1].max():.2f}]")

if __name__=="__main__": main()
