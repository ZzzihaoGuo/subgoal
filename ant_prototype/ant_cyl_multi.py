# Properly-scaled multi-agent: car_radius = 0.90 (leg cylinder). Crossing conflict,
# real u_ref + real manifold + skid-steer ant. Manifold keeps 0.90-cylinders apart (>=1.8m centers).
import os, sys, numpy as np, mujoco, jax, jax.numpy as jnp, pickle
sys.path.insert(0,"/home/a5l/zihao1996.a5l/project/subgoal")
from dgppo.env.lidar_env.lidar_bicycle_target import LidarBicycleTarget
from dgppo.env.lidar_env.base import LidarEnvState
AREA=12.0; N=3; RCYL=0.90
class LidarAnt(LidarBicycleTarget):
    PARAMS={**LidarBicycleTarget.PARAMS,"car_radius":RCYL,"comm_radius":6.0,
            "obs_len_range":[1.5,2.5],"n_obs":0,"default_area_size":AREA,
            "dist2goal":1.0,"top_k_rays":8,"n_rays":32,"m":0.1}
XML=os.path.join(os.path.dirname(__file__),"ant.xml"); mj=mujoco.MjModel.from_xml_path(XML)
PHI=np.deg2rad(np.array([45.,135.,225.,315.]));GRP=np.array([0.,1.,0.,1.]);ASGN=np.array([1.,-1.,-1.,1.])
CTRL_LEG=np.array([3,3,0,0,1,1,2,2]);CTRL_ISANK=np.array([0,1,0,1,0,1,0,1])
QADR_H=np.array([7,9,11,13]);QADR_A=np.array([8,10,12,14]);DADR_H=np.array([6,8,10,12]);DADR_A=np.array([7,9,11,13])
GP=np.load(os.path.join(os.path.dirname(__file__),"best_gait.npy"));f,hip_amp,ank_mid,ank_amp,kp,kd,phase,_=[float(x) for x in GP]
K=np.sin(PHI);FS,DT=5,0.05;SUBDT=DT/FS;LEFT=np.array([1.,1.,-1.,-1.]);VMAX=0.22
def yaw_of(q): w,x,y,z=q[3:7]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def torque(d,t,turn,fwd):
    th=2*np.pi*f*t+np.pi*GRP*phase; mult=1.0+turn*(-LEFT); hip_des=fwd*hip_amp*K*mult*np.cos(th)
    lift=ank_amp*np.maximum(0.,np.sin(th)); ank_des=ASGN*(ank_mid-lift)
    q_h,q_a=d.qpos[QADR_H],d.qpos[QADR_A]; dq_h,dq_a=d.qvel[DADR_H],d.qvel[DADR_A]
    return np.clip(np.where(CTRL_ISANK==1,(kp*(ank_des-q_a)-kd*dq_a)[CTRL_LEG],(kp*(hip_des-q_h)-kd*dq_h)[CTRL_LEG]),-1.,1.)
def new_ant():
    d=mujoco.MjData(mj); d.qpos[2]=0.32; d.qpos[QADR_A]=ASGN*ank_mid; mujoco.mj_forward(mj,d)
    for _ in range(40):
        for _ in range(FS): d.ctrl[:]=torque(d,0.,0.,1.); mujoco.mj_step(mj,d)
    return d
def main():
    env=LidarAnt(num_agents=N,area_size=AREA,max_step=3000,dt=DT)
    env.init_manifold(k=4,K=0.5,Kc=30.0,alpha_max=3.0,g_act_thresh=0.02,safety_margin=0.05,n_lookahead=1,w_slack=10.0)
    man_j=jax.jit(lambda g,u,s: env.get_manifold_action(g,u_ref=u,s_all=s))
    uref_j=jax.jit(lambda g,tp: env.u_ref(g,target_pos=tp,is_final_goal=True))
    starts=np.array([[2.,3.],[2.,6.],[2.,9.]]); goals=np.array([[10.,9.],[10.,6.],[10.,3.]])  # outer swap
    agent=jnp.array([[s[0],s[1],1.,0.,0.] for s in starts]); goal0=jnp.array([[g[0],g[1],1.,0.,0.] for g in goals])
    ants=[new_ant() for _ in range(N)]; offs=[starts[i]-ants[i].subtree_com[0][:2].copy() for i in range(N)]
    def build(a): return env.get_graph(LidarEnvState(a,goal0,None),None)
    s_all=env.manifold_init_slack(build(agent)); tg=0.; log=[]; mindists=[]
    for k in range(2600):
        graph=build(agent); nominal=uref_j(graph,goal0[:,:2]); safe,relax,s_all,_=man_j(graph,nominal,s_all)
        safe=np.array(safe); ag=np.array(graph.type_states(0,N)); newa=[]; geoms=[]
        for i in range(N):
            om,acc=float(safe[i,0]),float(safe[i,1]); v=ag[i,4]; th=np.arctan2(ag[i,3],ag[i,2])
            yaw_rate=10.0*v*om; v_des=float(np.clip(v+10.0*acc*DT,0.,VMAX))
            # skid-steer: also add direct heading correction toward safe heading (helps at low v)
            th_next=th+10.0*v*om*DT
            gdir=goals[i]-ag[i,:2]
            err_safe=np.arctan2(np.sin(th_next-th),np.cos(th_next-th))
            err_goal=np.arctan2(np.sin(np.arctan2(gdir[1],gdir[0])-th),np.cos(np.arctan2(gdir[1],gdir[0])-th))
            err=err_safe if abs(err_safe)>1e-3 else err_goal    # manifold heading if it acts, else goal
            turn=float(np.clip(2.44*yaw_rate+0.8*err,-0.8,0.8)); fwd=float(np.clip(v_des/VMAX,0.35,1.0))
            dist=float(np.hypot(*gdir))
            if dist<env.params["dist2goal"]: turn,fwd=0.,0.
            cp=ants[i].subtree_com[0].copy()
            for _ in range(FS): ants[i].ctrl[:]=torque(ants[i],tg,turn,fwd); mujoco.mj_step(mj,ants[i])
            mujoco.mj_forward(mj,ants[i]); com=ants[i].subtree_com[0].copy(); thn=yaw_of(ants[i].qpos)
            exy=com[:2]+offs[i]; vel=(com[:2]-cp[:2])/DT
            newa.append([exy[0],exy[1],np.cos(thn),np.sin(thn),float(np.hypot(*vel))])
            geoms.append((ants[i].geom_xpos.copy(),ants[i].geom_xmat.copy().reshape(-1,3,3),np.array(offs[i])))
        tg+=FS*SUBDT; agent=jnp.array(newa)
        pos=np.array([a[:2] for a in newa])
        pd=np.min([np.hypot(*(pos[a]-pos[b])) for a in range(N) for b in range(a+1,N)])
        mindists.append(pd)
        dists=[float(np.hypot(*(goals[i]-pos[i]))) for i in range(N)]
        rel=float(np.max(np.abs(np.array(relax)))) if relax is not None else 0.
        log.append((geoms,[tuple(p) for p in pos],dists,[a[2:4] for a in newa],rel,pd))
        if max(dists)<env.params["dist2goal"]: print("ALL REACHED step",k); break
    pickle.dump(log,open(os.path.join(os.path.dirname(__file__),"cyl_multi_log.pkl"),"wb"))
    np.save(os.path.join(os.path.dirname(__file__),"cyl_multi_meta.npy"),
            np.array([goals.flatten(),[RCYL,AREA,N,0,0,0]],dtype=object),allow_pickle=True)
    md=np.array(mindists)
    print(f"steps={len(log)} min_pairwise_dist={md.min():.2f} (safe threshold 2r={2*RCYL:.1f}) "
          f"final_dists={np.round(np.array(log[-1][2]),2)} max_relax={max(l[4] for l in log):.3f}")
if __name__=="__main__": main()
