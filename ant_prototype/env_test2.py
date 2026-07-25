import sys, time, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,"/home/a5l/zihao1996.a5l/project/subgoal")
from dgppo.env.lidar_env.lidar_ant import LidarAnt
env=LidarAnt(num_agents=3, max_step=256, dt=0.05)
env.init_manifold(k=3,K=0.5,Kc=30.0,alpha_max=3.0,g_act_thresh=0.02,safety_margin=0.05,n_lookahead=0,w_slack=10.0)
def rollout(key):
    graph=env.reset(key); goals=env.get_agent_goals(graph); s=env.manifold_init_slack(graph)
    def body(carry,_):
        graph,s=carry
        nominal=env.u_ref(graph,target_pos=goals[:,:2],is_final_goal=True)
        action,_,s,_=env.get_manifold_action(graph,u_ref=nominal,s_all=s)
        graph,r,c,done,_=env.step(graph,action)
        st=graph.type_states(0,3)
        d2g=jnp.linalg.norm(st[:,:2]-goals[:,:2],axis=1)
        return (graph,s),(d2g,st[:,4])   # dist2goal, speed
    (graph,_),(d2g,spd)=jax.lax.scan(body,(graph,s),None,length=256)
    return d2g, spd, goals, graph.type_states(0,3)
t0=time.time()
d2g,spd,goals,final=jax.jit(rollout)(jax.random.PRNGKey(0))
jax.block_until_ready(d2g); print(f"compiled+ran in {time.time()-t0:.1f}s")
d2g=np.array(d2g); spd=np.array(spd)
print("dist2goal  start:",np.round(d2g[0],2)," end:",np.round(d2g[-1],2))
print("speed      max over traj:",np.round(spd.max(0),3)," end:",np.round(spd[-1],3))
print("any nan:",bool(np.isnan(d2g).any()))
print("final agent[0]:",np.round(np.array(final)[0],2))
