import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,"/home/a5l/zihao1996.a5l/project/subgoal")
from dgppo.env.lidar_env.lidar_ant import LidarAnt
env=LidarAnt(num_agents=3, max_step=200, dt=0.05)
env.init_manifold(k=3,K=0.5,Kc=30.0,alpha_max=3.0,g_act_thresh=0.02,safety_margin=0.05,n_lookahead=0,w_slack=10.0)
print("state_dim",env.state_dim,"node_dim",env.node_dim,"edge_dim",env.edge_dim,"action_dim",env.action_dim,"n_cost",env.n_cost)
key=jax.random.PRNGKey(0)
graph=jax.jit(env.reset)(key)
print("reset OK; agent state[0]=",np.round(np.array(graph.type_states(0,3)[0]),3))
goals=env.get_agent_goals(graph)
s=env.manifold_init_slack(graph)
uref_j=jax.jit(lambda g,tp: env.u_ref(g,target_pos=tp,is_final_goal=True))
man_j=jax.jit(lambda g,u,s: env.get_manifold_action(g,u_ref=u,s_all=s))
step_j=jax.jit(env.step)
import time
d0=np.linalg.norm(np.array(graph.type_states(0,3))[:,:2]-np.array(goals)[:,:2],axis=1)
t0=time.time()
for k in range(150):
    nominal=uref_j(graph,goals[:,:2])
    action,_,s,_=man_j(graph,nominal,s)
    graph,r,c,done,_=step_j(graph,action)
    if k==0: print(f"first step compiled in {time.time()-t0:.1f}s")
    if jnp.any(jnp.isnan(graph.type_states(0,3))): print("NaN at step",k); break
pos=np.array(graph.type_states(0,3))[:,:2]; d1=np.linalg.norm(pos-np.array(goals)[:,:2],axis=1)
print(f"dist2goal start={np.round(d0,2)} -> end={np.round(d1,2)}")
print(f"agent speeds end={np.round(np.array(graph.type_states(0,3))[:,4],3)}")
print(f"150 steps in {time.time()-t0:.1f}s")
