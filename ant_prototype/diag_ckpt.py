import os, sys, yaml, numpy as np, jax, jax.numpy as jnp, jax.random as jr
sys.path.insert(0,"/home/a5l/zihao1996.a5l/project/subgoal")
from dgppo.env import ENV, LIDAR_ENVS
from dgppo.env.lidar_env.lidar_ant import LidarAnt
ENV['LidarAnt']=LidarAnt; LIDAR_ENVS.add('LidarAnt')
from dgppo.env import make_env
from dgppo.algo import make_algo
P="logs/LidarAnt/informarl_subgoal/seed0_725020512_SCWX"
config=yaml.load(open(os.path.join(P,"config.yaml")),Loader=yaml.UnsafeLoader)
env=make_env(env_id=config.env,num_agents=config.num_agents,num_obs=0,n_rays=config.n_rays,max_step=256)
def mk():
    return make_algo(algo=config.algo,env=env,node_dim=env.node_dim,edge_dim=env.edge_dim,
        state_dim=env.state_dim,action_dim=env.action_dim,n_agents=env.num_agents,cost_weight=config.cost_weight,
        actor_gnn_layers=config.actor_gnn_layers,Vl_gnn_layers=config.Vl_gnn_layers,
        Vh_gnn_layers=getattr(config,"Vh_gnn_layers",1),lr_actor=config.lr_actor,lr_Vl=config.lr_Vl,
        max_grad_norm=2.0,seed=config.seed,use_rnn=config.use_rnn,rnn_layers=config.rnn_layers,
        use_lstm=config.use_lstm,use_relative_subgoal=getattr(config,"relative_subgoal",True),
        max_delta=getattr(config,"max_delta",2.5))
graph=env.reset(jr.PRNGKey(1))
import jax.tree_util as jtu
def pnorm(params): return float(sum(jnp.sum(x**2) for x in jtu.tree_leaves(params)))
for step in (0,200,400):
    algo=mk(); algo.load(os.path.join(P,"models"),step)
    sg,_=algo.act(graph, algo.init_rnn_state)
    print(f"step {step}: policy_param_L2={pnorm(algo.params['policy']):.4f}  subgoal[0]={np.round(np.array(sg)[0],3)}")
