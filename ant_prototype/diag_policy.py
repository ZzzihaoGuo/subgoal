import os, sys, yaml, numpy as np, jax, jax.numpy as jnp, jax.random as jr
sys.path.insert(0,"/home/a5l/zihao1996.a5l/project/subgoal")
from dgppo.env import ENV, LIDAR_ENVS
from dgppo.env.lidar_env.lidar_ant import LidarAnt
ENV['LidarAnt']=LidarAnt; LIDAR_ENVS.add('LidarAnt')
from dgppo.env import make_env
from dgppo.algo import make_algo
P="logs/LidarAnt/informarl_subgoal/seed0_725020512_SCWX"
config=yaml.load(open(os.path.join(P,"config.yaml")),Loader=yaml.UnsafeLoader)
print("config: relative_subgoal =", getattr(config,'relative_subgoal','MISSING'), " max_delta =", getattr(config,'max_delta','MISSING'))
env=make_env(env_id=config.env,num_agents=config.num_agents,num_obs=0,n_rays=config.n_rays,max_step=256)
algo=make_algo(algo=config.algo,env=env,node_dim=env.node_dim,edge_dim=env.edge_dim,state_dim=env.state_dim,
    action_dim=env.action_dim,n_agents=env.num_agents,cost_weight=config.cost_weight,
    actor_gnn_layers=config.actor_gnn_layers,Vl_gnn_layers=config.Vl_gnn_layers,Vh_gnn_layers=getattr(config,"Vh_gnn_layers",1),
    lr_actor=config.lr_actor,lr_Vl=config.lr_Vl,max_grad_norm=2.0,seed=config.seed,use_rnn=config.use_rnn,
    rnn_layers=config.rnn_layers,use_lstm=config.use_lstm,
    use_relative_subgoal=getattr(config,"relative_subgoal",True),max_delta=getattr(config,"max_delta",2.5))
algo.load(os.path.join(P,"models"),1000)
pol=algo.policy
print("policy.use_relative_subgoal =", pol.use_relative_subgoal, " max_delta =", pol.max_delta, " area =", pol.area_size)
for seed in (1,1000,5):
    g=env.reset(jr.PRNGKey(seed))
    dist,_=pol.dist.apply(algo.params['policy'], g, algo.init_rnn_state, n_agents=env.num_agents)
    tanh=np.array(dist.mode())
    pos=np.array(g.type_states(0,env.num_agents))[:,:2]
    sg,_=algo.act(g, algo.init_rnn_state)
    print(f"\nseed {seed}: agent_pos={np.round(pos,2).tolist()}")
    print(f"   raw tanh(mode)={np.round(tanh,3).tolist()}")
    print(f"   subgoal={np.round(np.array(sg),2).tolist()}")
