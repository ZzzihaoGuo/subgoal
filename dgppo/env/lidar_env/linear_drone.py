import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy
import functools as ft

from typing import Optional, Tuple

from dgppo.utils.graph import EdgeBlock
from dgppo.utils.typing import Action, Array, Pos3d, Reward, State
from dgppo.utils.utils import jax_vmap, merge01
from dgppo.env.lidar_env.base import LidarEnv, LidarEnvState, LidarEnvGraphsTuple
from dgppo.env.obstacle import Sphere
from dgppo.env.utils import get_lidar, lqr, get_node_goal_rng, inside_obstacles


class LinearDrone(LidarEnv):
    """3D linear drone environment inheriting from LidarEnv.

    State: [x, y, z, vx, vy, vz] (6D)
    Action: [ax, ay, az] (3D)
    Obstacles: Spheres (3D)
    """

    AGENT = 0
    GOAL = 1
    OBS = 2

    GOAL_ASSIGNMENT = "target"  # agent_i <-> goal_i

    PARAMS = {
        "car_radius": 0.05,        # drone radius (reuse car_radius name for compatibility)
        "comm_radius": 0.5,
        "cbf_comm_radius": 100.0,
        "n_rays": 32,
        "obs_len_range": [0.08, 0.16],  # obstacle diameter range
        "n_obs": 4,
        "default_area_size": 0.8,
        "dist2goal": 0.02,
        "top_k_rays": 16,
        "m": 0.1,  # mass (used for compatibility, actual dynamics use _B)
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 256,
            dt: float = 0.03,
            params: dict = None,
            cbf_alpha: float = 10.0,
    ):
        area_size = LinearDrone.PARAMS["default_area_size"] if area_size is None else area_size
        # Skip LidarEnv.__init__'s 2D LQR setup, call MultiAgentEnv.__init__ directly
        # then do our own 3D LQR
        from dgppo.env.base import MultiAgentEnv
        MultiAgentEnv.__init__(self, num_agents, area_size, max_step, dt, params)

        # 3D linear drone dynamics: x_dot = A @ x + B @ u
        A_cont = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        A_cont[0, 3] = 1.0
        A_cont[1, 4] = 1.0
        A_cont[2, 5] = 1.0
        A_cont[3, 3] = -1.1
        A_cont[4, 4] = -1.1
        A_cont[5, 5] = -6.0
        A_discrete = scipy.linalg.expm(A_cont * self._dt)
        self._A_cont = A_cont
        self._A = A_discrete

        self._B = np.zeros((self.state_dim, self.action_dim), dtype=np.float32)
        self._B[3, 0] = 10.0
        self._B[4, 1] = 10.0
        self._B[5, 2] = 10.0

        self._Q = np.diag([5e1, 5e1, 5e1, 1e1, 1e1, 1e1]).astype(np.float32)
        self._R = np.eye(self.action_dim, dtype=np.float32)
        self._K = jnp.array(lqr(A_discrete, self._B, self._Q, self._R))

        # 3D sphere obstacles
        self.create_obstacles = jax_vmap(Sphere.create)
        self.num_goals = self._num_agents

        # CBF/manifold (not yet supported for 3D, but init slots)
        self.cbf_alpha = cbf_alpha
        self.k = 21
        self._cbf = None
        self._manifold = None
        self._manifold_init_slack = None
        self._safe_u_ref_jit = None
        self._get_min_lidar_dist_jit = None

    # ======== Dimensions (3D overrides) ========

    @property
    def state_dim(self) -> int:
        return 6  # x, y, z, vx, vy, vz

    @property
    def node_dim(self) -> int:
        return 9  # state_dim(6) + indicator(3): agent=001, goal=010, obs=100

    @property
    def edge_dim(self) -> int:
        return 6  # x_rel, y_rel, z_rel, vx_rel, vy_rel, vz_rel

    @property
    def action_dim(self) -> int:
        return 3  # ax, ay, az

    @property
    def n_cost(self) -> int:
        return 2  # agent-agent, agent-obstacle

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    # ======== Reset (3D) ========

    def reset(self, key: Array) -> LidarEnvGraphsTuple:
        # randomly generate sphere obstacles
        n_rng_obs = self._params["n_obs"]
        if n_rng_obs == 0:
            obstacles = None
        else:
            obstacle_key, key = jr.split(key, 2)
            obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 3), minval=0, maxval=self.area_size)
            r_key, key = jr.split(key, 2)
            obs_radius = jr.uniform(
                r_key, (n_rng_obs,),
                minval=self._params["obs_len_range"][0] / 2,
                maxval=self._params["obs_len_range"][1] / 2,
            )
            obstacles = self.create_obstacles(obs_pos, obs_radius)

        # randomly generate agent and goal positions (3D)
        states, goals = get_node_goal_rng(
            key, self.area_size, 3, self.num_agents,
            2.2 * self._params["car_radius"], obstacles,
        )

        # add zero velocity
        states = jnp.concatenate([states, jnp.zeros((self.num_agents, 3))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_goals, 3))], axis=1)

        env_states = LidarEnvState(states, goals, obstacles)
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    # ======== LiDAR (3D) ========

    def get_lidar_data(self, states, obstacles):
        if self._params["n_obs"] == 0 or obstacles is None:
            return None
        get_lidar_vmap = jax_vmap(
            ft.partial(
                get_lidar,
                obstacles=obstacles,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
                max_returns=self._params["top_k_rays"],
            )
        )
        lidar_data = get_lidar_vmap(states[:, :3])  # 3D positions
        assert lidar_data.shape == (self.num_agents, self._params["top_k_rays"], 3)
        return lidar_data

    # ======== Dynamics (3D linear drone) ========

    def agent_step_euler(self, agent_states, action):
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.matmul(agent_states, self._A_cont.T) + jnp.matmul(action, self._B.T)
        n_state = agent_states + x_dot * self._dt
        return self.clip_state(n_state)

    # ======== Reward ========

    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        reward = jnp.zeros(()).astype(jnp.float32)

        # distance to goal
        agent_pos = agent_states[:, :3]
        goal_pos = goals[:, :3]
        dist2goal = jnp.linalg.norm(agent_pos - goal_pos, axis=-1)
        reward -= dist2goal.mean() * 0.01

        # not reaching goal penalty
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001

        # action penalty
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001

        return reward

    # ======== Cost (3D, returns (n_agents, n_cost)) ========

    def get_cost(self, graph) -> Array:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_states[:, :3]

        # agent-agent collision
        dist = jnp.linalg.norm(
            jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1
        )
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost = self._params["car_radius"] * 2 - min_dist  # (n_agents,)

        # agent-obstacle collision
        if self._params["n_obs"] == 0:
            obs_cost = jnp.zeros(self.num_agents)
        else:
            n_rays = self._params["top_k_rays"]
            obs_states = graph.type_states(type_idx=2, n_type=n_rays * self.num_agents)
            obs_pos = obs_states[:, :3].reshape(self.num_agents, n_rays, 3)
            obs_dist = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)  # (n_agents, n_rays)
            obs_cost = self._params["car_radius"] - obs_dist.min(axis=1)  # (n_agents,)

        cost = jnp.stack([agent_cost, obs_cost], axis=1)  # (n_agents, 2)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)

        return cost

    # ======== Graph building (3D) ========

    def get_graph(self, state: LidarEnvState, lidar_data=None):
        n_rays = self._params["top_k_rays"]
        n_hits = n_rays * self.num_agents if self._params["n_obs"] > 0 and lidar_data is not None else 0
        n_nodes = self.num_agents + self.num_goals + n_hits

        if lidar_data is not None:
            lidar_data = merge01(lidar_data)  # (n_agents * top_k_rays, 3)

        # node features: state + indicator
        node_feats = jnp.zeros((n_nodes, self.node_dim))
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(state.agent)
        node_feats = node_feats.at[self.num_agents:self.num_agents + self.num_goals, :self.state_dim].set(state.goal)
        if lidar_data is not None:
            node_feats = node_feats.at[-n_hits:, :3].set(lidar_data)

        # indicators
        node_feats = node_feats.at[:self.num_agents, self.state_dim + 2].set(1.0)  # agent
        node_feats = node_feats.at[self.num_agents:self.num_agents + self.num_goals, self.state_dim + 1].set(1.0)  # goal
        if n_hits > 0:
            node_feats = node_feats.at[-n_hits:, self.state_dim].set(1.0)  # obs

        # node type
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(self.AGENT)
        node_type = node_type.at[self.num_agents:self.num_agents + self.num_goals].set(self.GOAL)
        if n_hits > 0:
            node_type = node_type.at[-n_hits:].set(self.OBS)

        # edge blocks
        edge_blocks = self.edge_blocks(state, lidar_data)

        # states for graph
        from dgppo.utils.graph import GetGraph
        states = jnp.concatenate([state.agent, state.goal], axis=0)
        if lidar_data is not None:
            lidar_states = jnp.concatenate(
                [lidar_data, jnp.zeros((n_hits, self.state_dim - 3))], axis=1
            )
            states = jnp.concatenate([states, lidar_states], axis=0)

        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=states,
        ).to_padded()

    def edge_blocks(self, state: LidarEnvState, lidar_data=None) -> list[EdgeBlock]:
        # agent-agent edges
        agent_pos = state.agent[:, :3]
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(self.num_agents) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent-goal edges: agent_i <-> goal_i 一一对应 (target mode)
        agent_goal_edges = []
        for i_agent in range(self.num_agents):
            agent_state_i = state.agent[i_agent]
            goal_state_i = state.goal[i_agent]
            agent_goal_feats_i = agent_state_i - goal_state_i
            agent_goal_edges.append(EdgeBlock(
                agent_goal_feats_i[None, None, :], jnp.ones((1, 1)),
                jnp.array([i_agent]), jnp.array([i_agent + self.num_agents])
            ))

        # agent-obs edges
        agent_obs_edges = []
        n_rays = self._params["top_k_rays"]
        if lidar_data is not None:
            n_hits = n_rays * self.num_agents
            id_obs = jnp.arange(self.num_agents + self.num_goals, self.num_agents + self.num_goals + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * n_rays, (i + 1) * n_rays)
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :3]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                agent_obs_mask = jnp.ones((1, n_rays))
                agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
                # pad to edge_dim
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((n_rays, self.edge_dim - 3))], axis=-1
                )
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
                )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges

    # ======== State/action limits (3D) ========

    def state_lim(self, state=None):
        low = jnp.array([0.0, 0.0, 0.0, -0.5, -0.5, -0.5])
        high = jnp.array([self.area_size, self.area_size, self.area_size, 0.5, 0.5, 0.5])
        return low, high

    def action_lim(self):
        return jnp.ones(3) * -1.0, jnp.ones(3) * 1.0

    # ======== u_ref (3D LQR, with subgoal support) ========

    def get_agent_goals(self, graph) -> Array:
        """返回 (n_agents, 3) 目标位置"""
        return graph.type_states(type_idx=1, n_type=self.num_goals)[:, :3]

    def u_ref(self, graph, target_pos=None, is_final_goal=False) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        if target_pos is None:
            goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        else:
            goal_pos = target_pos  # (n_agents, 3)
            agent_pos = agent[:, :3]

            direction = goal_pos - agent_pos
            dist = jnp.linalg.norm(direction, axis=-1, keepdims=True)
            direction_unit = jnp.where(dist > 1e-6, direction / dist, 0.0)

            # 与 LidarTarget 一致：subgoal 到达时近乎停住 (stop-and-go)
            max_vel = 1.5
            approach_dist = 0.02
            cruise_speed = max_vel * 0.2  # 中间 subgoal 不减速, 和远处同速恒速巡航
            desired_speed = jnp.where(
                dist > approach_dist,
                max_vel * 1,  # 远离 subgoal: 80% max_vel
                cruise_speed,
            )
            desired_vel = direction_unit * desired_speed
            desired_vel = jnp.where(
                is_final_goal,
                jnp.zeros_like(desired_vel),  # 最终目标: 速度为 0
                desired_vel,
            )
            goal = jnp.concatenate([goal_pos, desired_vel], axis=-1)

        error = goal - agent
        error_max = jnp.abs(
            error / (jnp.linalg.norm(error, axis=-1, keepdims=True) + 1e-8) * self._params["comm_radius"]
        )
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    # ======== Control affine dynamics (3D) ========

    def control_affine_dyn(self, state):
        assert state.ndim == 2
        f = jnp.matmul(state, self._A_cont.T)
        g = jnp.expand_dims(jnp.array(self._B), axis=0).repeat(state.shape[0], axis=0)
        return f, g

    # ======== Manifold support (3D) ========

    def state_to_pos_vel(self, state: Array) -> Array:
        """state is already [x, y, z, vx, vy, vz] — identity"""
        return state

    def get_pos_acc_jacobian(self, states: Array) -> Array:
        """G matrix: q̈ = G @ u, shape (n_agents, 3, 3)
        For linear drone: B[3:6, :] = diag(10, 10, 10), so G = 10 * I_3
        """
        n = states.shape[0]
        # B[3,0]=10, B[4,1]=10, B[5,2]=10
        G = jnp.array(self._B[3:, :])  # (3, 3)
        return jnp.broadcast_to(G, (n, 3, 3))

    def get_subgoal_shadow_cost(self, graph, subgoal_pos):
        """3D 环境不使用 2D LiDAR 锥形阴影检测，直接返回 0"""
        return jnp.zeros(self.num_agents)

    def init_manifold(self, k=None, K=0.5, Kc=100.0, s_min=0.1, alpha_max=50.0,
                      g_act_thresh=0.1, safety_margin=0.02, n_lookahead=2, w_slack=5.0):
        """初始化 3D manifold (dim_q=3)"""
        if k is None:
            k = self.k
        if self._manifold is None:
            from dgppo.algo.utils import get_manifold_fn
            print(f"Initializing 3D manifold (k={k}, dim_q=3)...")
            self._manifold, self._manifold_init_slack = get_manifold_fn(
                self, k=k, K=K, Kc=Kc, s_min=s_min,
                alpha_max=alpha_max, g_act_thresh=g_act_thresh,
                safety_margin=safety_margin, n_lookahead=n_lookahead, w_slack=w_slack,
                state_to_pos_vel=self.state_to_pos_vel,
                get_pos_acc_jacobian=self.get_pos_acc_jacobian,
                dim_q=3,
            )
        return self

    # ======== Render (3D) ========

    def render_video(self, rollout, video_path, Ta_is_unsafe=None, viz_opts=None, dpi=100, **kwargs):
        from dgppo.env.plot import render_lidar
        render_lidar(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=3,
            n_agent=self.num_agents,
            n_rays=self._params["top_k_rays"] if self._params["n_obs"] > 0 else 0,
            r=self._params["car_radius"],
            cost_components=self.cost_components,
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            n_goal=self.num_goals,
            dpi=dpi,
            **kwargs,
        )
