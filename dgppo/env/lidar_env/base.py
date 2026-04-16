import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import functools as ft
import einops as ei

from typing import NamedTuple, Tuple, Optional
from abc import ABC, abstractmethod

from jaxtyping import Float
from jaxproxqp.jaxproxqp import JaxProxQP

from ...trainer.data import Rollout
from ...utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ...utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState, Params
from ...utils.utils import merge01, jax_vmap, mask2index
from ..base import MultiAgentEnv
from dgppo.env.obstacle import Obstacle, Rectangle
from dgppo.env.plot import render_lidar
from dgppo.env.utils import get_lidar, lqr, get_node_goal_rng
from dgppo.algo.utils import get_pwise_cbf_fn, get_pwise_cbf_paper_fn, get_manifold_fn


class LidarEnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Obstacle

    @property
    def n_agent(self) -> int:
        return self.agent.shape[0]


LidarEnvGraphsTuple = GraphsTuple[State, LidarEnvState]


class LidarEnv(MultiAgentEnv, ABC):

    AGENT = 0
    GOAL = 1
    OBS = 2

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "cbf_comm_radius": 100,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.3],
        "n_obs": 3,
        "default_area_size": 1.5,
        "dist2goal": 0.01,
        "top_k_rays": 8,
        "m": 0.1,  # mass
    }

    def __init__(
            self,
            num_agents: int,
            area_size: Optional[float] = None,
            max_step: int = 256,
            dt: float = 0.03,
            params: dict = None,
            cbf_alpha: float = 10.0
    ):
        area_size = LidarEnv.PARAMS["default_area_size"] if area_size is None else area_size
        super(LidarEnv, self).__init__(num_agents, area_size, max_step, dt, params)
        
        ##==============================================================##
        A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        self._A = A * self._dt + np.eye(self.state_dim)
        self._B = (
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0 / self._params["m"], 0.0], [0.0, 1.0 / self._params["m"]]])
            * self._dt
        )
        # self._Q = np.eye(self.state_dim) * 5
        self._Q = np.diag([15.0, 15.0, 5.0, 5.0])
        self._R = np.eye(self.action_dim)
        self._K = jnp.array(lqr(self._A, self._B, self._Q, self._R))
        ##==============================================================##


        self.create_obstacles = jax_vmap(Rectangle.create)
        self.num_goals = self._num_agents

        # CBF 参数
        self.cbf_alpha = cbf_alpha
        self.k = 21  # 考虑最近的 k 个邻居
        self._cbf = None  # 延迟初始化
        self._manifold = None  # 延迟初始化: manifold 修正函数
        self._manifold_init_slack = None  # 延迟初始化: 松弛变量初始化函数
        self._safe_u_ref_jit = None  # JIT: CBF + QP
        self._get_min_lidar_dist_jit = None  # JIT: LiDAR 距离计算

    def state_to_pos_vel(self, state: Array) -> Array:
        """Convert state to [x, y, vx, vy] for manifold/CBF safety layer.
        Override in subclasses with different state representations."""
        return state  # default: state is already [x, y, vx, vy]

    def get_pos_acc_jacobian(self, state: Array) -> Array:
        """Return G matrix: q̈ = G(x) @ u, shape (n_agents, 2, action_dim).
        Override in subclasses with different dynamics."""
        # double integrator: q̈ = u / m, so G = (1/m) * I
        acc_scale = 1.0 / self._params["m"]
        n = state.shape[0]
        return jnp.broadcast_to(acc_scale * jnp.eye(2), (n, 2, 2))

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        return 7  # state dim (4) + indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_vel, vy_vel

    @property
    def action_dim(self) -> int:
        return 2  # ax, ay

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    def reset(self, key: Array) -> GraphsTuple:
        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        if n_rng_obs == 0:
            obstacles = None
        else:
            obstacle_key, key = jr.split(key, 2)
            obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
            length_key, key = jr.split(key, 2)
            obs_len = jr.uniform(
                length_key,
                (self._params["n_obs"], 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, key = jr.split(key, 2)
            obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        # randomly generate agent and goal
        states, goals = get_node_goal_rng(
            key, self.area_size, 2, self.num_agents, 2.2 * self.params["car_radius"], obstacles)
        states = jnp.concatenate(
            [states, jnp.zeros((self.num_agents, self.state_dim - states.shape[1]), dtype=states.dtype)], axis=1)
        goals = jnp.concatenate(
            [goals, jnp.zeros((self.num_goals, self.state_dim - goals.shape[1]), dtype=goals.dtype)], axis=1)

        assert states.shape == (self.num_agents, self.state_dim)
        assert goals.shape == (self.num_goals, self.state_dim)
        env_states = LidarEnvState(states, goals, obstacles)

        # get lidar data
        lidar_data = self.get_lidar_data(states, obstacles)

        return self.get_graph(env_states, lidar_data)

    def get_lidar_data(self, states: State, obstacles: Obstacle) -> Float[Array, "n_agent top_k_rays 2"]:
        lidar_data = None
        if self.params["n_obs"] > 0:
            get_lidar_vmap = jax_vmap(
                ft.partial(
                    get_lidar,
                    obstacles=obstacles,
                    num_beams=self._params["n_rays"],
                    sense_range=self._params["comm_radius"],
                    max_returns=self._params["top_k_rays"],
                )
            )
            lidar_data = get_lidar_vmap(states[:, :2])
            assert lidar_data.shape == (self.num_agents, self._params["top_k_rays"], 2)
        return lidar_data

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        """By default, use double integrator dynamics"""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = jnp.concatenate([agent_states[:, 2:], action * 10.], axis=1)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step(
            self, graph: LidarEnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[LidarEnvGraphsTuple, Reward, Cost, Done, Info]:
        # get information from graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        obstacles = graph.env_states.obstacle if self.params['n_obs'] > 0 else None

        # calculate next states
        action = self.clip_action(action)
        next_agent_states = self.agent_step_euler(agent_states, action)
        next_state = LidarEnvState(next_agent_states, goals, obstacles)
        lidar_data_next = self.get_lidar_data(next_agent_states, obstacles)
        info = {}

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        assert reward.shape == tuple()

        return self.get_graph(next_state, lidar_data_next), reward, cost, done, info

    @abstractmethod
    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        pass

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        min_dist = jnp.min(dist, axis=1)
        agent_cost: Array = self.params["car_radius"] * 2 - min_dist

        # collision between agents and obstacles
        if self.params['n_obs'] == 0:
            obs_cost = jnp.zeros((self.num_agents,)).astype(jnp.float32)
        else:
            obs_pos = graph.type_states(type_idx=2, n_type=self._params["top_k_rays"] * self.num_agents)[:, :2]
            obs_pos = jnp.reshape(obs_pos, (self.num_agents, self._params["top_k_rays"], 2))
            dist = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)  # (n_agent, top_k_rays)
            obs_cost: Array = self.params["car_radius"] - dist.min(axis=1)  # (n_agent,)

        cost = jnp.concatenate([agent_cost[:, None], obs_cost[:, None]], axis=1)
        assert cost.shape == (self.num_agents, self.n_cost)

        # add margin
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)

        return cost

    def get_subgoal_shadow_cost(self, graph: GraphsTuple, subgoal_pos: Array) -> Array:
        """
        检查subgoal是否落在LiDAR射线的阴影区（障碍物后方）

        原理：LiDAR只能看到障碍物面向agent的一面，障碍物后方是未知的危险区域。
        如果subgoal落在某条LiDAR射线的延长线上（锥形区域内）且距离超过LiDAR击中点，
        则认为subgoal可能在障碍物内部。

        Parameters
        ----------
        graph: GraphsTuple, 当前环境状态
        subgoal_pos: Array, shape (n_agents, 2), 每个agent的subgoal位置

        Returns
        -------
        cost: Array, shape (n_agents,), 每个agent的subgoal shadow cost
              -1 表示危险（subgoal在阴影区）
              0 表示安全
        """
        n_agents = self.num_agents
        # print("[DEBUG] LidarEnv.get_subgoal_shadow_cost called")  # 取消注释来调试

        # 如果没有障碍物，直接返回0
        if self.params['n_obs'] == 0:
            return jnp.zeros(n_agents)

        # 获取agent位置
        agent_pos = graph.type_states(type_idx=0, n_type=n_agents)[:, :2]  # (n_agents, 2)

        # 获取LiDAR击中点
        n_rays = self._params["top_k_rays"]
        obs_pos = graph.type_states(type_idx=2, n_type=n_rays * n_agents)[:, :2]
        obs_pos = jnp.reshape(obs_pos, (n_agents, n_rays, 2))  # (n_agents, n_rays, 2)

        # 锥形角度: θ = 360° / n_rays / 2 (弧度)
        theta = jnp.pi / self._params["n_rays"]  # 360° / n_rays / 2 = π / n_rays

        def check_single_agent(agent_p, subgoal_p, lidar_points):
            """检查单个agent的subgoal是否在阴影区"""
            # agent_p: (2,), subgoal_p: (2,), lidar_points: (n_rays, 2)

            # 计算agent到每个lidar点的方向和距离
            dir_to_lidar = lidar_points - agent_p  # (n_rays, 2)
            dist_to_lidar = jnp.linalg.norm(dir_to_lidar, axis=-1)  # (n_rays,)
            dir_to_lidar_norm = dir_to_lidar / (dist_to_lidar[:, None] + 1e-8)  # (n_rays, 2)

            # 计算agent到subgoal的方向和距离
            dir_to_subgoal = subgoal_p - agent_p  # (2,)
            dist_to_subgoal = jnp.linalg.norm(dir_to_subgoal)  # scalar
            dir_to_subgoal_norm = dir_to_subgoal / (dist_to_subgoal + 1e-8)  # (2,)

            # 计算每条lidar射线与subgoal方向的夹角
            # cos(angle) = dot(dir_lidar, dir_subgoal)
            cos_angles = jnp.sum(dir_to_lidar_norm * dir_to_subgoal_norm, axis=-1)  # (n_rays,)
            cos_angles = jnp.clip(cos_angles, -1.0, 1.0)
            angles = jnp.arccos(cos_angles)  # (n_rays,)

            # 判断条件：
            # 1. 夹角 < θ（在锥形区域内）
            # 2. subgoal距离 > lidar点距离 + car_radius（在阴影区，考虑agent半径作为安全边距）
            car_radius = self.params["car_radius"]
            in_cone = angles < theta
            in_shadow = dist_to_subgoal > (dist_to_lidar - car_radius) #  - car_radius

            # 任一lidar射线满足条件则危险
            is_dangerous = jnp.any(in_cone & in_shadow)

            return jnp.where(is_dangerous, -1.0, 0.0)

        # 对所有agent进行检查
        costs = jax_vmap(check_single_agent)(agent_pos, subgoal_pos, obs_pos)

        return costs

    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            show_subgoal: bool = False,
            subgoal_interval: int = 40,
            **kwargs
    ) -> None:
        render_lidar(rollout=rollout, video_path=video_path, side_length=self.area_size, dim=2, n_agent=self.num_agents,
                     n_rays=self.params["top_k_rays"] if self.params["n_obs"] > 0 else 0,
                     r=self.params["car_radius"], cost_components=self.cost_components,
                     Ta_is_unsafe=Ta_is_unsafe, viz_opts=viz_opts, n_goal=self.num_goals, dpi=dpi,
                     show_subgoal=show_subgoal, subgoal_interval=subgoal_interval, **kwargs)

    @abstractmethod
    def edge_blocks(self, state: LidarEnvState, lidar_data: Optional[Pos2d] = None) -> list[EdgeBlock]:
        pass

    def get_graph(self, state: LidarEnvState, lidar_data: Pos2d = None) -> GraphsTuple:
        n_hits = self._params["top_k_rays"] * self.num_agents if self.params["n_obs"] > 0 else 0
        n_nodes = self.num_agents + self.num_goals + n_hits

        if lidar_data is not None:
            lidar_data = merge01(lidar_data)

        # node features
        # states
        node_feats = jnp.zeros((self.num_agents + self.num_goals + n_hits, self.node_dim))
        node_feats = node_feats.at[: self.num_agents, :self.state_dim].set(state.agent)
        node_feats = node_feats.at[self.num_agents: self.num_agents + self.num_goals, :self.state_dim].set(state.goal)
        if lidar_data is not None:
            node_feats = node_feats.at[-n_hits:, :2].set(lidar_data)

        # indicators
        node_feats = node_feats.at[: self.num_agents, self.state_dim + 2].set(1.)  # agent
        node_feats = (
            node_feats.at[self.num_agents: self.num_agents + self.num_goals, self.state_dim + 1].set(1.))  # goal
        if n_hits > 0:
            node_feats = node_feats.at[-n_hits:, self.state_dim].set(1.)  # obs feats

        # node type
        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[: self.num_agents].set(LidarEnv.AGENT)
        node_type = node_type.at[self.num_agents: self.num_agents + self.num_goals].set(LidarEnv.GOAL)
        if n_hits > 0:
            node_type = node_type.at[-n_hits:].set(LidarEnv.OBS)

        # edge blocks
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        states = jnp.concatenate([state.agent, state.goal], axis=0)
        if lidar_data is not None:
            lidar_states = jnp.concatenate(
                [lidar_data, jnp.zeros((n_hits, self.state_dim - lidar_data.shape[1]))], axis=1)
            states = jnp.concatenate([states, lidar_states], axis=0)
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=states
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([0., 0., -0.5, -0.5])
        upper_lim = jnp.array([self.area_size, self.area_size, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.concatenate([state[:, 2:], jnp.zeros((state.shape[0], 2))], axis=1)
        g = jnp.concatenate([jnp.zeros((2, 2)), jnp.eye(2) / self._params['m']], axis=0)
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def get_agent_goals(self, graph: GraphsTuple) -> Array:
        """获取每个 agent 对应的目标位置, shape (num_agents, 2)
        默认直接从 graph 取 goal 节点。LidarLine 等环境可覆写此方法。
        """
        return graph.type_states(type_idx=1, n_type=self.num_goals)[:, :2]

    def u_ref(self, graph: GraphsTuple, target_pos: Optional[Array] = None, is_final_goal: bool = False) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        if target_pos is None:
            goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        else:
            goal_pos = target_pos
            agent_pos = agent[:, :2]

            # 计算方向和距离
            direction = goal_pos - agent_pos
            dist = jnp.linalg.norm(direction, axis=-1, keepdims=True)
            direction_unit = jnp.where(dist > 1e-6, direction / dist, 0.0)
            
            # 根据距离和是否是最终目标，设定期望速度
            max_vel = 0.7
            # 中间subgoal：保持恒定速度（最大速度的50%）
            cruise_speed = max_vel * 0.01
            # 或者根据距离调整：远离时加速，接近时减速到巡航速度
            approach_dist = 0.05
            desired_speed = jnp.where(
                dist > approach_dist,
                max_vel * 0.8,  # 远离subgoal：70%最大速度
                cruise_speed    # 接近subgoal：50%最大速度
            )
            desired_vel = direction_unit * desired_speed

            # 构造目标状态 [x, y, vx, vy]
            # 使用 jnp.where 替代 if/else
            desired_vel = jnp.where(
                is_final_goal,
                # jnp.zeros_like(desired_vel),  # 最终目标：速度为0
                # desired_vel                    # 中间subgoal：保持速度
                direction_unit * jnp.clip(dist * 0.0, 0.0, max_vel * 0.0),  # 最终目标：根据距离平滑减速到0                                                                  
                desired_vel                                                  # 中间subgoal：保持速度  
            )
            goal = jnp.concatenate([goal_pos, desired_vel], axis=-1)
    

        error = goal - agent
        error_max = jnp.abs(error / (jnp.linalg.norm(error, axis=-1, keepdims=True) + 1e-8) * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    def init_cbf(self, use_closed_form: bool = False, use_paper_cbf: bool = False, cbf_alpha1: float = 1.0, cbf_alpha2: float = 1.0, cbf_alpha: float = 10.0, **kwargs):
        """初始化 CBF 函数和 JIT 编译 - 必须在使用前调用

        Args:
            use_closed_form: 是否使用闭式解（极快，无QP求解），默认 False
            use_paper_cbf: 是否使用论文中的 relative-degree-2 CBF，默认 False
            cbf_alpha1: CBF 参数 α₁（仅当 use_paper_cbf=True 时使用）
            cbf_alpha2: CBF 参数 α₂（仅当 use_paper_cbf=True 时使用）

        Note:
            - 闭式解模式：用投影法替代 QP 求解，速度快 10-50x，精度略有损失
            - Paper CBF: 使用 relative-degree-2 CBF with conservative velocity approximation
        """
        if self._cbf is None:
            if use_paper_cbf:
                print(f"Initializing PAPER CBF function (alpha1={cbf_alpha1}, alpha2={cbf_alpha2})...")
                self._cbf = get_pwise_cbf_paper_fn(self, self.k, alpha1=cbf_alpha1, alpha2=cbf_alpha2)
                self._cbf_type = "paper"  # 标记使用的 CBF 类型
            else:
                print(f"Initializing CBF function (cbf_alpha={cbf_alpha})...")
                self._cbf = get_pwise_cbf_fn(self, self.k, cbf_alpha=cbf_alpha)
                self._cbf_type = "standard"

        if self._safe_u_ref_jit is None:
            if use_closed_form:
                print("JIT compiling safe_u_ref with CLOSED-FORM solver (fast mode)...")
                self._safe_u_ref_jit = jax.jit(self._safe_u_ref_impl_closed_form)
            else:
                print("JIT compiling safe_u_ref with QP solver...")
                self._safe_u_ref_jit = jax.jit(self._safe_u_ref_impl)

        if self._get_min_lidar_dist_jit is None:
            print("JIT compiling get_min_lidar_dist...")
            self._get_min_lidar_dist_jit = jax.jit(self._get_min_lidar_dist)

        return self

    def get_cbf(self, graph: GraphsTuple) -> tuple[Array, Array]:
        """获取 CBF 值"""
        # 注意：self._cbf 必须在 JIT 之前初始化（调用 init_cbf）
        result = self._cbf(graph)
        if len(result) == 3:  # Paper CBF returns (G, isobs, Gu)
            return result[0], result[1]  # Return only G and isobs for backward compatibility
        else:  # Standard CBF returns (h, isobs)
            return result[0], result[1]

    def get_qp_action(self, graph: GraphsTuple, u_ref: Optional[Action] = None, relax_penalty: float = 1e3) -> [Action, Array]:
        """获取 QP 过滤后的安全动作"""
        # Check if using paper CBF (which provides analytical Jacobian)
        if hasattr(self, '_cbf_type') and self._cbf_type == "paper":
            return self._get_qp_action_paper_cbf(graph, u_ref, relax_penalty)

        # Standard CBF path (autodiff-based)
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.num_agents)

        def h_aug(new_agent_state: State) -> tuple[Array, Array]:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = graph._replace(edges=new_state[graph.receivers] - new_state[graph.senders], states=new_state)
            ak_h_, ak_isobs_ = self.get_cbf(new_graph)
            return ak_h_, ak_isobs_

        def h(new_agent_state: State) -> Array:
            return h_aug(new_agent_state)[0]

        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        # (n_agents, k)
        ak_h, ak_isobs = h_aug(agent_state)
        # (n_agents, k | n_agents, nx)
        ak_hx = jax.jacfwd(h)(agent_state)

        a_dyn_f, a_dyn_g = self.control_affine_dyn(agent_state)
        ak_Lf_h = ei.einsum(ak_hx, a_dyn_f, "agent_i k agent_j nx, agent_j nx -> agent_i k")
        aka_Lg_h: Array = ei.einsum(ak_hx, a_dyn_g, "agent_i k agent_j nx, agent_j nx nu -> agent_i k agent_j nu")

        def index_fn(idx: int):
            k_Lg_h = aka_Lg_h[idx, :, idx]
            return k_Lg_h

        ak_Lg_h_self = jax_vmap(index_fn)(jnp.arange(self.num_agents))

        # 如果没有提供 u_ref，使用默认的

        au_ref = u_ref

        # (n_agents, ). 1 if agent-obs, 0.5 if agent-agent.
        ak_resp = jnp.where(ak_isobs, 1.0, 0.5)

        # construct QP
        au_opt, ar = jax_vmap(ft.partial(self._solve_qp_single, relax_penalty=relax_penalty))(
            ak_h, ak_Lf_h, ak_Lg_h_self, au_ref, ak_resp
        )
        return au_opt, ar

    def _get_qp_action_paper_cbf(self, graph: GraphsTuple, u_ref: Optional[Action] = None, relax_penalty: float = 1e3) -> [Action, Array]:
        """Paper CBF QP solver - uses pre-computed analytical Jacobian

        Paper CBF returns: (G_base, isobs, Gu) where:
        - G_base = ḧ(u=0) + α₂*ḣ + α₁*h (constraint value at u=0)
        - Gu = ∂G/∂u = -2*p_rel (analytical Jacobian)

        QP constraint becomes: G_base + Gu @ u >= 0
        """
        # Get paper CBF output with analytical Jacobian
        ak_G_base, ak_isobs, ak_Gu = self._cbf(graph)

        au_ref = u_ref
        ak_resp = jnp.where(ak_isobs, 1.0, 0.5)

        # Solve QP for each agent
        au_opt, ar = jax_vmap(ft.partial(self._solve_qp_single_paper_cbf, relax_penalty=relax_penalty))(
            ak_G_base, ak_Gu, au_ref, ak_resp
        )
        return au_opt, ar

    def _solve_qp_single(self, k_h, k_Lf_h, k_Lg_h, u_ref, k_responsibility: float, relax_penalty: float = 1e3):
        """单个 agent 的 QP 求解（JIT 兼容，无 assert）- Standard CBF"""
        n_qp_x = self.action_dim + self.k

        u_lb, u_ub = self.action_lim()

        H = jnp.eye(n_qp_x, dtype=jnp.float32)
        H = H.at[-self.k :, -self.k :].set(10.0)
        g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(self.k)], axis=0)

        k_C = -jnp.concatenate([k_Lg_h, jnp.eye(self.k)], axis=1)

        # Responsibility is one if agent-obs, half if agent-agent.
        k_b = k_responsibility * (k_Lf_h + self.cbf_alpha * k_h)

        r_lb = jnp.full(self.k, 0.0, dtype=jnp.float32)
        r_ub = jnp.full(self.k, jnp.inf, dtype=jnp.float32)

        l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        u_box = jnp.concatenate([u_ub, r_ub], axis=0)

        qp = JaxProxQP.QPModel.create(H, g, k_C, k_b, l_box, u_box)
        settings = JaxProxQP.Settings.default()
        settings.max_iter = 4

        settings.dua_gap_thresh_abs = None
        solver = JaxProxQP(qp, settings)
        sol = solver.solve()

        u_opt, r = sol.x[: self.action_dim], sol.x[-self.k :]

        return u_opt, r

    def _solve_qp_single_paper_cbf(self, k_G_base, k_Gu, u_ref, k_responsibility: float, relax_penalty: float = 1e3):
        """单个 agent 的 QP 求解 - Paper CBF (relative-degree-2)

        Paper CBF constraint: G_base + Gu @ u >= 0
        where G_base = ḧ(u=0) + α₂*ḣ + α₁*h, Gu = ∂G/∂u

        QP formulation:
        min  0.5 * ||u - u_ref||^2 + penalty * ||r||^2
        s.t. Gu @ u + r >= -G_base  (with responsibility weighting)
             u_lb <= u <= u_ub
             r >= 0
        """
        n_qp_x = self.action_dim + self.k
        u_lb, u_ub = self.action_lim()

        # Cost: minimize ||u - u_ref||^2 + penalty * ||r||^2
        H = jnp.eye(n_qp_x, dtype=jnp.float32)
        H = H.at[-self.k:, -self.k:].set(10.0)
        g = jnp.concatenate([-u_ref, relax_penalty * jnp.ones(self.k)], axis=0)

        # Constraint: -Gu @ u - r <= G_base * responsibility
        # In QP form: C @ x <= b, where x = [u, r]
        k_C = -jnp.concatenate([k_Gu, jnp.eye(self.k)], axis=1)  # (k, nu + k)
        k_b = k_responsibility * k_G_base  # (k,)

        # Box constraints
        r_lb = jnp.full(self.k, 0.0, dtype=jnp.float32)
        r_ub = jnp.full(self.k, jnp.inf, dtype=jnp.float32)
        l_box = jnp.concatenate([u_lb, r_lb], axis=0)
        u_box = jnp.concatenate([u_ub, r_ub], axis=0)

        # Solve QP
        qp = JaxProxQP.QPModel.create(H, g, k_C, k_b, l_box, u_box)
        settings = JaxProxQP.Settings.default()
        settings.max_iter = 4
        settings.dua_gap_thresh_abs = None
        solver = JaxProxQP(qp, settings)
        sol = solver.solve()

        u_opt = sol.x[:self.action_dim]
        r = sol.x[-self.k:]

        return u_opt, r

    def _solve_cbf_closed_form(self, k_h, k_Lf_h, k_Lg_h, u_ref, k_responsibility: float):
        """闭式 CBF 过滤（无 QP 求解器，极快）- Standard CBF

        CBF 约束: Lg_h @ u + Lf_h + α*h >= 0
        即: a @ u >= b, 其中 a = Lg_h, b = -Lf_h - α*h

        若违反，投影到约束边界: u_safe = u_ref + λ * a^T
        其中 λ = (b - a @ u_ref) / ||a||²
        """
        u_lb, u_ub = self.action_lim()

        # CBF 约束: Lg_h @ u >= -Lf_h - α*h (考虑责任系数)
        # k_margin[i] > 0 表示约束 i 满足
        k_b = -k_responsibility * (k_Lf_h + self.cbf_alpha * k_h)  # (k,)
        k_margin = (k_Lg_h @ u_ref) - k_b  # (k,)

        # 迭代处理每个违反的约束（最多 k 次）
        u_safe = u_ref

        def project_single_constraint(u_current, constraint_idx):
            """将 u 投影到单个约束边界"""
            a = k_Lg_h[constraint_idx]  # (nu,)
            b = k_b[constraint_idx]      # scalar

            margin = a @ u_current - b
            a_norm_sq = (a ** 2).sum() + 1e-8

            # 只在违反时修正 (margin < 0)
            lambda_proj = jnp.maximum(0, -margin / a_norm_sq)
            u_new = u_current + lambda_proj * a

            return jnp.clip(u_new, u_lb, u_ub)

        # 按违反程度排序，优先处理最严重的
        sorted_idx = jnp.argsort(k_margin)  # 最小（最违反）在前

        # 顺序投影（简单有效）
        def body_fn(i, u):
            idx = sorted_idx[i]
            return project_single_constraint(u, idx)

        u_safe = jax.lax.fori_loop(0, self.k, body_fn, u_safe)

        # 计算松弛量（用于兼容原接口）
        k_margin_final = (k_Lg_h @ u_safe) - k_b
        r = jnp.maximum(0, -k_margin_final)

        return u_safe, r

    def _solve_cbf_closed_form_paper(self, k_G_base, k_Gu, u_ref, k_responsibility: float):
        """闭式 CBF 过滤 - Paper CBF (relative-degree-2)

        Paper CBF 约束: G_base + Gu @ u >= 0
        即: a @ u >= b, 其中 a = Gu, b = -G_base

        若违反，投影到约束边界: u_safe = u_ref + λ * a^T
        其中 λ = (b - a @ u_ref) / ||a||²
        """
        u_lb, u_ub = self.action_lim()

        # Paper CBF 约束: Gu @ u >= -G_base (考虑责任系数)
        k_b = -k_responsibility * k_G_base  # (k,)
        k_margin = (k_Gu @ u_ref) - k_b  # (k,)

        u_safe = u_ref

        def project_single_constraint(u_current, constraint_idx):
            a = k_Gu[constraint_idx]  # (nu,)
            b = k_b[constraint_idx]    # scalar

            margin = a @ u_current - b
            a_norm_sq = (a ** 2).sum() + 1e-8

            lambda_proj = jnp.maximum(0, -margin / a_norm_sq)
            u_new = u_current + lambda_proj * a

            return jnp.clip(u_new, u_lb, u_ub)

        sorted_idx = jnp.argsort(k_margin)

        def body_fn(i, u):
            idx = sorted_idx[i]
            return project_single_constraint(u, idx)

        u_safe = jax.lax.fori_loop(0, self.k, body_fn, u_safe)

        k_margin_final = (k_Gu @ u_safe) - k_b
        r = jnp.maximum(0, -k_margin_final)

        return u_safe, r

    def get_qp_action_closed_form(self, graph: GraphsTuple, u_ref: Action) -> tuple[Action, Array]:
        """使用闭式解的安全动作（替代 get_qp_action）"""
        # Check if using paper CBF
        if hasattr(self, '_cbf_type') and self._cbf_type == "paper":
            return self._get_qp_action_closed_form_paper_cbf(graph, u_ref)

        # Standard CBF path
        agent_node_mask = graph.node_type == 0
        agent_node_id = mask2index(agent_node_mask, self.num_agents)

        def h_aug(new_agent_state: State) -> tuple[Array, Array]:
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = graph._replace(edges=new_state[graph.receivers] - new_state[graph.senders], states=new_state)
            ak_h_, ak_isobs_ = self.get_cbf(new_graph)
            return ak_h_, ak_isobs_

        def h(new_agent_state: State) -> Array:
            return h_aug(new_agent_state)[0]

        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        ak_h, ak_isobs = h_aug(agent_state)
        ak_hx = jax.jacfwd(h)(agent_state)

        a_dyn_f, a_dyn_g = self.control_affine_dyn(agent_state)
        ak_Lf_h = ei.einsum(ak_hx, a_dyn_f, "agent_i k agent_j nx, agent_j nx -> agent_i k")
        aka_Lg_h: Array = ei.einsum(ak_hx, a_dyn_g, "agent_i k agent_j nx, agent_j nx nu -> agent_i k agent_j nu")

        def index_fn(idx: int):
            k_Lg_h = aka_Lg_h[idx, :, idx]
            return k_Lg_h

        ak_Lg_h_self = jax_vmap(index_fn)(jnp.arange(self.num_agents))
        ak_resp = jnp.where(ak_isobs, 1.0, 0.5)

        # 使用闭式解
        au_opt, ar = jax_vmap(self._solve_cbf_closed_form)(
            ak_h, ak_Lf_h, ak_Lg_h_self, u_ref, ak_resp
        )
        return au_opt, ar

    def _get_qp_action_closed_form_paper_cbf(self, graph: GraphsTuple, u_ref: Action) -> tuple[Action, Array]:
        """Paper CBF closed-form solver"""
        ak_G_base, ak_isobs, ak_Gu = self._cbf(graph)
        ak_resp = jnp.where(ak_isobs, 1.0, 0.5)

        au_opt, ar = jax_vmap(self._solve_cbf_closed_form_paper)(
            ak_G_base, ak_Gu, u_ref, ak_resp
        )
        return au_opt, ar

    def init_manifold(self, k: int = None, K: float = 0.5, Kc: float = 100.0,
                      s_min: float = 0.1, alpha_max: float = 50.0, g_act_thresh: float = 0.1,
                      safety_margin: float = 0.02, n_lookahead: int = 2, w_slack: float = 5.0):
        """初始化 manifold 修正函数 (ATACOM v7)

        Args:
            k: 考虑最近的 k 个邻居，默认使用 self.k
            K: viability constraint 增益
            Kc: error correction 增益
            s_min: 松弛变量下界，防止 Jc 病态
            alpha_max: null space 控制量上界
            g_act_thresh: 约束激活阈值
            safety_margin: 安全边距，约束比碰撞判定更严格
            n_lookahead: 前瞻步数 (在预测位置评估约束)
            w_slack: slack 加权 (越大越优先通过 action 修正)
        """
        if k is None:
            k = self.k
        if self._manifold is None:
            print(f"Initializing manifold (k={k}, K={K}, Kc={Kc}, s_min={s_min}, "
                  f"alpha_max={alpha_max}, g_act_thresh={g_act_thresh}, "
                  f"safety_margin={safety_margin}, n_lookahead={n_lookahead}, w_slack={w_slack})...")
            self._manifold, self._manifold_init_slack = get_manifold_fn(
                self, k=k, K=K, Kc=Kc, s_min=s_min,
                alpha_max=alpha_max, g_act_thresh=g_act_thresh,
                safety_margin=safety_margin, n_lookahead=n_lookahead, w_slack=w_slack,
                state_to_pos_vel=self.state_to_pos_vel,
                get_pos_acc_jacobian=self.get_pos_acc_jacobian)
        return self

    def manifold_init_slack(self, graph: GraphsTuple):
        """初始化松弛变量 (从当前 graph 计算)"""
        if self._manifold_init_slack is None:
            raise RuntimeError("Must call init_manifold() before using manifold_init_slack")
        return self._manifold_init_slack(graph)

    def get_manifold_action(self, graph: GraphsTuple, u_ref: Action,
                            s_all=None) -> tuple[Action, Array, ...]:
        """Manifold-based 安全动作修正

        Parameters
        ----------
        graph : GraphsTuple
            当前环境状态图
        u_ref : Action
            nominal action, shape (n_agents, action_dim)
        s_all : Array, optional
            松弛变量, shape (n_agents, n_total_neighbors)

        Returns
        -------
        u_opt : Action
            修正后的安全动作, shape (n_agents, action_dim)
        relax : Array
            约束违反量, shape (n_agents, k)
        s_new : Array
            更新后的松弛变量, shape (n_agents, n_total_neighbors)
        """
        if self._manifold is None:
            raise RuntimeError("Must call init_manifold() before using get_manifold_action")
        u_opt, relax, s_new, debug_info = self._manifold(graph, u_ref, s_all)

        return u_opt, relax, s_new, debug_info
        # return u_ref, relax, s_new, debug_info

    def _safe_u_ref_impl_closed_form(self, graph: GraphsTuple, target_pos: Array, is_final_goal: bool) -> Action:
        """使用闭式解的 safe_u_ref"""
        nominal_action = self.u_ref(graph, target_pos=target_pos, is_final_goal=is_final_goal)
        action, _ = self.get_qp_action_closed_form(graph, u_ref=nominal_action)
        return action

    def _safe_u_ref_impl(self, graph: GraphsTuple, target_pos: Array, is_final_goal: bool) -> Action:
        """safe_u_ref 的内部实现（会被 JIT 编译）"""
        nominal_action = self.u_ref(graph, target_pos=target_pos, is_final_goal=is_final_goal)
        action, _ = self.get_qp_action(graph, u_ref=nominal_action)
        return action

    def _get_min_lidar_dist(self, graph: GraphsTuple) -> Array:
        """快速计算最近 LiDAR 距离（用于早期跳过判断）"""
        n_rays = self._params["top_k_rays"]
        r = self._params["car_radius"]

        # 获取 agent 和障碍物位置
        a_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]  # (n_agent, 2)
        obs_states = graph.type_states(type_idx=2, n_type=self.num_agents * n_rays)
        obs_pos = ei.rearrange(obs_states[:, :2], "(n_agent n_ray) d -> n_agent n_ray d", n_agent=self.num_agents)

        # 计算每个 agent 到其 LiDAR 点的距离
        # a_pos: (n_agent, 2), obs_pos: (n_agent, n_ray, 2)
        dist = jnp.linalg.norm(a_pos[:, None, :] - obs_pos, axis=-1)  # (n_agent, n_ray)
        dist = dist - 2 * r  # 减去两倍半径（agent + obstacle）

        return dist.min()

    def safe_u_ref(self, graph: GraphsTuple, target_pos: Optional[Array] = None, is_final_goal: bool = False) -> Action:
        """安全的参考控制器：u_ref + CBF过滤

        如果设置了 cbf_comm_radius，则只在有邻居在该范围内时才使用 CBF；
        否则总是使用 CBF。
        """
        if self._safe_u_ref_jit is None:
            raise RuntimeError("Must call init_cbf() before using safe_u_ref")

        return self._safe_u_ref_jit(graph, target_pos, is_final_goal)


