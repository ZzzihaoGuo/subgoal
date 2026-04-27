import functools as ft
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.linalg

from typing import Optional, Tuple

from dgppo.utils.graph import EdgeBlock, GetGraph
from dgppo.utils.typing import Action, Array, Reward
from dgppo.utils.utils import jax_vmap, merge01
from dgppo.env.base import MultiAgentEnv
from dgppo.env.lidar_env.base import LidarEnvState, LidarEnvGraphsTuple
from dgppo.env.lidar_env.linear_drone import LinearDrone
from dgppo.env.obstacle import Sphere
from dgppo.env.utils import RK4_step, get_lidar, get_node_goal_rng


def _get_rotmat(phi, theta, psi):
    c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
    c_th, s_th = jnp.cos(theta), jnp.sin(theta)
    c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
    return jnp.array([
        [c_psi * c_th, c_psi * s_th * s_phi - s_psi * c_phi, c_psi * s_th * c_phi + s_psi * s_phi],
        [s_psi * c_th, s_psi * s_th * s_phi + c_psi * c_phi, s_psi * s_th * c_phi - c_psi * s_phi],
        [-s_th,         c_th * s_phi,                         c_th * c_phi],
    ])


def _lqr_continuous(A, B, Q, R):
    """Continuous-time LQR: u = -K x, solves A^T P + P A - P B R^-1 B^T P + Q = 0."""
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K


class CrazyFlie(LinearDrone):
    """3D non-linear quadrotor (12-state) with LQR-wrapped 3-D velocity action.

    State (12): [x, y, z, psi, theta, phi, u, v, w, r, q, p]
      - (x, y, z): world-frame position
      - (psi, theta, phi): ZYX Euler angles (yaw, pitch, roll)
      - (u, v, w): body-frame linear velocity
      - (r, q, p): body-frame angular velocity (stored in this order)

    Action (3): [vx, vy, vz] in [-1, 1] (world-frame velocity targets, scaled).
    Yaw rate target is hard-wired to 0.

    Low-level: LQR tracks velocity targets → 4 motor thrusts.
    Manifold: dim_q = 3 (position), action_dim = 3 (matches).
    No CBF support.
    """

    AGENT = 0
    GOAL = 1
    OBS = 2

    GOAL_ASSIGNMENT = "target"

    X, Y, Z, PSI, THETA, PHI, U, V, W, R, Q, P = range(12)
    F_1, F_2, F_3, F_4 = range(4)
    L_PHI, L_THETA, L_PSI, L_P, L_Q, L_R, L_VX, L_VY, L_VZ = range(9)

    PARAMS = {
        "car_radius": 0.05,           # drone radius (named "car_radius" for base-class compatibility)
        "comm_radius": 1.0,
        "cbf_comm_radius": 100.0,
        "n_rays": 32,
        "obs_len_range": [0.08, 0.16],
        "n_obs": 4,
        "default_area_size": 0.8,
        "dist2goal": 0.05,
        "top_k_rays": 16,
        "m": 0.0299,
        "Ixx": 1.395e-5,
        "Iyy": 1.395e-5,
        "Izz": 2.173e-5,
        "CT": 3.1582e-10,
        "CD": 7.9379e-12,
        "d": 0.03973,
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
        area_size = self.PARAMS["default_area_size"] if area_size is None else area_size
        MultiAgentEnv.__init__(self, num_agents, area_size, max_step, dt, params)

        self.normalize_by_CT = True
        self.vel_targets_scale = jnp.array([2.0, 2.0, 0.5], dtype=jnp.float32)  # 3-dim (yaw cut)

        # Low-level LQR: vel-target → motor thrust
        self._K_ll = jnp.array(self._compute_K_ll())

        self.create_obstacles = jax_vmap(Sphere.create)
        self.num_goals = self._num_agents

        # CBF / manifold slots (no CBF used)
        self.cbf_alpha = cbf_alpha
        self.k = 21
        self._cbf = None
        self._manifold = None
        self._manifold_init_slack = None
        self._safe_u_ref_jit = None
        self._get_min_lidar_dist_jit = None

    # ========== Dimensions ==========

    @property
    def state_dim(self) -> int:
        return 12

    @property
    def node_dim(self) -> int:
        return 15  # 12 state + 3 indicator

    @property
    def edge_dim(self) -> int:
        return 12

    @property
    def action_dim(self) -> int:
        return 3  # yaw cut

    @property
    def n_cost(self) -> int:
        return 2

    @property
    def cost_components(self) -> Tuple[str, ...]:
        return "agent collisions", "obs collisions"

    # ========== Reset ==========

    def reset(self, key: Array) -> LidarEnvGraphsTuple:
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

        states_pos, goals_pos = get_node_goal_rng(
            key, self.area_size, 3, self.num_agents,
            2.2 * self._params["car_radius"], obstacles,
        )

        # full 12-dim state: [pos, euler=0, body_vel=0, body_omega=0]
        zero_rest = jnp.zeros((self.num_agents, self.state_dim - 3))
        states = jnp.concatenate([states_pos, zero_rest], axis=1)
        goals = jnp.concatenate([goals_pos, jnp.zeros((self.num_goals, self.state_dim - 3))], axis=1)

        env_states = LidarEnvState(states, goals, obstacles)
        lidar_data = self.get_lidar_data(states, obstacles)
        return self.get_graph(env_states, lidar_data)

    # ========== LiDAR (3D, same as LinearDrone) ==========

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
        lidar_data = get_lidar_vmap(states[:, :3])
        assert lidar_data.shape == (self.num_agents, self._params["top_k_rays"], 3)
        return lidar_data

    # ========== Dynamics (CrazyFlie 12-state non-linear) ==========

    @staticmethod
    def get_rotation_mat(x):
        return _get_rotmat(x[CrazyFlie.PHI], x[CrazyFlie.THETA], x[CrazyFlie.PSI])

    def _single_agent_f(self, x):
        Ixx, Iyy, Izz = self._params["Ixx"], self._params["Iyy"], self._params["Izz"]
        I = jnp.array([Ixx, Iyy, Izz])

        phi, theta = x[CrazyFlie.PHI], x[CrazyFlie.THETA]
        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_th = jnp.cos(theta)
        t_th = jnp.tan(theta)

        uvw = jnp.array([x[CrazyFlie.U], x[CrazyFlie.V], x[CrazyFlie.W]])
        pqr = jnp.array([x[CrazyFlie.P], x[CrazyFlie.Q], x[CrazyFlie.R]])

        R_W_cf = self.get_rotation_mat(x)
        v_W = R_W_cf @ uvw

        # Euler-angle dynamics: [d psi, d theta, d phi] from body pqr
        mat = jnp.array([
            [0.0, s_phi / c_th, c_phi / c_th],
            [0.0, c_phi,        -s_phi],
            [1.0, s_phi * t_th, c_phi * t_th],
        ])
        deuler_ypr = mat @ pqr

        # Body-frame linear acceleration (gravity + Coriolis)
        acc_cf_g = -R_W_cf[2, :] * 9.81
        acc_cf = -jnp.cross(pqr, uvw) + acc_cf_g

        # Body-frame angular acceleration (Euler's eq, drift only)
        pqr_dot = -jnp.cross(pqr, I * pqr) / I
        rpq_dot = pqr_dot[::-1]  # state layout is (r, q, p)

        return jnp.concatenate([v_W, deuler_ypr, acc_cf, rpq_dot])

    def _single_agent_gu(self, control):
        m, Ixx, Iyy, Izz = self._params["m"], self._params["Ixx"], self._params["Iyy"], self._params["Izz"]
        CT, CD, d = self._params["CT"], self._params["CD"], self._params["d"]
        if self.normalize_by_CT:
            CT, CD = 1.0, CD / CT

        w_term = jnp.sum(control)
        p_term = jnp.sum(control * jnp.array([-1.0, -1.0, 1.0, 1.0]))
        q_term = jnp.sum(control * jnp.array([-1.0, 1.0, 1.0, -1.0]))
        r_term = jnp.sum(control * jnp.array([-1.0, 1.0, -1.0, 1.0]))

        w_dot = CT * w_term / m
        p_dot = CT * np.sqrt(2) * d * p_term / Ixx
        q_dot = CT * np.sqrt(2) * d * q_term / Iyy
        r_dot = CD * r_term / Izz

        gu = jnp.zeros(self.state_dim)
        gu = gu.at[CrazyFlie.W].set(w_dot)
        gu = gu.at[CrazyFlie.P].set(p_dot)
        gu = gu.at[CrazyFlie.Q].set(q_dot)
        gu = gu.at[CrazyFlie.R].set(r_dot)
        return gu

    @property
    def u_eq(self):
        u_eq = jnp.zeros(4)
        u_eq = u_eq.at[CrazyFlie.F_1].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_2].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_3].set(self._params["m"] * 9.81 / 4)
        u_eq = u_eq.at[CrazyFlie.F_4].set(self._params["m"] * 9.81 / 4)
        if not self.normalize_by_CT:
            u_eq = u_eq / self._params["CT"]
        return u_eq

    def _xdot_ll(self, x, u):
        """Linearized body-rate + world-vel dynamics for LQR design (9-dim)."""
        m, Ixx, Iyy, Izz = self._params["m"], self._params["Ixx"], self._params["Iyy"], self._params["Izz"]
        CT, CD, d = self._params["CT"], self._params["CD"], self._params["d"]
        if self.normalize_by_CT:
            CT, CD = 1.0, CD / CT

        phi = x[CrazyFlie.L_PHI]
        theta = x[CrazyFlie.L_THETA]
        c_phi, s_phi = jnp.cos(phi), jnp.sin(phi)
        c_th = jnp.cos(theta)
        t_th = jnp.tan(theta)

        pqr = jnp.array([x[CrazyFlie.L_P], x[CrazyFlie.L_Q], x[CrazyFlie.L_R]])
        I = jnp.array([Ixx, Iyy, Izz])

        mat = jnp.array([
            [1.0, s_phi * t_th, c_phi * t_th],
            [0.0, c_phi,        -s_phi],
            [0.0, s_phi / c_th, c_phi / c_th],
        ])
        deuler_rpy = mat @ pqr

        R_W_cf = _get_rotmat(x[CrazyFlie.L_PHI], x[CrazyFlie.L_THETA], x[CrazyFlie.L_PSI])
        acc_W = jnp.array([0.0, 0.0, -9.81])

        pqr_dot = -jnp.cross(pqr, I * pqr) / I

        dw_du = CT * jnp.full(4, 1.0 / m)
        dp_du = CT * np.sqrt(2) * d * jnp.array([-1.0, -1.0, 1.0, 1.0]) / Ixx
        dq_du = CT * np.sqrt(2) * d * jnp.array([-1.0, 1.0, 1.0, -1.0]) / Iyy
        dr_du = CD * jnp.array([-1.0, 1.0, -1.0, 1.0]) / Izz

        pqr_dot_control = jnp.array([dp_du @ u, dq_du @ u, dr_du @ u])
        acc_W_control = R_W_cf @ jnp.array([0.0, 0.0, dw_du @ u])

        return jnp.concatenate([deuler_rpy, pqr_dot + pqr_dot_control, acc_W + acc_W_control])

    def thrust_from_motor(self):
        m, Ixx, Iyy, Izz = self._params["m"], self._params["Ixx"], self._params["Iyy"], self._params["Izz"]
        CT, CD, d = self._params["CT"], self._params["CD"], self._params["d"]
        if self.normalize_by_CT:
            CT, CD = 1.0, CD / CT
        dw_du = CT * np.full(4, 1.0 / m)
        dp_du = CT * np.sqrt(2) * d * np.array([-1.0, -1.0, 1.0, 1.0]) / Ixx
        dq_du = CT * np.sqrt(2) * d * np.array([-1.0, 1.0, 1.0, -1.0]) / Iyy
        dr_du = CD * np.array([-1.0, 1.0, -1.0, 1.0]) / Izz
        return np.stack([dw_du, dp_du, dq_du, dr_du], axis=0)

    def _compute_K_ll(self):
        def xdot(x, u):
            return self._xdot_ll(x, u + self.u_eq)

        x_zero = np.zeros(9)
        u_zero = np.zeros(4)
        A_ll, B_ll = jax.jacobian(xdot, argnums=(0, 1))(x_zero, u_zero)
        A_ll, B_ll = np.asarray(A_ll), np.asarray(B_ll)

        # Remove psi (state index L_PSI) since it's unobservable/circular
        A_ll = np.delete(np.delete(A_ll, CrazyFlie.L_PSI, axis=0), CrazyFlie.L_PSI, axis=1)
        B_ll = np.delete(B_ll, CrazyFlie.L_PSI, axis=0)

        #          [  phi,  theta,  p,   q,   r,   vx,   vy,   vz ]
        Q = np.diag([1.0,  1.0,    1.0, 1.0, 1.0, 10.0, 10.0, 20.0])
        R_thrust = 0.01 * np.diag([5.0, 1.0, 1.0, 1.0])
        T_fr_M = self.thrust_from_motor()
        R_motor = T_fr_M.T @ R_thrust @ T_fr_M

        K = _lqr_continuous(A_ll, B_ll, Q, R_motor)
        K = np.insert(K, CrazyFlie.L_PSI, 0, axis=1)  # add psi back as zero column
        return K

    def _get_ll_state(self, state):
        uvw = jnp.array([state[CrazyFlie.U], state[CrazyFlie.V], state[CrazyFlie.W]])
        R_W_cf = self.get_rotation_mat(state)
        v_W = R_W_cf @ uvw
        return jnp.array([
            state[CrazyFlie.PHI], state[CrazyFlie.THETA], state[CrazyFlie.PSI],
            state[CrazyFlie.P], state[CrazyFlie.Q], state[CrazyFlie.R],
            v_W[0], v_W[1], v_W[2],
        ])

    def _vel_targets_to_ll_state(self, vel_targets_4d):
        vx, vy, vz, r = vel_targets_4d
        return jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, r, vx, vy, vz])

    def _get_ll_controls(self, state, vel_targets_4d):
        ll_state = self._get_ll_state(state)
        ll_des = self._vel_targets_to_ll_state(vel_targets_4d)
        control = -self._K_ll @ (ll_state - ll_des) + self.u_eq
        return control

    def _agent_xdot_single_agent_hl(self, state, action_3d):
        """High-level closed-loop dynamics: x_dot = f(x) + g(x) @ LL_LQR(x, vel_targets)"""
        action_3d = self.clip_action(action_3d)
        vel_targets_3d = action_3d * self.vel_targets_scale
        vel_targets_4d = jnp.concatenate([vel_targets_3d, jnp.zeros(1)])  # yaw rate = 0
        control = self._get_ll_controls(state, vel_targets_4d)
        return self._single_agent_f(state) + self._single_agent_gu(control)

    def agent_step_euler(self, agent_states, action):
        """RK4 step for non-linear dynamics (kept method name for base-class compatibility)."""
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        def xdot_batch(x, u):
            return jax_vmap(self._agent_xdot_single_agent_hl)(x, u)

        n_state = RK4_step(xdot_batch, agent_states, action, self.dt)
        return self.clip_state(n_state)

    # ========== Reward (target-style, same as LinearDrone) ==========

    def get_reward(self, graph: LidarEnvGraphsTuple, action: Action) -> Reward:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_goals)
        reward = jnp.zeros(()).astype(jnp.float32)

        agent_pos = agent_states[:, :3]
        goal_pos = goals[:, :3]
        dist2goal = jnp.linalg.norm(agent_pos - goal_pos, axis=-1)
        reward -= dist2goal.mean() * 0.01
        reward -= jnp.where(dist2goal > self._params["dist2goal"], 1.0, 0.0).mean() * 0.001
        reward -= (jnp.linalg.norm(action, axis=1) ** 2).mean() * 0.0001
        return reward

    # ========== Cost (agent-agent + agent-obstacle collision, shape (n_agents, 2)) ==========

    def get_cost(self, graph) -> Array:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_states[:, :3]

        dist = jnp.linalg.norm(
            jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1
        )
        dist += jnp.eye(self.num_agents) * 1e6
        agent_cost = self._params["car_radius"] * 2 - dist.min(axis=1)

        if self._params["n_obs"] == 0:
            obs_cost = jnp.zeros(self.num_agents)
        else:
            n_rays = self._params["top_k_rays"]
            obs_states = graph.type_states(type_idx=2, n_type=n_rays * self.num_agents)
            obs_pos = obs_states[:, :3].reshape(self.num_agents, n_rays, 3)
            obs_dist = jnp.linalg.norm(obs_pos - agent_pos[:, None, :], axis=-1)
            obs_cost = self._params["car_radius"] - obs_dist.min(axis=1)

        cost = jnp.stack([agent_cost, obs_cost], axis=1)
        eps = 0.5
        cost = jnp.where(cost <= 0.0, cost - eps, cost + eps)
        cost = jnp.clip(cost, a_min=-1.0, a_max=1.0)
        return cost

    # ========== Graph ==========

    def get_graph(self, state: LidarEnvState, lidar_data=None):
        n_rays = self._params["top_k_rays"]
        n_hits = n_rays * self.num_agents if self._params["n_obs"] > 0 and lidar_data is not None else 0
        n_nodes = self.num_agents + self.num_goals + n_hits

        if lidar_data is not None:
            lidar_data = merge01(lidar_data)

        node_feats = jnp.zeros((n_nodes, self.node_dim))
        node_feats = node_feats.at[:self.num_agents, :self.state_dim].set(state.agent)
        node_feats = node_feats.at[self.num_agents:self.num_agents + self.num_goals, :self.state_dim].set(state.goal)
        if lidar_data is not None:
            node_feats = node_feats.at[-n_hits:, :3].set(lidar_data)

        node_feats = node_feats.at[:self.num_agents, self.state_dim + 2].set(1.0)
        node_feats = node_feats.at[self.num_agents:self.num_agents + self.num_goals, self.state_dim + 1].set(1.0)
        if n_hits > 0:
            node_feats = node_feats.at[-n_hits:, self.state_dim].set(1.0)

        node_type = -jnp.ones(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[:self.num_agents].set(self.AGENT)
        node_type = node_type.at[self.num_agents:self.num_agents + self.num_goals].set(self.GOAL)
        if n_hits > 0:
            node_type = node_type.at[-n_hits:].set(self.OBS)

        edge_blocks = self.edge_blocks(state, lidar_data)

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

    def edge_blocks(self, state: LidarEnvState, lidar_data=None):
        # agent-agent edges (gated by comm_radius on position only)
        agent_pos = state.agent[:, :3]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(self.num_agents) * (self._params["comm_radius"] + 1)
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent-goal edges: agent_i <-> goal_i 一一对应
        agent_goal_edges = []
        for i in range(self.num_agents):
            feats_i = state.agent[i] - state.goal[i]
            agent_goal_edges.append(EdgeBlock(
                feats_i[None, None, :], jnp.ones((1, 1)),
                jnp.array([i]), jnp.array([i + self.num_agents])
            ))

        # agent-obs edges
        agent_obs_edges = []
        n_rays = self._params["top_k_rays"]
        if lidar_data is not None:
            n_hits = n_rays * self.num_agents
            id_obs = jnp.arange(self.num_agents + self.num_goals,
                                 self.num_agents + self.num_goals + n_hits)
            for i in range(self.num_agents):
                id_hits = jnp.arange(i * n_rays, (i + 1) * n_rays)
                lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :3]
                lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
                active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
                mask = jnp.logical_and(jnp.ones((1, n_rays), dtype=bool), active_lidar)
                lidar_feats = jnp.concatenate(
                    [lidar_feats, jnp.zeros((n_rays, self.edge_dim - 3))], axis=-1
                )
                agent_obs_edges.append(
                    EdgeBlock(lidar_feats[None, :, :], mask, id_agent[i][None], id_obs[id_hits])
                )

        return [agent_agent_edges] + agent_goal_edges + agent_obs_edges

    # ========== Limits ==========

    def state_lim(self, state=None):
        low = jnp.array([
            -jnp.inf, -jnp.inf, -jnp.inf,                # x, y, z
            -jnp.inf, -jnp.pi / 4, -jnp.pi / 4,          # psi, theta, phi
            -0.3, -0.3, -0.3,                            # body u, v, w
            -10.0, -10.0, -10.0,                         # body r, q, p
        ])
        high = jnp.array([
            jnp.inf, jnp.inf, jnp.inf,
            jnp.inf, jnp.pi / 4, jnp.pi / 4,
            0.3, 0.3, 0.3,
            10.0, 10.0, 10.0,
        ])
        return low, high

    def action_lim(self):
        return jnp.ones(self.action_dim) * -1.0, jnp.ones(self.action_dim) * 1.0

    # ========== Reference controller (P-on-pos → 3D vel target) ==========

    def get_agent_goals(self, graph) -> Array:
        return graph.type_states(type_idx=1, n_type=self.num_goals)[:, :3]

    def u_ref(self, graph, target_pos=None, is_final_goal: bool = False) -> Action:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_states[:, :3]
        if target_pos is None:
            target_pos = graph.type_states(type_idx=1, n_type=self.num_goals)[:, :3]

        direction = target_pos - agent_pos
        dist = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        direction_unit = jnp.where(dist > 1e-6, direction / dist, 0.0)

        # vel_targets_scale = (2, 2, 0.5). Cap by tightest axis so per-axis clip can't distort
        # the heading toward XY when the desired direction has a Z component.
        max_speed = jnp.min(self.vel_targets_scale)
        p_gain = 2.0
        # Same P-control law for intermediate and final subgoals: cruises at max_speed when far,
        # decelerates linearly within max_speed/p_gain of the target. Avoids the abrupt
        # cruise→stop transition at subgoal switches that previously caused oscillation.
        desired_speed = jnp.clip(dist * p_gain, 0.0, max_speed)
        desired_vel_W = direction_unit * desired_speed
        action = desired_vel_W / self.vel_targets_scale[None, :]
        return self.clip_action(action)

    # ========== Manifold hooks ==========

    def state_to_pos_vel(self, state: Array) -> Array:
        """12-state → [x, y, z, vx_W, vy_W, vz_W]"""
        pos = state[:3]
        uvw = jnp.array([state[CrazyFlie.U], state[CrazyFlie.V], state[CrazyFlie.W]])
        R_W_cf = self.get_rotation_mat(state)
        v_W = R_W_cf @ uvw
        return jnp.concatenate([pos, v_W])

    def get_pos_acc_jacobian(self, states: Array) -> Array:
        """Effective closed-loop gain: u -> v_W -> a_W, shape (n, 3, 3).

        Instantaneous ∂a_W/∂u is degenerate at hover (lateral accel requires tilt,
        which is a second-order effect). Instead we model the LL LQR closure as a
        first-order filter: v_W settles to u * vel_targets_scale within ~settle_time,
        so effective G ≈ diag(vel_targets_scale / settle_time). Rotated to world
        frame via current yaw — pitch/roll are small after LQR, so R ≈ R_z(psi).
        """
        settle_time = 5.0 * self._dt  # ~5 steps for LL LQR to settle
        G_body = jnp.diag(self.vel_targets_scale / settle_time)  # (3, 3)

        def rotate(state):
            # Approximate world-frame gain via full rotation matrix
            R = self.get_rotation_mat(state)
            return R @ G_body
        return jax_vmap(rotate)(states)

    def get_subgoal_shadow_cost(self, graph, subgoal_pos):
        return jnp.zeros(self.num_agents)

    def init_manifold(self, k=None, K=0.5, Kc=100.0, s_min=0.1, alpha_max=50.0,
                      g_act_thresh=0.1, safety_margin=0.02, n_lookahead=2, w_slack=5.0):
        if k is None:
            k = self.k
        if self._manifold is None:
            from dgppo.algo.utils import get_manifold_fn
            print(f"Initializing 3D manifold (k={k}, dim_q=3) for CrazyFlie...")
            self._manifold, self._manifold_init_slack = get_manifold_fn(
                self, k=k, K=K, Kc=Kc, s_min=s_min,
                alpha_max=alpha_max, g_act_thresh=g_act_thresh,
                safety_margin=safety_margin, n_lookahead=n_lookahead, w_slack=w_slack,
                state_to_pos_vel=self.state_to_pos_vel,
                get_pos_acc_jacobian=self.get_pos_acc_jacobian,
                dim_q=3,
            )
        return self

    # ========== Control-affine (not used without CBF, stubbed) ==========

    def control_affine_dyn(self, state):
        raise NotImplementedError("CrazyFlie does not support CBF; control_affine_dyn is disabled.")

    # ========== Rendering ==========

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
