"""LidarAnt: MuJoCo-ant navigation env for the hierarchical subgoal + manifold stack.

The ant is a low-level executor under a BICYCLE CoM template (state [x,y,cosθ,sinθ,v]).
The GNN/u_ref/manifold all see the bicycle template unchanged; step() replaces the analytic
bicycle integration with an MJX ant rollout (skid-steer gait) and reads the CoM back.

First version: NO obstacles, random start/goal, NO agent-agent collision cost (pure navigation).

The ant's MJX Data rides inside env_states (AntEnvState). We carry the FULL mjx.Data (not just
qpos/qvel) because rebuilding Data each step loses the contact solver's warm-start and the ant
diverges after a few steps. Memory grows with n_env (the rollout stacks graphs over time) — use a
modest n_env for now; stripping Data from the stored history is a later optimization.
"""
import os
from typing import NamedTuple, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import mujoco
from mujoco import mjx

from dgppo.env.lidar_env.lidar_bicycle_target import LidarBicycleTarget
from dgppo.env.utils import get_node_goal_rng
from dgppo.utils.graph import GraphsTuple
from dgppo.utils.typing import Action, Array, Cost, Done, Info, Reward, State

_HERE = os.path.dirname(__file__)

_INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}


def tune_physics(m, integrator=None, iterations=None, ls_iterations=None):
    """Override the accuracy-oriented defaults that ship in the gym ant.xml.

    The stock file asks for RK4 (4 dynamics evaluations per timestep) and the C-MuJoCo solver
    defaults iterations=100 / ls_iterations=50, so ONE control step costs FS * 4 = 20 full
    forward-dynamics solves. That is the dominant cost of this env -- ~220x a bicycle step.
    Any argument left None keeps whatever the XML specified. See ant_prototype/bench_physics.py
    for the speed/fidelity trade-off measurements.
    """
    if integrator is not None:
        m.opt.integrator = _INTEGRATORS[str(integrator).lower()]
    if iterations is not None:
        m.opt.iterations = int(iterations)
    if ls_iterations is not None:
        m.opt.ls_iterations = int(ls_iterations)
    return m


class AntEnvState(NamedTuple):
    """Env state carrying the ant MJX Data. First 3 fields match LidarEnvState so inherited
    get_graph/edge_blocks keep working."""
    agent: State                    # (n_agent, 5) bicycle nav state [x,y,cosθ,sinθ,v]
    goal: State                     # (n_agent, 5)
    obstacle: object                # None (no obstacles)
    data: object                    # mjx.Data batched over agents
    gait_t: Array                   # () gait phase clock (seconds)


class LidarAnt(LidarBicycleTarget):

    PARAMS = {
        "car_radius": 0.90,          # leg-cylinder collision radius (measured)
        "comm_radius": 5.0,
        "n_rays": 32,
        "obs_len_range": [1.5, 2.5],
        "n_obs": 0,                  # no obstacles
        "default_area_size": 8.0,
        "dist2goal": 1.0,            # ≥ car_radius so "reached" = body covers the goal
        "top_k_rays": 8,
        "m": 0.1,
    }

    FS = 5                           # physics substeps per control step
    SETTLE = 40                      # standing settle steps in reset
    # --- physics speed knobs (see tune_physics and ant_prototype/bench_physics.py) -----
    # These are the stock ant.xml settings and cost 20 full dynamics solves per control step
    # (RK4 = 4 evals, times FS = 5), which is the dominant cost of this env.
    # "implicitfast" + 4/8 is ~an order of magnitude cheaper and was measured to keep the ant
    # upright and walking straight (z 0.525, up_min 0.999, drift 0.05 m over 10 s) -- BUT it
    # yields v_fwd 0.131 m/s and 8.5 deg/s of turn authority instead of the 0.22 / 19 that the
    # CEM gait produced under RK4. VMAX, R_MIN/R_MAX and u_ref's acc scaling are all calibrated
    # to 0.22, so switching integrator requires re-running the gait search (p05_gait.py) and
    # re-measuring VMAX first. Until then keep the stock values so the task stays reachable.
    INTEGRATOR = "rk4"
    SOLVER_ITER = 100
    SOLVER_LS_ITER = 50
    VMAX = 0.22                      # measured open-loop forward speed (m/s)
    V_MIN = 0.35 * 0.22              # min cruise so legs keep stepping (needed to turn)
    K_OM = 2.0                       # heading-error -> turn command gain
    # goal placed on a ring [R_MIN, R_MAX] around the start: far enough to be a real task,
    # near enough to reach in max_step (reachable ≈ VMAX·max_step·dt ≈ 2.8 m at max_step=256).
    # (R_MIN, R_MAX) is the natural curriculum knob — widen R_MAX over training.
    R_MIN = 1.5
    R_MAX = 2.5
    HEADING_NOISE = np.pi / 3         # start heading within ±60° of the goal direction

    def __init__(self, num_agents, area_size=None, max_step=256, dt=0.05, params=None, cbf_alpha=10.0):
        area_size = LidarAnt.PARAMS["default_area_size"] if area_size is None else area_size
        super().__init__(num_agents, area_size, max_step, dt, params, cbf_alpha)
        m = mujoco.MjModel.from_xml_path(os.path.join(_HERE, "ant.xml"))
        tune_physics(m, LidarAnt.INTEGRATOR, LidarAnt.SOLVER_ITER, LidarAnt.SOLVER_LS_ITER)
        self._mjx = mjx.put_model(m)
        # Control dt is fixed by the physics: FS substeps of mjx.step (each = model timestep).
        # Force it regardless of the dt make_env passes (make_env hardcodes 0.03).
        self._dt = LidarAnt.FS * float(m.opt.timestep)                    # 5 * 0.01 = 0.05 s
        gp = np.load(os.path.join(_HERE, "ant_best_gait.npy"))
        (self._f, self._hip_amp, self._ank_mid, self._ank_amp,
         self._kp, self._kd, self._phase, _) = [float(x) for x in gp]
        self._GRP = jnp.array([0., 1., 0., 1.])
        self._ASGN = jnp.array([1., -1., -1., 1.])
        self._LEFT = jnp.array([1., 1., -1., -1.])
        self._K = jnp.sin(jnp.deg2rad(jnp.array([45., 135., 225., 315.])))
        self._CTRL_LEG = jnp.array([3, 3, 0, 0, 1, 1, 2, 2])
        self._CTRL_ISANK = jnp.array([0, 1, 0, 1, 0, 1, 0, 1])
        self._QADR_H = jnp.array([7, 9, 11, 13]); self._QADR_A = jnp.array([8, 10, 12, 14])
        self._DADR_H = jnp.array([6, 8, 10, 12]); self._DADR_A = jnp.array([7, 9, 11, 13])
        self._SUBDT = float(m.opt.timestep)                              # gait phase advances with physics

    # ---------- ant low-level (MJX) ----------
    def _torque(self, dx, t, turn, fwd):
        th = 2 * jnp.pi * self._f * t + jnp.pi * self._GRP * self._phase
        mult = 1.0 + turn * (-self._LEFT)
        hip_des = fwd * self._hip_amp * self._K * mult * jnp.cos(th)
        lift = self._ank_amp * jnp.maximum(0., jnp.sin(th))
        ank_des = self._ASGN * (self._ank_mid - lift)
        q_h, q_a = dx.qpos[self._QADR_H], dx.qpos[self._QADR_A]
        dq_h, dq_a = dx.qvel[self._DADR_H], dx.qvel[self._DADR_A]
        tau_h = self._kp * (hip_des - q_h) - self._kd * dq_h
        tau_a = self._kp * (ank_des - q_a) - self._kd * dq_a
        return jnp.clip(jnp.where(self._CTRL_ISANK == 1, tau_a[self._CTRL_LEG], tau_h[self._CTRL_LEG]), -1., 1.)

    def _substeps(self, dx, turn, fwd, t0):
        def body(i, dx):
            t = t0 + i * self._SUBDT
            return mjx.step(self._mjx, dx.replace(ctrl=self._torque(dx, t, turn, fwd)))
        return jax.lax.fori_loop(0, LidarAnt.FS, body, dx)

    def _nav_of(self, dx, com0):
        com = dx.subtree_com[0]
        w, x, y, z = dx.qpos[3], dx.qpos[4], dx.qpos[5], dx.qpos[6]
        yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        vel = (com[:2] - com0[:2]) / self._dt
        return jnp.array([com[0], com[1], jnp.cos(yaw), jnp.sin(yaw), jnp.hypot(vel[0], vel[1])])

    def _make_standing(self, xy, yaw):
        """Fresh standing mjx.Data at world xy with body heading yaw (before settle)."""
        dx = mjx.make_data(self._mjx)
        qpos = dx.qpos.at[0:2].set(xy).at[2].set(0.32)
        qpos = qpos.at[3].set(jnp.cos(yaw / 2)).at[6].set(jnp.sin(yaw / 2))
        qpos = qpos.at[self._QADR_A].set(self._ASGN * self._ank_mid)
        return dx.replace(qpos=qpos)

    def _reset_one_ant(self, xy, yaw):
        dx = self._make_standing(xy, yaw)
        dx = jax.lax.fori_loop(0, LidarAnt.SETTLE, lambda _, d: self._substeps(d, 0.0, 0.0, 0.0), dx)
        return dx

    # ---------- nominal policy (forward-only; the ant can't reverse) ----------
    def u_ref(self, graph: GraphsTuple, target_pos: Optional[Array] = None, is_final_goal: bool = False) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)      # (n_agent, 5)
        if target_pos is None:
            goal = graph.type_states(type_idx=1, n_type=self.num_agents)[:, :2]
        else:
            goal = target_pos
        to = goal - agent[:, :2]
        dist = jnp.linalg.norm(to, axis=-1)
        th_goal = jnp.arctan2(to[:, 1], to[:, 0])
        th = jnp.arctan2(agent[:, 3], agent[:, 2])
        err = jnp.arctan2(jnp.sin(th_goal - th), jnp.cos(th_goal - th))    # wrapped heading error
        omega = jnp.clip(LidarAnt.K_OM * err, -1.0, 1.0)                   # turn command
        facing = jnp.clip(jnp.cos(err), 0.0, 1.0)
        near = jnp.logical_and(is_final_goal, dist < self.params["dist2goal"])
        v_target = jnp.where(near, 0.0, LidarAnt.V_MIN + (LidarAnt.VMAX - LidarAnt.V_MIN) * facing)
        acc = jnp.clip((v_target - agent[:, 4]) / (10.0 * self._dt), -1.0, 1.0)
        return self.clip_action(jnp.stack([omega, acc], axis=-1))

    # ---------- env API ----------
    def reset(self, key: Array) -> GraphsTuple:
        pos_key, key = jr.split(key)
        states, _ = get_node_goal_rng(                                     # use spread-out starts
            pos_key, self.area_size, 2, self.num_agents, 2.2 * self.params["car_radius"], None)
        # ring goal: start + R·(cosφ, sinφ), R ∈ [R_MIN, R_MAX]
        r_key, phi_key, key = jr.split(key, 3)
        R = jr.uniform(r_key, (self.num_agents,), minval=LidarAnt.R_MIN, maxval=LidarAnt.R_MAX)
        phi = jr.uniform(phi_key, (self.num_agents,), minval=0, maxval=2 * np.pi)
        goals = states + R[:, None] * jnp.stack([jnp.cos(phi), jnp.sin(phi)], axis=-1)
        goals = jnp.clip(goals, 0.0, self.area_size)                       # keep goals inside arena
        # start heading ~toward the goal (± HEADING_NOISE) so episodes aren't dominated by slow
        # open-loop U-turns; widen HEADING_NOISE over training as a curriculum.
        theta_key, key = jr.split(key)
        to_goal = jnp.arctan2(goals[:, 1] - states[:, 1], goals[:, 0] - states[:, 0])
        thetas = to_goal + jr.uniform(theta_key, (self.num_agents,),
                                      minval=-LidarAnt.HEADING_NOISE, maxval=LidarAnt.HEADING_NOISE)

        data = jax.vmap(self._reset_one_ant)(states, thetas)               # batched Data over agents
        com0 = jax.vmap(lambda d: d.subtree_com[0])(data)
        agent = jax.vmap(self._nav_of)(data, com0)                         # read settled nav state
        agent = agent.at[:, :2].set(states)                                # pin xy to intended start
        goal = jnp.concatenate([goals, jnp.zeros((self.num_agents, 3))], axis=1)

        env_states = AntEnvState(agent, goal, None, data, jnp.array(0.0))
        return self.get_graph(env_states)

    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False
             ) -> Tuple[GraphsTuple, Reward, Cost, Done, Info]:
        state: AntEnvState = graph.env_states
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)      # (n_agent, 5)
        action = self.clip_action(action)                                 # (n_agent, 2) [ω, acc]

        v = agent[:, 4]
        # turn decoupled from v (skid-steer can turn at low speed); fwd floored so legs keep
        # stepping (needed to turn) except when u_ref commands a stop (v_next -> 0 near goal).
        v_next = jnp.clip(v + 10.0 * action[:, 1] * self._dt, 0.0, LidarAnt.VMAX)
        turn = jnp.clip(0.8 * action[:, 0], -0.8, 0.8)
        fwd = jnp.clip(v_next / LidarAnt.VMAX, 0.0, 1.0)

        t0 = state.gait_t
        com0 = jax.vmap(lambda d: d.subtree_com[0])(state.data)
        new_data = jax.vmap(self._substeps, in_axes=(0, 0, 0, None))(state.data, turn, fwd, t0)
        nav = jax.vmap(self._nav_of)(new_data, com0)

        next_state = AntEnvState(nav, state.goal, None, new_data, t0 + self._dt)
        next_graph = self.get_graph(next_state)

        reward = self.get_reward(graph, action)
        cost = self.get_cost(graph)
        done = jnp.array(False)
        return next_graph, reward, cost, done, {}

    def get_cost(self, graph: GraphsTuple) -> Cost:
        # no collision volume between ants (pure navigation): always safe.
        # Return a clearly-negative cost so both `cost > 0` and `cost >= 0` unsafe-conventions
        # treat it as safe (get_cost==0 would be flagged unsafe by the `>= 0` convention).
        return -jnp.ones((self.num_agents, self.n_cost))

    def get_graph(self, state: AntEnvState, lidar_data=None) -> GraphsTuple:
        return super().get_graph(state, None)

    def lighten_graph(self, graph: GraphsTuple) -> GraphsTuple:
        # Drop the heavy mjx.Data (21.5 KB/ant, never read by training) from stored-history graphs.
        # The scan CARRY keeps the full Data for stepping; only the stacked OUTPUT is lightened.
        st: AntEnvState = graph.env_states
        return graph._replace(env_states=st._replace(data=None))
