import jax
import jax.numpy as jnp
import einops as ei
import functools as ft

from jaxtyping import Float

from ..utils.typing import Array, TFloat
from ..utils.utils import assert_shape
from ..utils.graph import GraphsTuple
from ..env.base import MultiAgentEnv


def compute_dec_ocp_gae(
    Tah_hs: Float[Array, "T a nh"],
    T_l: TFloat,
    Tp1ah_Vh: Float[Array, "Tp1 a nh"],
    Tp1_Vl: Float[Array, "Tp1"],
    disc_gamma: float,
    gae_lambda: float,
    discount_to_max: bool = True
) -> tuple[Float[Array, "T a nh"], TFloat]:
    """
    Compute GAE for MASOCP. Compute it using DP, starting at V(x_T) and working backwards.

    Returns
    -------
    Qhs: (T, a, nh),
    Ql: (T,)
    """
    T, n_agent, nh = Tah_hs.shape

    def loop(carry, inp):
        ii, hs, l, Vhs, Vl = inp  # hs: (a, nh), Vhs: (a, nh)
        next_Vhs_row, next_Vl_row, gae_coeffs = carry

        mask = assert_shape(jnp.arange(T + 1) < ii + 1, T + 1)
        mask_l = assert_shape(mask[:, None], (T + 1, 1))
        mask_h = assert_shape(mask[:, None, None], (T + 1, 1, 1))

        # DP for Vh.
        if discount_to_max:
            h_disc = hs.max(-1)  # (a,)
        else:
            h_disc = hs

        disc_to_h = (1 - disc_gamma) * h_disc[None, :, None] + disc_gamma * next_Vhs_row  # (T + 1, a, h)
        Vhs_row = assert_shape(mask_h * jnp.maximum(hs, disc_to_h), (T + 1, n_agent, nh), "Vhs_row")
        # DP for Vl. Clamp it to within J_max so it doesn't get out of hand.
        Vl_row = assert_shape(mask_l * (l + disc_gamma * next_Vl_row), (T + 1, n_agent))
        cat_V_row = assert_shape(jnp.concatenate([Vhs_row, Vl_row[:, :, None]], axis=-1), (T + 1, n_agent, nh + 1))

        Qs_GAE = assert_shape(ei.einsum(cat_V_row, gae_coeffs, "Tp1 na nhp2, Tp1 -> na nhp2"), (n_agent, nh + 1))

        # Setup Vs_row for next timestep.
        Vhs_row = Vhs_row.at[ii + 1, :].set(Vhs)
        Vl_row = Vl_row.at[ii + 1].set(Vl)

        #                            *  *        *   *             *     *
        # Update GAE coeffs. [1] -> [λ 1-λ] -> [λ² λ(1-λ) 1-λ] -> [λ³ λ²(1-λ) λ(1-λ) 1-λ]
        gae_coeffs = jnp.roll(gae_coeffs, 1)
        gae_coeffs = gae_coeffs.at[0].set(gae_lambda ** (ii + 1))
        gae_coeffs = gae_coeffs.at[1].set((gae_lambda ** ii) * (1 - gae_lambda))

        return (Vhs_row, Vl_row, gae_coeffs), Qs_GAE

    init_gae_coeffs = jnp.zeros(T + 1)
    init_gae_coeffs = init_gae_coeffs.at[0].set(1.0)

    Tah_Vh, T_Vl = Tp1ah_Vh[:-1], Tp1_Vl[:-1][:, None].repeat(n_agent, axis=1)
    Vh_final, Vl_final = Tp1ah_Vh[-1], Tp1_Vl[-1]

    init_Vhs = jnp.zeros((T + 1, n_agent, nh)).at[0, :].set(Vh_final)
    init_Vl = jnp.zeros(T + 1).at[0].set(Vl_final)[:, None].repeat(n_agent, axis=1)
    init_carry = (init_Vhs, init_Vl, init_gae_coeffs)

    ts = jnp.arange(T)[::-1]
    inps = (ts, Tah_hs, T_l, Tah_Vh, T_Vl)

    _, Qs_GAEs = jax.lax.scan(loop, init_carry, inps, reverse=True)
    Qhs_GAEs, Ql_GAEs = Qs_GAEs[:, :, :nh], Qs_GAEs[:, 0, nh]
    return assert_shape(Qhs_GAEs, (T, n_agent, nh)), assert_shape(Ql_GAEs, T)
    

def pwise_cbf_double_integrator_(state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int, cbf_alpha: float = 10.0):
    n_agent = len(a_state)

    pos = state[:2]
    all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
    all_obs_pos = all_obs_state[:, :2]

    # Only consider the k closest obstacles.
    o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
    # Remove self collisions
    o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)
    # Take the k closest obstacles.
    k_idx = jnp.argsort(o_dist_sq)[:k]
    k_dist_sq = o_dist_sq[k_idx]
    # Take radius into account.
    # agent-agent: 4r² (sum of radii = 2r), agent-obstacle: r² (LiDAR point on surface)
    k_isobs = k_idx >= n_agent
    k_safety_dist_sq = jnp.where(k_isobs, r ** 2, 4 * r ** 2)
    k_dist_sq = k_dist_sq - k_safety_dist_sq

    k_h0 = k_dist_sq

    k_xdiff = state[:2] - all_obs_state[k_idx][:, :2]
    k_vdiff = state[2:] - all_obs_state[k_idx][:, 2:]

    k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)

    k_h1 = k_h0_dot + cbf_alpha * k_h0

    return k_h1, k_isobs


def pwise_cbf_double_integrator(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int, cbf_alpha: float = 10.0):
    # (n_agents, 4)
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    # (n_obs, 4)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(ft.partial(pwise_cbf_double_integrator_, r=r, k=k, cbf_alpha=cbf_alpha), in_axes=(0, 0, 0, None))
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs


def get_pwise_cbf_fn(env: MultiAgentEnv, k: int = 3, cbf_alpha: float = 10.0):
    # TODO NEED TO ADD OTHER ENVS
    n_agent = env.num_agents
    # 注意：graph里存的是top_k_rays个障碍物点，不是n_rays
    n_rays = env.params["top_k_rays"]
    r = env.params["car_radius"]
    return ft.partial(pwise_cbf_double_integrator, r=r, n_agent=n_agent, n_rays=n_rays, k=k, cbf_alpha=cbf_alpha)


def pwise_cbf_paper_formulation_(
    state: Array,
    agent_idx: int,
    o_obs_state: Array,
    a_state: Array,
    r: float,
    k: int,
    m: float = 0.1,
    alpha1: float = 1.0,
    alpha2: float = 1.0
):
    """Paper's CBF formulation with relative-degree-2 dynamics

    Based on the paper's equations:
    h_ij(x_i, x_j) = ||p_rel,ij||^2 - (d_s + r_i + r_j)^2
    ḣ_ij(x_i, x_j) = 2 * p_rel,ij · v_rel,ij
    ḧ_ij(x_i, x_j, u_i) = 2||v_rel,ij||^2 + 2*p_rel,ij · (u_i / m)

    CBF constraint: G_ij = ḧ_ij + α₂ḣ_ij + α₁h_ij ≥ 0

    Args:
        state: agent state [px, py, vx, vy]
        agent_idx: index of current agent
        o_obs_state: obstacle states (n_ray, 4)
        a_state: all agent states (n_agent, 4)
        r: agent radius
        k: number of closest neighbors to consider
        m: agent mass (default: 0.1, so acceleration = u / m = 10 * u)
        alpha1, alpha2: CBF parameters

    Returns:
        k_G: CBF constraint values (k,)
        k_isobs: whether each neighbor is obstacle (k,)
        k_Gu: Jacobian w.r.t control input (k, 2)
    """
    n_agent = len(a_state)

    pos = state[:2]  # (2,)
    vel = state[2:]  # (2,)
    all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
    all_obs_pos = all_obs_state[:, :2]
    all_obs_vel = all_obs_state[:, 2:]

    # Distance to all neighbors
    o_dist_sq = ((pos - all_obs_pos) ** 2).sum(axis=-1)
    o_dist_sq = o_dist_sq.at[agent_idx].set(1e2)  # exclude self

    # Get k closest neighbors
    k_idx = jnp.argsort(o_dist_sq)[:k]

    # Compute relative states
    k_p_rel = pos - all_obs_pos[k_idx]  # p_rel,ij = p_i - p_j, (k, 2)
    k_v_rel = vel - all_obs_vel[k_idx]  # v_rel,ij, (k, 2)

    # Mark which neighbors are obstacles vs agents
    k_isobs = k_idx >= n_agent  # (k,)

    # Conservative velocity approximation: v̂_rel = -||v_rel|| * e_ij
    # where e_ij = p_rel / ||p_rel|| points from j to i
    # For obstacles (static): assume v_j = 0, so v_rel = v_i
    # For agents: use actual relative velocity
    k_p_rel_norm = jnp.linalg.norm(k_p_rel, axis=-1, keepdims=True) + 1e-8  # (k, 1)
    k_e_ij = k_p_rel / k_p_rel_norm  # (k, 2)

    # For obstacles, use only agent's velocity; for agents, use relative velocity
    k_v_for_approx = jnp.where(
        k_isobs[:, None],  # (k, 1) broadcast
        vel[None, :],  # Use agent's own velocity for obstacles
        k_v_rel  # Use relative velocity for other agents
    )
    k_v_rel_norm = jnp.linalg.norm(k_v_for_approx, axis=-1, keepdims=True)  # (k, 1)
    k_v_hat_rel = -k_v_rel_norm * k_e_ij  # (k, 2)

    # Barrier function: h_ij = ||p_rel||^2 - safety_dist²
    # Safety distance:
    # - Agent-agent: 2r (sum of radii), so (2r)² = 4r²
    # - Agent-obstacle: LiDAR point is on obstacle surface, so just r (agent radius), r²
    k_safety_dist_sq = jnp.where(k_isobs, r ** 2, 4 * r ** 2)
    k_h0 = o_dist_sq[k_idx] - k_safety_dist_sq  # (k,)

    # First time derivative: ḣ_ij = 2 * p_rel · v_rel
    # For obstacles, clamp ḣ to non-positive: min(ḣ, 0)
    # Approaching (ḣ < 0): use actual value, CBF triggers normally
    # Moving away (ḣ > 0): set to 0, don't let positive ḣ inflate G_base
    k_h0_dot_actual = 2 * (k_p_rel * k_v_rel).sum(axis=-1)  # (k,)
    # k_h0_dot = jnp.where(k_isobs, jnp.minimum(k_h0_dot_actual, 0.0), k_h0_dot_actual)  # (k,) clamp version
    k_h0_dot = k_h0_dot_actual  # no clamp: use actual h0_dot for all neighbors

    # Second time derivative (without control): ḧ_ij = 2||v_rel||^2 + 2*p_rel · u_i
    # For obstacles, only use normal velocity component: ḧ_base = 2*v_n²
    # Tangential velocity doesn't contribute to collision avoidance
    k_v_normal = (k_v_rel * k_e_ij).sum(axis=-1)  # (k,) radial velocity scalar
    k_v_normal_approaching = jnp.minimum(k_v_normal, 0.0)  # only count when approaching
    k_h0_ddot_obs = 2 * k_v_normal_approaching ** 2  # obstacles: normal approaching only
    k_h0_ddot_agent = 2 * (k_v_rel ** 2).sum(axis=-1)  # agents: full ||v_rel||²
    k_h0_ddot_base = jnp.where(k_isobs, k_h0_ddot_obs, k_h0_ddot_agent)  # (k,)

    # CBF constraint without control: G_ij(x_i, x_j, 0) = ḧ_ij(0) + α₂ḣ_ij + α₁h_ij
    k_G_base = k_h0_ddot_base + alpha2 * k_h0_dot + alpha1 * k_h0  # (k,)

    # Jacobian w.r.t control: ∂G/∂u = ∂ḧ/∂u = 2*p_rel / m
    # Since ẍ = u / m, we have ∂ḧ/∂u = 2 * p_rel / m
    k_Gu = 2 * k_p_rel / m  # (k, 2)

    return k_G_base, k_isobs, k_Gu


def pwise_cbf_paper_formulation(
    graph: GraphsTuple,
    r: float,
    n_agent: int,
    n_rays: int,
    k: int,
    m: float = 0.1,
    alpha1: float = 1.0,
    alpha2: float = 1.0
):
    """Paper's CBF formulation for all agents"""
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(
        ft.partial(pwise_cbf_paper_formulation_, r=r, k=k, m=m, alpha1=alpha1, alpha2=alpha2),
        in_axes=(0, 0, 0, None)
    )
    ak_G, ak_isobs, ak_Gu = fn(a_states, agent_idx, a_obs_states, a_states)

    return ak_G, ak_isobs, ak_Gu


def get_pwise_cbf_paper_fn(env: MultiAgentEnv, k: int = 3, alpha1: float = 1.0, alpha2: float = 1.0):
    """Get paper's CBF function with analytical Jacobian

    Args:
        env: environment
        k: number of closest neighbors to consider
        alpha1, alpha2: CBF parameters (should satisfy conditions from Theorem 1)
    """
    n_agent = env.num_agents
    n_rays = env.params["top_k_rays"]
    r = env.params["car_radius"]
    m = env.params["m"]  # agent mass
    return ft.partial(
        pwise_cbf_paper_formulation,
        r=r,
        n_agent=n_agent,
        n_rays=n_rays,
        k=k,
        m=m,
        alpha1=alpha1,
        alpha2=alpha2
    )
