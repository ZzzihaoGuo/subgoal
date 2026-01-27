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
    

def pwise_cbf_double_integrator_(state: Array, agent_idx: int, o_obs_state: Array, a_state: Array, r: float, k: int):
    n_agent = len(a_state)inter

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
    k_dist_sq = k_dist_sq - 4 * r ** 2

    k_h0 = k_dist_sq

    k_xdiff = state[:2] - all_obs_state[k_idx][:, :2]
    k_vdiff = state[2:] - all_obs_state[k_idx][:, 2:]

    k_h0_dot = 2 * (k_xdiff * k_vdiff).sum(axis=-1)

    k_h1 = k_h0_dot + 10.0 * k_h0

    k_isobs = k_idx >= n_agent

    return k_h1, k_isobs


def pwise_cbf_double_integrator(graph: GraphsTuple, r: float, n_agent: int, n_rays: int, k: int):
    # (n_agents, 4)
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    # (n_obs, 4)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(ft.partial(pwise_cbf_double_integrator_, r=r, k=k), in_axes=(0, 0, 0, None))
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs


def get_pwise_cbf_fn(env: MultiAgentEnv, k: int = 3):
    # TODO NEED TO ADD OTHER ENVS
    n_agent = env.num_agents
    # 注意：graph里存的是top_k_rays个障碍物点，不是n_rays
    n_rays = env.params["top_k_rays"]
    r = env.params["car_radius"]
    return ft.partial(pwise_cbf_double_integrator, r=r, n_agent=n_agent, n_rays=n_rays, k=k)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(ft.partial(pwise_cbf_double_integrator_, r=r, k=k), in_axes=(0, 0, 0, None))
    ak_h0, ak_isobs = fn(a_states, agent_idx, a_obs_states, a_states)
    return ak_h0, ak_isobs


def get_pwise_cbf_fn(env: MultiAgentEnv, k: int = 3):
    # TODO NEED TO ADD OTHER ENVS
    n_agent = env.num_agents
    # 注意：graph里存的是top_k_rays个障碍物点，不是n_rays
    n_rays = env.params["top_k_rays"]
    r = env.params["car_radius"]
    return ft.partial(pwise_cbf_double_integrator, r=r, n_agent=n_agent, n_rays=n_rays, k=k)


