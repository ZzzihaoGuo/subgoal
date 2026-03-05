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


def manifold_single_agent_(
    state: Array,
    agent_idx: int,
    o_obs_state: Array,
    a_state: Array,
    u_ref: Array,
    s_prev: Array,
    r: float,
    k: int,
    K: float = 0.2,
    Kc: float = 15.0,
    s_min: float = 0.1,
    alpha_max: float = 5.0,
    g_act_thresh: float = 0.1,
    dt: float = 0.03,
    safety_margin: float = 0.02,
    n_lookahead: int = 2,
    acc_scale: float = 10.0,
    w_slack: float = 5.0,
):
    """ATACOM manifold 安全修正 (v7)

    ATACOM + null space 法向/切向分解:
    - 特解 (a_comp + err): 安全修正, 推离障碍物
    - Null space: 恢复 u_ref, 但限制对抗安全的法向分量, 保留切向
    → 近距离直冲: 强修正; 斜行: 保留切向绕过; 安全: 不修正.

    Returns
    -------
    u_opt : Array, shape (2,)  — action 空间
    relax : Array, shape (k,)
    s_new : Array, shape (n_total,)
    """
    n_agent = len(a_state)
    dim_q = 2  # position dimension

    q = state[:2]   # position
    dq = state[2:]  # velocity

    all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
    all_obs_pos = all_obs_state[:, :2]
    all_obs_vel = all_obs_state[:, 2:]  # (n_total, 2)

    # ===== 1. 取 k 个最近邻 — 用 min(当前, 预测) 距离, 防止漏掉高速接近的障碍 =====
    o_dist_sq_now = ((q - all_obs_pos) ** 2).sum(axis=-1)
    o_dist_sq_now = o_dist_sq_now.at[agent_idx].set(1e6)
    all_rel_vel_sel = dq[None, :] - all_obs_vel  # (n_total, 2)
    lookahead_sel = dt * jnp.maximum(n_lookahead, 2)  # 至少看 2 步
    o_p_rel_pred = (q - all_obs_pos) + all_rel_vel_sel * lookahead_sel
    o_dist_sq_pred = (o_p_rel_pred ** 2).sum(axis=-1)
    o_dist_sq_pred = o_dist_sq_pred.at[agent_idx].set(1e6)
    # 取更危险的距离 (更小的那个)
    o_dist_sq = jnp.minimum(o_dist_sq_now, o_dist_sq_pred)
    k_idx = jnp.argsort(o_dist_sq)[:k]

    k_p_rel = q - all_obs_pos[k_idx]  # (k, 2) — 当前相对位置
    k_vel = all_obs_vel[k_idx]  # (k, 2)
    k_isobs = k_idx >= n_agent
    k_rel_vel = dq[None, :] - k_vel  # (k, 2)

    # ===== 动态安全半径: 基于接近速度的刹车距离 =====
    k_p_rel_norm = jnp.linalg.norm(k_p_rel, axis=-1, keepdims=True)  # (k, 1)
    k_p_rel_hat = k_p_rel / (k_p_rel_norm + 1e-8)  # (k, 2) 单位方向
    # 接近速度 (正值 = 在靠近)
    k_v_approach = jnp.maximum(0.0, -jnp.sum(k_rel_vel * k_p_rel_hat, axis=-1))  # (k,)
    # 刹车距离 = v² / (2 * a_max), a_max = acc_scale * action_clip(1.0)
    k_braking_dist = k_v_approach ** 2 / (2.0 * acc_scale + 1e-8)  # (k,)
    k_dynamic_margin = safety_margin + k_braking_dist  # (k,)

    k_base_r = jnp.where(k_isobs, r, 2 * r)  # (k,)
    k_safety_dist_sq = (k_base_r + k_dynamic_margin) ** 2  # (k,)

    # ===== 2. 约束值 — 取 max(g_current, g_pred), 同时保护当前和未来 =====
    k_g_now = k_safety_dist_sq - (k_p_rel ** 2).sum(axis=-1)  # 用真实当前距离

    # 预测位置约束 (lookahead)
    lookahead_dt = dt * n_lookahead
    k_p_rel_pred = k_p_rel + k_rel_vel * lookahead_dt  # 预测相对位置
    # 预测约束也用动态半径
    k_g_pred = k_safety_dist_sq - (k_p_rel_pred ** 2).sum(axis=-1)

    # 取更危险的那个 (g 越大越危险)
    k_g = jnp.maximum(k_g_now, k_g_pred)  # (k,)

    # ===== 3. 约束 Jacobian — 用当前位置 (稳定、方向正确) =====
    k_J_g = -2.0 * k_p_rel  # (k, 2)

    # ===== 4. Viability constraint =====
    k_dg_dt = jnp.sum(k_J_g * k_rel_vel, axis=-1)  # (k,) J_g @ (dq - v_j)
    k_g_viab = k_g + K * k_dg_dt  # (k,)

    # ===== 5. 松弛变量 =====
    k_s = s_prev[k_idx]  # (k,)
    k_s = jnp.maximum(k_s, s_min)

    # ===== 约束激活 — 平滑 ramp =====
    pos_active = jnp.clip((k_g + g_act_thresh) / g_act_thresh, 0.0, 1.0)
    viab_active = (jnp.clip(k_g_viab / g_act_thresh, 0.0, 1.0)
                   * (k_g > -g_act_thresh * 10).astype(jnp.float32))
    active_f = jnp.maximum(pos_active, viab_active)  # (k,) smooth [0,1]

    # ===== 6. 增广 Jacobian =====
    Jc_u = K * k_J_g * acc_scale * active_f[:, None]  # (k, 2)
    Jc_slack = jnp.diag(k_s) * active_f[:, None] + jnp.eye(k) * (1.0 - active_f[:, None])
    Jc = jnp.concatenate([Jc_u, Jc_slack], axis=1)  # (k, 2+k)

    # ===== 7. psi (drift) — 单侧: 只补偿 unsafe drift =====
    k_dJdt_dq = -2.0 * jnp.sum(k_rel_vel * k_rel_vel, axis=-1)  # -2*||v_rel||²
    k_psi_raw = k_dg_dt + K * k_dJdt_dq
    k_psi = jnp.maximum(k_psi_raw, 0.0) * active_f  # (k,)

    # ===== 8. SVD pseudo-inverse (damped, 加权: 优先调整 action 而非 slack) =====
    # 加权: slack 变化代价 = w_slack 倍 action 变化代价
    # → pseudo-inverse 优先用 action 修正, slack 只吸收残差
    W_inv = jnp.ones(dim_q + k)
    W_inv = W_inv.at[dim_q:].set(1.0 / w_slack)
    Jc_w = Jc * W_inv[None, :]  # 缩放 slack 列, 使其在 SVD 中权重更低

    U, S, Vh = jnp.linalg.svd(Jc_w, full_matrices=True)
    lambda_sq = 0.01
    S_inv = S / (S ** 2 + lambda_sq)
    Jc_w_pinv = (Vh[:k].T * S_inv[None, :]) @ U.T  # (2+k, k)
    # 还原加权: z = W_inv * Jc_w_pinv @ b
    Jc_pinv = W_inv[:, None] * Jc_w_pinv  # (2+k, k)
    Nc_w = Vh[k:].T  # null space of Jc_w
    Nc = W_inv[:, None] * Nc_w  # 还原到原空间

    # ===== 9. ATACOM 特解 =====
    a_comp = -Jc_pinv @ k_psi  # (2+k,)
    k_c = jnp.maximum(k_g_viab, 0.0) * active_f  # (k,) 只在 viability 违反时修正
    err = -Jc_pinv @ (Kc * k_c)  # (2+k,)

    # ===== 10. Null space + 法向/切向分解 =====
    Nc_q = Nc[:dim_q]  # (2, 2)
    target = u_ref - (a_comp[:dim_q] + err[:dim_q])  # (2,)
    alpha, _, _, _ = jnp.linalg.lstsq(Nc_q, target)  # (2,)
    alpha_norm = jnp.linalg.norm(alpha)
    alpha = jnp.where(alpha_norm > alpha_max, alpha * alpha_max / (alpha_norm + 1e-8), alpha)
    b_proj = Nc @ alpha  # (2+k,)

    # 约束法向: J_g 加权平均 (指向障碍物方向)
    normal_sum = jnp.sum(k_J_g * active_f[:, None], axis=0)  # (2,)
    normal_norm = jnp.linalg.norm(normal_sum)
    normal_dir = normal_sum / (normal_norm + 1e-8)  # 指向障碍物

    # 分解 null space action 为法向 + 切向
    b_action = b_proj[:dim_q]  # (2,)
    b_normal_coeff = jnp.dot(b_action, normal_dir)  # >0 = 朝向障碍物
    b_tangent = b_action - b_normal_coeff * normal_dir

    # 朝向障碍物的分量限制到 10%, 远离障碍物的分量保留
    b_normal_safe = jnp.where(b_normal_coeff > 0, 0.1 * b_normal_coeff, b_normal_coeff)
    b_action_safe = b_tangent + b_normal_safe * normal_dir

    # 只在有活跃约束时应用限制
    has_active = normal_norm > 1e-6
    b_action_final = jnp.where(has_active, b_action_safe, b_action)
    b_proj = b_proj.at[:dim_q].set(b_action_final)

    # ===== 11. 合成 =====
    ddq_ds = a_comp + b_proj + err  # (2+k,)
    u_opt = ddq_ds[:dim_q]  # action 空间

    # ===== 12. 松弛变量积分 =====
    ds = ddq_ds[dim_q:]
    k_s_new = k_s + ds * dt
    k_s_new = jnp.maximum(k_s_new, s_min)

    all_p_rel_all = q - all_obs_pos
    all_rel_vel = dq[None, :] - all_obs_vel
    all_J_g = -2.0 * all_p_rel_all
    all_dg_dt = jnp.sum(all_J_g * all_rel_vel, axis=-1)
    all_isobs = jnp.arange(len(all_obs_pos)) >= n_agent
    # 动态安全半径 (与 step 1 一致)
    all_p_rel_norm = jnp.linalg.norm(all_p_rel_all, axis=-1, keepdims=True)
    all_p_rel_hat = all_p_rel_all / (all_p_rel_norm + 1e-8)
    all_v_approach = jnp.maximum(0.0, -jnp.sum(all_rel_vel * all_p_rel_hat, axis=-1))
    all_braking_dist = all_v_approach ** 2 / (2.0 * acc_scale + 1e-8)
    all_dynamic_margin = safety_margin + all_braking_dist
    all_base_r = jnp.where(all_isobs, r, 2 * r)
    all_safety_sq = (all_base_r + all_dynamic_margin) ** 2
    all_g_now = all_safety_sq - (all_p_rel_all ** 2).sum(axis=-1)
    all_p_rel_pred = all_p_rel_all + all_rel_vel * lookahead_dt
    all_g_pred = all_safety_sq - (all_p_rel_pred ** 2).sum(axis=-1)
    all_g = jnp.maximum(all_g_now, all_g_pred)
    all_g_viab = all_g + K * all_dg_dt
    all_s_ideal = jnp.sqrt(jnp.maximum(-2.0 * all_g_viab, s_min ** 2))
    s_new = all_s_ideal
    s_new = s_new.at[k_idx].set(k_s_new)

    relax = k_c
    # debug info: [u_ref, u_opt, a_comp_u, err_u, b_proj_u, max_g_viab, max_active, min_obs_dist]
    debug_info = jnp.array([
        u_ref[0], u_ref[1],
        u_opt[0], u_opt[1],
        a_comp[0], a_comp[1],       # drift compensation (action part)
        err[0], err[1],             # error correction (action part)
        b_proj[0], b_proj[1],       # null space (action part)
        jnp.max(k_g_viab * active_f),  # max viability violation
        jnp.max(active_f),          # max activation
        jnp.min(o_dist_sq[k_idx]),  # min dist² to nearest neighbor
        jnp.max(k_c),              # max correction term
    ])
    return u_opt, relax, s_new, debug_info


def manifold_all_agents(
    graph: GraphsTuple,
    u_ref: Array,
    s_all: Array,
    r: float,
    n_agent: int,
    n_rays: int,
    k: int,
    K: float = 0.15,
    Kc: float = 8.0,
    s_min: float = 0.1,
    alpha_max: float = 5.0,
    g_act_thresh: float = 0.1,
    dt: float = 0.03,
    safety_margin: float = 0.02,
    n_lookahead: int = 2,
    acc_scale: float = 10.0,
    w_slack: float = 5.0,
):
    """对所有 agent 并行计算 manifold 修正"""
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    agent_idx = jnp.arange(n_agent)
    fn = jax.vmap(
        ft.partial(manifold_single_agent_, r=r, k=k, K=K, Kc=Kc, s_min=s_min,
                   alpha_max=alpha_max, g_act_thresh=g_act_thresh, dt=dt,
                   safety_margin=safety_margin, n_lookahead=n_lookahead,
                   acc_scale=acc_scale, w_slack=w_slack),
        in_axes=(0, 0, 0, None, 0, 0)
    )
    u_opt, relax, s_new, debug_info = fn(a_states, agent_idx, a_obs_states, a_states, u_ref, s_all)
    return u_opt, relax, s_new, debug_info


def manifold_init_slack(
    graph: GraphsTuple,
    r: float,
    n_agent: int,
    n_rays: int,
    K: float = 0.15,
    s_min: float = 0.1,
    safety_margin: float = 0.02,
    dt: float = 0.03,
    n_lookahead: int = 2,
    acc_scale: float = 10.0,
):
    """初始化松弛变量: s = sqrt(max(-2 * g_viab_pred, s_min²))"""
    a_states = graph.type_states(type_idx=0, n_type=n_agent)
    obs_states = graph.type_states(type_idx=2, n_type=n_agent * n_rays)
    a_obs_states = ei.rearrange(obs_states, "(n_agent n_ray) d -> n_agent n_ray d", n_agent=n_agent)

    def _init_single(state, agent_idx, o_obs_state, a_state):
        q = state[:2]
        dq = state[2:]
        all_obs_state = jnp.concatenate([a_state, o_obs_state], axis=0)
        all_obs_pos = all_obs_state[:, :2]
        all_obs_vel = all_obs_state[:, 2:]
        all_isobs = jnp.arange(len(all_obs_pos)) >= n_agent
        # 动态安全半径: 基于接近速度的刹车距离
        all_p_rel = q - all_obs_pos
        all_rel_vel = dq[None, :] - all_obs_vel
        all_p_rel_norm = jnp.linalg.norm(all_p_rel, axis=-1, keepdims=True)
        all_p_rel_hat = all_p_rel / (all_p_rel_norm + 1e-8)
        all_v_approach = jnp.maximum(0.0, -jnp.sum(all_rel_vel * all_p_rel_hat, axis=-1))
        all_braking_dist = all_v_approach ** 2 / (2.0 * acc_scale + 1e-8)
        all_dynamic_margin = safety_margin + all_braking_dist
        all_base_r = jnp.where(all_isobs, r, 2 * r)
        all_safety_sq = (all_base_r + all_dynamic_margin) ** 2
        # 当前约束 + 预测约束取 max
        all_g_now = all_safety_sq - (all_p_rel ** 2).sum(axis=-1)
        lookahead_dt = dt * n_lookahead
        all_p_rel_pred = all_p_rel + all_rel_vel * lookahead_dt
        all_g_pred = all_safety_sq - (all_p_rel_pred ** 2).sum(axis=-1)
        all_g = jnp.maximum(all_g_now, all_g_pred)
        all_J_g = -2.0 * all_p_rel
        all_dg_dt = jnp.sum(all_J_g * all_rel_vel, axis=-1)
        all_g_viab = all_g + K * all_dg_dt
        s = jnp.sqrt(jnp.maximum(-2.0 * all_g_viab, s_min ** 2))
        return s

    agent_idx = jnp.arange(n_agent)
    s_all = jax.vmap(_init_single, in_axes=(0, 0, 0, None))(
        a_states, agent_idx, a_obs_states, a_states
    )
    return s_all


def get_manifold_fn(env: MultiAgentEnv, k: int = 3, K: float = 0.15, Kc: float = 8.0,
                    s_min: float = 0.1, alpha_max: float = 5.0, g_act_thresh: float = 0.1,
                    safety_margin: float = 0.02, n_lookahead: int = 2, w_slack: float = 5.0):
    """工厂函数：创建 manifold 修正函数"""
    n_agent = env.num_agents
    n_rays = env.params["top_k_rays"]
    r = env.params["car_radius"]
    dt = env._dt
    acc_scale = 1.0 / env.params["m"]  # action → acceleration: ddq = u * acc_scale
    max_neighbors = (n_agent - 1) + n_rays
    k = min(k, max_neighbors)
    print(f"  manifold k clamped to {k} (max_neighbors={max_neighbors})")
    print(f"  manifold params: K={K}, Kc={Kc}, alpha_max={alpha_max}, "
          f"g_act_thresh={g_act_thresh}, safety_margin={safety_margin}, "
          f"dt={dt}, acc_scale={acc_scale}, n_lookahead={n_lookahead}")

    manifold_fn = ft.partial(
        manifold_all_agents,
        r=r,
        n_agent=n_agent,
        n_rays=n_rays,
        k=k,
        K=K,
        Kc=Kc,
        s_min=s_min,
        alpha_max=alpha_max,
        g_act_thresh=g_act_thresh,
        dt=dt,
        safety_margin=safety_margin,
        n_lookahead=n_lookahead,
        acc_scale=acc_scale,
        w_slack=w_slack,
    )

    init_slack_fn = ft.partial(
        manifold_init_slack,
        r=r,
        n_agent=n_agent,
        n_rays=n_rays,
        K=K,
        s_min=s_min,
        safety_margin=safety_margin,
        dt=dt,
        n_lookahead=n_lookahead,
        acc_scale=acc_scale,
    )

    return manifold_fn, init_slack_fn
