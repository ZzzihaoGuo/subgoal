"""Profile MATD3 training to identify bottlenecks

Usage:
    python profile_matd3.py
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import time
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.trainer_matd3_manifold import TrainerQMixManifold
import functools as ft


def profile_step_by_step():
    """Profile each component of a training step"""

    # Setup
    env = make_env('LidarTarget', num_agents=3, num_obs=3, max_step=128)
    env_test = make_env('LidarTarget', num_agents=3, num_obs=3, max_step=128)

    # Initialize manifold
    env.init_manifold(k=3, K=0.5, Kc=30.0, alpha_max=3.0,
                     g_act_thresh=0.02, safety_margin=0.02,
                     n_lookahead=0, w_slack=10.0)

    algo = make_algo(
        'informarl_matd3',
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=3,
        area_size=env.area_size,
        subgoal_interval=8,
        batch_size=32,
        use_rnn=False,
        seed=0
    )

    n_env = 64

    print("="*60)
    print("MATD3 Performance Profiling")
    print("="*60)
    print(f"n_env: {n_env}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # Setup rollout function
    from dgppo.trainer.utils import rollout_hierarchical_manifold

    init_rnn_state = algo.init_actor_rnn_state
    subgoal_interval = algo.subgoal_interval

    def rollout_fn_single(cur_params, cur_key, reach_thresh):
        return rollout_hierarchical_manifold(
            env,
            ft.partial(algo.step, params=cur_params, add_noise=True),
            init_rnn_state,
            cur_key,
            subgoal_interval=subgoal_interval,
            reach_thresh=reach_thresh,
        )

    def rollout_fn(cur_params, cur_keys, reach_thresh):
        return jax.vmap(ft.partial(rollout_fn_single, cur_params, reach_thresh=reach_thresh))(cur_keys)

    # Test WITHOUT JIT
    print("1. Testing rollout WITHOUT JIT...")
    key = jr.PRNGKey(0)
    keys = jr.split(key, n_env)

    start = time.time()
    rollouts_no_jit = rollout_fn(algo.params, keys, 0.01)
    jax.block_until_ready(rollouts_no_jit)
    time_no_jit = time.time() - start
    print(f"   Time: {time_no_jit:.3f}s")

    # Test WITH JIT (first call = compile + run)
    print("\n2. Testing rollout WITH JIT (first call)...")
    rollout_fn_jit = jax.jit(rollout_fn)

    start = time.time()
    rollouts_jit_1 = rollout_fn_jit(algo.params, keys, 0.01)
    jax.block_until_ready(rollouts_jit_1)
    time_jit_compile = time.time() - start
    print(f"   Time (compile + run): {time_jit_compile:.3f}s")

    # Test WITH JIT (second call = cached)
    print("\n3. Testing rollout WITH JIT (cached)...")
    start = time.time()
    rollouts_jit_2 = rollout_fn_jit(algo.params, keys, 0.01)
    jax.block_until_ready(rollouts_jit_2)
    time_jit_cached = time.time() - start
    print(f"   Time (cached): {time_jit_cached:.3f}s")

    # Test buffer operations
    print("\n4. Testing buffer operations...")
    from dgppo.trainer.buffer import ReplayBuffer
    buffer = ReplayBuffer(size=10000)

    start = time.time()
    buffer.append(rollouts_no_jit)
    time_append = time.time() - start
    print(f"   Append time: {time_append*1000:.2f}ms")

    # Add more data
    for _ in range(10):
        buffer.append(rollouts_no_jit)

    start = time.time()
    batch = buffer.sample(32)
    jax.block_until_ready(batch)
    time_sample = time.time() - start
    print(f"   Sample time: {time_sample*1000:.2f}ms")

    # Test update
    print("\n5. Testing network update...")
    start = time.time()
    update_info = algo.update(batch, 0)
    jax.block_until_ready(update_info)
    time_update = time.time() - start
    print(f"   Update time: {time_update:.3f}s")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Rollout (no JIT):    {time_no_jit:.3f}s  ❌ SLOW")
    print(f"Rollout (JIT cache): {time_jit_cached:.3f}s  {'✅ FAST' if time_jit_cached < 1 else '⚠️ SLOW'}")
    print(f"Buffer append:       {time_append*1000:.1f}ms")
    print(f"Buffer sample:       {time_sample*1000:.1f}ms")
    print(f"Network update:      {time_update:.3f}s")
    print()
    print(f"Speedup from JIT: {time_no_jit/time_jit_cached:.1f}x")
    print()

    # Estimate iteration time
    total_with_jit = time_jit_cached + time_append/1000 + time_sample/1000 + time_update
    total_no_jit = time_no_jit + time_append/1000 + time_sample/1000 + time_update

    print(f"Estimated per-iteration time:")
    print(f"  With JIT:    {total_with_jit:.3f}s  ({1/total_with_jit:.2f} it/s)")
    print(f"  Without JIT: {total_no_jit:.3f}s  ({1/total_no_jit:.2f} it/s)")

    if time_jit_cached > 1.0:
        print("\n⚠️  WARNING: JIT rollout still slow (>1s)")
        print("    Possible causes:")
        print("    - Manifold solver overhead")
        print("    - Too many environments")
        print("    - Graph operations overhead")
        print("    Suggestions:")
        print("    - Reduce n_env to 32")
        print("    - Simplify graph structure")


if __name__ == "__main__":
    profile_step_by_step()
