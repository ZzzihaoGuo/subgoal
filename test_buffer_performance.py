"""Compare performance of NumPy buffer vs Flashbax buffer

Usage:
    python test_buffer_performance.py
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import time
import numpy as np

from dgppo.trainer.buffer import ReplayBuffer
# from dgppo.trainer.buffer_flashbax import FlashbaxReplayBuffer
from dgppo.trainer.data import Rollout
from dgppo.utils.graph import GraphsTuple


def create_dummy_rollout(n_env=32, T=16, n_agents=3):
    """Create a dummy rollout for testing"""
    dummy_graph = GraphsTuple(
        nodes=jnp.zeros((n_env, n_agents, 10)),
        edges=jnp.zeros((n_env, n_agents, 8)),
        states=jnp.zeros((n_env, n_agents, 12)),
        n_node=jnp.array([n_agents] * n_env),
        n_edge=jnp.array([n_agents] * n_env),
        senders=jnp.tile(jnp.arange(n_agents), (n_env, 1)),
        receivers=jnp.tile(jnp.arange(n_agents), (n_env, 1)),
        node_type=jnp.zeros((n_env, n_agents)),
        env_states=jnp.zeros((n_env, 20)),
    )

    rollout = Rollout(
        graph=dummy_graph,
        actions=jnp.zeros((n_env, T, n_agents, 2)),
        rnn_states=jnp.zeros((n_env, T, n_agents, 64)),
        rewards=jnp.zeros((n_env, T, n_agents)),
        costs=jnp.zeros((n_env, T, n_agents)),
        dones=jnp.zeros((n_env, T), dtype=bool),
        log_pis=jnp.zeros((n_env, T, n_agents)),
        next_graph=dummy_graph,
        sparse_rewards=jnp.zeros((n_env, T)),
        dist2goal=jnp.zeros((n_env, T, n_agents)),
    )
    return rollout


def benchmark_numpy_buffer():
    """Benchmark NumPy-based buffer"""
    print("\n" + "="*60)
    print("Benchmarking NumPy Buffer")
    print("="*60)

    buffer = ReplayBuffer(size=1000)
    n_env = 32
    T = 16

    # Warmup
    rollout = create_dummy_rollout(n_env, T)
    buffer.append(rollout)

    # Benchmark append
    print("\n1. Testing APPEND performance...")
    n_appends = 100
    start = time.time()
    for i in range(n_appends):
        rollout = create_dummy_rollout(n_env, T)
        buffer.append(rollout)
        jax.block_until_ready(rollout)  # Ensure GPU sync
    jax.block_until_ready(buffer._buffer)
    append_time = time.time() - start
    print(f"  {n_appends} appends: {append_time:.3f}s ({append_time/n_appends*1000:.2f}ms per append)")

    # Benchmark sample
    print("\n2. Testing SAMPLE performance...")
    n_samples = 100
    key = jr.PRNGKey(0)
    start = time.time()
    for i in range(n_samples):
        key, subkey = jr.split(key)
        batch = buffer.sample(32)
        jax.block_until_ready(batch)
    sample_time = time.time() - start
    print(f"  {n_samples} samples: {sample_time:.3f}s ({sample_time/n_samples*1000:.2f}ms per sample)")

    print(f"\n  Buffer length: {buffer.length}")
    print(f"  Total time: {append_time + sample_time:.3f}s")

    return append_time, sample_time


# def benchmark_flashbax_buffer():
#     """Benchmark Flashbax-based buffer"""
#     print("\n" + "="*60)
#     print("Benchmarking Flashbax Buffer")
#     print("="*60)
#
#     buffer = FlashbaxReplayBuffer(
#         max_length=1000,
#         min_length=10,
#         sample_batch_size=32,
#         add_batch_size=32,
#     )
#     n_env = 32
#     T = 16
#
#     # Warmup
#     rollout = create_dummy_rollout(n_env, T)
#     buffer.add(rollout)
#
#     # Benchmark append
#     print("\n1. Testing ADD performance...")
#     n_appends = 100
#     start = time.time()
#     for i in range(n_appends):
#         rollout = create_dummy_rollout(n_env, T)
#         buffer.add(rollout)
#         jax.block_until_ready(rollout)
#     jax.block_until_ready(buffer._buffer_state)
#     append_time = time.time() - start
#     print(f"  {n_appends} adds: {append_time:.3f}s ({append_time/n_appends*1000:.2f}ms per add)")
#
#     # Benchmark sample
#     print("\n2. Testing SAMPLE performance...")
#     n_samples = 100
#     key = jr.PRNGKey(0)
#     start = time.time()
#     for i in range(n_samples):
#         key, subkey = jr.split(key)
#         batch = buffer.sample(subkey)
#         jax.block_until_ready(batch)
#     sample_time = time.time() - start
#     print(f"  {n_samples} samples: {sample_time:.3f}s ({sample_time/n_samples*1000:.2f}ms per sample)")
#
#     print(f"\n  Buffer length: {buffer.length}")
#     print(f"  Total time: {append_time + sample_time:.3f}s")
#
#     return append_time, sample_time


if __name__ == "__main__":
    print("Buffer Performance Comparison")
    print(f"JAX devices: {jax.devices()}")

    # NumPy buffer
    numpy_append, numpy_sample = benchmark_numpy_buffer()

    # Flashbax buffer
    # flashbax_append, flashbax_sample = benchmark_flashbax_buffer()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"NumPy Buffer:")
    print(f"  Append: {numpy_append/100*1000:.2f}ms")
    print(f"  Sample: {numpy_sample/100*1000:.2f}ms")
    print(f"  Total:  {(numpy_append + numpy_sample):.3f}s")

    # print(f"\nFlashbax Buffer:")
    # print(f"  Append: {flashbax_append/100*1000:.2f}ms")
    # print(f"  Sample: {flashbax_sample/100*1000:.2f}ms")
    # print(f"  Total:  {(flashbax_append + flashbax_sample):.3f}s")
    #
    # print(f"\nSpeedup:")
    # print(f"  Append: {numpy_append/flashbax_append:.2f}x")
    # print(f"  Sample: {numpy_sample/flashbax_sample:.2f}x")
    # print(f"  Total:  {(numpy_append + numpy_sample)/(flashbax_append + flashbax_sample):.2f}x")
