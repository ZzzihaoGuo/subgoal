"""Flashbax-based Replay Buffer for MATD3

Uses Flashbax for GPU-native, zero-copy replay buffer operations.
Much faster than NumPy-based buffer due to no CPU-GPU data transfer.
"""

import flashbax as fbx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Optional

from .data import Rollout


class FlashbaxReplayBuffer:
    """JAX-native replay buffer using Flashbax

    Advantages over NumPy buffer:
    - All operations on GPU (no CPU-GPU transfer)
    - Zero-copy sampling
    - JIT-friendly
    - Pre-allocated memory (no dynamic growth)
    """

    def __init__(
        self,
        max_length: int = 10000,
        min_length: int = 100,
        sample_batch_size: int = 32,
        add_batch_size: int = 1,
        sample_sequence_length: int = 1,
    ):
        """Initialize Flashbax replay buffer

        Args:
            max_length: Maximum number of episodes in buffer
            min_length: Minimum episodes before sampling
            sample_batch_size: Number of episodes to sample per batch
            add_batch_size: Number of episodes added per call (usually n_env)
            sample_sequence_length: Length of sampled sequences (1 for full episodes)
        """
        self.max_length = max_length
        self.min_length = min_length
        self.sample_batch_size = sample_batch_size
        self.add_batch_size = add_batch_size

        # Flashbax buffer (will be initialized on first add)
        self._buffer = None
        self._buffer_state = None

    def init(self, sample_rollout: Rollout):
        """Initialize buffer with a sample rollout to infer shapes

        Args:
            sample_rollout: A sample rollout to infer data structure (n_env, T, ...)
        """
        # Create flashbax flat buffer (stores individual episodes)
        # Each episode has shape (T, ...)
        self._buffer = fbx.make_flat_buffer(
            max_length=self.max_length,
            min_length=self.min_length,
            sample_batch_size=self.sample_batch_size,
        )

        # Initialize buffer state with sample episode (single env's data)
        # Take first env's trajectory: (T, ...)
        sample_single = jtu.tree_map(lambda x: x[0], sample_rollout)
        self._buffer_state = self._buffer.init(sample_single)

    def add(self, rollouts: Rollout):
        """Add rollouts to buffer

        Args:
            rollouts: Rollout with shape (n_env, T, ...) from parallel envs
        """
        if self._buffer is None:
            # First call: initialize buffer with sample data
            self.init(rollouts)

        # Flashbax flat buffer: add each episode one by one
        # rollouts has shape (n_env, T, ...), each env is one episode
        n_envs = jtu.tree_leaves(rollouts)[0].shape[0]

        for i in range(n_envs):
            # Extract single episode: (T, ...)
            single_episode = jtu.tree_map(lambda x: x[i], rollouts)
            # Add to buffer
            self._buffer_state = self._buffer.add(self._buffer_state, single_episode)

    def sample(self, key: jax.random.PRNGKey) -> Rollout:
        """Sample a batch from buffer

        Args:
            key: JAX random key

        Returns:
            Sampled rollout batch of shape (sample_batch_size, T, ...)
        """
        if self._buffer_state is None:
            raise ValueError("Buffer not initialized. Call add() first.")

        if not self.can_sample():
            raise ValueError(
                f"Buffer has {self.length} episodes, "
                f"need at least {self.min_length} to sample"
            )

        # Sample from buffer
        # Flashbax flat buffer returns (batch_size, T, ...)
        batch = self._buffer.sample(self._buffer_state, key).experience

        return batch

    def can_sample(self) -> bool:
        """Check if buffer has enough data to sample"""
        return self._buffer.can_sample(self._buffer_state)

    @property
    def length(self) -> int:
        """Return current number of episodes in buffer"""
        if self._buffer_state is None:
            return 0
        # Flashbax tracks number of added items
        return self._buffer_state.current_index

    @property
    def is_ready(self) -> bool:
        """Check if buffer has minimum required data"""
        return self.length >= self.min_length
