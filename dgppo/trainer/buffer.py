import jax.tree_util as jtu
import numpy as np

from abc import ABC, abstractproperty, abstractmethod
from .data import Rollout
from .utils import jax2np, np2jax
from ..utils.utils import tree_merge
from ..utils.typing import Array


class Buffer(ABC):

    def __init__(self, size: int):
        self._size = size

    @abstractmethod
    def append(self, rollout: Rollout):
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> Rollout:
        pass

    @abstractproperty
    def length(self) -> int:
        pass


class ReplayBuffer(Buffer):

    def __init__(self, size: int):
        """Initialize replay buffer with circular indexing

        Args:
            size: Maximum buffer size in terms of episodes (not transitions)
                  For hierarchical RL, each rollout has shape (n_env, T, ...)
                  so buffer stores up to 'size' episodes along axis 0
        """
        super(ReplayBuffer, self).__init__(size)
        self._buffer = None
        self._pointer = 0  # Current write position
        self._current_size = 0  # Current number of episodes stored

    def append(self, rollout: Rollout):
        """Append rollout to buffer using circular indexing

        Uses a circular buffer to avoid copying data on every append.
        This is O(1) instead of O(buffer_size).
        """
        rollout_np = jax2np(rollout)
        n_new_episodes = rollout_np.graph.nodes.shape[0]  # First dimension is batch

        if self._buffer is None:
            # First time: allocate buffer with max_size
            # Pre-allocate space for 'size' episodes
            self._buffer = rollout_np
            self._current_size = min(n_new_episodes, self._size)
            self._pointer = self._current_size % self._size
        else:
            # Add new rollouts using circular indexing
            for i in range(n_new_episodes):
                # Get single episode
                single_episode = jtu.tree_map(lambda x: x[i:i+1], rollout_np)

                if self._current_size < self._size:
                    # Buffer not full yet: concatenate
                    self._buffer = tree_merge([self._buffer, single_episode])
                    self._current_size += 1
                else:
                    # Buffer full: overwrite oldest episode (circular)
                    def replace_at_index(buf_arr, new_arr):
                        buf_arr[self._pointer] = new_arr[0]
                        return buf_arr

                    self._buffer = jtu.tree_map(
                        replace_at_index,
                        self._buffer,
                        single_episode
                    )

                self._pointer = (self._pointer + 1) % self._size

    def sample(self, batch_size: int) -> Rollout:
        """Sample batch_size episodes from buffer

        Note: batch_size should be much smaller than buffer.length to avoid
        sampling most of the buffer at once
        """
        if self._current_size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Sample from valid range [0, current_size)
        idx = np.random.randint(0, self._current_size, batch_size)
        return np2jax(self.get_data(idx))

    def get_data(self, idx: np.ndarray) -> Rollout:
        return jtu.tree_map(lambda x: x[idx], self._buffer)

    @property
    def length(self) -> int:
        """Return number of episodes in buffer (not transitions)"""
        return self._current_size
