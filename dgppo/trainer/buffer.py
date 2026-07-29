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
        """Initialize replay buffer

        Args:
            size: Maximum buffer size in terms of episodes (not transitions)
                  For hierarchical RL, each rollout has shape (n_env, T, ...)
                  so buffer stores up to 'size' episodes along axis 0
        """
        super(ReplayBuffer, self).__init__(size)
        self._buffer = None

    def append(self, rollout: Rollout):
        """Append rollout to buffer and trim if exceeds size

        The buffer stores rollouts along axis 0 (episode dimension).
        When buffer exceeds size, we keep only the most recent 'size' episodes.
        """
        if self._buffer is None:
            self._buffer = jax2np(rollout)
        else:
            self._buffer = tree_merge([self._buffer, jax2np(rollout)])

        # Trim buffer to keep only the most recent episodes
        # self._size is in terms of episodes (axis 0), not total transitions
        if self._buffer.length > self._size:
            self._buffer = jtu.tree_map(lambda x: x[-self._size:], self._buffer)

    def sample(self, batch_size: int) -> Rollout:
        """Sample batch_size episodes from buffer

        Note: batch_size should be much smaller than buffer.length to avoid
        sampling most of the buffer at once
        """
        idx = np.random.randint(0, self._buffer.length, batch_size)
        return np2jax(self.get_data(idx))

    def get_data(self, idx: np.ndarray) -> Rollout:
        return jtu.tree_map(lambda x: x[idx], self._buffer)

    @property
    def length(self) -> int:
        """Return number of episodes in buffer (not transitions)"""
        if self._buffer is None:
            return 0
        return self._buffer.length  # Use .length instead of .n_data
