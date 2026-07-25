import pathlib
import jax.numpy as jnp

from abc import ABC, abstractmethod, abstractproperty
from typing import NamedTuple, Optional, Tuple

from ..trainer.data import Rollout
from ..utils.graph import GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Reward, State


class StepResult(NamedTuple):
    graph: GraphsTuple
    reward: Reward
    cost: Cost
    done: Done
    info: Info


class RolloutResult(NamedTuple):
    Tp1_graph: GraphsTuple
    T_action: Action
    T_reward: Reward
    T_cost: Cost
    T_done: Done
    T_info: Info
    T_rnn_state: Array


class MultiAgentEnv(ABC):

    PARAMS = {}

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 128,
            dt: float = 0.03,
            params: Optional[dict] = None
    ):
        super(MultiAgentEnv, self).__init__()
        self._num_agents = num_agents
        self._dt = dt
        if params is None:
            params = self.PARAMS
        self._params = params
        self._t = 0
        self._max_step = max_step
        self._area_size = area_size

    @property
    def params(self) -> dict:
        return self._params

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def area_size(self) -> float:
        return self._area_size

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def max_episode_steps(self) -> int:
        return self._max_step

    @abstractproperty
    def n_cost(self) -> int:
        pass

    @abstractproperty
    def cost_components(self) -> Tuple[str, ...]:
        pass

    def clip_state(self, state: State) -> State:
        lower_limit, upper_limit = self.state_lim(state)
        return jnp.clip(state, lower_limit, upper_limit)

    def clip_action(self, action: Action) -> Action:
        lower_limit, upper_limit = self.action_lim()
        return jnp.clip(action, lower_limit, upper_limit)

    def lighten_graph(self, graph: GraphsTuple) -> GraphsTuple:
        """Return a storage-light copy of the graph for the rollout buffer (default: identity).
        Envs that carry heavy env_states (e.g. an MJX Data pytree) override this to strip it, so
        the scan-stacked history stays small. The scan CARRY keeps the full graph for stepping;
        only the stored OUTPUT is lightened. Training never reads the stripped fields."""
        return graph

    @abstractproperty
    def state_dim(self) -> int:
        pass

    @abstractproperty
    def node_dim(self) -> int:
        pass

    @abstractproperty
    def edge_dim(self) -> int:
        pass

    @abstractproperty
    def action_dim(self) -> int:
        pass

    @abstractmethod
    def reset(self, key: Array) -> GraphsTuple:
        pass

    @abstractmethod
    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False) -> StepResult:
        pass

    @abstractmethod
    def get_cost(self, graph: GraphsTuple) -> Cost:
        pass

    @abstractmethod
    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[State, State],
            limits of the state
        """
        pass

    @abstractmethod
    def action_lim(self) -> Tuple[Action, Action]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[Action, Action],
            limits of the action
        """
        pass

    @abstractmethod
    def get_graph(self, state: State) -> GraphsTuple:
        pass

    @abstractmethod
    def render_video(
            self,
            rollout: Rollout,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        pass

    def get_subgoal_shadow_cost(self, graph: GraphsTuple, subgoal_pos: Array) -> Array:
        """
        检查subgoal是否在障碍物阴影区（默认实现：返回0，表示安全）

        子类（如LidarEnv）可以重写此方法来实现具体的检查逻辑。

        Parameters
        ----------
        graph: GraphsTuple, 当前环境状态
        subgoal_pos: Array, shape (n_agents, 2), 每个agent的subgoal位置

        Returns
        -------
        cost: Array, shape (n_agents,), 每个agent的subgoal shadow cost
              -1 表示危险, 0 表示安全
        """
        return jnp.zeros(self.num_agents)
