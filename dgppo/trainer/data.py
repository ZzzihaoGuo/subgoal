from typing import NamedTuple, Optional

from ..utils.typing import Array
from ..utils.typing import Action, Reward, Cost, Done
from ..utils.graph import GraphsTuple


class Rollout(NamedTuple):
    graph: GraphsTuple
    actions: Action
    rnn_states: Array
    rewards: Reward
    costs: Cost
    dones: Done
    log_pis: Optional[Array]
    next_graph: GraphsTuple
    sparse_rewards: Optional[Reward] = None  # 自定义的稀疏奖励
    dist2goal: Optional[Array] = None  # 每个goal到最近agent的距离

    @property
    def length(self) -> int:
        return self.rewards.shape[0]

    @property
    def time_horizon(self) -> int:
        return self.rewards.shape[1]

    @property
    def num_agents(self) -> int:
        return self.rewards.shape[2]

    @property
    def n_data(self) -> int:
        return self.length * self.time_horizon
