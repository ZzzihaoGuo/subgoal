import flax.linen as nn
import functools as ft
import numpy as np
import jax.nn as jnn
import jax.numpy as jnp

from typing import Type, Tuple, Any
from abc import ABC, abstractproperty, abstractmethod

from .distribution import TanhTransformedDistribution, tfd
from ...utils.typing import Action, Array
from ...utils.graph import GraphsTuple
from ...nn.utils import default_nn_init, scaled_init
from ...nn.gnn import GNN, GraphTransformerGNN
from ...nn.rnn import RNN
from ...nn.mlp import MLP
from ...utils.typing import PRNGKey, Params


class PolicyNet(nn.Module):
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]
    rnn_cls: Type[RNN] = None

    @nn.compact
    def __call__(
            self, graph: GraphsTuple, rnn_state: Array, node_type: int = None, n_type: int = None
    ) -> [Array, Array]:
        x = self.gnn_cls()(graph, node_type, n_type)
        x = self.head_cls()(x)
        if self.rnn_cls is not None:
            x, rnn_state = self.rnn_cls()(x, rnn_state)
        return x, rnn_state


class PolicyDistribution(nn.Module, ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> [tfd.Distribution, Array]:
        pass

    @abstractproperty
    def nu(self) -> int:
        pass


class TanhNormal(PolicyDistribution):
    base_cls: Type[PolicyNet]
    _nu: int
    scale_final: float = 0.01
    std_dev_min: float = 1e-5
    std_dev_init: float = 0.5
    
    @property
    def std_dev_init_inv(self):
        # inverse of log(sum(exp())).
        inv = np.log(np.exp(self.std_dev_init) - 1)
        assert np.allclose(np.logaddexp(inv, 0), self.std_dev_init)
        return inv

    @nn.compact
    def __call__(
            self, obs: GraphsTuple, rnn_state: Array, n_agents: int, *args, **kwargs
    ) -> [tfd.Distribution, Array]:
        x, rnn_state = self.base_cls()(obs, rnn_state=rnn_state, node_type=0, n_type=n_agents)
        scaler_init = scaled_init(default_nn_init(), self.scale_final)
        feats_scaled = nn.Dense(64, kernel_init=scaler_init, name="ScaleHid")(x)

        means = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseMean")(feats_scaled)
        stds_trans = nn.Dense(self.nu, kernel_init=default_nn_init(), name="OutputDenseStdTrans")(feats_scaled)
        stds = jnn.softplus(stds_trans + self.std_dev_init_inv) + self.std_dev_min
        # stds = self.std_dev_min
        distribution = tfd.Normal(loc=means, scale=stds)
        return tfd.Independent(TanhTransformedDistribution(distribution), reinterpreted_batch_ndims=1), rnn_state

    @property
    def nu(self):
        return self._nu


class MultiAgentPolicy(ABC):

    def __init__(self, node_dim: int, edge_dim: int, n_agents: int, action_dim: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.action_dim = action_dim

    @abstractmethod
    def initialize_carry(self, key: PRNGKey) -> Array:
        pass

    @abstractmethod
    def get_action(self, params: Params, obs: GraphsTuple, rnn_state: Array) -> [Action, Array]:
        """
        Get action from the policy.

        Returns
        -------
        action: Action,
            The action to be taken by the agent.
        rnn_state: Array,
            The updated rnn states.
        """
        pass

    @abstractmethod
    def sample_action(
            self, params: Params, obs: GraphsTuple, rnn_state: Array, key: PRNGKey
    ) -> Tuple[Action, Array, Array]:
        """
        Sample action from the policy.

        Returns
        -------
        action: Action,
            The stochastic action to be taken by the agent.
        log_pi: Array,
            The log probability of the action.
        rnn_state: Array,
            The updated rnn states.
        """
        pass

    @abstractmethod
    def eval_action(
            self, params: Params, obs: GraphsTuple, action: Action, rnn_state: Array, key: PRNGKey
    ) -> Tuple[Array, Array, Array]:
        pass


class PPOPolicy(MultiAgentPolicy):

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            action_dim: int,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            gnn_layers: int = 1,
            gnn_out_dim: int = 16,
            use_lstm: bool = False,
    ):
        super().__init__(node_dim, edge_dim, n_agents, action_dim)
        self.gnn_out_dim = gnn_out_dim
        self.use_rnn = use_rnn
        self.gnn = ft.partial(
            GraphTransformerGNN,
            msg_dim=32,
            out_dim=gnn_out_dim,
            n_heads=3,
            n_layers=gnn_layers
        )
        self.head = ft.partial(
            MLP,
            hid_sizes=(64, 64),
            act=nn.relu,
            act_final=True,
            name='PolicyGNNHead'
        )
        if use_rnn:
            self.rnn_base = ft.partial(nn.LSTMCell if use_lstm else nn.GRUCell, features=64)
            self.rnn = ft.partial(
                RNN,
                rnn_cls=self.rnn_base,
                rnn_layers=rnn_layers
            )
            self.policy_base = ft.partial(
                PolicyNet,
                gnn_cls=self.gnn,
                head_cls=self.head,
                rnn_cls=self.rnn,
            )
            self.dist = TanhNormal(base_cls=self.policy_base, _nu=action_dim)
        else:
            self.policy_base = ft.partial(
                PolicyNet,
                gnn_cls=self.gnn,
                head_cls=self.head,
            )
            self.dist = TanhNormal(base_cls=self.policy_base, _nu=action_dim)

    def initialize_carry(self, key: PRNGKey) -> tuple[Array | Any, Array | Any] | Array:
        if self.use_rnn:
            return self.rnn_base().initialize_carry(key, (self.gnn_out_dim,))
        else:
            return jnp.zeros((self.gnn_out_dim,))

    def get_action(self, params: Params, obs: GraphsTuple, rnn_state: Array) -> [Action, Array]:
        dist, rnn_state = self.dist.apply(params, obs, rnn_state, n_agents=self.n_agents)
        action = dist.mode()
        return action, rnn_state

    def sample_action(
            self, params: Params, obs: GraphsTuple, rnn_state: Array, key: PRNGKey
    ) -> Tuple[Action, Array, Array]:
        rnn_state: Array
        dist, rnn_state = self.dist.apply(params, obs, rnn_state, n_agents=self.n_agents)
        action = dist.sample(seed=key)
        log_pi = dist.log_prob(action)
        return action, log_pi, rnn_state

    def eval_action(
            self, params: Params, obs: GraphsTuple, action: Action, rnn_state: Array, key: PRNGKey
    ) -> Tuple[Array, Array, Array]:
        rnn_state: Array
        dist, rnn_state = self.dist.apply(params, obs, rnn_state, n_agents=self.n_agents)
        log_pi = dist.log_prob(action)
        entropy = dist.entropy(seed=key)
        return log_pi, entropy, rnn_state
    

class SubgoalPolicy(MultiAgentPolicy):
    """
    输出 Subgoal 绝对坐标的 Policy
    复用 TanhNormal，但输出后做 (tanh + 1) / 2 * area_size 变换
    """
    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            subgoal_dim: int = 2,  # [x, y]
            area_size: float = 1.5,
            use_rnn: bool = True,
            rnn_layers: int = 1,
            gnn_layers: int = 2,
            gnn_out_dim: int = 64,
            use_lstm: bool = False,
            use_relative_subgoal: bool = False,  # 是否使用相对坐标
            max_delta: float = None,  # 相对模式下的最大偏移量，默认为 area_size / 4
    ):
        super().__init__(node_dim, edge_dim, n_agents, subgoal_dim)
        self.area_size = area_size
        self.subgoal_dim = subgoal_dim
        self.use_relative_subgoal = use_relative_subgoal
        self.max_delta = max_delta if max_delta is not None else area_size / 4
        self.gnn_out_dim = gnn_out_dim
        self.use_rnn = use_rnn
        
        # 复用 PPOPolicy 的结构
        self.gnn = ft.partial(
            GraphTransformerGNN,
            msg_dim=32,
            out_dim=gnn_out_dim,
            n_heads=3,
            n_layers=gnn_layers
        )
        self.head = ft.partial(
            MLP,
            hid_sizes=(64, 64),
            act=nn.relu,
            act_final=True,
            name='SubgoalPolicyHead'
        )
        
        if use_rnn:
            self.rnn_base = ft.partial(nn.LSTMCell if use_lstm else nn.GRUCell, features=64)
            self.rnn = ft.partial(RNN, rnn_cls=self.rnn_base, rnn_layers=rnn_layers)
            self.policy_base = ft.partial(
                PolicyNet,
                gnn_cls=self.gnn,
                head_cls=self.head,
                rnn_cls=self.rnn,
            )
            # 复用 TanhNormal
            self.dist = TanhNormal(base_cls=self.policy_base, _nu=subgoal_dim)
        else:
            self.policy_base = ft.partial(
                PolicyNet,
                gnn_cls=self.gnn,
                head_cls=self.head,
            )
            self.dist = TanhNormal(base_cls=self.policy_base, _nu=subgoal_dim)

    def initialize_carry(self, key: PRNGKey) -> Array:
        if self.use_rnn:
            return self.rnn_base().initialize_carry(key, (self.gnn_out_dim,))
        else:
            return jnp.zeros((self.gnn_out_dim,))

    def _transform_to_subgoal(self, action_tanh: Array, current_pos: Array = None) -> Array:
        """
        将 tanh 输出 [-1, 1] 转换为 subgoal 坐标

        Args:
            action_tanh: (n_agents, 2) in [-1, 1]
            current_pos: (n_agents, 2) 当前位置，仅在相对模式下使用

        Returns:
            subgoal: (n_agents, 2) in [0, area_size]
        """
        if self.use_relative_subgoal:
            # 相对模式: tanh [-1, 1] → delta [-max_delta, max_delta]
            # subgoal = current_pos + delta
            delta = action_tanh * self.max_delta
            subgoal = current_pos + delta
            # 裁剪到地图范围
            subgoal = jnp.clip(subgoal, 0, self.area_size)
        else:
            # 绝对模式（原来的方式）: tanh [-1, 1] → [0, area_size]
            subgoal = (action_tanh + 1.0) / 2.0 * self.area_size
        return subgoal

    def _transform_from_subgoal(self, subgoal: Array, current_pos: Array = None) -> Array:
        """
        反向变换：从 subgoal 坐标到 [-1, 1]
        用于 eval_action

        Args:
            subgoal: (n_agents, 2) in [0, area_size]
            current_pos: (n_agents, 2) 当前位置，仅在相对模式下使用

        Returns:
            action_tanh: (n_agents, 2) in [-1, 1]
        """
        if self.use_relative_subgoal:
            # 相对模式: delta = subgoal - current_pos → tanh = delta / max_delta
            delta = subgoal - current_pos
            action_tanh = delta / self.max_delta
            action_tanh = jnp.clip(action_tanh, -1, 1)
        else:
            # 绝对模式（原来的方式）
            action_tanh = subgoal / self.area_size * 2.0 - 1.0
        return action_tanh

    def _get_current_pos(self, obs: GraphsTuple) -> Array:
        """从 obs 中提取当前 agent 位置"""
        return obs.type_states(type_idx=0, n_type=self.n_agents)[:, :2]  # (n_agents, 2)

    def get_action(self, params: Params, obs: GraphsTuple, rnn_state: Array) -> [Action, Array]:
        """返回 subgoal: (n_agents, 2) - [x, y] 坐标"""
        dist, rnn_state = self.dist.apply(params, obs, rnn_state, n_agents=self.n_agents)
        action_tanh = dist.mode()  # (n_agents, 2) in [-1, 1]
        current_pos = self._get_current_pos(obs) if self.use_relative_subgoal else None
        subgoal = self._transform_to_subgoal(action_tanh, current_pos)
        return subgoal, rnn_state

    def sample_action(
            self, params: Params, obs: GraphsTuple, rnn_state: Array, key: PRNGKey
    ) -> Tuple[Action, Array, Array]:
        """采样 subgoal 并计算 log_prob"""
        dist, rnn_state = self.dist.apply(params, obs, rnn_state, n_agents=self.n_agents)
        action_tanh = dist.sample(seed=key)  # (n_agents, 2) in [-1, 1]
        log_prob = dist.log_prob(action_tanh)  # log_prob 是在 tanh 空间计算的
        current_pos = self._get_current_pos(obs) if self.use_relative_subgoal else None
        subgoal = self._transform_to_subgoal(action_tanh, current_pos)
        return subgoal, log_prob, rnn_state

    def eval_action(
            self, params: Params, obs: GraphsTuple, action: Action, rnn_state: Array, key: PRNGKey
    ) -> Tuple[Array, Array, Array]:
        """
        评估给定 subgoal 的 log_prob 和 entropy

        Args:
            action: subgoal in [0, area_size]
        """
        dist, rnn_state = self.dist.apply(params, obs, rnn_state, n_agents=self.n_agents)
        # 将 subgoal 转回 tanh 空间
        current_pos = self._get_current_pos(obs) if self.use_relative_subgoal else None
        action_tanh = self._transform_from_subgoal(action, current_pos)
        log_prob = dist.log_prob(action_tanh)
        entropy = dist.entropy()
        return log_prob, entropy, rnn_state