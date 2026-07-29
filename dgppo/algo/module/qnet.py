import functools as ft
import flax.linen as nn
import jax.numpy as jnp

from typing import Type

from ...nn.mlp import MLP
from ...nn.gnn import GraphTransformerGNN, GNN
from ...nn.rnn import RNN
from ...nn.utils import default_nn_init
from ...utils.typing import Array, Params, PRNGKey, Action
from ...utils.graph import GraphsTuple


class QStateFn(nn.Module):
    """Q-network for MATD3: Q(state, action) -> Q-value

    Processes graph state through GNN, concatenates with actions,
    then outputs Q-values for each agent.
    """
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]
    action_dim: int
    n_agents: int
    rnn_cls: Type[RNN] = None
    decompose: bool = False  # If True, output per-agent Q-values; else global Q

    @nn.compact
    def __call__(
            self,
            graph: GraphsTuple,
            actions: Action,  # (n_agents, action_dim)
            rnn_state: Array,
            n_agents: int,
            *args,
            **kwargs
    ) -> tuple[Array, Array]:
        """
        Args:
            graph: GraphsTuple containing state information
            actions: (n_agents, action_dim) - high-level subgoal actions
            rnn_state: (n_layers, n_carries, hid_size) or None
            n_agents: number of agents

        Returns:
            q_values: (n_agents, 1) if decompose else (1, 1)
            rnn_state: updated RNN state
        """
        # Process graph through GNN
        x = self.gnn_cls()(graph, node_type=0, n_type=n_agents)  # (n_agents, gnn_out_dim)

        # Concatenate with actions
        x = jnp.concatenate([x, actions], axis=-1)  # (n_agents, gnn_out_dim + action_dim)

        if not self.decompose:
            # Global Q-value: aggregate all agent info
            x = x.reshape(-1)  # Flatten
            x = x[None, :]  # (1, n_agents * (gnn_out_dim + action_dim))

        # Pass through MLP head
        x = self.head_cls()(x)  # (n_agents, hid_dim) or (1, hid_dim)

        # Pass through RNN if enabled
        if self.rnn_cls is not None:
            x, rnn_state = self.rnn_cls()(x, rnn_state)

        # Output Q-value
        q = nn.Dense(1, kernel_init=default_nn_init())(x)  # (n_agents, 1) or (1, 1)

        return q, rnn_state


class QNetwork:
    """Q-Network wrapper for MATD3

    Implements Q(state, action) -> Q-value for multi-agent setting.
    Can be configured for centralized or decentralized Q-learning.
    """

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            action_dim: int = 2,  # Subgoal dimension (x, y)
            use_rnn: bool = True,
            rnn_layers: int = 1,
            gnn_layers: int = 2,
            gnn_out_dim: int = 64,
            use_lstm: bool = False,
            decompose: bool = False,  # False for centralized Q
            n_heads: int = 3
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.gnn_out_dim = gnn_out_dim
        self.decompose = decompose

        # GNN for processing graph state
        self.gnn = ft.partial(
            GraphTransformerGNN,
            msg_dim=32,
            out_dim=gnn_out_dim,
            n_heads=n_heads,
            n_layers=gnn_layers
        )

        # MLP head for Q-value computation
        if decompose:
            # Decentralized: smaller network per agent
            self.head = ft.partial(
                MLP,
                hid_sizes=(128, 64),
                act=nn.relu,
                act_final=True,
                name='QNetHead'
            )
        else:
            # Centralized: network for global Q
            # Reduced from (256, 256) to (128, 128) for better efficiency
            # Still larger than V(s) network (64, 64) due to Q(s,a) input complexity
            self.head = ft.partial(
                MLP,
                hid_sizes=(128, 128),
                act=nn.relu,
                act_final=True,
                name='QNetHead'
            )

        # RNN for temporal modeling
        self.use_rnn = use_rnn
        if use_rnn:
            self.rnn_base = ft.partial(nn.LSTMCell if use_lstm else nn.GRUCell, features=64)
            self.rnn = ft.partial(
                RNN,
                rnn_cls=self.rnn_base,
                rnn_layers=rnn_layers
            )
        else:
            self.rnn = None

        # Create the Q-network module
        self.net = QStateFn(
            gnn_cls=self.gnn,
            head_cls=self.head,
            action_dim=action_dim,
            n_agents=n_agents,
            rnn_cls=self.rnn,
            decompose=decompose,
        )

    def initialize_carry(self, key: PRNGKey) -> Array:
        """Initialize RNN state"""
        if self.use_rnn:
            return self.rnn_base().initialize_carry(key, (self.gnn_out_dim,))
        else:
            return jnp.zeros((self.gnn_out_dim,))

    def get_q_value(
            self,
            params: Params,
            obs: GraphsTuple,
            actions: Action,
            rnn_state: Array
    ) -> tuple[Array, Array]:
        """Compute Q(s, a)

        Args:
            params: Network parameters
            obs: GraphsTuple state
            actions: (n_agents, action_dim) actions
            rnn_state: RNN hidden state

        Returns:
            q_values: (n_agents, 1) or (1, 1)
            rnn_state: Updated RNN state
        """
        q_values, rnn_state = self.net.apply(params, obs, actions, rnn_state, self.n_agents)
        return q_values, rnn_state
