"""Q-Networks for QMIX Algorithm

This module implements Q-networks for QMIX hierarchical multi-agent RL:
- QMixAgentNetwork: Individual agent Q-network outputting Q-values for discretized subgoal actions
- MixingNetwork: Hypernetwork-based mixing of individual Q-values into Q_tot
"""

import functools as ft
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from typing import Type

from ...nn.mlp import MLP
from ...nn.gnn import GraphTransformerGNN, GNN
from ...nn.rnn import RNN
from ...nn.utils import default_nn_init
from ...utils.typing import Array, Params, PRNGKey
from ...utils.graph import GraphsTuple


class QMixAgentNetwork(nn.Module):
    """Individual agent Q-network for QMIX

    Outputs Q-values for discretized subgoal grid positions.
    For continuous subgoal space [0, area_size]^2, we discretize into n_bins x n_bins grid.
    """
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]
    rnn_cls: Type[RNN] = None
    n_actions: int = 100  # Number of discrete subgoal positions (default: 10x10 grid)

    @nn.compact
    def __call__(
            self,
            graph: GraphsTuple,
            rnn_state: Array,
            dones: Array,  # (n_agents,) - for RNN reset
            n_agents: int,
    ) -> tuple[Array, Array]:
        """
        Args:
            graph: GraphsTuple containing state information
            rnn_state: (n_agents, rnn_layers * hidden_size) or None
            dones: (n_agents,) - episode done flags for RNN reset
            n_agents: number of agents

        Returns:
            q_vals: (n_agents, n_actions) - Q-values for each discrete subgoal position
            rnn_state: updated RNN state
        """
        # Process graph through GNN
        x = self.gnn_cls()(graph, node_type=0, n_type=n_agents)  # (n_agents, gnn_out_dim)

        # Pass through MLP head
        x = self.head_cls()(x)  # (n_agents, hid_dim)

        # Pass through RNN if enabled
        if self.rnn_cls is not None:
            # Reset RNN state where dones == True
            # TODO: Handle RNN reset properly with scan
            x, rnn_state = self.rnn_cls()(x, rnn_state)

        # Output Q-values for all discrete actions
        q_vals = nn.Dense(
            self.n_actions,
            kernel_init=default_nn_init(),
            name='QOutput'
        )(x)  # (n_agents, n_actions)

        return q_vals, rnn_state


class HyperNetwork(nn.Module):
    """Hypernetwork for generating weights of mixing network

    Takes global state as input and outputs weights for the mixing network.
    Based on JaxMARL QMIX implementation.
    """
    hidden_dim: int
    output_dim: int
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, global_state: Array) -> Array:
        """
        Args:
            global_state: (batch_size, state_dim) - global state information

        Returns:
            weights: (batch_size, output_dim) - weights for mixing network
        """
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.orthogonal(self.init_scale),
            bias_init=nn.initializers.constant(0.0),
        )(global_state)
        x = nn.relu(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=nn.initializers.orthogonal(self.init_scale),
            bias_init=nn.initializers.constant(0.0),
        )(x)
        return x


class MixingNetwork(nn.Module):
    """QMIX Mixing Network

    Mixes individual agent Q-values into a joint Q_tot using hypernetworks.
    Enforces monotonicity constraint: ∂Q_tot/∂Q_i >= 0 by using abs() on weights.

    Based on the original QMIX paper and JaxMARL implementation.
    """
    embedding_dim: int = 32
    hypernet_hidden_dim: int = 64
    init_scale: float = 1.0

    @nn.compact
    def __call__(
            self,
            q_vals: Array,  # (n_agents, batch_size) - individual Q-values
            global_state: Array,  # (batch_size, state_dim) - global state
            n_agents: int,
    ) -> Array:
        """
        Args:
            q_vals: (n_agents, batch_size) - chosen Q-values for each agent
            global_state: (batch_size, state_dim) - global state from graph.__all__
            n_agents: number of agents

        Returns:
            q_tot: (batch_size,) - mixed total Q-value
        """
        batch_size = global_state.shape[0]

        # Transpose q_vals to (batch_size, n_agents) for easier processing
        q_vals = q_vals.T  # (batch_size, n_agents)

        # Generate mixing network weights using hypernetworks
        # First layer: (n_agents, embedding_dim)
        w_1 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim * n_agents,
            init_scale=self.init_scale,
        )(global_state)
        b_1 = nn.Dense(
            self.embedding_dim,
            kernel_init=nn.initializers.orthogonal(self.init_scale),
            bias_init=nn.initializers.constant(0.0),
        )(global_state)

        # Second layer: (embedding_dim, 1)
        w_2 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim,
            init_scale=self.init_scale,
        )(global_state)
        b_2 = HyperNetwork(
            hidden_dim=self.embedding_dim,
            output_dim=1,
            init_scale=self.init_scale
        )(global_state)

        # Reshape and enforce monotonicity with abs()
        w_1 = jnp.abs(w_1.reshape(batch_size, n_agents, self.embedding_dim))
        b_1 = b_1.reshape(batch_size, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(batch_size, self.embedding_dim, 1))
        b_2 = b_2.reshape(batch_size, 1, 1)

        # Mix: Q_tot = w_2^T * ELU(w_1^T * Q + b_1) + b_2
        # First layer
        hidden = nn.elu(jnp.matmul(q_vals[:, None, :], w_1) + b_1)  # (batch_size, 1, embedding_dim)

        # Second layer
        q_tot = jnp.matmul(hidden, w_2) + b_2  # (batch_size, 1, 1)

        return q_tot.squeeze()  # (batch_size,)


class QMixNetwork:
    """Wrapper for QMIX Q-network with discretized subgoal actions

    For hierarchical RL with continuous subgoal space [0, area_size]^2,
    we discretize into n_bins x n_bins grid (default: 10x10 = 100 actions).

    Example:
        area_size = 1.5, n_bins = 10
        grid_positions = [(0.075, 0.075), (0.075, 0.225), ..., (1.425, 1.425)]
        Each agent outputs Q(s, a) for a ∈ {0, 1, ..., 99}
    """

    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            n_agents: int,
            area_size: float = 1.5,
            n_bins: int = 10,  # Discretization resolution (10x10 grid)
            use_rnn: bool = True,
            rnn_layers: int = 1,
            gnn_layers: int = 2,
            gnn_out_dim: int = 64,
            use_lstm: bool = False,
            n_heads: int = 3,
            # Mixing network params
            mixer_embed_dim: int = 32,
            mixer_hypernet_hidden: int = 64,
    ):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents
        self.area_size = area_size
        self.n_bins = n_bins
        self.n_actions = n_bins * n_bins  # Total discrete subgoal positions
        self.gnn_out_dim = gnn_out_dim
        self.use_rnn = use_rnn

        # Create discrete subgoal grid
        self.subgoal_grid = self._create_subgoal_grid()

        # GNN for processing graph state
        self.gnn = ft.partial(
            GraphTransformerGNN,
            msg_dim=32,
            out_dim=gnn_out_dim,
            n_heads=n_heads,
            n_layers=gnn_layers
        )

        # MLP head - match PPO/MATD3 size for fair comparison
        self.head = ft.partial(
            MLP,
            hid_sizes=(64, 64),
            act=nn.relu,
            act_final=True,
            name='QMixAgentHead'
        )

        # RNN for temporal modeling
        if use_rnn:
            self.rnn_base = ft.partial(
                nn.LSTMCell if use_lstm else nn.GRUCell,
                features=64
            )
            self.rnn = ft.partial(
                RNN,
                rnn_cls=self.rnn_base,
                rnn_layers=rnn_layers
            )
        else:
            self.rnn = None

        # Agent Q-network
        self.agent_net = QMixAgentNetwork(
            gnn_cls=self.gnn,
            head_cls=self.head,
            rnn_cls=self.rnn,
            n_actions=self.n_actions,
        )

        # Mixing network
        self.mixer = MixingNetwork(
            embedding_dim=mixer_embed_dim,
            hypernet_hidden_dim=mixer_hypernet_hidden,
        )

    def _create_subgoal_grid(self) -> Array:
        """Create discretized subgoal grid positions

        Returns:
            grid: (n_actions, 2) - grid positions in [0, area_size]^2
        """
        # Create evenly spaced grid
        bin_size = self.area_size / self.n_bins
        grid_1d = jnp.linspace(
            bin_size / 2,  # Center of first bin
            self.area_size - bin_size / 2,  # Center of last bin
            self.n_bins
        )

        # Create 2D grid via meshgrid
        xx, yy = jnp.meshgrid(grid_1d, grid_1d)
        grid = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)  # (n_actions, 2)

        return grid

    def action_to_subgoal(self, actions: Array) -> Array:
        """Convert discrete action indices to continuous subgoal positions

        Args:
            actions: (n_agents,) - discrete action indices in [0, n_actions)

        Returns:
            subgoals: (n_agents, 2) - continuous subgoal positions
        """
        return self.subgoal_grid[actions]

    def subgoal_to_action(self, subgoals: Array) -> Array:
        """Convert continuous subgoal positions to nearest discrete action indices

        Args:
            subgoals: (n_agents, 2) - continuous subgoal positions

        Returns:
            actions: (n_agents,) - discrete action indices
        """
        # Find nearest grid point for each subgoal
        # distances: (n_agents, n_actions)
        distances = jnp.sum(
            (subgoals[:, None, :] - self.subgoal_grid[None, :, :]) ** 2,
            axis=-1
        )
        actions = jnp.argmin(distances, axis=-1)  # (n_agents,)
        return actions

    def initialize_carry(self, key: PRNGKey) -> Array:
        """Initialize RNN hidden state"""
        if self.use_rnn:
            return self.rnn_base().initialize_carry(key, (self.gnn_out_dim,))
        else:
            return jnp.zeros((self.gnn_out_dim,))
