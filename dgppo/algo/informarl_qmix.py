"""InforMARL_QMIX: QMIX for Hierarchical Multi-Agent RL

Implements QMIX (Monotonic Value Function Factorization) as the high-level
algorithm for hierarchical MARL with Manifold/CBF safety controller at low-level.

Key differences from InforMARL_SUB (PPO-based):
- Off-policy learning with replay buffer (handled by trainer)
- Discrete subgoal actions (discretized grid)
- Value decomposition: Q_tot = Mixer(Q_1, ..., Q_n, global_state)
- ε-greedy exploration instead of stochastic policy
- DQN-style target network updates
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
import functools as ft
import os
import pickle

from typing import Optional, Tuple
from flax.training.train_state import TrainState

from .base import Algorithm
from ..utils.typing import Action, Params, PRNGKey, Array
from ..utils.graph import GraphsTuple
from ..utils.utils import tree_index, jax_vmap
from ..trainer.data import Rollout
from ..env.base import MultiAgentEnv
from .module.qnet_qmix import QMixNetwork


class InforMARL_QMIX(Algorithm):
    """QMIX for hierarchical multi-agent RL

    High-level: QMIX outputs discrete subgoal actions
    Low-level: Manifold/CBF safety controller tracks subgoals
    """

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,  # subgoal_dim, typically 2 for (x, y)
            n_agents: int,
            # Hierarchical RL params
            subgoal_interval: int = 40,
            area_size: float = 1.5,
            n_bins: int = 10,  # Discretization: 10x10 grid = 100 actions
            # QMIX params
            gamma: float = 0.99,
            lr: float = 5e-4,
            target_update_interval: int = 200,  # Hard update every N steps
            tau: float = 0.005,  # Soft update coefficient (if using soft update)
            eps_start: float = 1.0,
            eps_end: float = 0.05,
            eps_decay: float = 0.995,  # Exponential decay
            max_grad_norm: float = 10.0,
            # Network params
            use_rnn: bool = True,
            rnn_layers: int = 1,
            gnn_layers: int = 2,
            use_lstm: bool = False,
            mixer_embed_dim: int = 32,
            mixer_hypernet_hidden: int = 64,
            # Misc
            seed: int = 0,
            use_soft_update: bool = True,  # Use soft or hard target updates
            **kwargs
    ):
        super(InforMARL_QMIX, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # Save hyperparameters
        self.subgoal_interval = subgoal_interval
        self.area_size = area_size
        self.n_bins = n_bins
        self.gamma = gamma
        self.lr = lr
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.max_grad_norm = max_grad_norm
        self.use_rnn = use_rnn
        self.rnn_layers = rnn_layers
        self.use_lstm = use_lstm
        self.use_soft_update = use_soft_update
        self.seed = seed

        # Epsilon schedule (exponential decay)
        self.epsilon = eps_start
        self.epsilon_min = eps_end

        # Set nominal graph for initialization
        nominal_graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.nominal_graph = nominal_graph

        # Initialize Q-network and mixer
        self.qnet = QMixNetwork(
            node_dim=node_dim,
            edge_dim=edge_dim,
            n_agents=n_agents,
            area_size=area_size,
            n_bins=n_bins,
            use_rnn=use_rnn,
            rnn_layers=rnn_layers,
            gnn_layers=gnn_layers,
            gnn_out_dim=64,
            use_lstm=use_lstm,
            mixer_embed_dim=mixer_embed_dim,
            mixer_hypernet_hidden=mixer_hypernet_hidden,
        )

        # Initialize RNN states
        key = jr.PRNGKey(seed)
        rnn_state_key, key = jr.split(key)
        rnn_state_keys = jr.split(rnn_state_key, n_agents)
        init_rnn_state = jax_vmap(self.qnet.initialize_carry)(rnn_state_keys)

        if isinstance(init_rnn_state, tuple):
            init_rnn_state = jnp.stack(init_rnn_state, axis=1)
        else:
            init_rnn_state = jnp.expand_dims(init_rnn_state, axis=1)

        # (n_rnn_layers, n_agents, n_carries, rnn_state_dim)
        self.init_rnn_state = init_rnn_state[None, :, :, :].repeat(rnn_layers, axis=0)

        # Initialize Q-network parameters
        q_key, mixer_key, key = jr.split(key, 3)

        # Dummy inputs for initialization
        dummy_dones = jnp.zeros((n_agents,), dtype=bool)
        q_params = self.qnet.agent_net.init(
            q_key,
            nominal_graph,
            self.init_rnn_state,
            dummy_dones,
            n_agents
        )

        # Mixer initialization
        dummy_q_vals = jnp.zeros((n_agents, 1))  # (n_agents, batch_size=1)
        dummy_global_state = nominal_graph.env_states[None, :]  # (1, state_dim)
        mixer_params = self.qnet.mixer.init(
            mixer_key,
            dummy_q_vals,
            dummy_global_state,
            n_agents
        )

        # Combine params
        qmix_params = {
            "q_network": q_params,
            "mixer": mixer_params
        }

        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=lr)
        )

        # Create train states
        self.train_state = TrainState.create(
            apply_fn=None,  # We'll use methods directly
            params=qmix_params,
            tx=optimizer
        )

        # Target network (deep copy)
        self.target_params = jtu.tree_map(lambda x: x.copy(), qmix_params)

        # Update counter for target network
        self.update_counter = 0

        self.key = key

    @property
    def config(self) -> dict:
        return {
            'subgoal_interval': self.subgoal_interval,
            'area_size': self.area_size,
            'n_bins': self.n_bins,
            'gamma': self.gamma,
            'lr': self.lr,
            'target_update_interval': self.target_update_interval,
            'tau': self.tau,
            'eps_start': self.eps_start,
            'eps_end': self.eps_end,
            'eps_decay': self.eps_decay,
            'max_grad_norm': self.max_grad_norm,
            'use_rnn': self.use_rnn,
            'rnn_layers': self.rnn_layers,
            'use_lstm': self.use_lstm,
            'use_soft_update': self.use_soft_update,
            'seed': self.seed,
        }

    @property
    def params(self) -> Params:
        return self.train_state.params

    def get_q_values(
            self,
            params: Params,
            graph: GraphsTuple,
            rnn_state: Array,
            dones: Array,
    ) -> Tuple[Array, Array]:
        """Get Q-values for all discrete actions

        Args:
            params: Q-network parameters
            graph: Current graph observation
            rnn_state: RNN hidden state
            dones: Episode done flags

        Returns:
            q_vals: (n_agents, n_actions) - Q-values for each action
            rnn_state: Updated RNN state
        """
        q_vals, rnn_state = self.qnet.agent_net.apply(
            params["q_network"],
            graph,
            rnn_state,
            dones,
            self.n_agents
        )
        return q_vals, rnn_state

    def get_greedy_actions(
            self,
            q_vals: Array,
    ) -> Array:
        """Select greedy actions from Q-values

        Args:
            q_vals: (n_agents, n_actions) - Q-values

        Returns:
            actions: (n_agents,) - Discrete action indices
        """
        return jnp.argmax(q_vals, axis=-1)

    def epsilon_greedy(
            self,
            key: PRNGKey,
            q_vals: Array,
            epsilon: float,
    ) -> Array:
        """ε-greedy action selection

        Args:
            key: Random key
            q_vals: (n_agents, n_actions) - Q-values
            epsilon: Exploration probability

        Returns:
            actions: (n_agents,) - Selected action indices
        """
        keys = jr.split(key, self.n_agents)
        greedy_actions = self.get_greedy_actions(q_vals)

        def select_action(agent_key, agent_q_vals, greedy_action):
            rand_key, choice_key = jr.split(agent_key)
            # Explore: random action
            random_action = jr.randint(rand_key, (), 0, self.qnet.n_actions)
            # Exploit: greedy action
            return jnp.where(
                jr.uniform(choice_key) < epsilon,
                random_action,
                greedy_action
            )

        actions = jax.vmap(select_action)(keys, q_vals, greedy_actions)
        return actions

    def act(
            self,
            graph: GraphsTuple,
            rnn_state: Array,
            params: Optional[Params] = None,
    ) -> Tuple[Action, Array]:
        """Greedy action selection (for evaluation)

        Returns:
            subgoals: (n_agents, 2) - Continuous subgoal positions
            rnn_state: Updated RNN state
        """
        if params is None:
            params = self.params

        dones = jnp.zeros((self.n_agents,), dtype=bool)
        q_vals, rnn_state = self.get_q_values(params, graph, rnn_state, dones)
        action_indices = self.get_greedy_actions(q_vals)
        subgoals = self.qnet.action_to_subgoal(action_indices)

        return subgoals, rnn_state

    def step(
            self,
            graph: GraphsTuple,
            rnn_state: Array,
            key: PRNGKey,
            params: Optional[Params] = None,
            add_noise: bool = True,
    ) -> Tuple[Action, Array, Array]:
        """ε-greedy action selection (for training)

        Returns:
            subgoals: (n_agents, 2) - Continuous subgoal positions
            action_indices: (n_agents,) - Discrete action indices (for logging)
            rnn_state: Updated RNN state
        """
        if params is None:
            params = self.params

        dones = jnp.zeros((self.n_agents,), dtype=bool)
        q_vals, rnn_state = self.get_q_values(params, graph, rnn_state, dones)

        if add_noise:
            action_indices = self.epsilon_greedy(key, q_vals, self.epsilon)
        else:
            action_indices = self.get_greedy_actions(q_vals)

        subgoals = self.qnet.action_to_subgoal(action_indices)

        # Return action_indices as dummy log_pi (not used in QMIX)
        return subgoals, action_indices.astype(jnp.float32), rnn_state

    def collect(self, params: Params, b_key: PRNGKey, step: int = 0) -> Rollout:
        """Collect rollouts - handled by trainer with replay buffer

        This method exists to satisfy the Algorithm interface but is not used directly.
        Data collection is handled by the trainer (TrainerQMixManifold).
        """
        raise NotImplementedError(
            "QMIX uses off-policy learning with replay buffer. "
            "Data collection is handled by TrainerQMixManifold.collect_rollouts()."
        )

    def compute_td_targets(
            self,
            rollout: Rollout,
    ) -> Array:
        """Compute TD targets for QMIX update

        Uses double Q-learning: actions selected by online Q, evaluated by target Q.

        Args:
            rollout: (b, T) - Batch of transitions

        Returns:
            targets: (b, T) - TD targets for Q_tot
        """
        b, T = rollout.dones.shape

        # Get Q-values from target network for next states
        # Shape: (b, T, n_agents, n_actions)
        def get_next_q(graph, rnn_state):
            dones = jnp.zeros((self.n_agents,), dtype=bool)
            q_vals, _ = self.get_q_values(
                self.target_params,
                graph,
                rnn_state,
                dones
            )
            return q_vals

        # Scan over time to get next Q-values
        def scan_next_q(rnn_init, graphs):
            def body(rnn, graph):
                q_vals, new_rnn = get_next_q(graph, rnn)
                return new_rnn, q_vals
            _, q_vals_seq = jax.lax.scan(body, rnn_init, graphs)
            return q_vals_seq

        bT_next_q_vals = jax.vmap(scan_next_q)(
            self.init_rnn_state.repeat(b, axis=1),
            rollout.next_graph
        )  # (b, T, n_agents, n_actions)

        # Select greedy actions
        bTa_next_actions = jax.vmap(jax.vmap(self.get_greedy_actions))(
            bT_next_q_vals
        )  # (b, T, n_agents)

        # Get chosen Q-values
        bTa_next_q = jnp.take_along_axis(
            bT_next_q_vals,
            bTa_next_actions[..., None],
            axis=-1
        ).squeeze(-1)  # (b, T, n_agents)

        # Mix next Q-values with target mixer
        def mix_q(q_vals, global_state):
            # q_vals: (n_agents,), global_state: (state_dim,)
            return self.qnet.mixer.apply(
                self.target_params["mixer"],
                q_vals[:, None],  # (n_agents, 1)
                global_state[None, :],  # (1, state_dim)
                self.n_agents
            ).squeeze()

        bT_next_q_tot = jax.vmap(jax.vmap(mix_q))(
            bTa_next_q,
            rollout.next_graph.env_states
        )  # (b, T)

        # Compute TD targets: r + γ * (1 - done) * Q_next
        bT_targets = (
            rollout.sparse_rewards +
            self.gamma * (1 - rollout.dones.astype(jnp.float32)) * bT_next_q_tot
        )

        return bT_targets

    @ft.partial(jax.jit, static_argnums=(0,), donate_argnames=("train_state",))
    def update(
            self,
            train_state: TrainState,
            rollout: Rollout,
    ) -> Tuple[TrainState, dict]:
        """Update Q-network and mixer using sampled batch

        Args:
            train_state: Current training state
            rollout: (b, T) - Batch sampled from replay buffer

        Returns:
            train_state: Updated training state
            info: Training metrics
        """
        b, T = rollout.dones.shape

        # Compute TD targets
        bT_targets = self.compute_td_targets(rollout)

        def loss_fn(params):
            # Get current Q-values
            def get_q_and_mix(graph, actions, global_state, rnn_state):
                # Get Q-values for all actions
                dones = jnp.zeros((self.n_agents,), dtype=bool)
                q_vals, new_rnn = self.get_q_values(
                    params,
                    graph,
                    rnn_state,
                    dones
                )  # (n_agents, n_actions)

                # Select Q-values for chosen actions
                chosen_q = jnp.take_along_axis(
                    q_vals,
                    actions.astype(jnp.int32)[:, None],
                    axis=-1
                ).squeeze(-1)  # (n_agents,)

                # Mix Q-values
                q_tot = self.qnet.mixer.apply(
                    params["mixer"],
                    chosen_q[:, None],  # (n_agents, 1)
                    global_state[None, :],  # (1, state_dim)
                    self.n_agents
                ).squeeze()

                return q_tot, new_rnn

            # Scan over time
            def scan_q(rnn_init, data):
                graphs, actions, global_states = data
                def body(rnn, inp):
                    g, a, s = inp
                    q_tot, new_rnn = get_q_and_mix(g, a, s, rnn)
                    return new_rnn, q_tot
                _, q_tots = jax.lax.scan(body, rnn_init, (graphs, actions, global_states))
                return q_tots

            # Vmap over batch
            bT_q_tot = jax.vmap(scan_q)(
                self.init_rnn_state.repeat(b, axis=1),
                (rollout.graph, rollout.actions, rollout.graph.env_states)
            )  # (b, T)

            # TD error loss
            td_error = bT_q_tot - jax.lax.stop_gradient(bT_targets)
            loss = jnp.mean(td_error ** 2)

            return loss, {
                'q_tot_mean': jnp.mean(bT_q_tot),
                'target_mean': jnp.mean(bT_targets),
                'td_error_abs_mean': jnp.mean(jnp.abs(td_error)),
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            train_state.params
        )

        # Apply gradients
        train_state = train_state.apply_gradients(grads=grads)

        # Compute gradient norm
        grad_norm = optax.global_norm(grads)

        info = {
            'qmix/loss': loss,
            'qmix/grad_norm': grad_norm,
            'qmix/epsilon': self.epsilon,
            **{f'qmix/{k}': v for k, v in metrics.items()}
        }

        return train_state, info

    def update_target_network(self):
        """Update target network parameters"""
        if self.use_soft_update:
            # Soft update: θ_target ← τ*θ + (1-τ)*θ_target
            self.target_params = jtu.tree_map(
                lambda target, online: self.tau * online + (1 - self.tau) * target,
                self.target_params,
                self.train_state.params
            )
        else:
            # Hard update: θ_target ← θ
            self.target_params = jtu.tree_map(
                lambda x: x.copy(),
                self.train_state.params
            )

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.eps_decay)

    def save(self, save_dir: str, step: int):
        """Save model parameters"""
        model_dir = os.path.join(save_dir, str(step))
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, 'qmix.pkl'), 'wb') as f:
            pickle.dump(self.train_state.params, f)

        with open(os.path.join(model_dir, 'target.pkl'), 'wb') as f:
            pickle.dump(self.target_params, f)

    def load(self, load_dir: str, step: int):
        """Load model parameters"""
        path = os.path.join(load_dir, str(step))

        with open(os.path.join(path, 'qmix.pkl'), 'rb') as f:
            params = pickle.load(f)
            self.train_state = self.train_state.replace(params=params)

        with open(os.path.join(path, 'target.pkl'), 'rb') as f:
            self.target_params = pickle.load(f)
