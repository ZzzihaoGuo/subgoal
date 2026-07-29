import jax.numpy as jnp
import jax.random as jr
import optax
import os
import jax
import functools as ft
import jax.tree_util as jtu
import numpy as np
import pickle

from typing import Optional, Tuple
from flax.training.train_state import TrainState
from jax import lax

from .base import Algorithm
from ..utils.typing import Action, Params, PRNGKey, Array
from ..utils.graph import GraphsTuple
from ..utils.utils import tree_index, jax_vmap
from ..trainer.data import Rollout
from ..trainer.utils import has_any_nan_or_inf, compute_norm_and_clip
from ..env.base import MultiAgentEnv
from ..algo.module.qnet import QNetwork
from ..algo.module.policy import SubgoalPolicy


class InforMARL_MATD3(Algorithm):
    """Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3)

    Hierarchical RL with:
    - High-level: MATD3 (off-policy) for subgoal generation
    - Low-level: Fixed LQR + CBF/Manifold safety controller

    Key features:
    - Deterministic actor for subgoal targets
    - Double Q-learning with two critic networks
    - Target policy smoothing for robustness
    - Delayed policy updates (update actor every policy_delay steps)
    - Replay buffer for off-policy learning
    """

    def __init__(
            self,
            env: MultiAgentEnv,
            node_dim: int,
            edge_dim: int,
            state_dim: int,
            action_dim: int,
            n_agents: int,
            subgoal_interval: int = 40,
            area_size: float = 1.5,

            # TD3 hyperparameters
            gamma: float = 0.99,
            tau: float = 0.005,  # Soft update coefficient for target networks
            policy_delay: int = 2,  # Delay policy update by this many critic updates
            target_noise: float = 0.2,  # Noise added to target policy
            noise_clip: float = 0.5,  # Clip target noise
            exploration_noise: float = 0.1,  # Exploration noise during training

            # Network architecture
            actor_gnn_layers: int = 2,
            critic_gnn_layers: int = 2,

            # Optimization
            lr_actor: float = 1e-4,
            lr_critic: float = 3e-4,
            batch_size: int = 256,
            max_grad_norm: float = 2.0,

            # RNN settings
            use_rnn: bool = True,
            rnn_layers: int = 1,
            use_lstm: bool = False,

            # Subgoal settings
            use_relative_subgoal: bool = False,
            max_delta: float = None,

            seed: int = 0,
            **kwargs
    ):
        super(InforMARL_MATD3, self).__init__(
            env=env,
            node_dim=node_dim,
            edge_dim=edge_dim,
            action_dim=action_dim,
            n_agents=n_agents
        )

        # Save hyperparameters
        self.subgoal_interval = subgoal_interval
        self.area_size = area_size
        self.use_relative_subgoal = use_relative_subgoal
        self.max_delta = max_delta

        # TD3 hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise

        # Network config
        self.actor_gnn_layers = actor_gnn_layers
        self.critic_gnn_layers = critic_gnn_layers

        # Optimization
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # RNN
        self.use_rnn = use_rnn
        self.rnn_layers = rnn_layers
        self.use_lstm = use_lstm

        self.seed = seed

        # Set up nominal graph for network initialization
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

        # Initialize PRNG key
        key = jr.PRNGKey(seed)

        # ===== Actor (Deterministic Policy) =====
        self.actor = SubgoalPolicy(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            subgoal_dim=self.action_dim,
            area_size=self.area_size,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.actor_gnn_layers,
            gnn_out_dim=64,
            use_lstm=self.use_lstm,
            use_relative_subgoal=self.use_relative_subgoal,
            max_delta=self.max_delta
        )

        # Initialize actor RNN state
        rnn_state_key, key = jr.split(key)
        rnn_state_key = jr.split(rnn_state_key, self.n_agents)
        init_rnn_state = jax_vmap(self.actor.initialize_carry)(rnn_state_key)
        if type(init_rnn_state) is tuple:
            init_rnn_state = jnp.stack(init_rnn_state, axis=1)
        else:
            init_rnn_state = jnp.expand_dims(init_rnn_state, axis=1)
        self.init_actor_rnn_state = init_rnn_state[None, :, :, :].repeat(self.rnn_layers, axis=0)

        # Initialize actor parameters
        actor_key, key = jr.split(key)
        actor_params = self.actor.dist.init(
            actor_key, nominal_graph, self.init_actor_rnn_state, self.n_agents
        )
        actor_optim = optax.adam(learning_rate=lr_actor)
        self.actor_optim = optax.apply_if_finite(actor_optim, 1_000_000)
        self.actor_train_state = TrainState.create(
            apply_fn=self.actor.get_action,
            params=actor_params,
            tx=self.actor_optim
        )

        # Actor target network (frozen copy)
        self.actor_target_params = actor_params

        # ===== Critic Networks (Q1 and Q2) =====
        self.Q1 = QNetwork(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.critic_gnn_layers,
            gnn_out_dim=64,
            use_lstm=self.use_lstm,
            decompose=False,  # Centralized Q-value
            n_heads=3
        )

        self.Q2 = QNetwork(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            n_agents=self.n_agents,
            action_dim=self.action_dim,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            gnn_layers=self.critic_gnn_layers,
            gnn_out_dim=64,
            use_lstm=self.use_lstm,
            decompose=False,
            n_heads=3
        )

        # Initialize critic RNN states
        Q_rnn_state_key, key = jr.split(key)
        init_Q_rnn_state = self.Q1.initialize_carry(Q_rnn_state_key)
        if type(init_Q_rnn_state) is tuple:
            init_Q_rnn_state = jnp.stack(init_Q_rnn_state, axis=0)
        else:
            init_Q_rnn_state = init_Q_rnn_state[None, :]
        self.init_Q_rnn_state = init_Q_rnn_state[None, :, :].repeat(self.rnn_layers, axis=0)[:, None, :, :]

        # Initialize Q1
        Q1_key, key = jr.split(key)
        nominal_action = jnp.zeros((n_agents, action_dim))
        Q1_params = self.Q1.net.init(Q1_key, nominal_graph, nominal_action, self.init_Q_rnn_state, self.n_agents)
        Q1_optim = optax.adam(learning_rate=lr_critic)
        self.Q1_optim = optax.apply_if_finite(Q1_optim, 1_000_000)
        self.Q1_train_state = TrainState.create(
            apply_fn=self.Q1.get_q_value,
            params=Q1_params,
            tx=self.Q1_optim
        )

        # Initialize Q2
        Q2_key, key = jr.split(key)
        Q2_params = self.Q2.net.init(Q2_key, nominal_graph, nominal_action, self.init_Q_rnn_state, self.n_agents)
        Q2_optim = optax.adam(learning_rate=lr_critic)
        self.Q2_optim = optax.apply_if_finite(Q2_optim, 1_000_000)
        self.Q2_train_state = TrainState.create(
            apply_fn=self.Q2.get_q_value,
            params=Q2_params,
            tx=self.Q2_optim
        )

        # Critic target networks (frozen copies)
        self.Q1_target_params = Q1_params
        self.Q2_target_params = Q2_params

        # Set up key
        self.key = key

        # Update counter for delayed policy updates
        self.update_step = 0

    @property
    def config(self) -> dict:
        return {
            'subgoal_interval': self.subgoal_interval,
            'area_size': self.area_size,
            'gamma': self.gamma,
            'tau': self.tau,
            'policy_delay': self.policy_delay,
            'target_noise': self.target_noise,
            'noise_clip': self.noise_clip,
            'exploration_noise': self.exploration_noise,
            'actor_gnn_layers': self.actor_gnn_layers,
            'critic_gnn_layers': self.critic_gnn_layers,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'batch_size': self.batch_size,
            'max_grad_norm': self.max_grad_norm,
            'seed': self.seed,
            'use_rnn': self.use_rnn,
            'rnn_layers': self.rnn_layers,
            'use_lstm': self.use_lstm,
            'use_relative_subgoal': self.use_relative_subgoal,
            'max_delta': self.max_delta
        }

    @property
    def params(self) -> Params:
        return {
            "actor": self.actor_train_state.params,
            "actor_target": self.actor_target_params,
            "Q1": self.Q1_train_state.params,
            "Q2": self.Q2_train_state.params,
            "Q1_target": self.Q1_target_params,
            "Q2_target": self.Q2_target_params,
        }

    def act(
            self,
            graph: GraphsTuple,
            rnn_state: Array,
            params: Optional[Params] = None,
    ) -> Tuple[Action, Array]:
        """Deterministic action (for evaluation)"""
        if params is None:
            params = self.params
        action, rnn_state = self.actor.get_action(params["actor"], graph, rnn_state)
        return action, rnn_state

    def step(
            self,
            graph: GraphsTuple,
            rnn_state: Array,
            key: PRNGKey,
            params: Optional[Params] = None,
            add_noise: bool = True,
    ) -> Tuple[Action, Array, Array]:
        """Action with exploration noise (for training)"""
        if params is None:
            params = self.params

        # Get deterministic action
        action, rnn_state = self.actor.get_action(params["actor"], graph, rnn_state)

        # Add exploration noise
        if add_noise:
            noise = jr.normal(key, action.shape) * self.exploration_noise
            # Clip action to valid range [0, area_size]
            action = jnp.clip(action + noise, 0, self.area_size)

        # For off-policy, we don't compute log_pi during collection
        log_pi = jnp.zeros((self.n_agents,))

        return action, log_pi, rnn_state

    def collect(self, params: Params, key: PRNGKey, step: int = 0) -> Rollout:
        """Collect rollouts - Note: actual collection is done in TrainerMATD3

        This method exists to satisfy the Algorithm interface, but MATD3
        uses off-policy learning where collection is handled by the trainer
        with a replay buffer. This method should not be called directly.

        Args:
            params: Network parameters
            key: Random key
            step: Training step (unused)

        Returns:
            Empty rollout (placeholder)

        Raises:
            NotImplementedError: This method should not be called for MATD3
        """
        raise NotImplementedError(
            "MATD3 uses off-policy learning with replay buffer. "
            "Data collection is handled by TrainerMATD3.collect_rollouts(). "
            "Do not call algo.collect() directly."
        )

    def soft_update(self, tau: float) -> None:
        """Soft update target networks: θ_target = τ*θ + (1-τ)*θ_target"""
        self.actor_target_params = jtu.tree_map(
            lambda x, y: tau * x + (1 - tau) * y,
            self.actor_train_state.params,
            self.actor_target_params
        )
        self.Q1_target_params = jtu.tree_map(
            lambda x, y: tau * x + (1 - tau) * y,
            self.Q1_train_state.params,
            self.Q1_target_params
        )
        self.Q2_target_params = jtu.tree_map(
            lambda x, y: tau * x + (1 - tau) * y,
            self.Q2_train_state.params,
            self.Q2_target_params
        )

    def update(self, rollout: Rollout, step: int) -> dict:
        """Update networks using sampled batch from replay buffer

        This is called after sampling a minibatch from the replay buffer.

        Args:
            rollout: Sampled batch from replay buffer (batch_size, T, ...)
            step: Current training step

        Returns:
            Dictionary of training metrics
        """
        key, self.key = jr.split(self.key)

        # Remove env_state from rollout
        graph_clean = rollout.graph._replace(env_states=None)
        next_graph_clean = rollout.next_graph._replace(env_states=None)
        rollout = rollout._replace(graph=graph_clean, next_graph=next_graph_clean)

        # Update critics
        Q1_train_state, Q2_train_state, critic_info = self.update_critics(
            self.Q1_train_state,
            self.Q2_train_state,
            rollout,
            key
        )
        self.Q1_train_state = Q1_train_state
        self.Q2_train_state = Q2_train_state

        update_info = critic_info

        # Delayed policy update
        self.update_step += 1
        if self.update_step % self.policy_delay == 0:
            # Update actor
            actor_train_state, actor_info = self.update_actor(
                self.actor_train_state,
                self.Q1_train_state,
                rollout
            )
            self.actor_train_state = actor_train_state
            update_info.update(actor_info)

            # Soft update target networks
            self.soft_update(self.tau)

        return update_info

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_critics(
            self,
            Q1_train_state: TrainState,
            Q2_train_state: TrainState,
            rollout: Rollout,
            key: PRNGKey
    ) -> Tuple[TrainState, TrainState, dict]:
        """Update both critic networks using TD3 loss

        TD3 uses double Q-learning with clipped target:
        target = r + γ * min(Q1_target(s', a'), Q2_target(s', a'))
        where a' = actor_target(s') + clipped_noise
        """
        b, T, a, _ = rollout.actions.shape

        def compute_targets(next_graph, reward, done):
            """Compute TD target with target policy smoothing"""
            # Get next action from target actor
            next_action, _ = self.actor.get_action(
                self.actor_target_params, next_graph, self.init_actor_rnn_state
            )

            # Add clipped noise to target action (target policy smoothing)
            noise = jr.normal(key, next_action.shape) * self.target_noise
            noise = jnp.clip(noise, -self.noise_clip, self.noise_clip)
            next_action = jnp.clip(next_action + noise, 0, self.area_size)

            # Compute target Q-values using target networks
            q1_target, _ = self.Q1.get_q_value(
                self.Q1_target_params, next_graph, next_action, self.init_Q_rnn_state
            )
            q2_target, _ = self.Q2.get_q_value(
                self.Q2_target_params, next_graph, next_action, self.init_Q_rnn_state
            )

            # Take minimum (clipped double Q-learning)
            q_target = jnp.minimum(q1_target, q2_target).squeeze()  # (1,) -> scalar

            # TD target: r + γ * (1 - done) * Q_target(s', a')
            target = reward + self.gamma * (1.0 - done) * q_target

            return target

        # Compute targets for all transitions
        bT_targets = jax.vmap(jax.vmap(compute_targets))(
            rollout.next_graph, rollout.sparse_rewards, rollout.dones[:, :, 0]
        )  # (b, T)

        def critic_loss_fn(Q_params, Q_net):
            """Compute MSE loss for one critic"""
            def compute_q(graph, action):
                q, _ = Q_net.get_q_value(Q_params, graph, action, self.init_Q_rnn_state)
                return q.squeeze()

            bT_q_values = jax.vmap(jax.vmap(compute_q))(rollout.graph, rollout.actions)
            loss = optax.l2_loss(bT_q_values, bT_targets).mean()
            return loss

        # Update Q1
        loss_Q1, grad_Q1 = jax.value_and_grad(lambda p: critic_loss_fn(p, self.Q1))(Q1_train_state.params)
        Q1_has_nan = has_any_nan_or_inf(grad_Q1).astype(jnp.float32)
        grad_Q1, grad_norm_Q1 = compute_norm_and_clip(grad_Q1, self.max_grad_norm)
        Q1_train_state = Q1_train_state.apply_gradients(grads=grad_Q1)

        # Update Q2
        loss_Q2, grad_Q2 = jax.value_and_grad(lambda p: critic_loss_fn(p, self.Q2))(Q2_train_state.params)
        Q2_has_nan = has_any_nan_or_inf(grad_Q2).astype(jnp.float32)
        grad_Q2, grad_norm_Q2 = compute_norm_and_clip(grad_Q2, self.max_grad_norm)
        Q2_train_state = Q2_train_state.apply_gradients(grads=grad_Q2)

        info = {
            'critic/Q1_loss': loss_Q1,
            'critic/Q2_loss': loss_Q2,
            'critic/Q1_grad_norm': grad_norm_Q1,
            'critic/Q2_grad_norm': grad_norm_Q2,
            'critic/Q1_has_nan': Q1_has_nan,
            'critic/Q2_has_nan': Q2_has_nan,
            'critic/target_mean': bT_targets.mean(),
            'critic/target_std': bT_targets.std(),
        }

        return Q1_train_state, Q2_train_state, info

    @ft.partial(jax.jit, static_argnums=(0,))
    def update_actor(
            self,
            actor_train_state: TrainState,
            Q1_train_state: TrainState,
            rollout: Rollout
    ) -> Tuple[TrainState, dict]:
        """Update actor by maximizing Q1(s, actor(s))

        Actor loss: -mean(Q1(s, actor(s)))
        """
        def actor_loss_fn(actor_params):
            """Compute policy gradient loss"""
            def compute_q_for_policy(graph):
                # Get action from current actor
                action, _ = self.actor.get_action(actor_params, graph, self.init_actor_rnn_state)
                # Evaluate Q1
                q, _ = self.Q1.get_q_value(Q1_train_state.params, graph, action, self.init_Q_rnn_state)
                return q.squeeze()

            bT_q_values = jax.vmap(jax.vmap(compute_q_for_policy))(rollout.graph)
            # Maximize Q => minimize -Q
            loss = -bT_q_values.mean()
            return loss

        loss, grad = jax.value_and_grad(actor_loss_fn)(actor_train_state.params)
        actor_has_nan = has_any_nan_or_inf(grad).astype(jnp.float32)
        grad, grad_norm = compute_norm_and_clip(grad, self.max_grad_norm)
        actor_train_state = actor_train_state.apply_gradients(grads=grad)

        info = {
            'actor/loss': loss,
            'actor/grad_norm': grad_norm,
            'actor/has_nan': actor_has_nan,
        }

        return actor_train_state, info

    def save(self, save_dir: str, step: int):
        """Save all network parameters"""
        model_dir = os.path.join(save_dir, str(step))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        pickle.dump(self.actor_train_state.params, open(os.path.join(model_dir, 'actor.pkl'), 'wb'))
        pickle.dump(self.actor_target_params, open(os.path.join(model_dir, 'actor_target.pkl'), 'wb'))
        pickle.dump(self.Q1_train_state.params, open(os.path.join(model_dir, 'Q1.pkl'), 'wb'))
        pickle.dump(self.Q2_train_state.params, open(os.path.join(model_dir, 'Q2.pkl'), 'wb'))
        pickle.dump(self.Q1_target_params, open(os.path.join(model_dir, 'Q1_target.pkl'), 'wb'))
        pickle.dump(self.Q2_target_params, open(os.path.join(model_dir, 'Q2_target.pkl'), 'wb'))

    def load(self, load_dir: str, step: int):
        """Load all network parameters"""
        path = os.path.join(load_dir, str(step))

        self.actor_train_state = self.actor_train_state.replace(
            params=pickle.load(open(os.path.join(path, 'actor.pkl'), 'rb'))
        )
        self.actor_target_params = pickle.load(open(os.path.join(path, 'actor_target.pkl'), 'rb'))
        self.Q1_train_state = self.Q1_train_state.replace(
            params=pickle.load(open(os.path.join(path, 'Q1.pkl'), 'rb'))
        )
        self.Q2_train_state = self.Q2_train_state.replace(
            params=pickle.load(open(os.path.join(path, 'Q2.pkl'), 'rb'))
        )
        self.Q1_target_params = pickle.load(open(os.path.join(path, 'Q1_target.pkl'), 'rb'))
        self.Q2_target_params = pickle.load(open(os.path.join(path, 'Q2_target.pkl'), 'rb'))
