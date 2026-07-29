import argparse
import datetime
import os
import numpy as np
import wandb
import yaml
import jax
import jax.random as jr
import time

# Suppress jaxproxqp debug output
from loguru import logger
logger.disable("jaxproxqp")

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.trainer_matd3 import TrainerMATD3
from dgppo.trainer.utils import is_connected


def train(args):
    print(f"> Running train_matd3.py {args}")

    # Set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if not is_connected():
        os.environ["WANDB_MODE"] = "offline"
    np.random.seed(args.seed)
    if args.debug:
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["JAX_DISABLE_JIT"] = "True"

    # Create environments
    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
        max_step=args.max_step,
        cbf_alpha=args.cbf_std_alpha2,
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
        max_step=args.max_step,
        cbf_alpha=args.cbf_std_alpha2,
    )

    # Create MATD3 algorithm
    algo = make_algo(
        algo='informarl_matd3',
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        area_size=env.area_size,

        # Hierarchical RL parameters
        subgoal_interval=args.subgoal_interval,
        use_relative_subgoal=args.relative_subgoal,
        max_delta=args.max_delta,

        # TD3 hyperparameters
        gamma=args.gamma,
        tau=args.tau,
        policy_delay=args.policy_delay,
        target_noise=args.target_noise,
        noise_clip=args.noise_clip,
        exploration_noise=args.exploration_noise,

        # Network architecture
        actor_gnn_layers=args.actor_gnn_layers,
        critic_gnn_layers=args.critic_gnn_layers,

        # Optimization
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,

        # RNN settings
        use_rnn=not args.no_rnn,
        rnn_layers=args.rnn_layers,
        use_lstm=args.use_lstm,

        seed=args.seed,
    )

    # Generate a 4 letter random identifier for the run
    rng_ = np.random.default_rng()
    rand_id = "".join([chr(rng_.integers(65, 91)) for _ in range(4)])

    # Set up logger
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%m%d%H%M%S")
    if not args.debug:
        if not os.path.exists(f"{args.log_dir}/{args.env}/matd3"):
            os.makedirs(f"{args.log_dir}/{args.env}/matd3", exist_ok=True)
    start_time = int(start_time)
    while os.path.exists(f"{args.log_dir}/{args.env}/matd3/seed{args.seed}_{start_time}_{rand_id}"):
        start_time += 1

    log_dir = f"{args.log_dir}/{args.env}/matd3/seed{args.seed}_{start_time}_{rand_id}"
    run_name = "matd3_seed{:03}_{}_{}".format(args.seed, start_time, rand_id)
    if args.name is not None:
        run_name = "{}_{}_seed{:03}_{}_{}".format(run_name, args.name, args.seed, start_time, rand_id)

    # Get training parameters
    train_params = {
        "run_name": run_name,
        "training_steps": args.steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval
    }

    # Create MATD3 trainer
    trainer = TrainerMATD3(
        env=env,
        env_test=env_test,
        algo=algo,
        gamma=args.gamma,
        log_dir=log_dir,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        seed=args.seed,
        params=train_params,
        save_log=not args.debug,
        # MATD3-specific
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        updates_per_step=args.updates_per_step,
    )

    # Save config
    wandb.config.update(args)
    wandb.config.update(algo.config, allow_val_change=True)
    if not args.debug:
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(vars(args), f)
            yaml.dump(algo.config, f)

    # ========== Warmup: Pre-compile CBF and JIT functions ==========
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    # Initialize CBF function and JIT compile
    env.init_cbf(
        use_closed_form=args.use_cbf_closed_form,
        use_paper_cbf=args.use_paper_cbf,
        cbf_alpha1=args.cbf_alpha1,
        cbf_alpha2=args.cbf_alpha2,
        cbf_alpha=args.cbf_std_alpha1,
    )
    env_test.init_cbf(
        use_closed_form=args.use_cbf_closed_form,
        use_paper_cbf=args.use_paper_cbf,
        cbf_alpha1=args.cbf_alpha1,
        cbf_alpha2=args.cbf_alpha2,
        cbf_alpha=args.cbf_std_alpha1,
    )

    # Warm up safe_u_ref (trigger first JIT compilation)
    print("Warming up CBF controller (first JIT compile)...")
    warmup_key = jr.PRNGKey(42)
    warmup_graph = env.reset(warmup_key)
    target_pos = warmup_graph.type_states(type_idx=1, n_type=env.num_agents)[:, :2]

    start = time.time()
    action = env.safe_u_ref(warmup_graph, target_pos=target_pos, is_final_goal=False)
    jax.block_until_ready(action)
    print(f"First compile: {time.time() - start:.2f}s")

    start = time.time()
    action = env.safe_u_ref(warmup_graph, target_pos=target_pos, is_final_goal=False)
    jax.block_until_ready(action)
    print(f"Cached call: {time.time() - start:.4f}s")
    print("CBF warmup complete!")
    # ===============================================================

    # Start training
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train MATD3 for hierarchical multi-agent RL")

    # Required arguments
    parser.add_argument("--env", type=str, default="LidarSpread", help="Environment name")
    parser.add_argument("-n", "--num-agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--obs", type=int, default=3, help="Number of obstacles")

    # Training arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--name", type=str, default=None, help="Custom run name suffix")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode (no logging, no JIT)")

    # MATD3-specific hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient for target networks")
    parser.add_argument("--policy-delay", type=int, default=2, help="Delay policy updates by this many critic updates")
    parser.add_argument("--target-noise", type=float, default=0.2, help="Noise added to target policy")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="Clip target noise")
    parser.add_argument("--exploration-noise", type=float, default=0.1, help="Exploration noise during training")

    # Replay buffer
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--min-buffer-size", type=int, default=1000, help="Start training after this many transitions")
    parser.add_argument("--updates-per-step", type=int, default=1, help="Number of gradient updates per environment step")

    # Environment arguments
    parser.add_argument("--n-rays", type=int, default=32, help="Number of LiDAR rays")
    parser.add_argument("--full-observation", action="store_true", default=False, help="Use full observability")
    parser.add_argument("--max-step", type=int, default=128, help="Max timesteps per episode")
    parser.add_argument("--cbf-std-alpha2", type=float, default=50.0, help="Standard CBF parameter α₂")

    # CBF solver arguments
    parser.add_argument("--use-paper-cbf", action="store_true", default=True,
                        help="Use paper's relative-degree-2 CBF")
    parser.add_argument("--cbf-alpha1", type=float, default=40.0, help="CBF parameter α₁ for paper CBF")
    parser.add_argument("--cbf-alpha2", type=float, default=30.0, help="CBF parameter α₂ for paper CBF")
    parser.add_argument("--use-cbf-closed-form", action="store_true", default=True,
                        help="Use closed-form CBF solver (faster)")
    parser.add_argument("--cbf-std-alpha1", type=float, default=10.0, help="Standard CBF parameter α₁")

    # Subgoal mode arguments
    parser.add_argument("--subgoal-interval", type=int, default=8,
                        help="Hierarchical RL: steps between subgoal generation")
    parser.add_argument("--relative-subgoal", action="store_true", default=True,
                        help="Use relative subgoal instead of absolute coordinates")
    parser.add_argument("--max-delta", type=float, default=0.2,
                        help="Max delta for relative subgoal mode")

    # Network architecture
    parser.add_argument("--actor-gnn-layers", type=int, default=2, help="Number of GNN layers in actor")
    parser.add_argument("--critic-gnn-layers", type=int, default=2, help="Number of GNN layers in critic")
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--lr-critic", type=float, default=3e-4, help="Critic learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=2.0, help="Max gradient norm for clipping")
    parser.add_argument("--rnn-layers", type=int, default=1, help="Number of RNN layers")
    parser.add_argument("--use-lstm", action="store_true", default=False, help="Use LSTM instead of GRU")
    parser.add_argument("--no-rnn", action="store_true", default=False, help="Disable RNN")

    # Training settings
    parser.add_argument("--n-env-train", type=int, default=128, help="Number of parallel training environments")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for updates")
    parser.add_argument("--n-env-test", type=int, default=32, help="Number of parallel test environments")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--eval-epi", type=int, default=1, help="Number of evaluation episodes")
    parser.add_argument("--save-interval", type=int, default=1000, help="Model save interval")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
