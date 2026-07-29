"""Training script for QMIX with Manifold safety controller

Usage:
    python train_qmix_manifold.py --env LidarTarget --algo informarl_qmix -n 3 --obs 3
"""

import argparse
import datetime
import os
import numpy as np
import wandb
import yaml

# Suppress jaxproxqp debug output
from loguru import logger
logger.disable("jaxproxqp")

from dgppo.algo import make_algo
from dgppo.env import make_env
from dgppo.trainer.trainer_qmix_manifold import TrainerQMixManifold
from dgppo.trainer.utils import is_connected


def train(args):
    print(f"> Running train_qmix_manifold.py with args: {args}")

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
    )
    env_test = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        n_rays=args.n_rays,
        full_observation=args.full_observation,
        max_step=args.max_step,
    )

    # Create QMIX algorithm
    algo = make_algo(
        algo=args.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        area_size=env.area_size,
        # Hierarchical RL params
        subgoal_interval=args.subgoal_interval,
        n_bins=args.n_bins,
        # QMIX params
        gamma=args.gamma,
        lr=args.lr,
        target_update_interval=args.target_update_interval,
        tau=args.tau,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        max_grad_norm=args.max_grad_norm,
        # Network params
        use_rnn=not args.no_rnn,
        rnn_layers=args.rnn_layers,
        gnn_layers=args.gnn_layers,
        use_lstm=args.use_lstm,
        mixer_embed_dim=args.mixer_embed_dim,
        mixer_hypernet_hidden=args.mixer_hypernet_hidden,
        # Misc
        seed=args.seed,
        use_soft_update=args.use_soft_update,
    )

    # Generate a 4 letter random identifier for the run
    rng_ = np.random.default_rng()
    rand_id = "".join([chr(rng_.integers(65, 91)) for _ in range(4)])

    # Set up logger
    start_time = datetime.datetime.now()
    start_time = start_time.strftime("%m%d%H%M%S")
    if not args.debug:
        os.makedirs(f"{args.log_dir}/{args.env}/{args.algo}", exist_ok=True)

    start_time = int(start_time)
    while os.path.exists(f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}_{rand_id}"):
        start_time += 1

    log_dir = f"{args.log_dir}/{args.env}/{args.algo}/seed{args.seed}_{start_time}_{rand_id}"
    run_name = "{}_seed{:03}_{}_{}".format(args.algo, args.seed, start_time, rand_id)
    if args.name is not None:
        run_name = "{}_{}".format(args.name, run_name)

    # Get training parameters
    train_params = {
        "run_name": run_name,
        "training_steps": args.steps,
        "eval_interval": args.eval_interval,
        "eval_epi": args.eval_epi,
        "save_interval": args.save_interval
    }

    # Create QMIX trainer
    trainer = TrainerQMixManifold(
        env=env,
        env_test=env_test,
        algo=algo,
        gamma=args.gamma,
        log_dir=log_dir,
        n_env_train=args.n_env_train,
        n_env_test=args.n_env_test,
        seed=args.seed,
        params=train_params,
        # QMIX-specific params
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        save_log=not args.debug,
    )

    # Save config
    wandb.config.update(args)
    wandb.config.update(algo.config, allow_val_change=True)
    if not args.debug:
        with open(f"{log_dir}/config.yaml", "w") as f:
            yaml.dump(vars(args), f)
            yaml.dump(algo.config, f)

    # ========== Warmup: Precompile Manifold (ATACOM) and JIT functions ==========
    import jax
    import jax.random as jr
    import time

    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")

    # 1. Initialize manifold function
    print("Initializing manifold controller...")
    env.init_manifold(
        k=args.topk,
        K=args.viab_gain,
        Kc=args.err_gain,
        alpha_max=args.alpha_max,
        g_act_thresh=args.g_act_thresh,
        safety_margin=args.safety_margin,
        n_lookahead=args.n_lookahead,
        w_slack=args.w_slack,
    )
    env_test.init_manifold(
        k=args.topk,
        K=args.viab_gain,
        Kc=args.err_gain,
        alpha_max=args.alpha_max,
        g_act_thresh=args.g_act_thresh,
        safety_margin=args.safety_margin,
        n_lookahead=args.n_lookahead,
        w_slack=args.w_slack,
    )

    # 2. Warmup manifold (trigger first JIT compile)
    print("Warming up manifold controller (first JIT compile)...")
    warmup_key = jr.PRNGKey(42)
    warmup_graph = env.reset(warmup_key)
    warmup_s = env.manifold_init_slack(warmup_graph)
    target_pos = env.get_agent_goals(warmup_graph)
    nominal = env.u_ref(warmup_graph, target_pos=target_pos, is_final_goal=False)

    start = time.time()
    action, _, _, _ = env.get_manifold_action(warmup_graph, u_ref=nominal, s_all=warmup_s)
    jax.block_until_ready(action)
    print(f"First compile: {time.time() - start:.2f}s")

    start = time.time()
    action, _, _, _ = env.get_manifold_action(warmup_graph, u_ref=nominal, s_all=warmup_s)
    jax.block_until_ready(action)
    print(f"Cached call: {time.time() - start:.4f}s")
    print("Manifold warmup complete!")

    print(f"\n========== QMIX Configuration ==========")
    print(f"Subgoal discretization: {args.n_bins}x{args.n_bins} grid = {args.n_bins**2} actions")
    print(f"Replay buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning starts: {args.learning_starts}")
    print(f"Epsilon: {args.eps_start} -> {args.eps_end} (decay: {args.eps_decay})")
    print(f"Target update: every {args.target_update_interval} steps (tau={args.tau})")
    print(f"========================================\n")
    # ====================================================

    # Start training
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train QMIX with Manifold safety")

    # Required arguments
    parser.add_argument("--env", type=str, default="LidarTarget",
                        help="Environment: LidarTarget, LidarSpread, etc.")
    parser.add_argument("-n", "--num-agents", type=int, default=3,
                        help="Number of agents")
    parser.add_argument("--algo", type=str, default="informarl_qmix",
                        help="Algorithm name (should be informarl_qmix)")
    parser.add_argument("--obs", type=int, default=3,
                        help="Number of obstacles")

    # Manifold parameters
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--viab-gain", type=float, default=0.5)
    parser.add_argument("--err-gain", type=float, default=30.0)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--g-act-thresh", type=float, default=0.02)
    parser.add_argument("--safety-margin", type=float, default=0.02)
    parser.add_argument("--n-lookahead", type=int, default=0)
    parser.add_argument("--w-slack", type=float, default=10.0)

    # QMIX-specific arguments
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Discretization bins per dimension (10x10=100 actions)")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size (number of episodes)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-starts", type=int, default=1000,
                        help="Start learning after N environment steps")
    parser.add_argument("--train-freq", type=int, default=1,
                        help="Train every N collection steps")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="Number of gradient steps per training call")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    parser.add_argument("--target-update-interval", type=int, default=200,
                        help="Update target network every N steps")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Soft update coefficient (if using soft update)")
    parser.add_argument("--eps-start", type=float, default=1.0,
                        help="Initial epsilon for exploration")
    parser.add_argument("--eps-end", type=float, default=0.05,
                        help="Final epsilon for exploration")
    parser.add_argument("--eps-decay", type=float, default=0.995,
                        help="Epsilon decay rate (exponential)")
    parser.add_argument("--max-grad-norm", type=float, default=10.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--use-soft-update", action="store_true", default=True,
                        help="Use soft target updates (vs hard updates)")

    # Network arguments
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--no-rnn", action="store_true", default=False)
    parser.add_argument("--use-lstm", action="store_true", default=False)
    parser.add_argument("--mixer-embed-dim", type=int, default=32)
    parser.add_argument("--mixer-hypernet-hidden", type=int, default=64)

    # Environment arguments
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200000,
                        help="Total training steps")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--full-observation", action="store_true", default=False)
    parser.add_argument("--subgoal-interval", type=int, default=20,
                        help="Steps between subgoal updates")
    parser.add_argument("--max-step", type=int, default=256)

    # Training arguments
    parser.add_argument("--n-env-train", type=int, default=128,
                        help="Number of parallel training environments (lower for QMIX)")
    parser.add_argument("--n-env-test", type=int, default=8,
                        help="Number of parallel test environments")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-epi", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
