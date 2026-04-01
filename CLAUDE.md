# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

JAX implementation of DGPPO (Discrete GCBF Proximal Policy Optimization) - a multi-agent reinforcement learning algorithm for safe optimal control (ICLR 2025). Contains multiple algorithms, environments across three simulation engines, and training/testing infrastructure. Active development on two experimental branches: hierarchical subgoal RL with CBF safety and manifold-based (ATACOM) safety.

## Quick Start Commands

### Environment Setup
```bash
conda create -n dgppo python=3.10 && conda activate dgppo
pip install -r requirements.txt
pip install -e .
```

### Training
```bash
# Standard training (all four arguments required: --env, --algo, -n, --obs)
python train.py --env LidarSpread --algo dgppo -n 3 --obs 3

# Debug mode (no logging, JAX JIT disabled)
python train.py --env LidarSpread --algo dgppo -n 3 --obs 3 --debug

# Hierarchical subgoal training (CBF-based safety)
python train_try_1.py --env LidarSpread --algo informarl_subgoal -n 3 --obs 2 --subgoal-interval 20

# Manifold/ATACOM training
python train_try_manifold.py --env LidarTarget --algo informarl_subgoal -n 3 --obs 3
```

### Evaluation
```bash
# Standard evaluation
python test.py --path ./logs/LidarSpread/dgppo/seed0_xxxxxxxxxx_XXXX

# Manifold evaluation (reports reward, distance, safe_rate, success metrics)
python test_manifold.py --path <log-path>
```

### Parameter Sweeps
```bash
python sweep_manifold.py  # Automated sweep over manifold parameters
```

## Training Scripts

There are three training entry points, each using a different trainer and rollout strategy:

| Script | Trainer | Rollout | Algorithms | Key Differences |
|--------|---------|---------|------------|-----------------|
| `train.py` | `trainer.Trainer` | `rollout()` | dgppo, informarl, informarl_lagr, hcbfcrpo | Standard RL loop, 128 envs, 200k steps |
| `train_try_1.py` | `trainer_subgoal.Trainer` | `rollout_hierarchical()` | informarl_subgoal | CBF warmup, 2048 envs, 400k steps, rnn_step=1 |
| `train_try_manifold.py` | `trainer_subgoal_manifold.Trainer` | `rollout_hierarchical_manifold()` | informarl_subgoal | Manifold/ATACOM warmup, 2048 envs, 200k steps |

The subgoal variants auto-compute `batch_size = n_env_train * subgoal_interval` and use much higher parallelism (2048 vs 128 envs) with shorter RNN chunks (1 vs 16 steps).

## Algorithm Inheritance Hierarchy

```
Algorithm (dgppo/algo/base.py)
  └── InforMARL (informarl.py) - MAPPO + GNN baseline
        ├── InforMARL_SUB (informarl_subgoal.py) - Hierarchical subgoal variant
        │     Uses SubgoalPolicy; outputs subgoal targets every subgoal_interval steps
        └── InforMARLLagr (informarl_lagr.py) - Lagrangian-constrained, adds Vh value network
              ├── DGPPO (dgppo.py) - Learned GCBF + CBF loss scheduling
              └── HCBFCRPO (hcbfcrpo.py) - Hand-crafted CBF variant
```

Algorithm factory: `dgppo/algo/__init__.py`. Policy/value networks: `dgppo/algo/module/policy.py` and `value.py`.

## Key Architecture

### Graph-Based Multi-Agent State
- All algorithms use `GraphsTuple` (`dgppo/utils/graph.py`) for state representation
- Nodes typed as: agent=0, goal=1, obstacle=2
- Edges: relative states between entities within communication radius
- GNN layers (`dgppo/nn/gnn.py`) process interactions via message passing

### Neural Network Pipeline
- **Policy**: GNN → RNN (GRU default, LSTM with `--use-lstm`) → TanhNormal distribution
- **Value Networks**: Vl (reward value), Vh (constraint value, only in Lagr/DGPPO variants)
- **SubgoalPolicy** (`dgppo/algo/module/policy.py`): outputs target positions for low-level controller

### Environment Interface (`dgppo/env/base.py`)
All environments implement: `reset(key)` → GraphsTuple, `step(graph, action)` → (graph, reward, cost, done, info), `get_cost(graph)`.

### Three Simulation Engines
- **MPE** (`dgppo/env/mpe/`): Particle-based, double integrator dynamics. 6 envs: Target, Spread, Formation, Line, Corridor, ConnectSpread
- **LidarEnv** (`dgppo/env/lidar_env/`): LiDAR sensors, double integrator or bicycle dynamics. 4 envs: Target, Spread, Line, BicycleTarget
- **VMAS** (`dgppo/env/vmas/`): Contact physics via physax engine (`dgppo/env/vmas/physax/`), full observability. 2 envs: ReverseTransport, Wheel

### Safety Controllers (in `dgppo/env/lidar_env/base.py`)
The LidarEnv base class contains two safety controller implementations:
- **CBF solver**: `init_cbf()` / `safe_u_ref()` — uses `jaxproxqp` QP solver for control barrier function safety
- **Manifold/ATACOM solver**: `init_manifold()` / `get_manifold_action()` — adaptive target adaptation with parameters: topk, viab_gain, err_gain, alpha_max, g_act_thresh, safety_margin, n_lookahead, w_slack
- **LQR reference**: `u_ref()` — baseline linear-quadratic reference controller

### Trainer Architecture (`dgppo/trainer/`)
- `data.py` — Rollout namedtuple definition
- `buffer.py` — Experience buffer
- `utils.py` — Contains all rollout functions (`rollout`, `rollout_hierarchical`, `rollout_hierarchical_manifold`), test rollout functions, and global coefficients (`GOAL_REWARD_COEF`, `SUBGOAL_BONUS_THRESH`, `SUBGOAL_SHADOW_COEF`)

## Required Arguments

| Arg | Description |
|-----|-------------|
| `--env` | Environment name (e.g., LidarSpread, MPETarget, VMASWheel) |
| `--algo` | Algorithm (dgppo, informarl, informarl_lagr, informarl_subgoal, hcbfcrpo) |
| `-n` | Number of agents |
| `--obs` | Number of obstacles |

## Performance Tips
- Single agent: Use `--lr-actor 1e-5 --lr-Vl 3e-4 --lr-Vh 3e-4`
- GPU memory issues: Reduce `--n-env-train` and `--batch-size`
- VMAS: Add `--no-rnn` for faster training
- `--debug` disables JIT for easier debugging but is much slower

## Output Structure
Training creates: `./logs/<env>/<algo>/seed<seed>_<timestamp>_<4-char-id>/`
- `models/` - Checkpointed parameters
- `config.yaml` - Training configuration
- `videos/` - Evaluation videos

## JAX Patterns
- Functional programming with pure functions throughout
- State management via PyTrees and dataclasses
- Vectorization via `jax.vmap` for parallel environments
- Both CBF and manifold solvers require a JIT warmup step at training start (see `init_cbf()` / `init_manifold()` calls in training scripts)
