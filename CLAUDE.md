# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the JAX-based implementation of DGPPO (Discrete GCBF Proximal Policy Optimization) - a multi-agent reinforcement learning algorithm for safe optimal control (ICLR 2025). The repository contains implementations of multiple algorithms, environments across different simulation engines, and comprehensive training/testing infrastructure.

## Project Structure

```
dgppo/
├── dgppo/                 # Main package
│   ├── algo/             # Algorithm implementations
│   │   ├── dgppo.py      # Main DGPPO algorithm
│   │   ├── informarl.py  # InforMARL baseline (MAPPO + GNN)
│   │   ├── informarl_lagr.py    # Lagrangian-constrained variant
│   │   ├── informarl_subgoal.py # Hierarchical subgoal variant
│   │   ├── hcbfcrpo.py   # Hand-crafted CBF variant
│   │   └── module/       # Neural network modules (policy, value, distribution)
│   ├── env/              # Environment implementations
│   │   ├── mpe/          # Multi-Agent Particle Environment
│   │   ├── lidar_env/    # LiDAR-based environments
│   │   └── vmas/         # Vectorized Multi-Agent Simulator (with physax engine)
│   ├── nn/               # Neural network architectures (GNN, MLP, RNN)
│   ├── trainer/          # Training infrastructure
│   └── utils/            # Utilities (graph operations, typing)
├── train.py              # Main training script
├── train_try_1.py        # Training script with subgoal support
├── test.py               # Model evaluation and video generation
└── media/                # Environment visualizations and videos
```

## Quick Start Commands

### Training
```bash
# Basic training (required arguments)
python train.py --env LidarSpread --algo dgppo -n 3 --obs 3

# Training with specific hyperparameters
python train.py --env MPESpread --algo informarl -n 5 --obs 2 --seed 42 --steps 100000

# Debug mode (no logging, JAX JIT disabled)
python train.py --env VMASWheel --algo hcbfcrpo -n 4 --obs 0 --debug

# Full observation mode (for MPE/LidarEnv)
python train.py --env MPEFormation --algo dgppo -n 6 --obs 3 --full-observation

# Hierarchical subgoal training (use train_try_1.py)
python train_try_1.py --env LidarSpread --algo informarl_subgoal -n 3 --obs 2 --subgoal-interval 40
```

### Evaluation
```bash
# Basic evaluation
python test.py --path ./logs/LidarSpread/dgppo/seed0_xxxxxxxxxx_XXXX

# Evaluation with custom parameters
python test.py --path ./logs/MPESpread/informarl/seed42_xxxxxxxxxx_XXXX --epi 10 --no-video

# CPU-only evaluation
python test.py --path ./logs/VMASWheel/hcbfcrpo/seed0_xxxxxxxxxx_XXXX --cpu --epi 20

# Stochastic policy evaluation
python test.py --path ./logs/LidarLine/dgppo/seed0_xxxxxxxxxx_XXXX --stochastic
```

## Environment Setup

### Dependencies
```bash
# Create conda environment
conda create -n dgppo python=3.10
conda activate dgppo

# Install requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### JAX/CUDA Configuration
The project requires JAX with CUDA support. Key environment variables:
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` (set automatically in scripts)
- For offline mode: `WANDB_MODE=offline`
- For debug mode: `JAX_DISABLE_JIT=True`

## Algorithm Architecture

### Core Algorithms
- **DGPPO** (`dgppo`): Main algorithm with learned discrete GCBF constraints
- **InforMARL** (`informarl`): MAPPO baseline with GNN message passing
- **InforMARLLagr** (`informarl_lagr`): Lagrangian-constrained variant with max-over-time cost
- **InforMARL_SUB** (`informarl_subgoal`): Hierarchical subgoal variant with LQR tracking
- **HCBFCRPO** (`hcbfcrpo`): Hand-crafted CBF variant for comparison

### Neural Network Components
- **GNN-based policies**: Graph neural networks for multi-agent coordination
- **RNN support**: GRU/LSTM for partial observability (optional with `--no-rnn`)
- **Value functions**: Separate local (Vl) and global (Vh) value networks
- **Safety constraints**: CBF-based safety filtering in action space

### Environment Engines

#### MPE (Multi-Agent Particle Environment)
- **Environments**: MPETarget, MPESpread, MPEFormation, MPELine, MPECorridor, MPEConnectSpread
- **Dynamics**: Double integrator with continuous action space
- **Observation**: Limited range unless `--full-observation` is used

#### LidarEnv
- **Environments**: LidarTarget, LidarSpread, LidarLine, LidarBicycleTarget
- **Sensors**: LiDAR-based observation with configurable `--n-rays` (default: 32)
- **Dynamics**: Double integrator or bicycle model

#### VMAS (Vectorized Multi-Agent Simulator)  
- **Environments**: VMASReverseTransport, VMASWheel
- **Features**: Contact dynamics, full observability
- **Performance**: Use `--no-rnn` flag to accelerate training

## Training Configuration

### Required Arguments
- `--env`: Environment name (e.g., LidarSpread, MPETarget, VMASWheel)
- `--algo`: Algorithm (dgppo, informarl, informarl_lagr, informarl_subgoal, hcbfcrpo)
- `-n`: Number of agents
- `--obs`: Number of obstacles

### Key Hyperparameters
- `--cbf-weight`: CBF loss weight for DGPPO/HCBFCRPO (default: 1.0)
- `--cbf-eps`: CBF constraint tolerance (default: 0.01)
- `--alpha`: Class-κ function parameter (default: 10.0)
- `--clip-eps`: PPO clipping parameter (default: 0.25)
- `--subgoal-interval`: Steps between subgoal generation for hierarchical RL (default: 40)
- Learning rates: `--lr-actor` (3e-4), `--lr-Vl` (1e-3), `--lr-Vh` (1e-3)

### Performance Tuning
- Single agent: Use lower learning rates (1e-5 for actor, 3e-4 for critics)
- GPU memory issues: Reduce `--n-env-train` (default: 128) and `--batch-size` (default: 16384)
- VMAS environments: Add `--no-rnn` for faster training

## Output Structure

### Training Logs
- Location: `./logs/<env>/<algo>/seed<seed>_<timestamp>_<random_id>/`
- Contents: Model checkpoints, config.yaml, wandb logs
- Intervals: Evaluation every 50 steps, saving every 50 steps (configurable)

### Evaluation Results
- Videos: `<log_path>/videos/<step>/` (MP4 format with safety visualization)
- Metrics: Reward, cost, safety rate printed to console
- CSV logging available with `--log` flag

## Development Notes

### JAX Patterns
- All environments follow functional programming with pure functions
- State management through PyTrees and dataclasses
- JIT compilation for performance (`jax.jit` applied automatically)

### Graph Operations
- Multi-agent coordination via GraphsTuple structures
- Dynamic graph connectivity based on communication radius
- GNN layers process agent interactions and environmental constraints

### Safety Framework
- Cost functions return per-agent constraint violations
- CBF enforces safety through action filtering
- Safety rate computed as 1 - max(constraint_violations)

### Debugging
- Use `--debug` flag to disable logging and JIT compilation
- `ipdb.launch_ipdb_on_exception()` wrapper in main scripts
- Rich logging for training progress visualization

### Custom Environments
To create a custom environment: inherit from an existing environment class in one of the three engines, define your reward function, graph connection, and dynamics, then register the new environment in `dgppo/env/__init__.py`.