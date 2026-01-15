# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

JAX implementation of DGPPO (Discrete GCBF Proximal Policy Optimization) - a multi-agent reinforcement learning algorithm for safe optimal control (ICLR 2025). Contains multiple algorithms, environments across different simulation engines, and training/testing infrastructure.

## Quick Start Commands

### Training
```bash
# Basic training (all four arguments required)
python train.py --env LidarSpread --algo dgppo -n 3 --obs 3

# Debug mode (no logging, JAX JIT disabled)
python train.py --env LidarSpread --algo dgppo -n 3 --obs 3 --debug

# Hierarchical subgoal training
python train_try_1.py --env LidarSpread --algo informarl_subgoal -n 3 --obs 2 --subgoal-interval 20
```

### Evaluation
```bash
python test.py --path ./logs/LidarSpread/dgppo/seed0_xxxxxxxxxx_XXXX
```

### Environment Setup
```bash
conda create -n dgppo python=3.10 && conda activate dgppo
pip install -r requirements.txt
pip install -e .
```

## Algorithm Inheritance Hierarchy

```
Algorithm (base.py)
  └── InforMARL (informarl.py) - MAPPO + GNN baseline
        ├── InforMARL_SUB (informarl_subgoal.py) - Hierarchical subgoal variant
        └── InforMARLLagr (informarl_lagr.py) - Lagrangian-constrained
              ├── DGPPO (dgppo.py) - Main algorithm with learned GCBF
              └── HCBFCRPO (hcbfcrpo.py) - Hand-crafted CBF variant
```

## Key Architecture

### Graph-Based Multi-Agent Coordination
- All algorithms use `GraphsTuple` (`dgppo/utils/graph.py`) for multi-agent state representation
- Nodes: agent states, goals, obstacles (typed via `node_type`: agent=0, goal=1, obstacle=2)
- Edges: relative states between connected entities within communication radius
- GNN layers (`dgppo/nn/gnn.py`) process agent interactions

### Environment Interface (`dgppo/env/base.py`)
All environments implement:
- `reset(key)` → GraphsTuple
- `step(graph, action)` → (graph, reward, cost, done, info)
- `get_cost(graph)` → per-agent constraint violations

### Three Simulation Engines
- **MPE**: Particle-based, double integrator dynamics (6 envs)
- **LidarEnv**: LiDAR sensors, double integrator or bicycle dynamics (4 envs)
- **VMAS**: Contact physics via physax engine, full observability (2 envs)

### Neural Network Components
- **Policy**: GNN → RNN (optional) → TanhNormal distribution
- **Value Networks**: Vl (local reward), Vh (constraint value for DGPPO/Lagr variants)
- **RNN**: GRU by default, LSTM with `--use-lstm`

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

## Output Structure
Training creates: `./logs/<env>/<algo>/seed<seed>_<timestamp>_<4-char-id>/`
- `models/` - Checkpointed parameters
- `config.yaml` - Training configuration
- `videos/` - Evaluation videos

## JAX Patterns
- Functional programming with pure functions throughout
- State management via PyTrees and dataclasses
- `--debug` flag disables JIT for easier debugging
- Vectorization via `jax.vmap` for parallel environments

## Adding Custom Environments
1. Inherit from existing environment class (MPE/LidarEnv/VMAS base)
2. Define `PARAMS`, `reset()`, `step()`, `get_cost()`, `get_graph()`
3. Register in `dgppo/env/__init__.py`
