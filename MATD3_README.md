# MATD3 Implementation for Hierarchical Multi-Agent RL

## Overview

This branch implements **Multi-Agent Twin Delayed DDPG (MATD3)** for hierarchical reinforcement learning with safety constraints.

### Architecture

```
┌─────────────────────────────────────────────────┐
│  High-Level (OFF-POLICY: MATD3)                │
│  - Deterministic actor π(s)                     │
│  - Double Q-critics: Q1(s,a), Q2(s,a)           │
│  - Target networks with soft updates            │
│  - Outputs: subgoal (n_agents, 2) every N steps │
│  - Replay buffer for off-policy learning        │
└──────────────────┬──────────────────────────────┘
                   │ subgoal
                   ▼
┌─────────────────────────────────────────────────┐
│  Low-Level (FIXED: LQR + CBF/Manifold)         │
│  - Reference controller: u_ref(current, target) │
│  - Safety filter: CBF or manifold constraints   │
│  - Outputs: safe actions every step             │
└─────────────────────────────────────────────────┘
```

## Key Features

### 1. **Off-Policy Learning**
- **Replay Buffer**: Stores (s, a, r, s', done) transitions
- **Sample Efficiency**: Reuses past experiences
- **Exploration**: Gaussian noise on deterministic policy

### 2. **TD3 Improvements**
- **Double Q-Learning**: min(Q1, Q2) to reduce overestimation
- **Target Policy Smoothing**: Add noise to target actions
- **Delayed Policy Updates**: Update actor every 2 critic updates

### 3. **Multi-Agent Extensions**
- **Centralized Critic**: Global Q-value Q(s, a1,...,an)
- **Decentralized Actors**: Each agent has subgoal target
- **Graph Neural Network**: Process agent interactions via GNN

## Files Created

```
dgppo/
├── algo/
│   ├── informarl_matd3.py      # MATD3 algorithm implementation
│   ├── module/
│   │   └── qnet.py             # Q-network (state-action value)
│   └── __init__.py             # Register MATD3 in factory
├── trainer/
│   └── trainer_matd3.py        # MATD3 trainer with replay buffer
train_matd3.py                   # Training script
```

## Implementation Details

### MATD3 Algorithm (`informarl_matd3.py`)

**Networks:**
- **Actor**: `SubgoalPolicy` (deterministic after mode())
  - Input: GraphsTuple (state)
  - Output: (n_agents, 2) subgoal coordinates

- **Critic**: `QNetwork` × 2 (Q1, Q2)
  - Input: GraphsTuple (state) + actions
  - Output: scalar Q-value (centralized)

- **Target Networks**: Frozen copies updated via soft update
  - `θ_target ← τ*θ + (1-τ)*θ_target` with τ=0.005

**Training Loop:**
```python
# 1. Collect experience with exploration noise
action = actor(state) + ε, ε ~ N(0, σ²)
rollout = env.step(action)

# 2. Store in replay buffer
replay_buffer.append(rollout)

# 3. Sample minibatch
batch = replay_buffer.sample(batch_size)

# 4. Update critics (every step)
target = r + γ * min(Q1_target(s', a'), Q2_target(s', a'))
loss_Q = MSE(Q(s,a), target)

# 5. Update actor (every policy_delay steps)
loss_actor = -mean(Q1(s, actor(s)))

# 6. Soft update targets
θ_target ← τ*θ + (1-τ)*θ_target
```

### Q-Network (`qnet.py`)

```python
class QNetwork:
    def __call__(self, graph, actions, rnn_state):
        # Process state through GNN
        x = GNN(graph)  # (n_agents, gnn_dim)

        # Concatenate with actions
        x = concat(x, actions)  # (n_agents, gnn_dim + action_dim)

        # Centralized Q-value
        x = flatten(x)  # (n_agents * (gnn_dim + action_dim),)
        x = MLP(64, 64)(x)    # Same size as PPO V-network for fair comparison
        x = Dense(1)(x)       # (1,) - global Q-value

        return q_value
```

**Network Size:**
- Q-network MLP: (64, 64) - **same as PPO V-network**
- Rationale: Fair comparison, similar parameter count
- Structure difference: Q takes concat(state, action) while V only takes state
- Total params: ~80K (comparable to PPO's 80K)

### Trainer (`trainer_matd3.py`)

**Key Differences from On-Policy Trainer:**

| Feature | On-Policy (PPO) | Off-Policy (MATD3) |
|---------|-----------------|-------------------|
| Data Collection | Discard after update | Store in replay buffer |
| Sampling | Use collected batch | Sample from buffer |
| Exploration | Stochastic policy | Deterministic + noise |
| Updates | 1 update per collection | Multiple updates per step |

**Hyperparameters:**
```python
buffer_size = 100000          # Replay buffer capacity
min_buffer_size = 1000        # Start training after this
updates_per_step = 1          # Gradient steps per env step
exploration_noise = 0.1       # σ for Gaussian noise
```

## Usage

### Basic Training

```bash
python train_matd3.py --env LidarSpread -n 3 --obs 3 \
    --subgoal-interval 8 \
    --steps 200000
```

### Advanced Configuration

```bash
python train_matd3.py \
    --env LidarTarget \
    -n 5 --obs 4 \
    --subgoal-interval 8 \
    --max-step 128 \
    \
    # TD3 hyperparameters
    --gamma 0.99 \
    --tau 0.005 \
    --policy-delay 2 \
    --target-noise 0.2 \
    --noise-clip 0.5 \
    --exploration-noise 0.1 \
    \
    # Replay buffer
    --buffer-size 100000 \
    --min-buffer-size 1000 \
    --updates-per-step 1 \
    \
    # Network architecture
    --actor-gnn-layers 2 \
    --critic-gnn-layers 2 \
    --lr-actor 1e-4 \
    --lr-critic 3e-4 \
    --batch-size 256 \
    \
    # Training
    --n-env-train 128 \
    --steps 200000 \
    --eval-interval 100 \
    --seed 0
```

### Debug Mode

```bash
python train_matd3.py --env LidarSpread -n 3 --obs 2 --debug
# Disables: logging, JIT compilation
```

## Expected Behavior

### Training Progress

```
Warming up CBF controller (first JIT compile)...
First compile: 2.35s
Cached call: 0.0012s
CBF warmup complete!

Training MATD3: 0%|          | 0/200000 [00:00<?, ?it/s]
Collecting initial data: 128/1000
Collecting initial data: 256/1000
...
Collecting initial data: 1024/1000
step:   0, time:    15s, reward:  -125.4567, sparse:   -89.2341, cost:   12.3456, unsafe:   0.45, dist: 0.3421, buffer: 1024
step: 100, time:   156s, reward:   -98.7654, sparse:   -67.8912, cost:    8.9012, unsafe:   0.32, dist: 0.2891, buffer: 13824
step: 200, time:   289s, reward:   -76.5432, sparse:   -52.3456, cost:    5.6789, unsafe:   0.21, dist: 0.1987, buffer: 26624
...
```

### Metrics Logged to WandB

**Evaluation:**
- `eval/reward`: Total environment reward
- `eval/sparse_reward`: High-level sparse reward
- `eval/cost`: Maximum safety violation
- `eval/unsafe_frac`: Fraction of unsafe trajectories
- `eval/final_dist2goal`: Final distance to goals
- `buffer/size`: Current replay buffer size

**Training:**
- `critic/Q1_loss`: Q1 network loss
- `critic/Q2_loss`: Q2 network loss
- `critic/target_mean`: Mean TD target value
- `actor/loss`: Policy gradient loss (every policy_delay steps)
- `actor/grad_norm`: Actor gradient norm

## Comparison to InforMARL_SUB (On-Policy PPO)

| Aspect | InforMARL_SUB | MATD3 |
|--------|---------------|-------|
| **Algorithm** | PPO (on-policy) | TD3 (off-policy) |
| **Policy** | Stochastic (TanhNormal) | Deterministic + noise |
| **Value Network** | V(s) | Q(s, a) |
| **Data Usage** | Single-use | Replay buffer |
| **Sample Efficiency** | Lower | Higher |
| **Exploration** | Policy entropy | Action noise |
| **Stability** | Higher (clipped updates) | Lower (critic overestimation) |
| **Parallelism** | 2048 envs | 128 envs (less needed) |

**When to use MATD3:**
- ✓ Sample efficiency is critical
- ✓ Deterministic policies preferred
- ✓ Limited environment interactions
- ✓ Willing to tune hyperparameters

**When to use InforMARL_SUB:**
- ✓ Stability is critical
- ✓ Massive parallelism available
- ✓ Stochastic exploration needed
- ✓ Simple hyperparameter tuning

## Troubleshooting

### Common Issues

**1. Replay buffer not filling:**
```python
# Check: buffer/size in WandB
# Solution: Ensure n_env_train > 0, rollouts are being collected
```

**2. Q-values exploding:**
```python
# Check: critic/target_mean growing unbounded
# Solution: Reduce lr_critic, increase tau, check reward scaling
```

**3. Policy not improving:**
```python
# Check: actor/loss not decreasing
# Solution: Increase exploration_noise, reduce policy_delay, check Q-network
```

**4. OOM errors:**
```python
# Solution: Reduce buffer_size, n_env_train, or batch_size
```

### Debugging Tips

```bash
# 1. Check network initialization
python -c "from dgppo.algo import make_algo; ..."

# 2. Verify rollout collection
python train_matd3.py --debug --steps 10

# 3. Monitor gradient norms
# Look at critic/*_grad_norm in WandB

# 4. Check replay buffer
# Look at buffer/size metric
```

## Future Extensions

### Possible Improvements

1. **Prioritized Experience Replay (PER)**
   - Weight transitions by TD error
   - Improve sample efficiency

2. **N-step Returns**
   - Use multi-step bootstrapping
   - Better credit assignment

3. **Recurrent Critics**
   - Enable temporal modeling in Q-networks
   - Handle partial observability

4. **QMIX Value Decomposition**
   - Factorize Q_tot into per-agent Q_i
   - Learn decentralized execution

5. **Soft Actor-Critic (SAC)**
   - Maximum entropy RL
   - Automatic exploration tuning

## References

- **TD3**: [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)
- **MADDPG**: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
- **CleanRL**: [JAX TD3 Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/td3_continuous_action_jax.py)
- **JaxMARL**: [Multi-Agent RL in JAX](https://github.com/FLAIROx/JaxMARL)

---

**Generated with Claude Code** 🤖
