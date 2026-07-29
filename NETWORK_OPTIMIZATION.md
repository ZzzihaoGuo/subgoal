# MATD3 网络架构优化方案

## 当前架构分析

### 参数量对比

| Component | InforMARL_SUB (PPO) | MATD3 (当前) | Ratio |
|-----------|---------------------|--------------|-------|
| **Actor** | ~50K | ~50K | 1.0x |
| **Critic/Value** | Vl: ~30K | Q1+Q2: ~300K | 10x ⚠️ |
| **Total** | ~80K | ~350K | **4.4x** |

### 为什么 Q-network 更大？

```python
# V(s) - 只需要状态
state → GNN(64) → MLP(64,64) → V
Input dim: 64

# Q(s, a) - 需要状态+动作
state → GNN(64) → concat(feat, action) → flatten → MLP(256,256) → Q
Input dim: n_agents × (64 + 2) = n_agents × 66

# 3 agents 例子:
# V(s): 64维输入
# Q(s,a): 198维输入 (3×66)
```

**关键问题：输入维度差距 3x，但网络容量差距 10x！**

---

## 优化方案

### 方案 1: 缩小 Q-network MLP ⭐⭐⭐⭐⭐

**最简单，推荐先试**

```python
# 当前
critic MLP: (256, 256)  # ~150K params per Q

# 优化
critic MLP: (128, 128)  # ~40K params per Q
# 或
critic MLP: (128, 64)   # ~25K params per Q
```

**优点：**
- 简单，只改一行代码
- 参数量降到 ~100K (vs 350K)
- 仍然足够表达能力

**实现：**
```python
# dgppo/algo/module/qnet.py, line 123
self.head = ft.partial(
    MLP,
    hid_sizes=(128, 128),  # 从 (256, 256) 改为 (128, 128)
    act=nn.relu,
    act_final=True,
    name='QNetHead'
)
```

---

### 方案 2: 分解 Q-value (Decentralized Q) ⭐⭐⭐

**每个 agent 独立的 Q_i(s_i, a_i)**

```python
# 当前 (Centralized Q)
Q_tot(s, a1, a2, ..., an) = scalar

# 优化 (Decentralized Q)
Q_i(s_i, a_i) = scalar per agent
Q_tot = sum(Q_i) or mean(Q_i)
```

**架构变化：**
```python
# 不 flatten，保持 per-agent 维度
x = concat(GNN_output, actions)  # (n_agents, 64+2)
x = MLP(x)                       # (n_agents, hidden)
q = Dense(1)(x)                  # (n_agents, 1)
q_tot = q.mean()                 # scalar
```

**优点：**
- 参数量大幅减少 (~50K per Q)
- 可扩展性好（agents 数量变化不影响网络）
- 更接近 QMIX 的思想

**缺点：**
- 丢失部分全局协调信息
- 可能需要调整学习率

---

### 方案 3: 共享 GNN Encoder ⭐⭐⭐⭐

**Actor 和 Critic 共享 GNN 权重**

```python
# 当前
Actor: GNN_actor → MLP_actor → action
Q1:    GNN_q1 → concat(feat, action) → MLP_q1 → Q
Q2:    GNN_q2 → concat(feat, action) → MLP_q2 → Q

# 优化
shared_gnn = GNN_shared
Actor: shared_gnn → MLP_actor → action
Q1:    shared_gnn → concat(feat, action) → MLP_q1 → Q
Q2:    shared_gnn → concat(feat, action) → MLP_q2 → Q
```

**优点：**
- 节省 ~30K params (2个 GNN)
- 更好的特征共享
- 训练更稳定

**缺点：**
- 实现复杂度增加
- 需要小心处理 target network

---

### 方案 4: 使用 Dueling Q-Network ⭐⭐

**分解 Q(s,a) = V(s) + A(s,a)**

```python
# 标准 Q
Q(s, a) = MLP(concat(s, a))

# Dueling Q
V(s) = MLP_V(s)
A(s, a) = MLP_A(concat(s, a))
Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
```

**优点：**
- 更好的值估计
- 特别适合某些状态下动作差异不大的情况

**缺点：**
- 参数量不变（甚至增加）
- 主要提升性能而非减少参数

---

## 推荐实施步骤

### 第一阶段：快速验证 (1天)

```bash
# 1. 缩小 Q-network MLP: (256,256) → (128,128)
# 修改 dgppo/algo/module/qnet.py line 123

# 2. 测试训练
python train_matd3.py --env LidarSpread -n 3 --obs 2 \
    --steps 50000 --eval-interval 500
```

**预期结果：**
- 参数量：350K → ~140K
- 训练速度：提升 20-30%
- 性能：应该相近（可能略降）

---

### 第二阶段：架构优化 (3-5天)

如果第一阶段性能下降明显，试试：

```bash
# 选项 A: 分解 Q (方案 2)
# 实现 decompose=True 模式

# 选项 B: 共享 GNN (方案 3)
# 实现 shared_encoder 参数
```

---

## 网络大小对照表

| Configuration | Actor | Q1 | Q2 | Total | vs PPO |
|---------------|-------|----|----|-------|--------|
| **PPO Baseline** | 50K | - | - | 80K (Vl: 30K) | 1.0x |
| **MATD3 Current** | 50K | 150K | 150K | 350K | 4.4x |
| **MATD3 Small Q** | 50K | 40K | 40K | 130K | 1.6x ⭐ |
| **MATD3 Decompose** | 50K | 50K | 50K | 150K | 1.9x |
| **MATD3 Shared GNN** | 50K | 120K | 120K | 290K | 3.6x |

---

## 代码修改示例

### 方案 1：缩小 Q-network

```python
# File: dgppo/algo/module/qnet.py

# Line 111-118, 修改前:
if decompose:
    self.head = ft.partial(
        MLP,
        hid_sizes=(128, 64),  # 去中心化已经较小
        act=nn.relu,
        act_final=True,
        name='QNetHead'
    )
else:
    self.head = ft.partial(
        MLP,
        hid_sizes=(256, 256),  # ← 这里太大
        act=nn.relu,
        act_final=True,
        name='QNetHead'
    )

# 修改后:
if decompose:
    self.head = ft.partial(
        MLP,
        hid_sizes=(128, 64),
        act=nn.relu,
        act_final=True,
        name='QNetHead'
    )
else:
    self.head = ft.partial(
        MLP,
        hid_sizes=(128, 128),  # ← 改小到 (128, 128)
        act=nn.relu,
        act_final=True,
        name='QNetHead'
    )
```

### 方案 2：分解 Q-value

```python
# File: dgppo/algo/informarl_matd3.py

# 创建算法时设置 decompose=True
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
    decompose=True,  # ← 改为 True
    n_heads=3
)

# 修改 Q-network 输出处理
# File: dgppo/algo/module/qnet.py, line 42
def __call__(self, graph, actions, rnn_state, n_agents):
    x = self.gnn_cls()(graph, node_type=0, n_type=n_agents)
    x = jnp.concatenate([x, actions], axis=-1)  # (n_agents, gnn_dim+action_dim)

    if not self.decompose:
        x = x.reshape(-1)[None, :]  # Flatten
    # else: 保持 (n_agents, feat_dim)

    x = self.head_cls()(x)
    if self.rnn_cls is not None:
        x, rnn_state = self.rnn_cls()(x, rnn_state)

    q = nn.Dense(1, kernel_init=default_nn_init())(x)

    if self.decompose:
        q = q.mean(axis=0, keepdims=True)  # (n_agents, 1) → (1, 1)

    return q, rnn_state
```

---

## 性能影响预测

### 训练速度

| Configuration | Forward Pass | Backward Pass | Memory |
|---------------|--------------|---------------|--------|
| Current (256,256) | 1.0x | 1.0x | 1.0x |
| Small Q (128,128) | **0.7x** | **0.7x** | **0.6x** |
| Decompose | **0.6x** | **0.6x** | **0.5x** |

### 学习性能

| Configuration | Sample Efficiency | Final Performance | Stability |
|---------------|-------------------|-------------------|-----------|
| Current | Baseline | Baseline | Baseline |
| Small Q | -5% ~ +0% | -2% ~ +2% | Similar |
| Decompose | -10% ~ -5% | -5% ~ 0% | Slightly worse |

---

## 结论

**推荐顺序：**

1. **先试方案 1** (缩小 Q-network)
   - 最简单，1行代码
   - 性能影响小
   - 速度提升明显

2. **如果需要进一步优化，试方案 2** (分解 Q)
   - 更大的参数节省
   - 可扩展性更好
   - 需要调参

3. **方案 3 和 4** 作为高级优化
   - 适合已经调好基础版本后
   - 追求极致性能

**关键指标监控：**
- `critic/Q1_loss`, `critic/Q2_loss` - 不应该明显增大
- `eval/sparse_reward` - 最终性能
- Training time per step - 训练速度

---

生成于 Claude Code 🤖
