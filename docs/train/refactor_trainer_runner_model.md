# Trainer 与 Model 解耦 — 简化版思路（评审）

目标：**少做几种文件、少改几处代码**，就能换模型或换训练方式，而不是「一种模型一个 `runner_*.py` + `train_world` 里写死调用」。

---

## 1. 现在哪里别扭

- **`FMPRunner`** 里堆了太多事：拼网络、找权重路径、噪声调度、训练 `forward`、采样、存盘……变体一多就要**复制整个 runner 文件**。
- **`train_world.py`** 直接按固定参数去调 `action_expert(...)`，和某一种 Runner **绑死**了。

---

## 2. 想变成什么样（两句话）

1. **`train_world`（Trainer）**：只负责「循环、优化器、日志、存 checkpoint」，**不管**你里面是 causal 还是 128dim，只要对方提供一个 **`training_step(batch) -> loss`****（或等价的小接口）**。
2. **「Runner」**：收缩成**一个「World 训练模块」类**（名字随意），里面再组装 `WorldPolicy`、加载权重等；**新变体优先改这里或改 `models/`，不要去动 Trainer 主循环**。

不必一上来就做完整注册表、多层 `builders/`；可以先把概念收敛成 **「Trainer + 一个 Module」** 两层。

---

## 3. 关系示意（极简）

```text
train_world.py          ← 薄：Accelerate + for batch in loader
        │
        ▼
TrainModule             ← 承接现在 FMPRunner 里「batch → loss」那一段
        │
        ├── models/...  ← 纯网络（Policy、Expert）
        └── 调度 / 噪声  ← 仍可在 Module 里，或以后再拆小文件
```

---

## 4. 目录可以怎么摆（从简）

不必一次拆成很多包，**先做三处**就够理清边界：

| 位置 | 放什么 |
|------|--------|
| `wam/trainers/train_world.py` | 只调 `module.training_step(...)`，不再写一长串 `forward` 参数 |
| `wam/trainers/world_train_module.py`（新文件，名字可改） | 从现有 `FMPRunner` **搬**「组装 + forward 算 loss」；对外暴露 `training_step` |
| `models/` | 继续放 `WorldPolicy`、Expert 等**纯网络**；逐步把「读盘、if WoW」从 runner 挪到 Module 或单独 `load_weights.py` |

等这条线稳了，再考虑要不要 **`registry` / yml 里 `trainer_id`**，那是第二步优化，不是第一步必做。

---

## 5. 迁移：两步就够

| 步骤 | 做什么 |
|------|--------|
| **①** | 新建 `TrainModule`，把 `FMPRunner.forward` + 构造逻辑挪进去；`FMPRunner` 可先变成**薄包装**或直接弃用 |
| **②** | 改 `train_world`：构造 `TrainModule`，训练循环里只调 `training_step`；采样函数同样只依赖这个 Module |

每步做完跑一次 **同配置、同 loss 是否对齐** 即可。

---

## 6. 风险（只记两条）

- **存盘 / 断点**：尽量保持 `state_dict` 键名与现在一致，或加版本字段，避免旧 checkpoint 读不进。
- **EMA / DDP**：明确「参与训练、被 `optimizer` 更新的那个 `nn.Module`」是哪一个，避免包错层。

---

## 7. 小结

- **核心**：Trainer 只认「**一个会算 loss 的 Module**」，不再认「某一个 Runner 类名」。
- **实现**：先 **一个文件 + 两步迁移**，不必先上复杂目录与 registry；需要时再在文档里加第二阶段设计即可。

---

## 8. 落地状态（第一步）

已将「batch → 编码 → `policy` forward → loss」迁入 **`wam/trainers/world_train_module.py`** 的 `TrainModule.training_step`；`wam/trainers/train_world.py` 训练循环内改为只调用该方法。policy 仍为 `FMPRunner`（后续可换构造方式而少动 Trainer）。

---

*简化版评审稿；与 `WAM/wam/trainers/train_world.py`、`WAM/wam/runners/runner_world_casual_128dim.py` 现状对照阅读即可。*
