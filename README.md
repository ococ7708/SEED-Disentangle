# 解耦实验 README（启动说明 · 三数据集三组员）

日期：2026-11-09。范围：`experiments_dtrm/` 下的 CAST-e3 多源解耦实验。本文件是**实验启动说明**：模型、候选、数据集分配、命令、要拿到的结果。

## 1. 模型（一句话看懂）

```
DE [N,3,62,5] → CAST-e3 → h[128]
  ├─ S(h) = s[128]  → 14 个源 adapter(128→32→128) → 共享投影 128→512 → C1-E2 头 → 距离加权融合 logits[K]
  └─ Q(stopgrad(h)) = q[128] → q 身份头（训练期，不回传 CAST）
  s → 按真实情绪类别选择的身份头（训练期）
```

- 推理只用 `s` 路径 + 融合。
- 选定：`record_minmax` 归一化（转导，逐被试 [-1,1]）+ **target-best** 选模（对齐多数跨被试论文）。
- 训练用源标签；目标标签只进入 checkpoint 选择与最终评价（不参与反传）。

## 2. 三个候选（≤3，就是我们自己的设计）

| 版本 | 训练内容 | 作用 |
|---|---|---|
| **P0** | S(h) + 情绪 CE + **融合 CE** + C1 原正则 | 同容量任务基线 |
| **P1** | P0 + **表示层**类条件身份对抗 `I(S;R\|Y)` | 作用位置对照（攻击 s） |
| **P3** | P0 + **决策层**类条件身份对抗 `I(T;R\|Y)`，`t=Aᵀℓ` + **冻结 P0 教师 L_keep** | 主候选（攻击真实决策 + 任务保留） |

三者的差别只在**对抗作用位置**（P0 无 / P1 表示 / P3 决策）和**任务保留**（P3 有），其余同归一化、同划分、同容量、同训练预算。

## 3. 我们的创新（不是抄论文）

1. **诊断发现（本项目独有）**：在 CAST-e3 多源模型上，身份信息存在于**表示 s**（线性探针 0.83），但**无条件决策 t=Aᵀℓ 已接近身份不变**（0.071≈1/14 随机）；只有**类条件的决策依赖**残留（0.199）。
2. **由此得到的方法选择**：正确的干预目标不是"让整个 shared 向量不可识别身份"（表示层），而是**类条件的决策依赖 `I(T;R|Y)`**（决策层）。这解释了为什么之前表示层解耦（正交/GRL/MI）不涨点。
3. **P3 的具体形式**：类条件身份头接收**真实融合 t**（不是独立分支 logits），并用**冻结 P0 教师 KL**（`L_keep`）保住任务预测。
4. **与现有工具的区别**：类条件对抗、GRL、教师 KL 都是已有工具；本文的贡献是"诊断驱动的干预位置选择（决策层类条件）+ 任务保留"这一组合与证据链，不是新造模块。

## 4. 数据集分配（三个组员）

| 组员 | 数据集 | 配置 | 类别 |
|---|---|---|---|
| 组员 A | **SEED** | `configs/seed_family/seed.json` | 3 类（neg/neu/pos） |
| 组员 B | **SEED-IV** | `configs/seed_family/seediv.json` | 4 类（neu/sad/fear/happy） |
| 组员 C | **SEED-V** | `configs/seed_family/seedv.json` | 5 类 |

- SEED-VII 本机无数据，本轮不跑。
- 每个数据集内部：P0 / P1 / P3 × 3 sessions × 15 targets × seeds。

## 5. 运行命令

环境：`C:/Users/oc200/anaconda3/envs/sgda_py311/python.exe`，默认 CUDA。

```powershell
$py = 'C:/Users/oc200/anaconda3/envs/sgda_py311/python.exe'

# 0) 数据检查
& $py train_seed_family_disentangle.py --config configs/seed_family/seed.json --check-data

# 1) 先跑 P0（必须最先：P3 的冻结教师 = 同 fold 的 P0 源侧 checkpoint）
& $py train_seed_family_disentangle.py --config configs/seed_family/seed.json --variant P0

# 2) 再跑 P1（表示层对照）和 P3（主候选；教师自动按同 session/seed/target 定位 P0 的 source_best.pt）
& $py train_seed_family_disentangle.py --config configs/seed_family/seed.json --variant P1
& $py train_seed_family_disentangle.py --config configs/seed_family/seed.json --variant P3
```

- 短检查：`--sessions 1 --seeds 42 --targets 1 --epochs 12 --steps 1`（只验链路，不是收敛结果）。
- 正式：默认 3 sessions × 3 seeds × 全部 target。
- **P3 教师是 per-fold 的 `source_best.pt`（源侧选模，不是 target-best）**：必须在同 output_root 下先跑完 P0；P3 会自动按 (dataset, session, seed, target, epochs, steps) 匹配，不匹配直接报错，不会静默复用错误教师。

## 6. 需要实验拿到的结果（交付清单）

每个数据集、每个 variant 都要报告下面这张表（按 session × seed 汇总，被试为统计单位）：

| 指标 | 说明 |
|---|---|
| **target accuracy**（主指标） | target-best checkpoint 上的未见被试准确率 |
| macro-F1 / balanced accuracy | 类别不平衡下的补充 |
| per-class recall + confusion matrix | 查是否因某类崩溃 |
| 逐被试 accuracy | 配对比较用 |

**三组对比要回答（按优先级）：**

1. **P1 vs P0**：表示层类条件对抗有没有用（作用位置对照）。
2. **P3 vs P0（核心）**：决策层类条件对抗 + L_keep，是否在**保住准确率**的同时降低类条件身份依赖。
3. **P3 vs P1**：决策层是否优于表示层。

**判定标准（写进结论前必须满足）：**

- 主指标：P3 相对 P0 的配对均值差、逐被试差异、多 seed 稳定性（不能只看单 seed）。
- 机制：对 P3 的 checkpoint 复跑**类条件身份探针**（同 trial 互斥、类别×被试平衡），确认 `t` 的类条件身份可读性下降。
- **"身份探针下降但准确率也下降"不算成功**；"准确率提高但机制不支持"只能报性能、不能强称解耦。

## 7. 可行性说明

- 每 run：200 epoch × 30 step；单 run 估计 20-40 分钟（14 分支 + C1 + 融合 CE）。
- 满配 `3 sessions × 3 seeds × 15 targets × 3 variants = 405 run/数据集`，单数据集约 5-8 天（GPU 连续）。
- **建议分两阶段**：
  1. **Phase 1（先做）**：3 sessions × 1 seed × 15 targets × 3 variants = 135 run（约 2-3 天），拿到方向。
  2. **Phase 2（确认后）**：对 P0 和 P3 补 seeds 43/44，做多 seed 稳定性。

## 8. 输出位置

`experiments_dtrm/results/seed_family_disentangle/<dataset>/<variant>/<hash>/targetXX/`：
- `run.json`（生效配置、划分、源码哈希、选模规则）
- `history.json`（每 epoch source-dev 与 target）
- `target_best.pt` / `last.pt`
- `metrics.json`（最终 target 指标、selected_epoch、混淆矩阵）
- `COMPLETE`

## 9. 代码清单

- `experiments_dtrm/train_seed_family_disentangle.py` — P0/P1/P2/P3 训练入口
- `experiments_dtrm/conditional_disentangler.py` — 条件解耦模块（s 对抗 + t 对抗 + 类内去相关）
- `experiments_dtrm/seed_family_data.py` — trial-safe DE 缓存接口 + 归一化
- `experiments_dtrm/configs/seed_family/{seed,seediv,seedv,seedvii}.json` — 四套配置
- `scripts/import_seed_family_de.py` — MAT → NPZ 导入器

## 10. 环境依赖（必须核对）

本仓库只含**实验代码与配置**，依赖外部项目/数据：

| 依赖 | 配置字段 | 默认相对路径 | 说明 |
|---|---|---|---|
| SGDA 模型代码 | `sgda_root` | `../SGDA_py3.11` | 提供 `GeoSemSTDA` / `MultiSourceAffectiveMetric` 等 |
| SEED 系列数据缓存 | `data.cache_root` | `../SGDA_py3.11/data/...` | NPZ: x/y/trial/window |
| 本地 CLIP 文本模型 | `clip_root` | `../local_clip_model` | 离线生成类别原型 |
| Python 环境 | — | `sgda_py311` | torch 2.7 + transformers + sklearn |

仓库应放在与 `SGDA_py3.11`、`local_clip_model` 同级的位置（路径才与默认配置一致），否则用 `--cache-root` 或改配置。

> 本地运行示例：`python train_seed_family_disentangle.py --config configs/seed_family/seediv.json --check-data`
