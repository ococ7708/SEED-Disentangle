# SEED-IV 跨被试情绪识别：任务保护解耦实验（组员交付说明）

## 一、实验目标（一句话）

在 CAST-e3 多源模型上验证：**任务保护下的决策条件解耦**能否在跨被试（LOSO）场景提供独立收益。

## 二、三组实验（A / B / C）

| 组 | 模型 | 回答的问题 |
|---|---|---|
| **A** | P0：CAST 多源骨干 + C1(E2) + 融合 CE | 当前匹配基线 |
| **B** | A + 教师排序保护（关闭身份对抗） | 任务保护本身是否有效 |
| **C** | P3：B + 融合相对 logits 的条件身份对抗 | 解耦是否提供独立收益 |

三组**唯一差别**在"任务保护 / 对抗"的开关；归一化、数据划分、选模、训练预算完全相同。

## 三、实验范式（必须一致）

| 项 | 值 |
|---|---|
| 数据集 | SEED-IV，DE 特征 [T=3, 62, 5] |
| 划分 | 跨被试 LOSO（留一被试） |
| 数据管线 | **旧管线（全部源数据，`--pipeline old`）** |
| 归一化 | record_minmax（逐被试 [-1,1]） |
| 选模 | **target-best**（逐 epoch 目标准确率选 checkpoint） |
| 训练 | 只用源标签；目标标签仅用于选模 |
| 教师 | per-fold 冻结 P0（`last.pt`），**不用 target-best 教师** |

> **注意**：这是"转导预处理 + 目标标签选模"协议，论文里必须明确写"目标标签参与 epoch 选择"。

## 四、运行命令

```powershell
# 环境：C:/Users/oc200/anaconda3/envs/sgda_py311/python.exe（torch 2.7 + transformers + sklearn）
$py = 'C:/Users/oc200/anaconda3/envs/sgda_py311/python.exe'
$s = 'train_seed_family_disentangle.py'
$c = 'configs/seed_family/seediv.json'

# 数据检查
& $py $s --config $c --pipeline old --check-data

# A 组（必须先跑，B/C 要用它的 P0 当冻结教师）
& $py $s --config $c --variant P0 --pipeline old --sessions 1 --seeds 42

# B 组（排序保护，关对抗）
& $py $s --config $c --variant P0 --keep-only --pipeline old --sessions 1 --seeds 42

# C 组（P3：排序保护 + 融合相对 logits 条件身份对抗）
& $py $s --config $c --variant P3 --pipeline old --sessions 1 --seeds 42
```

- 默认：session 1、全部 15 个 target、seed 42、200 epoch × 30 step。
- 短检查：加 `--targets 1 --epochs 12 --steps 1`（只验链路，不报告）。
- 输出：`results/seed_family_disentangle/seediv/{P0,P3}/<hash>/targetXX/`。

## 五、交付内容（每个组员）

每组跑完 15 个 target 后，汇总成**一张表**（被试为统计单位）：

| target | A acc | B acc | C acc |
|---|---|---|---|

再加：

1. **配对均值差**：`B−A`、`C−B`、`C−A`（同 target 配对，报告均值 + 标准差）；
2. 每组 macro-F1、balanced accuracy、混淆矩阵（从 `metrics.json` 取）；
3. 确认每组 `metrics.json` 的 `selected_epoch` 是 target-best 选出的。

## 六、判定标准（写结论前必须满足）

- 主指标：**target accuracy**（target-best）；
- `C > B` 说明解耦（对抗）有独立收益；`B > A` 说明任务保护本身有收益；
- **只报 target-best，不报 source-best**（source-best 是教师选模用的，不进主表）；
- 收益需在 15 target 上配对稳定，不能只看单 target。

## 七、环境依赖

| 依赖 | 默认路径 | 说明 |
|---|---|---|
| 模型实现 | `models/`（仓库内置） | `geosem_stda.py`、`context_affective_metric.py` 已 vendor |
| SEED-IV 数据缓存 | `../SGDA_py3.11/data/seediv_fourclass/seediv` | NPZ: x/y/trial/window |
| 本地 CLIP | `../local_clip_model` | 离线生成类别原型 |
| Python | `sgda_py311` | torch 2.7 + transformers + sklearn |

路径可用 `--cache-root` / `--clip-root` / `--sgda-root` 覆盖。

## 八、后续（第一轮 A/B/C 出结果后）

若 C 组有稳定收益，再做**源侧组外专家（OOF）**：训练时排除整名源被试，在留出被试上评估，提供"未见源被试"的反馈。这是独立训练支路，不改 A/B/C 主表。

---

**代码清单**：`train_seed_family_disentangle.py`（训练入口）、`conditional_disentangler.py`（条件解耦模块）、`seed_family_data.py`（数据接口）、`models/`（模型实现）、`configs/seed_family/`（配置）。
