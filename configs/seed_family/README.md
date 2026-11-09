# SEED系列解耦实验配置

完整方案、公式、协议边界、数据格式和命令见项目 `docs/解耦优化完整方案与SEED系列实验接口_20261109.md`。

在项目根目录执行：

```powershell
$py = 'C:/Users/oc200/anaconda3/envs/sgda_py311/python.exe'
& $py experiments_dtrm/train_seed_family_disentangle.py --config experiments_dtrm/configs/seed_family/seediv.json --check-data
& $py experiments_dtrm/train_seed_family_disentangle.py --config experiments_dtrm/configs/seed_family/seediv.json --variant P0 --sessions 1 --seeds 42 --targets 1
& $py experiments_dtrm/train_seed_family_disentangle.py --config experiments_dtrm/configs/seed_family/seediv.json --variant P1 --sessions 1 --seeds 42 --targets 1
```

`seed.json`/`seediv.json`/`seedv.json`的本机缓存已检查。SEED-V为五分类、16被试、每个LOSO折15个源adapter；三个session分别运行。`seedvii.json`需先准备data/seedvii缓存，或用--cache-root指定实际位置。SEED-VII七分类使用四session合并，不使用单session冒充完整七分类。

SEED-V运行：

```powershell
& $py experiments_dtrm/train_seed_family_disentangle.py --config experiments_dtrm/configs/seed_family/seedv.json --check-data
& $py experiments_dtrm/train_seed_family_disentangle.py --config experiments_dtrm/configs/seed_family/seedv.json --variant P0 --sessions 1 --seeds 42 --targets 1
& $py experiments_dtrm/train_seed_family_disentangle.py --config experiments_dtrm/configs/seed_family/seedv.json --variant P1 --sessions 1 --seeds 42 --targets 1
```

SEED-VII原论文核查笔记：`docs/SEED_V接入与SEED_VII原论文协议核查_20261109.md`。

默认source_global_zscore+源trial验证选模。复查旧转导预处理可用--normalization record_minmax，须同时重跑P0/P1。正式默认3seed×全部session组×全部target；未在本轮启动完整矩阵。

MAT导入的trial_index_example.csv仅演示字段，键名和标签必须按真实文件核实；导入器不从原始EEG提取DE、不猜测官方标签编号。
