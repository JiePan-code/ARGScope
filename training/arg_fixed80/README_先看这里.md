# ARGScope：新的 80:20 划分，Full + 三组消融

本包只交付代码。正式训练由你自己运行；不使用补充材料的旧 Full 指标，也不复用旧划分上训练出的融合模型、原型、Teacher 或 Student。

## 放在哪里

把整个 `ARGScope_ablation_suite_80_20` 文件夹放到：

```text
E:\1. ARGmodel\1. NonARG_training\new model result8.24\1.training dataset\
```

该文件夹与已经生成的 `split_80_20_grouped`、`ESM2_EMBEDDING`、`DNABERT2_embedding` 是同级目录。不要只取出其中一个文件运行，也不要覆盖旧的 `ARGScope_ablation_suite`。

## 怎么运行

1. 双击 `00_run_preflight.bat`，确认出现 `PREFLIGHT PASS`。
2. 双击 `RUN_ALL_8_MODELS.bat`，自动依次运行新的 Full 和三个消融，每组都有 Protein、DNA。
3. 八组模型全部完成后，双击 `05_summarize_results.bat`，汇总本次新结果。

也可以分别运行 `01_run_FULL.bat`、`02_run_WO_GATE.bat`、`03_run_WO_GRAPH.bat`、`04_run_WO_TRANSFER.bat`；每个入口会自动补齐它需要的上游阶段。一次只运行一个训练入口，不要同时开多个，以免争用 GPU 或输出目录。

BAT 默认使用 `D:\Anaconda\envs\dnabert2_5060\python.exe`。已在该环境检查 CUDA 与两个原始 embedding 文件。如果电脑路径变化，修改 BAT 内的 `PYTHON_EXE` 即可。无需重新提取 embedding。

## 固定划分

所有阶段直接读取 `split_80_20_grouped/split_manifest.csv` 中的名单，不再随机重切验证集。

| 集合 | 总数 | ARG | nonARG |
|---|---:|---:|---:|
| 外部 train（含 validation） | 18,296 | 9,148 | 9,148 |
| 实际 model-train | 14,636 | 7,318 | 7,318 |
| validation | 3,660 | 1,830 | 1,830 |
| test | 4,574 | 2,287 | 2,287 |

外部 train:test 是 80:20。model-train 和 test 都包含 820 个 ARG subtype、23 个 ARG Type；nonARG 另外计一类。验证集不要求包含只有两条样本的所有稀有 subtype。

新划分按相同蛋白序列或相同 DNA 序列的连通组进行分组；三个集合间没有相同 ID、相同蛋白序列或相同 DNA 序列。这不等同于完成了同源序列隔离，不能宣称 homology-independent 测试。

固定划分指纹：`6d2d3cf7ad252f255dedf4e4bbeae693f5ea0c5112f39daec331aaa975d93991`。

## 四组实验究竟改变什么

| 实验 | 融合 / 图 / 迁移设置 |
|---|---|
| Full | 学习式 gate + model-train raw prototypes + residual GAT + 重新训练 Teacher + Student 知识迁移 |
| w/o Gate | 仅将学习式 gate 改为固定 0.5/0.5；保留标准化、两侧 512D projection、融合 LayerNorm；单独重建融合、raw、GAT、Teacher、Student |
| w/o Graph | 使用本次新 Full 的融合结果与 model-train raw prototypes；raw 直接替代 GAT-refined prototypes，重训 Teacher、Student |
| w/o Transfer | 保留 Student 架构与 binary/subtype CE；不训练 Teacher，KD、feature alignment、prototype alignment、head anchor 五项系数全为零；不拷贝学习得到的 Teacher head |

w/o Transfer 中仍有一个未训练的随机 Teacher 容器，供旧 Student 接口提供形状兼容的随机初始化；两个分类 head 均重新随机初始化，并未训练或传递学习得到的知识。旧基类日志/文件名可能仍带 Teacher/GAT 字样，实际实验身份以本次 `experiment_provenance` 和 `COMPLETE.json` 为准。未删掉计算中的零权重诊断项，以保留原实现。

这里的 w/o Gate **不是直接拼接原始 1280D+768D embedding，也不是删除 projection**，沿用之前确认的单模块消融定义。

## 保留的训练设置

五个核心模型脚本由旧套件原样复制；改变的是统一划分接口、输入/输出连接与消融入口。正式模式不做新的参数搜索、不通过测试指标选择 epoch 或参数。

- Fusion：原设置上限 100 epochs，patience=12。
- GAT：原设置 K=[5,10,15,20]，每个 K 上限 300 epochs，patience=30，按验证集选择。
- Protein：沿用原 Protein 参数文件中的已固定设置；Teacher 上限 100 epochs，Student 200 epochs。`protein_parameters.csv` 只保留参数及固定阈值，不含旧测试指标。
- DNA：沿用原 DNA 核心脚本固定设置；Teacher、Student 均为 300 epochs 上限。
- 两种 Student 的 ARG 判定阈值均保留 0.80。不同模态本来具有各自参数，不强行改成相同值；同一模态的 Full 与三个消融则保持相同设置（除所消融项）。
- 标准化统计量、raw prototypes 等只使用 model-train；验证集用于原有选择逻辑，测试仅用于报告或保存原代码的表征快照。

注意：沿用历史固定超参数，不代表在这套新划分上重新进行了独立的嵌套调参。如果论文需要强调“测试从未参与过任何历史参数确定”，还需要另外核查历史调参样本来源。本包不会自动重新搜索参数。

## 结果在哪里

正式结果统一写到同级的新文件夹：

```text
1.training dataset\ARGScope_80_20_results\
  full\         fusion\ raw\ graph\ protein\ dna\
  wo_gate\      fusion\ raw\ graph\ protein\ dna\
  wo_graph\     protein\ dna\
  wo_transfer\  protein\ dna\
  logs\
  metrics_运行时间\
```

各组 `protein` 和 `dna` 文件夹内都有 `test_sequence_predictions.csv`、实际使用的 `split_manifest.csv`、训练日志/模型等。全程日志另保存在 `logs`。

汇总只读这八组新预测文件，并核对每组恰好为固定测试集的 4,574 个 ID。输出 Accuracy、Precision、Recall、F1、MCC：

- `00_all_models_summary.csv`：Full + 消融总表，56 行。
- `01_ARG_nonARG_ARG_positive_five_metrics.csv`：ARG 为阳性、nonARG 为阴性的二分类，8 行。
- `02_Type_overall_macro_weighted_micro.csv`：Type 三种平均，24 行。
- `03_subtype_overall_macro_weighted_micro.csv`：subtype 三种平均，24 行。
- `04_ARG_nonARG_macro_weighted_micro_optional.csv`：二分类的三种平均，作为额外核查，不混同于 ARG-positive 指标。

Type/subtype 主表沿用此前的端到端口径，**包含 nonARG**，不只挑出预测正确的 ARG。Accuracy 和 MCC 是整体值，三个平均行中保持相同；只有 Precision/Recall/F1 随平均方式变化。没有新增逐类表；以总表为主。

## 中断与重复运行

已正常完成且划分、代码、依赖文件及输出哈希均相符的阶段，会显示 `VERIFIED COMPLETE` 并跳过；这只是复用**本套新划分**已经完成的阶段，不复用旧实验。

未完成的阶段若已写入文件会停止，不覆盖、不删除。请先将错误消息指出的那一个阶段文件夹改名保存，例如把 `full\protein` 改为 `full\protein_interrupted`，再运行原入口，从该阶段重新训练。没有实现恢复到中断 epoch。不要改动已经完成的模型后继续沿用其下游结果。

不要添加 `--overwrite`，不要手动挪入旧 `.pt` 或旧预测文件。更换代码、原始 embedding 或划分后，需要另起新的完整结果目录。

## 检查与短测试

`test_fixed_pipeline.py` 是自动回归检查；`preflight.py` 只检查，不训练。`--smoke-test` 仅用于开发时一轮衔接测试，会写到另一个带 SMOKE 字样的目录，绝不能用于论文。正常 BAT 不包含这个选项，正式训练次数不会被缩短。

如需只跑一种模态，可在有 CUDA 的 Python 下运行 `run_pipeline.py --variant all --modality protein` 或 `--modality dna`。脚本文件夹不在数据根目录下时，显式传入 `--data-root`；`--output-root` 可指定另一处新的输出目录。
