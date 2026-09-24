# Type7: TimesNet with Dual-Hierarchy Convolution

独立、精简的 Type7 训练代码。模型来自原实验目录的 `models/type7.py`，训练与评估来自 `TimesNetDHAblationMulti.py`。仅保留 Type7 及其必要依赖；整个目录可以单独复制、运行或上传 GitHub。

## 文件结构

```text
mini/
├── datasets/            # 8 组完整 SFD 数据、对应层级矩阵、数据校验清单
├── type7.py             # 双层级卷积与误差感知残差修正模型
├── backbone.py          # TimesNet 主干、层级一致性损失、BU 辅助损失
├── layers/              # 必需的 embedding 和 inception 层
├── train.py             # 训练、早停、最佳权重加载、测试及多种子汇总
├── requirements.txt
├── LICENSE              # 保留原实验仓库许可证
└── .gitignore
```

## 安装与运行

在 Python 3.11 环境中，进入本目录：

```bash
python -m pip install -r requirements.txt
python train.py --smoke-test --device cpu
python train.py
```

默认自动选择 CUDA（可用时）或 CPU。依赖版本取自本机通过验证的环境；GPU 训练还需匹配的 CUDA 版 PyTorch 与驱动。代码不依赖父目录、DUET 框架或原实验环境中的其他 Python 文件。

完整训练默认运行 3 个种子（42、43、44），每次最多 50 轮，验证集连续 3 轮未改善则早停。默认数据为 `mixedhumid_sfd`。原训练文件默认模型原为 `type4`，此版本按目标模型固定为 **Type7**。

```bash
# 切换数据集
python train.py --dataset cold_sfd

# 单次训练 / 自定义参数
python train.py --runs 1 --epochs 50 --seq 24 --pred-len 24 --dh-steps 1 --seed 42

# 指定输出位置，避免覆盖其他配置的结果
python train.py --dataset marine_sfd --output-dir outputs/marine

python train.py --help
```

数据路径始终相对于 `train.py` 所在目录解析，可从其他工作目录调用。`--output-dir` 的相对路径按当前工作目录解析；默认输出到本项目 `outputs/`。

## 保留的实验设置

| 设置 | 默认值 |
| --- | --- |
| 历史长度 / 预测长度 | 24 / 24 |
| 按时间划分训练 / 验证 / 测试 | 70% / 10% / 20% |
| 时间层级相邻倍率 / 累计倍率 | `[1, 2, 3]` / `[1, 2, 6]` |
| 批大小 / 学习率 | 64 / 0.001 |
| 主损失 | Huber |
| 一致性 / BU 辅助损失权重 | 0.01 / 0.01 |
| TimesNet d_model / d_ff / 层数 | 16 / 64 / 2 |
| top_k / num_kernels / dropout | 5 / 6 / 0.1 |
| 层级隐层维度 / 卷积步数 | 16 / 1 |
| 最大残差门值 | 0.5 |

标准化统计量和时间尺度统计量仅用训练集拟合。验证与测试窗口可使用分割点之前的历史输入，预测目标位于对应划分内。时间尺度按原代码做求和聚合；历史长度和预测长度必须能被 6 整除。

每次运行保存最佳模型 `outputs/*_best.pth`，并重新加载后评估测试集。`outputs/results.json` 保存本次配置、各次运行的原始指标和 BU/TD 层级指标表；终端打印均值与样本标准差（ddof=1，单次运行标准差记为 0）。再次使用相同输出目录和配置会覆盖同名文件。

`--smoke-test` 使用完整数据拟合统计量，但训练、验证、测试各只取一个最多 2 个样本的批次，运行 1 轮、1 个种子，输出到 `outputs/smoke/`。它仅用于安装与流程检查，不能作为论文结果。

## 已执行的验证

- Python 3.11，torch 2.13.0+cu132、numpy 2.4.6、pandas 2.3.3、scikit-learn 1.7.2。
- 对比原始 Type7：相同种子下参数初始化、state_dict、预测、复合损失和参数梯度逐元素完全一致（CPU）。
- 8 组数据与原文件逐字节一致；数值有限，层级矩阵为有效的树/森林。
- CPU 快速端到端测试通过。
- CUDA 上使用 mixedhumid_sfd 全部数据完成 1 轮训练、最佳权重重载与完整测试：normalized MSE 约 0.090291，MAE 约 0.215858。这是可运行性验证，不是完整论文实验复现。

## 代码来源

`type7.py` 保留原模型计算过程；修正了原注释中残留的 Type6 名称。`backbone.py` 从 `TimesNetDH.py` 提取实际使用的主干和两个损失。`layers/` 从原仓库 `ts_benchmark/baselines/time_series_library/layers/` 提取必要类，并保留原仓库许可证。训练部分删除其他模型注册、无关诊断和实验代码，新增命令行参数、结果 JSON、输入检查与快速验证选项。

数据来源及发布信息见 [datasets/README.md](datasets/README.md)。本目录不包含训练产物、缓存、已有检查点或其他消融模型。
