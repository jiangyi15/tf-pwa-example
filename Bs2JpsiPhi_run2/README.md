# Bs2JpsiPhi Run2 分析流程

## 概述

本项目使用 [tf-pwa](https://github.com/jiangyi15/tf-pwa) 对 $B_s^0 \to J/\psi \phi$ 衰变进行时间依赖的振幅分析（time-dependent amplitude analysis）。分析基于 LHCb Run2 数据（2015–2018）。

## 通用参数

所有 `.py` 脚本都支持以下命令行参数：


| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--years` | 处理的数据年份，逗号分隔 | `2015,2016,2017,2018` |
| `--trigger` | Trigger category | `all/unbiased/biased` |
| `--seed` | 随机数种子，用于生成伪数据和 MC | `10` |



**示例：**
```bash
python create_Monte_Carlo.py --years 2018 --trigger all
python cut_Monte_Carlo.py --years 2018 --trigger all --seed 10
```


## 文件命名约定

输出文件遵循以下命名格式：```{prefix}{variable}{years}_{trigger}.npy```

例如：`MC_weight_cut_2018_all.npy`

## 分析流程

### 1. 数据准备

| 脚本 | 说明 |
|------|------|
| `create_real_data.py` | 从 ROOT 文件读取真实数据，保存为 `.npy` 格式 |
| `create_Monte_Carlo.py` | 从 full simulation MC ROOT 文件读取事例，计算振幅修正权重 |
| `create_Monte_Carlo_time.py` | 同上，但**直接使用 ROOT 中的 reconstructed time**，不额外生成 |
| `create_pseudo_data.py` | 从 full simulation MC ROOT 文件读取事例，生成伪数据（pseudo data）和伪 MC （pseudo MC），用于验证分析流程的正确性 |

### 2. 时间分辨率处理

| 脚本 | 说明 |
|------|------|
| `cut_Monte_Carlo.py` | 对 MC 做 time > 0 cut，生成 time resolution 样本 |
| `cut_Monte_Carlo_time.py` | 同上，针对 `Monte_Carlo_time` 流程 |
| `cut_pseudo_data.py` | 对 pseudo data 做 time > 0 cut |

### 3. CP 对称翻倍

| 脚本 | 说明 |
|------|------|
| `double_Monte_Carlo.py` | 将 MC 样本翻倍（第二份 tag 取反），用于 CP 对称分析 |
| `double_Monte_Carlo_time.py` | 同上，针对 `Monte_Carlo_time` 流程 |
| `double_pseudo_data.py` | 对 pseudo data 和 pseudo MC 分别翻倍 |

### 4. 拟合

| 脚本 | 说明 |
|------|------|
| `fit_conv_Monte_Carlo.py` | 使用 full simulation MC 做 phase space 归一化的拟合 |
| `fit_conv_Monte_Carlo_time.py` | 同上，但使用 ROOT 自带的 time |
| `fit_conv_pseudo_data.py` | 使用 pseudo data 进行拟合验证 |

## 归一化方式对比

本项目支持三种不同的 phase space 归一化方式：

| 命名前缀 | 归一化方式 | 说明 |
|----------|-----------|------|
| `phsp_fast` | Toy MC 归一化 | 使用快速 toy sample 做 phase space 积分 |
| `Monte_Carlo` | Full simulation 归一化 | 使用完整的 full simulation MC 做归一化 |
| `pseudo_data` | 自验证归一化 | 将 full simulation MC 随机分成两部分：一部分作为 pseudo data，另一部分作为归一化 MC。用于验证分析流程的正确性 |

## Time 处理方式对比

| 命名 | Time 来源 | 说明 |
|------|----------|------|
| 不带 `time` | Rejection sampling 生成 | 通过指数衰减 + time acceptance + resolution smear 生成 toy time |
| 带 `time` | ROOT 直接读取 | 直接使用 full simulation ROOT 文件中的 reconstructed time 变量，不额外生成 |

## 配置文件

| 文件 | 说明 |
|------|------|
| `config_conv_Monte_Carlo.yml` | Full simulation MC 拟合配置 |
| `config_conv_Monte_Carlo_time.yml` | Full simulation MC (ROOT time) 拟合配置 |
| `config_conv_pseudo_data.yml` | Pseudo data 拟合配置 |

## 参数文件

| 文件 | 说明 |
|------|------|
| `final_params_decay.json` | 初始振幅参数（共振态质量、宽度、耦合系数等） |
| `final_params_*.json` | 各流程拟合后的最终参数 |

## 输出

拟合结果保存在 `figure_{fit_name}_{years}_{trigger}/` 目录下，包含：
- 各变量的拟合分布图（time, cos θ_K, φ_K, cos θ_μ 等）
- 按 tag 分类的分布图（tagp_, tagm_ 前缀）
- Pull distribution 子图
