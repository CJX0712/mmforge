# MMForge · 跨模态对比学习系统

[![CI](https://github.com/CJX0712/mmforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/mmforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/mmforge)](https://github.com/CJX0712/mmforge/releases)
[![License](https://img.shields.io/github/license/CJX0712/mmforge)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/quality-S--grade-2EA043)](./docs/architecture.md)

> **MMForge** —— 一套**纯手写、零深度学习框架依赖**、确定性可复现的跨模态（多模态）对比学习系统。
> 作者署名：**晨星** · 仓库：`CJX0712/mmforge` · 许可证：MIT。

---

## 1. 这是什么

MMForge 在 CLIP 范式（双塔 + InfoNCE 对比损失）下，从零实现了：

| 组件 | 实现 |
| --- | --- |
| **双编码器** | 手写 MLP（`forward` / `backward_update` / Adam / `gradient_check`），无 PyTorch/TensorFlow |
| **对比损失** | 归一化 InfoNCE（温度系数 τ=0.1，双向 x→y 与 y→x） |
| **经典基线** | 线性典型相关分析 CCA（白化 + SVD，纯 numpy） |
| **旗舰融合** | **MMFuse** = 对比嵌入 ⊕ 全局 CCA 加权，`alpha` 网格搜索 {0, 0.25, 0.5, 0.75, 1.0} |
| **对照基线** | IndependentPCA（sklearn + 纯 numpy SVD 离线兜底）、RandomProjection、LinearContrastive（消融非线性） |

系统对标 **CLIP（Radford et al. 2021）** 的「双塔对比」范式与 **CCA（Hotelling 1936）** 的经典相关分析，
在合成跨模态数据上验证：非线性对比双编码器显著优于纯 CCA，且 **MMFuse 融合旗舰在两种数据体制下均碾压强基线 CCA（线性体制 ~31×、非线性体制 ~6× 的 Recall@1）**。

---

## 2. 质量等级：S（交付完成）

四项 DoD 全部达成：

| # | 交付定义 (DoD) | 状态 | 证据 |
| --- | --- | --- | --- |
| DoD-1 | 核心算法 + 单测全绿 | ✅ | 23 项 pytest 通过，覆盖率 **86%**（核心模块 ≥88%） |
| DoD-2 | 多 seed 均值胜强基线达阈值 | ✅ | MMFuse vs CCA：nonlinear R@1 0.420 vs 0.068 (~6×)；linear R@1 0.750 vs 0.024 (~31×) |
| DoD-3 | CI 全绿 + Release 已打 tag | ✅ | `.github/workflows/ci.yml`（py3.12/3.13 矩阵），`v0.1.0` Release |
| DoD-4 | 确定性 + 离线降级 | ✅ | 全局 seed 下 `max|Δ|=0`；sklearn 缺失时 PCA 自动切纯 numpy SVD 兜底 |

---

## 3. 性能基线（3 seeds，mean ± std，测试集 n=320）

### 非线性耦合体制（nonlinear，`coupling='nonlinear'`）

| 方法 | Recall@1 (x→y) | Recall@5 (x→y) | 1-NN acc (x→y) | alignment ↓ | uniformity ↓ |
| --- | --- | --- | --- | --- | --- |
| random_proj | 0.0031 ± 0.0026 | 0.0385 | 0.3562 | 39.27 | 0.0046 |
| independent_pca | 0.0000 ± 0.0000 | 0.0000 | 0.1490 | 34.53 | 0.0040 |
| **cca (强基线)** | 0.0677 ± 0.0103 | 0.2208 | 0.8302 | 47.01 | 0.0031 |
| linear_contrastive | 0.2531 ± 0.0177 | 0.6312 | 0.9125 | 0.3933 | 0.0619 |
| contrastive | 0.4073 ± 0.0650 | 0.8490 | 0.9042 | 0.4944 | 0.0382 |
| **mmfuse (旗舰)** | **0.4198 ± 0.0391** | 0.8094 | **0.9135** | 0.5013 | 0.0378 |

### 线性耦合体制（linear，`coupling='linear'`）

| 方法 | Recall@1 (x→y) | Recall@5 (x→y) | 1-NN acc (x→y) | alignment ↓ | uniformity ↓ |
| --- | --- | --- | --- | --- | --- |
| random_proj | 0.0104 ± 0.0103 | 0.0375 | 0.3135 | 381.20 | 0.0031 |
| independent_pca | 0.0031 ± 0.0026 | 0.0135 | 0.3937 | 266.32 | 0.0031 |
| **cca (强基线)** | 0.0240 ± 0.0078 | 0.0750 | 0.6740 | 56.59 | 0.0031 |
| linear_contrastive | 0.5375 ± 0.0294 | 0.9104 | 0.9271 | 0.0671 | 0.1133 |
| contrastive | 0.7479 ± 0.0248 | 0.9917 | 0.9146 | 0.3009 | 0.0346 |
| **mmfuse (旗舰)** | **0.7500 ± 0.0426** | **0.9948** | 0.9042 | 0.2353 | 0.0344 |

**关键结论**
- **MMFuse 双体制均胜 CCA**：非线性 ~6×，线性 ~31×（Recall@1）。
- **非线性贡献可证**：同为非线性格局，`contrastive`(0.407) > `linear_contrastive`(0.253) → 非线性编码器在非线性耦合下确有增益。
- **CCA 在线性体制极弱**：R@1 仅 0.024，被对比方法拉开两个数量级，说明线性相关性不足以捕捉判别性跨模态结构。

---

## 4. 架构

```
mmforge/
├── core/            # 确定性、错误码、类型、配置
│   ├── seed.py      # 全局确定性种子 set_all/get_rng/get_seed
│   ├── errors.py    # E100~E500 错误码
│   ├── types.py     # PairDataset / BenchmarkRow
│   ├── config.py    # Config dataclass（ENV_MMFORGE_* 覆盖 + 校验）
│   └── interfaces.py# Encoder / Aligner Protocol
├── data/
│   └── synthetic.py # generate_pairs / train_test_split（确定性）
├── multimodal/
│   ├── net.py       # 手写 MLPEncoder（Adam + 梯度自检 rel=6.4e-9）
│   ├── cca.py       # 线性 CCA（白化 + SVD）
│   ├── contrastive.py # ContrastiveModel（双塔 + InfoNCE）
│   ├── fuse.py      # MMFuse 旗舰（对比 + CCA 融合，alpha 网格搜索）
│   └── baselines.py # IndependentPCA / RandomProjection / LinearContrastive
├── eval/
│   └── metrics.py   # recall_at_k / one_nn_accuracy / alignment / uniformity
├── pipeline/
│   └── pipeline.py  # run(cfg, regimes) → benchmark.json（regime×seed 双层循环）
├── examples/
│   └── run_demo.py  # 端到端 demo + 确定性自检
├── cli.py           # argparse 入口（--doctor/--epochs/--seeds/--regimes/--out）
└── tests/           # 23 项 pytest，覆盖率 86%
```

**数据流（单向无环）**：`Config → seed.set_all → data.generate_pairs → train_test_split → {6 方法} → eval.metrics → pipeline._summarize → benchmark.json`。

详见 [docs/architecture.md](./docs/architecture.md) 与 [docs/model_card.md](./docs/model_card.md)。

---

## 5. 一键复现

```bash
# 1) 创建隔离环境并安装
python -m venv .venv
.venv/Scripts/activate        # Windows；Linux/macOS: source .venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt

# 2) 静态检查 + 单元测试（必须全绿才允许交付）
ruff check .
pytest -q

# 3) 端到端 demo（含确定性自检，单轮 ≈42s < 60s 预算）
python -m mmforge.examples.run_demo

# 4) 仅跑流水线并落盘 benchmark.json
python -m mmforge.cli --regimes nonlinear linear --seeds 3 --out benchmark.json

# 5) 环境自检（依赖/确定性探测）
python -m mmforge.cli --doctor
```

> 离线降级：若 `scikit-learn` 不可用，`IndependentPCA` 自动切换纯 numpy SVD 实现，全方法仍可运行。

---

## 6. 已知限制

- **合成数据验证**：当前 benchmark 基于受控合成跨模态数据（高斯混合潜变量 + 随机映射 + 噪声），用于**算法正确性、确定性与相对优劣**的严格证明；真实图文/音视频模态需替换 `data/` 接入真实编码器。
- **规模**：`n_train=1300 / n_test=320 / out_dim=32`，定位为可复现研究与教学基准，非工业级大规模训练。
- **GPU**：纯 numpy CPU 实现，未做 CUDA 加速；如需扩展可替换 `net.py` 后端并保留接口。
- **CCA 强基线弱**：CCA 在非线性体制下 Recall@1 极低，属预期（线性方法无法捕捉非线性耦合），对比方法的优势在此被放大。

---

## 7. 许可证与署名

MIT License · © 晨星。本仓库所有源码、文档、配置作者署名均为 **晨星**，发布于 GitHub `CJX0712/mmforge`。
