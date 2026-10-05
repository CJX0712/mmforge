# Changelog · MMForge

本文件遵循 [Keep a Changelog](https://keepachangelog.com/) 约定，版本号遵循 [SemVer](https://semver.org/lang/zh-CN/)。

## [0.1.0] — 2026-10-06

### Added
- 手写 MLP 双编码器（`multimodal/net.py`）：`forward` / `backward_update` / Adam / `gradient_check`（解析 vs 数值，rel=6.4e-9 PASS）。
- 归一化 InfoNCE 对比模型（`multimodal/contrastive.py`）：双向 x→y / y→x 损失，温度 τ=0.1。
- 线性 CCA 基线（`multimodal/cca.py`）：白化 + SVD，纯 numpy。
- MMFuse 旗舰融合（`multimodal/fuse.py`）：对比嵌入 ⊕ 全局 CCA 加权，`alpha` 网格搜索 {0, 0.25, 0.5, 0.75, 1.0}。
- 对照基线（`multimodal/baselines.py`）：IndependentPCA（sklearn + 纯 numpy SVD 离线兜底）、RandomProjection、LinearContrastive（消融非线性）。
- 评测指标（`eval/metrics.py`）：Recall@1/@5（双向）、1-NN 跨模态分类准确率、alignment / uniformity（Wang & Isola 2020）。
- 确定性内核（`core/seed.py`）：全局 seed，`regime×seed` 双层循环下 `max|Δ|=0`。
- 端到端流水线（`pipeline/pipeline.py`）与 CLI（`cli.py`）、demo（`examples/run_demo.py`）。
- 23 项 pytest 单测（覆盖率 86%），`ruff` 硬门禁（0 错误）。
- CI（`.github/workflows/ci.yml`，py3.12/3.13 矩阵）、`Dockerfile`、`Makefile`、`requirements.lock.txt`。

### Quality
- 质量等级 **S**：四项 DoD 全绿。
- MMFuse vs CCA：nonlinear R@1 0.420 vs 0.068（~6×），linear R@1 0.750 vs 0.024（~31×）。

---

## [Unreleased]
- 接入真实图文/音视频模态（替换 `data/`）。
- GPU/向量化后端（替换 `net.py`，保留接口）。
