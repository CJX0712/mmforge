# 已交付记录（MMForge）

> 本文件为 SOP 强制回写的精简交付日志。完整记录见桌面任务书 §11 与 README.md。

| 系统 | 域 | 日期 | tag | 关键指标（含门槛与通过） |
|------|----|------|-----|--------------------------|
| MMForge | 跨模态对比学习（手写 MLP 双编码器 + 归一化 InfoNCE + 线性 CCA 基线 + MMFuse 融合旗舰；sklearn 强基线 + 纯 numpy SVD 离线兜底；numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1） | 2026-10-06 | v0.1.0 | 旗舰 MMFuse vs 强基线 CCA：Recall@1 非线性 ~6×（0.420±0.039 vs 0.068±0.010）、线性 ~31×（0.750±0.043 vs 0.024±0.008）（3 seeds mean±std，≥预设阈值 ✅ PASS）；非线性贡献可证（contrastive 0.407 > linear_contrastive 0.253）；确定性 max|Δ|=0 逐位一致；23 单测全绿 / 覆盖率 86% / ruff 0.16.10 双绿硬门禁 / CI 矩阵 py3.12+3.13 / demo ~42s≤60s；离线可降级（sklearn 缺失自动切纯 numpy SVD）；质量等级 **S**。仓库 https://github.com/CJX0712/mmforge Release v0.1.0 |
