# Model Card · MMForge

> 作者：晨星 · 仓库 `CJX0712/mmforge` · MIT License · 版本 v0.1.0

## 模型细节

- **范式**：跨模态双塔对比学习（CLIP 范式），归一化 InfoNCE 损失。
- **旗舰**：MMFuse = 对比嵌入（手写 MLP 双编码器） ⊕ 全局线性 CCA 加权融合，`alpha` 经网格搜索选定。
- **强基线**：线性 CCA（Hotelling 1936，白化 + SVD）。
- **对照**：IndependentPCA、RandomProjection、LinearContrastive（消融非线性）。
- **参数量**：MLP 编码器 `hidden=(64,64)`，`out_dim=32`，约数千可训参数（CPU 即可训练）。
- **训练**：Adam，lr=1e-2，epochs=100，batch_size=256，τ=0.1。

## 数据

- **类型**：合成配对跨模态数据（受控基准，用于算法正确性/确定性/相对优劣证明）。
- **生成**：潜变量 `z ~ 高斯混合`（z_dim=4，n_classes 类，class_sep 间隔）→ 各模态独立随机映射（tanh 非线性 或 线性） + 高斯噪声（noise=0.15）。
- **规模**：n_train=1300，n_test=320（额外 val 用于 alpha 调参）。
- **两种体制**：`nonlinear`（非线性耦合）、`linear`（线性耦合），覆盖不同难度。
- **切分**：确定性 train/val/test，互不泄漏（单测校验）。

## 评测指标

| 指标 | 含义 |
| --- | --- |
| Recall@1 / Recall@5 | 跨模态检索：给定 x 检索最相似 y 是否命中（双向 x→y / y→x） |
| 1-NN accuracy | 跨模态 1-NN 分类准确率（x→y / x→x） |
| alignment（↓） | 正样本对嵌入距离（Wang & Isola 2020，越低越好） |
| uniformity（↓） | 嵌入分布均匀性（越高越均匀，越接近 0 越好） |

## 结果摘要（3 seeds，mean ± std）

### nonlinear 体制
- **MMFuse** R@1(x→y) = **0.420 ± 0.039**，1-NN = 0.914
- CCA（强基线）R@1(x→y) = 0.068 ± 0.010
- 优势：**~6×**

### linear 体制
- **MMFuse** R@1(x→y) = **0.750 ± 0.043**，R@5 = 0.995
- CCA（强基线）R@1(x→y) = 0.024 ± 0.008
- 优势：**~31×**

### 消融结论
- 非线性贡献：nonlinear 体制下 `contrastive`(0.407) > `linear_contrastive`(0.253) → 非线性编码器确有增益。
- CCA 在非线性体制下 Recall@1 极低（0.068）属预期（线性方法无法捕捉非线性耦合）。

## 使用场景与局限

**适用**：跨模态检索/对齐的教学与研究基准、手写可审计对比学习实现、离线无框架环境。
**局限**：基于合成数据，未接入真实图文/音视频；规模与算力为研究级，非工业级；纯 CPU numpy，未做 GPU 加速。

## 伦理与风险

- 合成数据无隐私/偏见风险。
- 作为研究基准，不应直接用于高风险决策（如医疗/司法跨模态判定）而未经真实数据验证。
