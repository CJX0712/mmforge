# 架构说明 · MMForge

> 作者：晨星 · 仓库 `CJX0712/mmforge` · MIT License

本文档描述 MMForge 的模块职责、数据流与确定性/可复现设计。性能数字见 [../README.md](./README.md)。

## 1. 设计原则

1. **零深度学习框架依赖**：所有前向/反向/优化均为手写 numpy，便于审计、教学与离线运行。
2. **单向无环数据流**：`Config → seed → data → models → eval → summarize`，无回边，保证可复现与可测试。
3. **确定性优先**：单一全局 RNG（PCG64），每次 `set_all(seed)` 重置；`regime×seed` 双层循环下重跑 `max|Δ|=0`。
4. **离线可降级**：`scikit-learn` 缺失时 `IndependentPCA` 自动切纯 numpy SVD，全方法仍可运行。

## 2. 模块职责

### core/
| 文件 | 职责 |
| --- | --- |
| `seed.py` | `set_all(seed)` 设 `random` + `np.random` + 单一 `default_rng`；`get_rng()` / `get_seed()` |
| `errors.py` | `ConfigError` / `DataError` / `ModelError` / `TrainingError` / `EvalError`（E100~E500） |
| `types.py` | `PairDataset`(X, Y, labels, regime, meta) + `BenchmarkRow` dataclass |
| `config.py` | `Config` dataclass；`ENV_MMFORGE_*` 环境变量覆盖；字段校验 |
| `interfaces.py` | `Encoder` / `Aligner` Protocol（统一语义：分数越大越相似） |

### data/
- `synthetic.py`：`generate_pairs(rng, n, z_dim, x_dim, y_dim, coupling, noise, n_classes, class_sep)` 生成配对跨模态数据；`train_test_split` 确定性切分（train/val/test，互不泄漏）。

### multimodal/
| 文件 | 职责 |
| --- | --- |
| `net.py` | 手写 `MLPEncoder`（forward / backward_update / Adam / gradient_check），`l2_normalize` |
| `cca.py` | `CCA`（fit / similarity / encode_x / encode_y），白化 + SVD，正则化 |
| `contrastive.py` | `ContrastiveModel`（fx, gy 两塔，`_infonce`，train/fit，encode_*，similarity） |
| `fuse.py` | `MMFuse` 旗舰：对比 + CCA 融合，`_tune_alpha` 网格搜索 |
| `baselines.py` | `IndependentPCA` / `RandomProjection` / `LinearContrastive` |

### eval/
- `metrics.py`：`recall_at_k` / `recall1` / `recall5` / `one_nn_accuracy_q` / `alignment` / `uniformity`。

### pipeline/
- `pipeline.py`：`run(cfg, regimes, out_path)` 双层循环 `regime × seed`，`set_all` 重置，生成→切分→6 方法→落盘 `benchmark.json`；`_summarize` 聚合 `mean ± std`。

## 3. 数据流（单向无环）

```
            ┌──────────┐
            │  Config  │  (ENV_MMFORGE_* 覆盖 + 校验)
            └────┬─────┘
                 │
            ┌────▼─────┐
            │ seed.set_all(seed)  ← 全局确定性重置
            └────┬─────┘
                 │
            ┌────▼──────────┐
            │ data.generate_pairs → train_test_split  (train/val/test 不泄漏)
            └────┬──────────┘
                 │
   ┌─────────────┼───────────────────────────────────────┐
   │             │ 6 方法并行独立训练/编码                  │
   ▼             ▼                                         ▼
random_proj  independent_pca  cca  linear_contrastive  contrastive  mmfuse
   │             │            │           │                 │          │
   └─────────────┴────────────┴───────────┴─────────────────┴──────────┘
                 │
            ┌────▼─────┐
            │ eval.metrics  (Recall@1/@5 双向, 1-NN, alignment, uniformity)
            └────┬─────┘
                 │
            ┌────▼──────────┐
            │ pipeline._summarize → benchmark.json (mean ± std)
            └───────────────┘
```

## 4. 确定性保证

- 每个 `(regime, seed)` 组合开始时调用 `set_all(seed)`，重置 `random` / `np.random` / `default_rng`。
- 数据切分使用确定性索引（基于 RNG 的洗牌种子固定），train/test 无样本泄漏（单测 `test_data` 校验）。
- demo 重跑两次取 `max|Δ|`，实测 **= 0**（逐位一致）。

## 5. 梯度正确性

`MLPEncoder.gradient_check` 双校验：
1. 归一化 Jacobian 解析 vs 数值（中心差分），rel ≈ 6.4e-9。
2. 权重有限差分反向核对。

该自检作为单测 `test_net` 的一部分，确保手写反向传播正确（而非仅前向可用）。

## 6. 性能预算

单轮 demo（3 seeds × 2 regimes，6 方法）≈ **42s**（CPU，纯 numpy），低于 60s 交付预算。
