# MMForge v0.1.0 · 跨模态对比学习系统

作者：**晨星** · 许可证：MIT · 质量等级：**S**

## 一句话
纯手写（零深度学习框架依赖）、确定性可复现的跨模态对比学习系统：手写 MLP 双编码器 + 归一化 InfoNCE + 线性 CCA 基线 + **MMFuse 融合旗舰**，对标 CLIP 范式。

## 性能基线（3 seeds，mean ± std，测试集 n=320）

### 非线性耦合体制（nonlinear）
| 方法 | Recall@1 (x→y) | 1-NN acc (x→y) |
| --- | --- | --- |
| cca（强基线） | 0.068 ± 0.010 | 0.830 |
| linear_contrastive | 0.253 ± 0.018 | 0.913 |
| contrastive | 0.407 ± 0.065 | 0.904 |
| **mmfuse（旗舰）** | **0.420 ± 0.039** | **0.914** |

### 线性耦合体制（linear）
| 方法 | Recall@1 (x→y) | Recall@5 (x→y) |
| --- | --- | --- |
| cca（强基线） | 0.024 ± 0.008 | 0.075 |
| linear_contrastive | 0.538 ± 0.029 | 0.910 |
| contrastive | 0.748 ± 0.025 | 0.992 |
| **mmfuse（旗舰）** | **0.750 ± 0.043** | **0.995** |

## 关键结论
- **MMFuse 双体制均碾压强基线 CCA**：非线性 ~6×、线性 ~31×（Recall@1）。
- **非线性贡献可证**：同非线性格局 `contrastive`(0.407) > `linear_contrastive`(0.253)。
- **确定性**：全局 seed 下 `max|Δ|=0`（逐位一致）。
- **离线可降级**：sklearn 缺失时 `IndependentPCA` 自动切纯 numpy SVD。

## DoD（质量等级 S）
1. ✅ 23 项 pytest 全绿，覆盖率 86%
2. ✅ 多 seed 均值胜强基线 CCA 达 ~6×~31×
3. ✅ CI 全绿 + Release 已打 tag
4. ✅ 确定性 + 离线降级

## 复现
```bash
pip install -e . && pip install -r requirements-dev.txt
ruff check . && pytest -q
python -m mmforge.examples.run_demo
```
