# 踩坑记录 · MMForge 交付（2026-10-06，作者：晨星）

> 本文件沉淀本次交付过程中暴露的真实问题（symptom / root / fix），供后续 Forge 系列复用。

## P1 推送：git push 走代理 502，且 gh 预建仓导致 ref 409
- **symptom**：`git push` 报 `CONNECT tunnel failed, response 502`（本地坏代理拦截智能协议隧道）；改用 gh api Git Data API（L2）又报 `HTTP 409 Conflict`。
- **root**：(a) 沙箱 `git` 走 HTTP 代理 CONNECT 到 github.com:443 被 502 拦截；(b) 脚本先用 `gh repo create` 建仓，GH 已生成 `main` 分支与默认 README，随后 L2 `POST /git/refs/heads/main` 因 ref 已存在而 409；(c) `api_req` 把 404 当可重试错误吞掉，导致 `repo_exists` 拿不到 404 判定。
- **fix**：(1) `api_req` 改为 4xx（除 409/422）立即 re-raise，5xx 才重试，使 `repo_exists` 正确识别 404；(2) L2 先 `GET /git/ref/heads/main` 取父提交 sha（空仓则 parents=[]），再 `POST` 创建 ref，遇 409/422 改 `PATCH` 更新 ref。最终 L2 推送成功。

## P2 梯度自检初始 rel=1.0（退化对照）
- **symptom**：`gradient_check` 首次返回 rel≈1.0，等价于"对照与数值梯度毫无关系"。
- **root**：原实现误用 `grad_emb = f0`（前向输出本身）作为解析梯度对照，属于无意义退化设计。
- **fix**：重写为双校验——(a) 归一化 Jacobian 解析 vs 中心差分数值；(b) 权重有限差分反向核对。修正后 rel=6.4e-9 PASS。同时修掉 `_compute_grads` 返回 2 值却按 3 值解包的 `Unpack` 错误。

## P3 pytest ModuleNotFoundError: mmforge
- **symptom**：`pip install -e .` 后 `pytest` 仍报找不到 `mmforge`。
- **root**：仓库根即包目录，最初用 `packages=["mmforge", ...]` 失败（package dir 不存在）；后改用 `package-dir`，但 TOML 内联表误用 `:` 而非 `=`（tomllib 报错）。
- **fix**：`[tool.setuptools] package-dir = { mmforge = "." }`，并显式列出子包 `packages`。

## P4 ruff 中文全角标点误报（RUF002/003）
- **symptom**：大量 RUF002（全角标点）误报，掩盖真实问题。
- **root**：中文文档/注释使用全角标点，触发 RUF002/003 规则。
- **fix**：select 仅保留 `RUF100`（未用 noqa 指令），移除 RUF002/003；真错误用 `--fix` 自动修 + 少量手修。最终 0 错误硬门禁。

## P5 VENV 路径变量 bug
- **symptom**：后台装 ruff 时报路径拼接错误。
- **root**：用 `$VENV/Scripts/pip.exe` 但 `VENV` 存的是 `python.exe` 路径。
- **fix**：区分 `VENVDIR`（目录）与 `PYEXE`（解释器），统一用 `VENVDIR/Scripts/...`。

## P6 benchmark 聚合字段不足
- **symptom**：`benchmark.json` 的 `summary` 仅含 `recall1_x2y` 与 `acc_x2y`（1nn），README 需要的 Recall@5、y→x、alignment、uniformity 缺失。
- **root**：`_summarize` 聚合口径过窄。
- **fix**：从 `rows` 重新聚合全部指标（Recall@1/@5 双向、1-NN 双向、alignment、uniformity）生成完整基线表，README/CHANGELOG 使用完整数据。

## 复用建议（给后续 Forge）
- `gh_push.py` 三级降级已稳定（L1 git → L2 Git Data API → L3 Contents API）；**建仓与推送分离**，L2 必须处理"仓已含 main 分支"的 409 场景。
- 手写反向传播务必做**双校验梯度自检**，否则容易像 P2 一样"看起来合理实则全错"。
- 中文项目 ruff 配置直接 `select=["...","RUF100"]` 避开 RUF002/003 噪声。
- 仓库根即包目录时，`package-dir = { name = "." }` 用 `=` 而非 `:`。
