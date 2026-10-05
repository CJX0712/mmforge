#!/usr/bin/env python3
"""
gh_push.py — MMForge 三级降级推送脚本（作者：晨星）
将本地仓库推送到 GitHub（默认 CJX0712/<repo>），并打 tag + 创建 Release。

三级降级：
  L1  git CLI        : git init / add / commit / push（最可靠，优先）
  L2  Git Data API   : blobs -> tree -> commit -> ref（无 git 时）
  L3  Contents API   : 逐文件 create/update（L2 仍失败时）

用法：
  python scripts/gh_push.py --repo mmforge --local . \
      --description "MMForge · 跨模态对比学习系统" \
      --tag v0.1.0 --release "<benchmark 摘要>"
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"

# 跳过的目录/文件（.git 永远跳过；其余与 .gitignore 对齐）
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules",
             ".pytest_cache", ".ruff_cache", ".mypy_cache", ".eggs",
             "build", "dist", ".coverage", "*.egg-info"}
SKIP_SUFFIX = (".pyc", ".pyo", ".pt", ".ckpt", ".onnx")
SKIP_NAMES = {".DS_Store", "benchmark_determinism.json"}


def log(msg: str) -> None:
    print(f"[gh_push] {msg}", flush=True)


def run(cmd, check=True, cwd=None):
    log("exec: " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\n{r.stderr}")
    return r


def get_token() -> str:
    # 优先用 gh 已登录的 token
    try:
        r = run(["gh", "auth", "token"], check=False)
        t = r.stdout.strip()
        if t:
            return t
    except Exception:
        pass
    t = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not t:
        raise RuntimeError("未找到 GitHub token：请先 `gh auth login` 或设置 GH_TOKEN")
    return t


def get_owner(token: str) -> str:
    data = api_get("/user", token)
    return data["login"]


def api_req(method: str, path: str, token: str, body=None, retries=3):
    url = API + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "mmforge-gh-push",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    last = None
    for i in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (409, 422):  # 冲突/已存在，交给调用方判断
                raise
            if 500 <= e.code < 600:  # 服务端错误可重试
                detail = e.read().decode("utf-8", "replace")
                log(f"HTTP {e.code} on {method} {path}: {detail[:300]} (retry)")
                time.sleep(2 ** i)
                continue
            raise  # 其余 4xx（如 404）直接抛出，由调用方决定
        except Exception as e:
            last = e
            log(f"network error on {method} {path}: {e}")
            time.sleep(2 ** i)
    raise RuntimeError(f"api {method} {path} failed: {last}")


def api_get(path, token):
    return api_req("GET", path, token)


def api_post(path, token, body):
    return api_req("POST", path, token, body)


def api_put(path, token, body):
    return api_req("PUT", path, token, body)


def api_patch(path, token, body):
    return api_req("PATCH", path, token, body)


def repo_exists(full: str, token: str) -> bool:
    try:
        api_get(f"/repos/{full}", token)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def create_repo(name: str, token: str, description: str) -> None:
    # 优先用 gh（可正确处理默认分支/可见性）
    r = run(["gh", "repo", "create", name, "--public",
             "--description", description, "--confirm"], check=False)
    if r.returncode == 0:
        log(f"gh repo create {name} OK")
        return
    log(f"gh repo create 失败({r.returncode})，改用 API")
    try:
        api_post("/user/repos", token,
                 {"name": name, "description": description, "public": True,
                  "auto_init": False})
        log("API create repo OK")
    except urllib.error.HTTPError as e:
        if e.code == 422:
            log("repo 已存在（422），继续")
        else:
            raise


def collect_files(local: Path) -> list[Path]:
    out = []
    for p in local.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(local)
        parts = set(rel.parts)
        if parts & SKIP_DIRS:
            continue
        if rel.name in SKIP_NAMES or rel.name in SKIP_DIRS:
            continue
        if rel.suffix in SKIP_SUFFIX:
            continue
        out.append(p)
    out.sort()
    return out


# ---------------- L1: git CLI ----------------
def l1_git(local: Path, full: str, token: str) -> bool:
    try:
        if not (local / ".git").exists():
            run(["git", "init", "-q"], cwd=str(local))
        run(["git", "config", "user.email", "bot@mmforge.local"], cwd=str(local))
        run(["git", "config", "user.name", "晨星"], cwd=str(local))
        run(["git", "add", "-A"], cwd=str(local))
        # 提交（若无变化则跳过）
        st = run(["git", "status", "--porcelain"], cwd=str(local), check=False)
        if st.stdout.strip():
            run(["git", "commit", "-q", "-m", "MMForge v0.1.0 · 跨模态对比学习系统（作者：晨星）"],
                cwd=str(local))
        else:
            log("L1: 无文件变更，跳过提交")
        run(["git", "branch", "-M", "main"], cwd=str(local))
        remote_url = f"https://{token}@github.com/{full}.git"
        # 设置或替换 origin
        with contextlib.suppress(Exception):
            run(["git", "remote", "remove", "origin"], cwd=str(local), check=False)
        run(["git", "remote", "add", "origin", remote_url], cwd=str(local))
        run(["git", "push", "--force", "-u", "origin", "main"], cwd=str(local))
        log("L1 git push 成功")
        return True
    except Exception as e:
        log(f"L1 git 失败：{e}")
        return False


# ---------------- L2: Git Data API ----------------
def l2_gitdata(local: Path, full: str, files: list[Path], token: str) -> bool:
    try:
        log("L2: Git Data API 推送")
        blobs = {}
        for p in files:
            rel = str(p.relative_to(local)).replace(os.sep, "/")
            content = p.read_bytes()
            b64 = base64.b64encode(content).decode("ascii")
            res = api_post(f"/repos/{full}/git/blobs", token,
                           {"content": b64, "encoding": "base64"})
            blobs[rel] = res["sha"]
        tree = [{"path": rel, "mode": "100644", "type": "blob", "sha": sha}
                for rel, sha in blobs.items()]
        tree_res = api_post(f"/repos/{full}/git/trees", token, {"tree": tree})
        tree_sha = tree_res["sha"]
        commit = api_post(f"/repos/{full}/git/commits", token,
                          {"message": "MMForge v0.1.0 · 跨模态对比学习系统（作者：晨星）",
                           "tree": tree_sha, "parents": []})
        commit_sha = commit["sha"]
        # 创建或更新 main 分支引用
        try:
            api_post(f"/repos/{full}/git/refs", token,
                     {"ref": "refs/heads/main", "sha": commit_sha})
        except urllib.error.HTTPError as e:
            if e.code == 422:  # ref 已存在，更新
                api_patch(f"/repos/{full}/git/refs/heads/main", token,
                          {"sha": commit_sha})
            else:
                raise
        log("L2 Git Data API 推送成功")
        return True
    except Exception as e:
        log(f"L2 失败：{e}")
        return False


# ---------------- L3: Contents API ----------------
def l3_contents(local: Path, full: str, files: list[Path], token: str) -> bool:
    try:
        log("L3: Contents API 逐文件推送")
        ok = 0
        for p in files:
            rel = str(p.relative_to(local)).replace(os.sep, "/")
            content = base64.b64encode(p.read_bytes()).decode("ascii")
            # 检查是否已存在
            sha = None
            try:
                existing = api_get(f"/repos/{full}/contents/{rel}", token)
                sha = existing.get("sha")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
            body = {"message": f"add {rel} (MMForge)", "content": content,
                    "branch": "main"}
            if sha:
                body["sha"] = sha
            if sha:
                api_put(f"/repos/{full}/contents/{rel}", token, body)
            else:
                api_post(f"/repos/{full}/contents/{rel}", token, body)
            ok += 1
        log(f"L3 完成，写入 {ok}/{len(files)} 个文件")
        return ok > 0
    except Exception as e:
        log(f"L3 失败：{e}")
        return False


# ---------------- tag + release ----------------
def tag_and_release(full: str, tag: str, release: str, token: str) -> None:
    try:
        # 取默认分支头
        repo = api_get(f"/repos/{full}", token)
        branch = repo.get("default_branch", "main")
        ref = api_get(f"/repos/{full}/git/ref/heads/{branch}", token)
        head_sha = ref["object"]["sha"]
        # 创建 tag ref
        try:
            api_post(f"/repos/{full}/git/refs", token,
                     {"ref": f"refs/tags/{tag}", "sha": head_sha})
            log(f"tag {tag} 创建成功")
        except urllib.error.HTTPError as e:
            if e.code == 422:
                log(f"tag {tag} 已存在，跳过")
            else:
                raise
        # 创建 release
        try:
            api_post(f"/repos/{full}/releases", token,
                     {"tag_name": tag, "name": tag, "body": release,
                      "draft": False, "prerelease": False})
            log(f"Release {tag} 创建成功")
        except urllib.error.HTTPError as e:
            if e.code == 422:
                log(f"Release {tag} 已存在，跳过")
            else:
                raise
    except Exception as e:
        log(f"tag/release 步骤异常（非致命）：{e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="MMForge 三级降级推送")
    ap.add_argument("--repo", required=True, help="仓库名（如 mmforge）")
    ap.add_argument("--local", default=".", help="本地仓库路径")
    ap.add_argument("--description", default="MMForge · 跨模态对比学习系统")
    ap.add_argument("--tag", default="v0.1.0")
    ap.add_argument("--release", default="")
    ap.add_argument("--release-file", default="", help="从文件读取 Release 说明（优先于 --release）")
    ap.add_argument("--no-release", action="store_true")
    args = ap.parse_args()

    release = args.release
    if args.release_file and os.path.isfile(args.release_file):
        release = Path(args.release_file).read_text(encoding="utf-8")

    local = Path(args.local).resolve()
    token = get_token()
    owner = get_owner(token)
    full = f"{owner}/{args.repo}"
    log(f"目标仓库：{full}")

    if not repo_exists(full, token):
        create_repo(args.repo, token, args.description)
    else:
        log("仓库已存在，直接推送")

    files = collect_files(local)
    log(f"待推送文件数：{len(files)}")

    pushed = False
    for level, fn in (
        ("L1-git", lambda: l1_git(local, full, token)),
        ("L2-gitdata", lambda: l2_gitdata(local, full, files, token)),
        ("L3-contents", lambda: l3_contents(local, full, files, token)),
    ):
        log(f"尝试 {level} ...")
        if fn():
            pushed = True
            break
        else:
            log(f"{level} 失败，降级到下一級")

    if not pushed:
        log("三级全部失败，终止")
        return 1

    if not args.no_release:
        tag_and_release(full, args.tag, args.release, token)

    log(f"完成。仓库地址：https://github.com/{full}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
