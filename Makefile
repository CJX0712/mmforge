# MMForge Makefile · 作者：晨星
# 用法： make <target>
PYTHON ?= python
VENV ?= .venv

.PHONY: help install dev lint test demo doctor benchmark clean

help:
	@echo "MMForge targets:"
	@echo "  install  创建 venv 并安装运行依赖"
	@echo "  dev      安装开发依赖（ruff/pytest）"
	@echo "  lint     运行 ruff 静态检查"
	@echo "  test     运行 pytest（含覆盖率）"
	@echo "  demo     运行端到端 demo（确定性自检）"
	@echo "  doctor   环境自检"
	@echo "  benchmark 运行流水线并落盘 benchmark.json"
	@echo "  clean    清理缓存"

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/pip.exe install -r requirements.txt
	$(VENV)/Scripts/pip.exe install -e .

dev:
	$(VENV)/Scripts/pip.exe install -r requirements-dev.txt

lint:
	$(VENV)/Scripts/python.exe -m ruff check .

test:
	$(VENV)/Scripts/python.exe -m pytest -q --cov=mmforge

demo:
	$(VENV)/Scripts/python.exe -m mmforge.examples.run_demo

doctor:
	$(VENV)/Scripts/python.exe -m mmforge.cli --doctor

benchmark:
	$(VENV)/Scripts/python.exe -m mmforge.cli --regimes nonlinear linear --seeds 3 --out benchmark.json

clean:
	-rm -rf .ruff_cache .pytest_cache .coverage build dist *.egg-info
	-find . -type d -name __pycache__ -exec rm -rf {} +
