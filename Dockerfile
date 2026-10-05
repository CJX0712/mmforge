# MMForge · 跨模态对比学习系统
# 纯 numpy 实现，CPU 即可运行。作者：晨星 · MIT License
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 先装依赖以利用层缓存
COPY requirements.lock.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# 安装项目（仓库根即包目录）
COPY . .
RUN pip install --no-cache-dir -e .

# 默认运行端到端 demo（含确定性自检）
CMD ["python", "-m", "mmforge.examples.run_demo"]
