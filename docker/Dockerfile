# Team image: one pinned environment for development, evaluation and the final submission run.
# Code and data are bind-mounted at runtime, never baked in -- the data is not redistributable.
FROM python:3.11.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MTAH_DATA_DIR=/workspace/data

# libgomp1: OpenMP runtime for lightgbm / scikit-learn. ffmpeg: video inspection and frame export.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.in docker/requirements.lock.tx[t] /tmp/req/
RUN if [ -f /tmp/req/requirements.lock.txt ]; then \
        pip install -r /tmp/req/requirements.lock.txt; \
    else \
        pip install -r /tmp/req/requirements.in; \
    fi

WORKDIR /workspace
CMD ["bash"]
