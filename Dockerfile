# syntax=docker/dockerfile:1.7
#
# Rorschach + lichess-bot, single image. Multi-stage to keep runtime lean.
#
# Build:   docker build -t rorschach .
# Run:     see compose.yaml

ARG PYTHON_VERSION=3.11
ARG LICHESS_BOT_REF=master

# ---------- builder ----------
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# git is needed because maia3 is a git+https dependency.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/rorschach

# Resolve deps first for layer caching; project install happens after sources copy.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY rorschach ./rorschach
COPY bin ./bin
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# lichess-bot lives next to us; install its deps into the same venv.
ARG LICHESS_BOT_REF
RUN git clone --depth 1 --branch "${LICHESS_BOT_REF}" \
        https://github.com/lichess-bot-devs/lichess-bot.git /opt/lichess-bot \
 && rm -rf /opt/lichess-bot/.git
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/rorschach/.venv/bin/python \
        -r /opt/lichess-bot/requirements.txt

# ---------- runtime ----------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/rorschach/.venv/bin:${PATH}" \
    HF_HOME=/data/huggingface

# libgomp1: torch CPU wheel pulls libgomp at runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/rorschach /opt/rorschach
COPY --from=builder /opt/lichess-bot /opt/lichess-bot

RUN chmod +x /opt/rorschach/bin/patricia \
 && mkdir -p /data/huggingface

WORKDIR /opt/lichess-bot

# Mount your config.yml here (see configs/lichess-bot.docker.yml.example).
# Provide LICHESS_TOKEN via env for the opening explorer.
CMD ["python", "/opt/lichess-bot/lichess-bot.py"]
