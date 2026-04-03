FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    GLAMA_VERSION="1.0.0" \
    PYTHONUNBUFFERED=1

# Install Node.js (for mcp-proxy) and Python/uv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g mcp-proxy@6.4.3 \
    && node --version \
    && curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="/usr/local/bin" sh \
    && uv python install 3.12 --default \
    && ln -s $(uv python find) /usr/local/bin/python \
    && python --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app
COPY . .
RUN uv sync --extra all

CMD ["mcp-proxy","--","/app/.venv/bin/humane-proxy","mcp-serve"]

