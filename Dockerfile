FROM node:24-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git config --global user.name "arena" \
    && git config --global user.email "arena@localhost" \
    && git config --global init.defaultBranch main

# Install Node.js-based agent CLIs globally in a single layer
RUN npm install -g \
    @anthropic-ai/claude-code \
    @google/gemini-cli \
    @vibe-kit/grok-cli \
    @openai/codex \
    @qwen-code/qwen-code \
    @moonshot-ai/kimi-code \
    mmx-cli

# Install Cursor CLI (installs agent symlink to ~/.local/bin)
RUN curl https://cursor.com/install -fsS | bash
ENV PATH="/root/.local/bin:${PATH}"

# Configure Qwen Code to use OPENROUTER_API_KEY (set as HF Space secret at runtime)
RUN mkdir -p /root/.qwen && echo '{"security":{"auth":{"selectedType":"openai"}},"model":{"name":"qwen/qwen3-coder-plus"},"modelProviders":{"openai":[{"id":"qwen/qwen3-coder-plus","name":"Qwen3-Coder-Plus via OpenRouter","envKey":"OPENROUTER_API_KEY","baseUrl":"https://openrouter.ai/api/v1"}]}}' > /root/.qwen/settings.json

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY . .

EXPOSE 7860
CMD ["sh", "-c", "\
  mkdir -p /root/.kimi-code && \
  printf 'default_model = \"kimi-code/kimi-for-coding\"\\ndefault_permission_mode = \"auto\"\\n\\n[providers.kimi-code]\\ntype = \"openai\"\\nbase_url = \"https://openrouter.ai/api/v1\"\\napi_key = \"%s\"\\n\\n[models.\"kimi-code/kimi-for-coding\"]\\nprovider = \"kimi-code\"\\nmodel = \"moonshotai/kimi-k2\"\\nmax_context_size = 262144\\n' \"$OPENROUTER_API_KEY\" > /root/.kimi-code/config.toml && \
  echo 'Kimi Code config written.' && \
  if [ -n \"$MINIMAX_API_KEY\" ]; then \
    if mmx auth login --api-key \"$MINIMAX_API_KEY\" --quiet --no-color --non-interactive; then \
      echo 'MiniMax CLI auth configured.'; \
    else \
      echo 'MiniMax CLI auth failed; continuing without MiniMax auth.'; \
    fi; \
  else \
    echo 'MINIMAX_API_KEY not set; skipping MiniMax CLI auth.'; \
  fi && \
  exec node --no-deprecation app.js"]
