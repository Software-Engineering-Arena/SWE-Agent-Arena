FROM node:22-slim

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
    @qwen-code/qwen-code

# Install Kimi Code CLI via pre-built single binary
RUN curl -fsSL \
    "https://github.com/MoonshotAI/kimi-cli/releases/download/1.41.0/kimi-1.41.0-x86_64-unknown-linux-gnu.tar.gz" \
    -o /tmp/kimi.tar.gz \
  && tar -tzf /tmp/kimi.tar.gz \
  && tar -xzf /tmp/kimi.tar.gz -C /tmp \
  && find /tmp -maxdepth 3 -name "kimi" -type f -exec mv {} /usr/local/bin/kimi \; \
  && chmod +x /usr/local/bin/kimi \
  && rm -f /tmp/kimi.tar.gz

# Configure Qwen Code to use OPENROUTER_API_KEY (set as HF Space secret at runtime)
RUN mkdir -p /root/.qwen && echo '{"security":{"auth":{"selectedType":"openai"}},"model":{"name":"qwen/qwen3-coder-plus"},"modelProviders":{"openai":[{"id":"qwen/qwen3-coder-plus","name":"Qwen3-Coder-Plus via OpenRouter","envKey":"OPENROUTER_API_KEY","baseUrl":"https://openrouter.ai/api/v1"}]}}' > /root/.qwen/settings.json

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY . .

EXPOSE 7860
CMD ["sh", "-c", "mkdir -p /root/.kimi && printf 'default_model = \"kimi\"\\n\\n[providers.kimi]\\ntype = \"openai\"\\nbase_url = \"https://openrouter.ai/api/v1\"\\napi_key = \"%s\"\\n\\n[models.kimi]\\nprovider = \"kimi\"\\nmodel = \"~moonshotai/kimi-latest\"\\n' \"$OPENROUTER_API_KEY\" > /root/.kimi/config.toml && echo 'Kimi config written.' && exec node app.js"]
