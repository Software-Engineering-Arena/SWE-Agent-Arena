FROM node:20-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
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

# Install Kimi Code CLI via official installer
RUN curl -fsSL code.kimi.com/install.sh | bash
RUN ls -la /root/.local/bin/ && which kimi || echo "kimi NOT FOUND after install"
RUN ln -sf /root/.local/bin/kimi /usr/local/bin/kimi
ENV PATH="/root/.local/bin:${PATH}"

# Configure Qwen Code to use OPENROUTER_API_KEY (set as HF Space secret at runtime)
RUN mkdir -p /root/.qwen && echo '{"security":{"auth":{"selectedType":"openai"}},"model":{"name":"qwen/qwen3-coder-plus"},"modelProviders":{"openai":[{"id":"qwen/qwen3-coder-plus","name":"Qwen3-Coder-Plus via OpenRouter","envKey":"OPENROUTER_API_KEY","baseUrl":"https://openrouter.ai/api/v1"}]}}' > /root/.qwen/settings.json

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY . .

EXPOSE 7860
CMD ["node", "app.js"]
