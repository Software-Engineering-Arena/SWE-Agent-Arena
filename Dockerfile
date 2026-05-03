FROM node:20-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js-based agent CLIs globally in a single layer
RUN npm install -g \
    @anthropic-ai/claude-code \
    @google/gemini-cli \
    @vibe-kit/grok-cli \
    @openai/codex \
    @qwen-code/qwen-code

# Install Kimi Code CLI via official installer
RUN curl -L code.kimi.com/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"

# Configure Qwen Code to use OPENROUTER_API_KEY (set as HF Space secret at runtime)
RUN mkdir -p /root/.qwen && echo '{"security":{"auth":{"selectedType":"openai"}},"model":{"name":"qwen/qwen3-coder"},"modelProviders":{"openai":[{"id":"openrouter-qwen","name":"Qwen3-Coder via OpenRouter","envKey":"OPENROUTER_API_KEY","baseUrl":"https://openrouter.ai/api/v1"}]}}' > /root/.qwen/settings.json

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY . .

EXPOSE 7860
CMD ["node", "app.js"]
