FROM node:20-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install agent CLIs globally in a single layer
RUN npm install -g \
    @anthropic-ai/claude-code \
    @augmentcode/auggie \
    @google/gemini-cli \
    @openai/codex \
    # @vibe-kit/grok-cli \

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY . .

EXPOSE 7860
CMD ["node", "app.js"]
