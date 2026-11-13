# ============================================================================
# VoiceLab AI — Zero-Cost Production Dockerfile
# Compatible with Free-Tier Container Hosts (Render Free, Hugging Face Spaces
# Free 2 vCPU / 16 GB RAM Docker Tier, Railway Trial, or Self-Hosted Docker)
# ============================================================================
FROM node:22-bookworm-slim

# Install FFmpeg & FFprobe for real acoustic DSP, pitch/formant analysis, and WAV/MP3 encoding
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY package.json ./
RUN npm install --legacy-peer-deps --no-audit --no-fund

# Copy application source
COPY . .

# Build optimized Vite frontend bundle into dist/
RUN npm run build

# Default production environment
ENV NODE_ENV=production
ENV PORT=3000

EXPOSE 3000

# Run the production Express server natively on Node 22
CMD ["npm", "run", "start"]
