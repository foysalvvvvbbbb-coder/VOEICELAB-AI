# VoiceLab AI (v3.3.0)

**Same-Language AI Voice Cloning Studio (`bn`, `hi`, `en`) & Universal Multilingual Audio-to-SRT Workstation**

VoiceLab AI is a full-stack TypeScript, React 19, Express 4, and FFmpeg DSP web application engineered for zero-cost (`৳0`) operation and strict audio/subtitle integrity.

---

## Core Features

1. **Same-Language Voice Cloning Studio (Bengali, Hindi, English)**
   - **Strict Same-Language Enforcement**: Validates Unicode character ranges (`U+0980–U+09FF` for Bengali, `U+0900–U+097F` for Hindi Devanagari, and Latin for English) while permitting shared Indic Danda punctuation (`।`, `॥`). Blocks cross-language synthesis.
   - **Mode 1 — Cloud Persona-Conditioned TTS + FFmpeg Acoustic Transfer**: Extracts fundamental frequency ($F_0$), harmonic warmth, and a 16-D spectral envelope via FFmpeg/FFprobe DSP, conditions Gemini 3.8 TTS on the closest vocal register persona, applies bounded pitch/spectral transfer (`±3.5` semitones max), and reports separated speaker, spectral, and ASR back-check metrics.
   - **Mode 2 — Local Neural Zero-Shot Node (`127.0.0.1:8001`)**: Connects strictly to a local loopback FastAPI node (`scripts/local_neural_node.py`) running **AI4Bharat IndicF5** (Bengali & Hindi), **OpenVoice V2** (English), and **SpeechBrain ECAPA-TDNN** speaker verification. Never silently falls back to Cloud TTS when Mode 2 is selected.
2. **Universal Multilingual Audio-to-SRT Generator**
   - Transcribes real uploaded audio files (`WAV`, `MP3`, `M4A`, `FLAC`, `OGG`, `AAC`, `WEBM` up to 50 MB) in Bengali, Hindi, English, Arabic, and 15+ languages.
   - **Exact Ordered Token Sequence Verification (`compareTokenSequenceIntegrity`)**: Computes a Longest Common Subsequence (LCS) dynamic-programming alignment between the recognized source transcript and combined SRT output to detect missing, duplicated, reordered, inserted, or normalization-mismatched tokens.
   - **Natural Subtitle Segmentation**: Aligns subtitle boundaries to word-level ASR timestamps and 20ms PCM acoustic pauses while enforcing language-aware line limits (`<= 38` chars/line for Bengali/Hindi/Arabic, `<= 42` chars/line for English/Latin, max 2 lines/subtitle).

---

## Architecture & Same-Origin Routing

- **Frontend**: React 19 + TypeScript + Tailwind CSS v4 + Vite (`src/`), compiled into `dist/`.
- **Backend**: Express 4 running natively on Node.js 22 (`server.ts`, `src/server/aiPipelines.ts`, `src/server/audioDsp.ts`).
- **Same-Origin `/api/*` Routing**: Both frontend assets and `/api/*` endpoints are served from the same Express origin. `src/utils/apiClient.ts` uses relative `/api/*` paths by default (`VITE_API_BASE_URL=""`), and `server.ts` enforces a strict JSON 404 barrier on `/api/*` before the SPA fallback.

---

## Installation, Development & Production Build

### Prerequisites
- **Node.js**: `v22.x` or later
- **FFmpeg & FFprobe**: Installed on system `PATH` (`ffmpeg -version`)

### 1. Local Development
```bash
# Install dependencies
npm install

# Create local environment file from template
cp .env.example .env
# Set GEMINI_API_KEY in .env

# Start full-stack development server (Express + Vite middleware on http://localhost:3000)
npm run dev
```

### 2. TypeScript Check & Production Build
```bash
# Verify TypeScript types (0 errors required)
npm run lint

# Build optimized frontend bundle into dist/
npm run build

# Start production Express server serving dist/ and /api/* on port 3000
npm run start:prod
```

---

## Environment Variables

| Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `GEMINI_API_KEY` | **Yes** (for Cloud Mode & ASR) | *Auto-injected in AI Studio* | Google Gemini API key (server-side only; never exposed to client). |
| `APP_URL` | Optional | `http://localhost:3000` | Public HTTPS origin allowed by the strict CORS policy. |
| `PORT` | Optional | `3000` | HTTP port for Express (`server.ts` respects `DEFAULT_APP_PORT=3000` inside AI Studio containers). |
| `NODE_ENV` | Optional | `development` | Set to `production` to serve static compiled assets from `dist/`. |
| `SESSION_SECRET` | Optional | *Auto-generated 256-bit CSPRNG* | 32+ byte secret used to sign `HttpOnly` `voicelab_sid` session cookies with `HMAC-SHA256`. |
| `VITE_API_BASE_URL` | Optional | `""` (empty) | Leave empty (`""`) for same-origin `/api/*` routing. |

---

## Private GitHub Repository Setup & CI Workflow

### Option A — Export Directly from Google AI Studio UI
1. Click the **Export to GitHub** (or **Sync to GitHub**) icon in the top-right toolbar of Google AI Studio.
2. Authorize your GitHub account when prompted.
3. Enter repository name **`voicelab-ai`** and select **Private** visibility.
4. Click **Create & Push** to push the codebase to your private repository.

### Option B — Manual Push from Local Terminal
If you download the project archive to your machine:
```bash
git init -b main
git add .
git status   # Verify .env and audio files are excluded by .gitignore
git commit -m "feat: VoiceLab AI v3.3.0 production release"
gh repo create voicelab-ai --private --source=. --remote=origin --push
```

### Automated GitHub Actions CI (`.github/workflows/ci.yml`)
On every push or pull request to `main`/`master`, the included workflow runs on `ubuntu-latest` (free tier) and executes:
1. **Tracked-File Secret & Media Scan** (blocks any `.env`, `.pem`, `.key`, audio files, or hardcoded API keys).
2. **Node.js 22 + FFmpeg Installation**.
3. **Dependency Verification (`npm install`)**.
4. **TypeScript Validation (`npm run lint`)**.
5. **Production Vite Build (`npm run build`)**.
6. **Production Server Smoke Test (`GET /api/health` & `GET /`)**.
