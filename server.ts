import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import os from 'os';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { createRemoteJWKSet, jwtVerify } from 'jose';
import type {
  AudioSignalAnalysis,
  CloningLanguageCode,
  SavedVoiceProfile,
  SavedVoiceSourceType,
  VoiceAudioQualityMode,
  VoiceCloneResult
} from './src/types/voicelab.ts';
import {
  extractAuto10SecVoiceCloneReference,
  inspectAndValidateAudioFile
} from './src/server/audioDsp.ts';
import {
  analyzeReferenceVoiceLanguageAndIdentity,
  assertSafeNeuralNodeUrl,
  executeMultilingualSrtPipeline,
  executeSameLanguageVoiceCloning,
  extractQuotaExhaustedInfo,
  getOrCreateAuthorizedSampleAudio,
  prepareCanonicalSpeakerPromptWav,
  QuotaExhaustedError,
  resolveRuntimeGeminiApiKey
} from './src/server/aiPipelines.ts';
import {
  checkCloudConditioningExists,
  deleteSavedVoiceFromFirestore,
  extractBearerTokenFromAuthHeader,
  fetchSingleSavedVoiceFromFirestore,
  hydrateSavedVoiceFilesFromFirestore,
  incrementSavedVoiceUsageInFirestore,
  listUserSavedVoicesFromFirestore,
  persistSavedVoiceAndAudioToFirestore,
  renameSavedVoiceInFirestore
} from './src/server/firestoreVaultBridge.ts';

const execFileAsync = promisify(execFile);

function resolveOwnerEmailAllowlist(): Set<string> {
  const configured = [process.env.OWNER_EMAIL || '', process.env.OWNER_EMAIL_ALLOWLIST || '']
    .join(',')
    .split(',')
    .map((entry) => entry.trim().toLowerCase())
    .filter((entry) => entry.includes('@'));
  if (configured.length > 0) {
    return new Set(configured);
  }
  // Default server-side owner identity when OWNER_EMAIL env var is unset
  return new Set(['foysalvvvvbbbb@gmail.com']);
}

const OWNER_EMAIL_ALLOWLIST = resolveOwnerEmailAllowlist();
const PRIMARY_DEFAULT_OWNER_EMAIL = Array.from(OWNER_EMAIL_ALLOWLIST)[0];

function resolveFirebaseProjectId(): string {
  try {
    const cfgPath = path.resolve(process.cwd(), 'firebase-applet-config.json');
    if (fs.existsSync(cfgPath)) {
      const parsed = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
      if (typeof parsed?.projectId === 'string' && parsed.projectId.trim()) {
        return parsed.projectId.trim();
      }
    }
  } catch {
    // Fallback cleanly
  }
  return '';
}

let cachedFirebaseProjectId = resolveFirebaseProjectId();
function getFirebaseProjectId(): string {
  if (cachedFirebaseProjectId) return cachedFirebaseProjectId;
  cachedFirebaseProjectId = resolveFirebaseProjectId();
  return cachedFirebaseProjectId;
}
const FIREBASE_JWKS = createRemoteJWKSet(
  new URL(
    'https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com'
  )
);

const PORT =
  process.env.PORT && process.env.PORT !== process.env.NGINX_PORT
    ? Number(process.env.PORT)
    : Number(process.env.DEFAULT_APP_PORT) || 3000;
const IS_PRODUCTION =
  process.env.NODE_ENV === 'production' || process.env.SERVE_STATIC_DIST === 'true';
const LEGACY_TEMP_ROOT = path.join(os.tmpdir(), 'voicelab_ai_storage');

// Permanent workspace-anchored persistent root (survives server restarts & never wiped by TTL)
const PERSISTENT_DATA_ROOT = path.resolve(process.cwd(), '.voicelab_persistent');
const UPLOAD_DIR = path.join(PERSISTENT_DATA_ROOT, 'uploads');
const GENERATED_DIR = path.join(PERSISTENT_DATA_ROOT, 'generated');
const SAMPLES_DIR = path.join(PERSISTENT_DATA_ROOT, 'samples');
const VOICE_VAULT_DIR = path.join(PERSISTENT_DATA_ROOT, 'voice_vault');
const SAVED_VOICES_INDEX_FILE = path.join(PERSISTENT_DATA_ROOT, 'saved_voices_index.json');

for (const dir of [
  PERSISTENT_DATA_ROOT,
  UPLOAD_DIR,
  GENERATED_DIR,
  SAMPLES_DIR,
  VOICE_VAULT_DIR
]) {
  fs.mkdirSync(dir, { recursive: true });
}

function isPlaceholderSecret(val: string | undefined): boolean {
  if (!val) return true;
  const trimmed = val.trim();
  if (trimmed.length < 32) return true;
  return /^(REPLACE_WITH_|MY_|YOUR_|CHANGE_ME|DEFAULT|PLACEHOLDER)/i.test(trimmed);
}

/**
 * Resolves a cryptographically strong 256-bit (32-byte) SESSION_SECRET that is
 * deterministic across server restarts and published Cloud Run revisions without
 * hardcoding credentials or relying on ephemeral /tmp or .env files.
 */
function resolveProductionSessionSecret(): {
  secret: string;
  source: 'env-secret' | 'hkdf-runtime-derived' | 'ephemeral-csprng';
  entropyBits: number;
} {
  const envVal = process.env.SESSION_SECRET?.trim();
  if (envVal && !isPlaceholderSecret(envVal)) {
    return {
      secret: envVal,
      source: 'env-secret',
      entropyBits: Math.min(512, envVal.length * 4)
    };
  }

  // Derive a deterministic, one-way 256-bit signing key via RFC 5869 HKDF-SHA256
  // from the runtime-injected secret and container applet identity.
  const runtimeKey = resolveRuntimeGeminiApiKey();
  const appletSalt =
    process.env.APPLET_ID ||
    process.env.K_CONFIGURATION ||
    process.env.K_SERVICE ||
    'voicelab-ai-session-salt';

  if (runtimeKey && runtimeKey !== 'MY_GEMINI_API_KEY' && runtimeKey.length > 16) {
    const derivedBuf = crypto.hkdfSync(
      'sha256',
      Buffer.from(runtimeKey, 'utf8'),
      Buffer.from(appletSalt, 'utf8'),
      Buffer.from('voicelab-ai-hmac-sha256-session-cookie-signing-key-v3.3', 'utf8'),
      32
    );
    const derivedHex = Buffer.from(derivedBuf).toString('hex');
    return {
      secret: derivedHex,
      source: 'hkdf-runtime-derived',
      entropyBits: 256
    };
  }

  const generated = crypto.randomBytes(32).toString('hex');
  return {
    secret: generated,
    source: 'ephemeral-csprng',
    entropyBits: 256
  };
}

const SESSION_CONFIG = resolveProductionSessionSecret();
const SESSION_SECRET = SESSION_CONFIG.secret;
const SESSION_TTL_SECONDS = 7 * 24 * 60 * 60;
const SESSION_TTL_MS = SESSION_TTL_SECONDS * 1000;

// Runtime binary verification state
let ffmpegBinaryVerified = false;
let ffprobeBinaryVerified = false;
let ffmpegVersionSummary = 'checking';

async function verifyMediaBinaries(): Promise<void> {
  try {
    const { stdout } = await execFileAsync('ffmpeg', ['-version'], { timeout: 4000 });
    ffmpegBinaryVerified = true;
    ffmpegVersionSummary = (stdout.split('\n')[0] || 'ffmpeg verified').trim();
  } catch {
    ffmpegBinaryVerified = false;
    ffmpegVersionSummary = 'unavailable';
  }
  try {
    await execFileAsync('ffprobe', ['-version'], { timeout: 4000 });
    ffprobeBinaryVerified = true;
  } catch {
    ffprobeBinaryVerified = false;
  }
}

// In-memory registry of temporary uploaded/sample audio files with timestamps for automatic TTL cleanup
const uploadedFilesRegistry = new Map<
  string,
  { filePath: string; analysis: AudioSignalAnalysis; createdAtMs: number }
>();

// Real-time operation progress registry & SSE subscribers for Live Activity Bar
interface ServerOperationProgress {
  operationId: string;
  stageIndex: number;
  stageLabel: string;
  stageDetail?: string;
  progressPct: number | null;
  progressSource: 'backend-chunks' | 'indeterminate-inference' | 'completed';
  completedUnits?: number;
  totalUnits?: number;
  status: 'running' | 'succeeded' | 'failed';
  updatedAtMs: number;
}

const operationProgressRegistry = new Map<string, ServerOperationProgress>();
const operationSseSubscribers = new Map<string, Set<express.Response>>();

function extractOperationIdFromReq(req: express.Request): string | null {
  const raw = req.headers['x-voicelab-operation-id'] || req.query.operationId;
  if (!raw || typeof raw !== 'string') return null;
  const trimmed = raw.trim();
  if (!trimmed || trimmed.length > 96 || !/^[a-zA-Z0-9_.-]+$/.test(trimmed)) return null;
  return trimmed;
}

function emitServerOperationProgress(
  operationId: string | null,
  update: Omit<ServerOperationProgress, 'operationId' | 'updatedAtMs'>
): void {
  if (!operationId) return;
  const payload: ServerOperationProgress = {
    operationId,
    ...update,
    updatedAtMs: Date.now()
  };
  operationProgressRegistry.set(operationId, payload);
  const subs = operationSseSubscribers.get(operationId);
  if (subs && subs.size > 0) {
    const frame = `data: ${JSON.stringify(payload)}\n\n`;
    for (const res of subs) {
      try {
        res.write(frame);
      } catch {
        subs.delete(res);
      }
    }
    if (update.status === 'succeeded' || update.status === 'failed') {
      for (const res of subs) {
        try {
          res.end();
        } catch {
          // Ignore
        }
      }
      operationSseSubscribers.delete(operationId);
    }
  }
}

const ALLOWED_EXTENSIONS = new Set(['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.aac']);
const SAFE_IDENTIFIER_REGEX = /^[a-zA-Z0-9_.-]+$/;

/**
 * Resolves a filename strictly inside a designated directory, blocking any path traversal attempt.
 */
function resolveSafeChildPath(baseDir: string, rawIdentifier: string): string | null {
  if (!rawIdentifier || typeof rawIdentifier !== 'string') return null;
  const trimmed = rawIdentifier.trim();
  if (!trimmed || trimmed.includes('..') || trimmed.includes('/') || trimmed.includes('\\')) {
    return null;
  }
  const cleanName = path.basename(trimmed);
  if (!SAFE_IDENTIFIER_REGEX.test(cleanName)) {
    return null;
  }
  const resolvedBase = path.resolve(baseDir);
  const resolvedTarget = path.resolve(resolvedBase, cleanName);
  if (!resolvedTarget.startsWith(resolvedBase + path.sep)) {
    return null;
  }
  return resolvedTarget;
}

/**
 * Inspects the first 16 bytes of an uploaded file to ensure it has a genuine audio/media container
 * signature (WAV/RIFF, MP3/ID3/MPEG sync, FLAC, OGG, M4A/MP4 ftyp, AAC ADTS, or WebM/Matroska).
 * Prevents disguised scripts, HTML, or executables from passing upload validation.
 */
async function verifyAudioFileMagicBytes(filePath: string): Promise<boolean> {
  let fd: fs.promises.FileHandle | null = null;
  try {
    fd = await fs.promises.open(filePath, 'r');
    const buf = Buffer.alloc(16);
    const { bytesRead } = await fd.read(buf, 0, 16, 0);
    if (bytesRead < 4) return false;

    // 1. RIFF....WAVE (WAV)
    if (
      buf.toString('ascii', 0, 4) === 'RIFF' &&
      bytesRead >= 12 &&
      buf.toString('ascii', 8, 12) === 'WAVE'
    ) {
      return true;
    }
    // 2. ID3 tag (MP3)
    if (buf.toString('ascii', 0, 3) === 'ID3') {
      return true;
    }
    // 3. MPEG Audio Frame Sync / ADTS AAC (0xFF followed by 0xE0..0xFF)
    if (buf[0] === 0xff && (buf[1] & 0xe0) === 0xe0) {
      return true;
    }
    // 4. FLAC ('fLaC')
    if (buf.toString('ascii', 0, 4) === 'fLaC') {
      return true;
    }
    // 5. Ogg ('OggS')
    if (buf.toString('ascii', 0, 4) === 'OggS') {
      return true;
    }
    // 6. MP4 / M4A ('ftyp' at byte 4)
    if (bytesRead >= 8 && buf.toString('ascii', 4, 8) === 'ftyp') {
      return true;
    }
    // 7. WebM / Matroska EBML header (0x1A 0x45 0xDF 0xA3)
    if (buf[0] === 0x1a && buf[1] === 0x45 && buf[2] === 0xdf && buf[3] === 0xa3) {
      return true;
    }
    return false;
  } catch {
    return false;
  } finally {
    if (fd) {
      await fd.close().catch(() => {});
    }
  }
}

// Cryptographic Session Token helpers (HMAC-SHA256 signed HttpOnly cookie)
function signSessionToken(sessionId: string, issuedAtMs: number): string {
  const payload = `${sessionId}.${issuedAtMs}`;
  const sig = crypto.createHmac('sha256', SESSION_SECRET).update(payload).digest('base64url');
  return `${payload}.${sig}`;
}

interface VerifiedSessionState {
  valid: boolean;
  expired: boolean;
  explicitlyInvalid: boolean;
  sessionId: string;
  issuedAtMs: number;
  signedToken: string;
}

function verifySessionToken(rawToken: string | undefined): VerifiedSessionState | null {
  if (!rawToken || typeof rawToken !== 'string') return null;
  const trimmed = rawToken.trim();
  if (!trimmed) return null;

  const parts = trimmed.split('.');
  if (parts.length !== 3) {
    return {
      valid: false,
      expired: false,
      explicitlyInvalid: true,
      sessionId: '',
      issuedAtMs: 0,
      signedToken: ''
    };
  }
  const [sessionId, issuedAtStr, sig] = parts;
  if (!/^sess_[a-f0-9]{24}$/.test(sessionId)) {
    return {
      valid: false,
      expired: false,
      explicitlyInvalid: true,
      sessionId: '',
      issuedAtMs: 0,
      signedToken: ''
    };
  }
  const issuedAtMs = Number(issuedAtStr);
  if (!Number.isFinite(issuedAtMs) || issuedAtMs <= 0) {
    return {
      valid: false,
      expired: false,
      explicitlyInvalid: true,
      sessionId: '',
      issuedAtMs: 0,
      signedToken: ''
    };
  }

  const expectedPayload = `${sessionId}.${issuedAtMs}`;
  const expectedSig = crypto
    .createHmac('sha256', SESSION_SECRET)
    .update(expectedPayload)
    .digest('base64url');

  const sigBuf = Buffer.from(sig);
  const expBuf = Buffer.from(expectedSig);
  if (sigBuf.length !== expBuf.length || !crypto.timingSafeEqual(sigBuf, expBuf)) {
    return {
      valid: false,
      expired: false,
      explicitlyInvalid: true,
      sessionId: '',
      issuedAtMs: 0,
      signedToken: ''
    };
  }

  if (Date.now() - issuedAtMs > SESSION_TTL_MS) {
    return {
      valid: false,
      expired: true,
      explicitlyInvalid: true,
      sessionId,
      issuedAtMs,
      signedToken: trimmed
    };
  }

  return {
    valid: true,
    expired: false,
    explicitlyInvalid: false,
    sessionId,
    issuedAtMs,
    signedToken: trimmed
  };
}

/**
 * Determines whether the request originated over HTTPS / public Cloud Run ingress,
 * accounting for container Nginx setting X-Forwarded-Proto: http while passing X-Forwarded-Host: *.run.app.
 */
function isHttpsPublicRequest(req: express.Request): boolean {
  if (req.secure) return true;
  const proto = String(req.headers['x-forwarded-proto'] || '').toLowerCase();
  if (proto === 'https') return true;
  const forwardedHost = String(req.headers['x-forwarded-host'] || '').toLowerCase();
  if (
    forwardedHost.endsWith('.run.app') ||
    (forwardedHost &&
      !forwardedHost.startsWith('localhost') &&
      !forwardedHost.startsWith('127.0.0.1'))
  ) {
    return true;
  }
  const originOrRef = String(req.headers.origin || req.headers.referer || '').toLowerCase();
  if (originOrRef.startsWith('https://')) return true;
  return IS_PRODUCTION;
}

function buildSessionCookieHeader(signedToken: string, isHttps: boolean): string {
  const cookieParts = [
    `voicelab_sid=${encodeURIComponent(signedToken)}`,
    'Path=/',
    'HttpOnly',
    `Max-Age=${SESSION_TTL_SECONDS}`
  ];
  if (isHttps) {
    // Cross-site iframe (aistudio.google.com -> *.run.app) & top-level HTTPS require SameSite=None; Secure; Partitioned
    cookieParts.push('SameSite=None', 'Secure', 'Partitioned');
  } else {
    cookieParts.push('SameSite=Lax');
  }
  return cookieParts.join('; ');
}

function parseCookies(cookieHeader: string | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  if (!cookieHeader) return out;
  for (const pair of cookieHeader.split(';')) {
    const idx = pair.indexOf('=');
    if (idx > 0) {
      const k = pair.slice(0, idx).trim();
      const v = pair.slice(idx + 1).trim();
      out[k] = decodeURIComponent(v);
    }
  }
  return out;
}

// Sliding-Window Rate Limiter per IP
interface RateBucket {
  timestamps: number[];
}
const generalRateBuckets = new Map<string, RateBucket>();
const computeRateBuckets = new Map<string, RateBucket>();

function checkSlidingRateLimit(
  store: Map<string, RateBucket>,
  clientKey: string,
  maxRequests: number,
  windowMs: number
): { allowed: boolean; remaining: number; resetSeconds: number } {
  const now = Date.now();
  const cutoff = now - windowMs;
  let bucket = store.get(clientKey);
  if (!bucket) {
    bucket = { timestamps: [] };
    store.set(clientKey, bucket);
  }
  bucket.timestamps = bucket.timestamps.filter((t) => t > cutoff);
  if (bucket.timestamps.length >= maxRequests) {
    const oldest = bucket.timestamps[0] || now;
    const resetSeconds = Math.max(1, Math.ceil((oldest + windowMs - now) / 1000));
    return { allowed: false, remaining: 0, resetSeconds };
  }
  bucket.timestamps.push(now);
  return {
    allowed: true,
    remaining: Math.max(0, maxRequests - bucket.timestamps.length),
    resetSeconds: Math.ceil(windowMs / 1000)
  };
}

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, UPLOAD_DIR);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname || '').toLowerCase() || '.wav';
    const safeExt = ALLOWED_EXTENSIONS.has(ext) ? ext : '.wav';
    const uniqueId = `aud_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
    cb(null, `${uniqueId}${safeExt}`);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 150 * 1024 * 1024, // 150 MB max upload size (long reference voices auto-trim to 10s)
    files: 1
  },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname || '').toLowerCase();
    const mimeOk =
      file.mimetype.startsWith('audio/') ||
      file.mimetype === 'video/webm' ||
      file.mimetype === 'video/mp4' ||
      file.mimetype === 'application/octet-stream';
    if (ALLOWED_EXTENSIONS.has(ext) && mimeOk) {
      cb(null, true);
    } else {
      cb(
        new Error(
          'Unsupported audio format. Allowed formats: WAV, MP3, M4A, FLAC, OGG, AAC, WEBM.'
        )
      );
    }
  }
});

// Sidecar persistence helper so server restarts never lose uploaded audio references
async function saveUploadedFileEntry(
  fileId: string,
  filePath: string,
  analysis: AudioSignalAnalysis
): Promise<void> {
  const safeId = path.basename(fileId.trim());
  if (!SAFE_IDENTIFIER_REGEX.test(safeId)) return;
  const createdAtMs = Date.now();
  uploadedFilesRegistry.set(safeId, { filePath, analysis, createdAtMs });
  const metaPath = resolveSafeChildPath(UPLOAD_DIR, `${safeId}.meta.json`);
  if (metaPath) {
    await fs.promises
      .writeFile(metaPath, JSON.stringify({ filePath, analysis, createdAtMs }), 'utf8')
      .catch(() => {});
  }
}

async function resolveUploadedFileEntry(
  fileId: string | undefined
): Promise<{ filePath: string; analysis: AudioSignalAnalysis; createdAtMs: number } | null> {
  if (!fileId || typeof fileId !== 'string') return null;
  const cleanId = path.basename(fileId.trim());
  if (!SAFE_IDENTIFIER_REGEX.test(cleanId)) return null;

  const inMem = uploadedFilesRegistry.get(cleanId);
  if (inMem && fs.existsSync(inMem.filePath)) {
    return inMem;
  }

  // Check disk sidecar metadata if server restarted after upload
  const metaPath = resolveSafeChildPath(UPLOAD_DIR, `${cleanId}.meta.json`);
  if (metaPath && fs.existsSync(metaPath)) {
    try {
      const raw = await fs.promises.readFile(metaPath, 'utf8');
      const parsed = JSON.parse(raw);
      const candidatePath = resolveSafeChildPath(
        UPLOAD_DIR,
        path.basename(String(parsed?.filePath || ''))
      );
      if (candidatePath && fs.existsSync(candidatePath) && parsed?.analysis) {
        const entry = {
          filePath: candidatePath,
          analysis: parsed.analysis as AudioSignalAnalysis,
          createdAtMs: Number(parsed.createdAtMs) || Date.now()
        };
        uploadedFilesRegistry.set(cleanId, entry);
        return entry;
      }
    } catch {
      // Fall through to directory scan
    }
  }

  // Fallback scan of UPLOAD_DIR for matching fileId prefix
  try {
    const files = await fs.promises.readdir(UPLOAD_DIR);
    const matchFile = files.find(
      (f) => f.startsWith(`${cleanId}.`) && !f.endsWith('.meta.json')
    );
    if (matchFile) {
      const fullPath = resolveSafeChildPath(UPLOAD_DIR, matchFile);
      if (fullPath && fs.existsSync(fullPath)) {
        const previewUrl = `/api/audio/uploaded/${cleanId}`;
        const analysis = await inspectAndValidateAudioFile(
          fullPath,
          cleanId,
          matchFile,
          'audio/wav',
          previewUrl
        );
        if (analysis.isValid) {
          await saveUploadedFileEntry(cleanId, fullPath, analysis);
          return uploadedFilesRegistry.get(cleanId) || null;
        }
      }
    }
  } catch {
    // Ignore scan errors
  }

  return null;
}

// Wrap Multer middleware so any upload error always returns valid JSON instead of Express HTML
function handleAudioUploadMiddleware(
  req: express.Request,
  res: express.Response,
  next: express.NextFunction
) {
  const contentType = String(req.headers['content-type'] || '').toLowerCase();
  if (!contentType.includes('multipart/form-data')) {
    next();
    return;
  }
  upload.single('audio')(req, res, (err: any) => {
    if (err) {
      const message =
        err?.code === 'LIMIT_FILE_SIZE'
          ? 'Audio file exceeds the 150 MB maximum upload limit.'
          : err?.message || 'Audio file upload failed validation.';
      res.status(400).type('application/json').json({
        error: message
      });
      return;
    }
    next();
  });
}

// Periodic in-memory maintenance (rate limits & old SSE progress only — NEVER deletes saved/uploaded/generated user audio)
const cleanupIntervalTimer = setInterval(() => {
  const now = Date.now();
  // Prune stale rate limit buckets
  for (const store of [generalRateBuckets, computeRateBuckets]) {
    for (const [key, bucket] of store.entries()) {
      if (bucket.timestamps.every((t) => now - t > 120_000)) {
        store.delete(key);
      }
    }
  }
  // Prune settled operation progress records older than 10 minutes
  for (const [opId, state] of operationProgressRegistry.entries()) {
    if (now - state.updatedAtMs > 10 * 60 * 1000) {
      operationProgressRegistry.delete(opId);
      operationSseSubscribers.delete(opId);
    }
  }
}, 15 * 60 * 1000);
cleanupIntervalTimer.unref?.();

async function startServer() {
  void verifyMediaBinaries();

  const app = express();
  app.disable('x-powered-by');
  // Trust only loopback (Nginx 127.0.0.1) and internal Cloud Run private hops; never arbitrary public X-Forwarded-*
  app.set('trust proxy', ['loopback', 'linklocal', 'uniquelocal']);

  // 1. Security Headers & Restricted CORS Policy
  app.use((req, res, next) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    res.setHeader('X-Permitted-Cross-Domain-Policies', 'none');
    res.setHeader('Permissions-Policy', 'microphone=(self)');

    const origin = req.headers.origin;
    if (origin) {
      const host = req.headers.host || '';
      const forwardedHost = String(req.headers['x-forwarded-host'] || '');
      const configuredAppUrl = (process.env.APP_URL || '').trim();
      let isAllowedOrigin = false;
      try {
        const parsedOrigin = new URL(origin);
        if (parsedOrigin.host === host || (forwardedHost && parsedOrigin.host === forwardedHost)) {
          isAllowedOrigin = true;
        } else if (configuredAppUrl && origin.startsWith(configuredAppUrl)) {
          isAllowedOrigin = true;
        } else if (
          parsedOrigin.hostname === 'localhost' ||
          parsedOrigin.hostname === '127.0.0.1' ||
          parsedOrigin.hostname.endsWith('.run.app')
        ) {
          isAllowedOrigin = true;
        }
      } catch {
        isAllowedOrigin = false;
      }

      if (isAllowedOrigin) {
        res.setHeader('Access-Control-Allow-Origin', origin);
        res.setHeader('Access-Control-Allow-Credentials', 'true');
        res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PATCH, DELETE, OPTIONS');
        res.setHeader(
          'Access-Control-Allow-Headers',
          'Content-Type, Accept, Authorization, X-Requested-With, X-VoiceLab-Session, X-VoiceLab-Operation-Id'
        );
        res.setHeader('Access-Control-Expose-Headers', 'X-VoiceLab-Session-Token');
        res.setHeader('Vary', 'Origin');
      } else if (req.path.startsWith('/api/')) {
        res.status(400).type('application/json').json({
          error: 'Cross-origin request blocked by VoiceLab AI security policy.'
        });
        return;
      }
    }

    if (req.method === 'OPTIONS') {
      res.status(204).end();
      return;
    }
    next();
  });

  // 2. Cryptographic Session Middleware (Executes BEFORE all API routes and Multer uploads)
  app.use((req, res, next) => {
    const cookies = parseCookies(req.headers.cookie);
    const cookieToken = cookies['voicelab_sid'];
    const headerToken =
      typeof req.headers['x-voicelab-session'] === 'string'
        ? req.headers['x-voicelab-session'].trim()
        : undefined;

    const verifiedCookie = verifySessionToken(cookieToken);
    const verifiedHeader = verifySessionToken(headerToken);
    const isHttps = isHttpsPublicRequest(req);

    // Prefer valid cookie, then valid header
    const activeVerified =
      verifiedCookie?.valid ? verifiedCookie : verifiedHeader?.valid ? verifiedHeader : null;

    // Detect if the client explicitly supplied a forged or expired token without any valid fallback
    const hadExplicitInvalidToken =
      !activeVerified &&
      Boolean(verifiedCookie?.explicitlyInvalid || verifiedHeader?.explicitlyInvalid);
    const hadExpiredToken =
      !activeVerified && Boolean(verifiedCookie?.expired || verifiedHeader?.expired);

    if (activeVerified) {
      (req as any).voicelabSession = {
        ...activeVerified,
        hadExplicitInvalidToken: false,
        hadExpiredToken: false
      };
      // Ensure browser cookie is refreshed if request authenticated via X-VoiceLab-Session header
      if (!verifiedCookie?.valid) {
        res.setHeader(
          'Set-Cookie',
          buildSessionCookieHeader(activeVerified.signedToken, isHttps)
        );
      }
      res.setHeader('X-VoiceLab-Session-Token', activeVerified.signedToken);
    } else if (req.method === 'GET' || req.method === 'HEAD') {
      // Establish a fresh session on initial page load (GET /) or session bootstrap (GET /api/auth/session)
      const newSessionId = `sess_${crypto.randomBytes(12).toString('hex')}`;
      const issuedAtMs = Date.now();
      const signedToken = signSessionToken(newSessionId, issuedAtMs);
      (req as any).voicelabSession = {
        valid: true,
        expired: false,
        explicitlyInvalid: false,
        sessionId: newSessionId,
        issuedAtMs,
        signedToken
      };
      res.setHeader('Set-Cookie', buildSessionCookieHeader(signedToken, isHttps));
      res.setHeader('X-VoiceLab-Session-Token', signedToken);
    } else {
      // Unauthenticated, forged, or expired session on state-mutating request (POST / DELETE)
      (req as any).voicelabSession = {
        valid: false,
        expired: hadExpiredToken,
        explicitlyInvalid: hadExplicitInvalidToken,
        sessionId: '',
        issuedAtMs: 0,
        signedToken: ''
      };
    }
    next();
  });

  // Protected Route Authentication Guard — enforces valid session and returns HTTP 401 JSON on unauthenticated/invalid/expired credentials
  function requireAuthenticatedSession(
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
  ) {
    const sess = (req as any).voicelabSession as VerifiedSessionState | undefined;
    if (!sess || !sess.valid) {
      res.status(401).type('application/json').json({
        error: sess?.expired
          ? 'Session expired. Please refresh your session and try again.'
          : 'Authentication required: Valid session cookie or token was not provided.',
        code: sess?.expired ? 'SESSION_EXPIRED' : 'SESSION_UNAUTHORIZED'
      });
      return;
    }
    next();
  }

  app.use(express.json({ limit: '15mb' }));

  // 3. Structured Secret-Safe API Logging, Cache-Control & General Rate Limiting
  app.use('/api', (req, res, next) => {
    const startMs = Date.now();
    res.setHeader('Cache-Control', 'no-store');

    const safePath = req.originalUrl.split('?')[0];
    const isExemptReadPath =
      (req.method === 'GET' || req.method === 'HEAD') &&
      (safePath.startsWith('/api/operations/') ||
        safePath.startsWith('/api/voice-clone/jobs/') ||
        safePath.startsWith('/api/audio/') ||
        safePath === '/api/health' ||
        safePath === '/api/auth/session');

    if (!isExemptReadPath) {
      const clientIp = String(req.ip || req.socket.remoteAddress || '127.0.0.1');
      const sess = (req as any).voicelabSession as VerifiedSessionState | undefined;
      const bucketKey = `${clientIp}::${sess?.sessionId || 'anon'}`;
      const rateStatus = checkSlidingRateLimit(generalRateBuckets, bucketKey, 360, 60_000);
      res.setHeader('X-RateLimit-Limit', '360');
      res.setHeader('X-RateLimit-Remaining', String(rateStatus.remaining));
      res.setHeader('X-RateLimit-Reset', String(rateStatus.resetSeconds));

      if (!rateStatus.allowed) {
        res.status(429).type('application/json').json({
          error: 'Rate limit exceeded. Please wait before sending additional API requests.',
          retryAfterSeconds: rateStatus.resetSeconds
        });
        return;
      }
    }

    res.on('finish', () => {
      const elapsedMs = Date.now() - startMs;
      if (!safePath.startsWith('/api/operations/') && !safePath.startsWith('/api/voice-clone/jobs/')) {
        console.log(
          `[VoiceLab API] ${req.method} ${safePath} -> HTTP ${res.statusCode} (${elapsedMs}ms)`
        );
      }
    });

    next();
  });

  // Server-side Governance, Ownership, Feature Paywall, Payment & Token Store (Fail-Closed Owner Security)
  interface ServerGovernancePolicy {
    maintenanceMode: boolean;
    maintenanceNotice: string;
    allowVoiceCloning: boolean;
    allowSrtGeneration: boolean;
    requireAuthForCompute: boolean;
    maxScriptChars: number;
    rateLimitPerMinute: number;
    enforceZeroGeminiQuota: boolean;
  }

  interface ServerOwnershipState {
    primaryOwnerUid: string;
    primaryOwnerEmail: string;
    organizationName: string;
    ownershipLocked: boolean;
  }

  type FeatureGateKey = 'voice_cloning' | 'long_script' | 'srt_generator';
  type PackageFeatureKey = FeatureGateKey | 'pro_bundle';
  type PaymentMethodId = 'bkash' | 'nagad';

  interface ServerFeatureFlagConfig {
    key: FeatureGateKey;
    title: string;
    description: string;
    enabled: boolean;
    isPaid: boolean;
    tokenCost: number;
  }

  interface ServerFeaturePackage {
    id: string;
    featureKey: PackageFeatureKey;
    title: string;
    description: string;
    priceBdt: number;
    durationDays: number;
    enabled: boolean;
  }

  interface ServerTokenPackage {
    id: string;
    title: string;
    description: string;
    tokens: number;
    priceBdt: number;
    enabled: boolean;
  }

  interface ServerPaymentSettings {
    bkashEnabled: boolean;
    bkashNumber: string;
    bkashAccountType: 'Personal' | 'Merchant' | 'Agent';
    bkashInstructions: string;
    nagadEnabled: boolean;
    nagadNumber: string;
    nagadAccountType: 'Personal' | 'Merchant' | 'Agent';
    nagadInstructions: string;
  }

  interface ServerUserRecord {
    uid: string;
    email: string;
    displayName: string;
    role: 'owner' | 'user';
    accountStatus: 'active' | 'suspended';
    createdAt: string;
    lastSeenAt: string;
  }

  interface ServerUserEntitlement {
    uid: string;
    email: string;
    tokenBalance: number;
    voiceCloningUnlocked: boolean;
    voiceCloningExpiresAtMs: number;
    longScriptUnlocked: boolean;
    longScriptExpiresAtMs: number;
    srtGeneratorUnlocked: boolean;
    srtGeneratorExpiresAtMs: number;
    proBundleUnlocked: boolean;
    proBundleExpiresAtMs: number;
    updatedAt: string;
  }

  interface ServerPaymentRequest {
    orderId: string;
    userUid: string;
    userEmail: string;
    userName: string;
    itemType: 'feature_package' | 'token_package';
    itemId: string;
    itemTitle: string;
    featureKey?: PackageFeatureKey;
    amountBdt: number;
    tokensGranted: number;
    durationDays: number;
    paymentMethod: PaymentMethodId;
    senderNumber: string;
    trxId: string;
    receiverNumber: string;
    note: string;
    status: 'pending' | 'approved' | 'rejected';
    credited: boolean;
    reviewedByUid: string;
    rejectionReason: string;
    createdAt: string;
    updatedAt: string;
  }

  interface ServerTokenLedgerEntry {
    entryId: string;
    userUid: string;
    userEmail: string;
    delta: number;
    balanceAfter: number;
    type: 'purchase_credit' | 'admin_grant' | 'admin_revoke' | 'feature_debit' | 'refund';
    reason: string;
    actorUid: string;
    createdAt: string;
  }

  interface ServerAuditLogEntry {
    logId: string;
    actorUid: string;
    actorEmail: string;
    action: string;
    target: string;
    details: string;
    createdAt: string;
  }

  const GOVERNANCE_STATE_FILE = path.join(PERSISTENT_DATA_ROOT, 'server_governance_state.json');
  const LEGACY_GOVERNANCE_STATE_FILE = path.join(LEGACY_TEMP_ROOT, 'server_governance_state.json');
  if (!fs.existsSync(GOVERNANCE_STATE_FILE) && fs.existsSync(LEGACY_GOVERNANCE_STATE_FILE)) {
    try {
      fs.copyFileSync(LEGACY_GOVERNANCE_STATE_FILE, GOVERNANCE_STATE_FILE);
    } catch {
      // Non-fatal migration fallback
    }
  }

  let serverGovernancePolicy: ServerGovernancePolicy = {
    maintenanceMode: false,
    maintenanceNotice: 'VoiceLab AI is currently undergoing scheduled owner maintenance.',
    allowVoiceCloning: true,
    allowSrtGeneration: true,
    requireAuthForCompute: false,
    maxScriptChars: 12000,
    rateLimitPerMinute: 25,
    enforceZeroGeminiQuota: true
  };

  let serverOwnershipState: ServerOwnershipState = {
    primaryOwnerUid: '',
    primaryOwnerEmail: PRIMARY_DEFAULT_OWNER_EMAIL,
    organizationName: 'VoiceLab AI',
    ownershipLocked: true
  };

  let serverFeatureConfigs: Record<FeatureGateKey, ServerFeatureFlagConfig> = {
    voice_cloning: {
      key: 'voice_cloning',
      title: 'Voice Cloning Studio',
      description: 'Same-language neural voice cloning for Bengali, Hindi, and English.',
      enabled: true,
      isPaid: false,
      tokenCost: 0
    },
    long_script: {
      key: 'long_script',
      title: 'Long-Script Pro (>1,000 Chars)',
      description: 'Auto-chunked synthesis for scripts longer than 1,000 spoken characters.',
      enabled: true,
      isPaid: false,
      tokenCost: 0
    },
    srt_generator: {
      key: 'srt_generator',
      title: 'Audio to SRT Generator',
      description: 'Multilingual audio transcription and SubRip subtitle timing.',
      enabled: true,
      isPaid: false,
      tokenCost: 0
    }
  };

  let serverFeaturePackages: ServerFeaturePackage[] = [
    {
      id: 'pkg_voice_cloning',
      featureKey: 'voice_cloning',
      title: 'Voice Cloning Pass',
      description: 'Full access to same-language neural Voice Cloning Studio',
      priceBdt: 299,
      durationDays: 30,
      enabled: true
    },
    {
      id: 'pkg_long_script',
      featureKey: 'long_script',
      title: 'Long-Script Pro (>1,000 Chars)',
      description: 'Synthesize long narration scripts beyond 1,000 characters',
      priceBdt: 249,
      durationDays: 30,
      enabled: true
    },
    {
      id: 'pkg_srt_generator',
      featureKey: 'srt_generator',
      title: 'Audio to SRT Generator',
      description: 'Multilingual speech-to-subtitle workflow & SRT export',
      priceBdt: 199,
      durationDays: 30,
      enabled: true
    },
    {
      id: 'pkg_pro_bundle',
      featureKey: 'pro_bundle',
      title: 'All-in-One Pro Bundle',
      description: 'Unlock Voice Cloning, Long-Script Pro, and Audio-to-SRT together',
      priceBdt: 499,
      durationDays: 30,
      enabled: true
    }
  ];

  let serverTokenPackages: ServerTokenPackage[] = [
    {
      id: 'tok_starter_50',
      title: 'Starter 50 Tokens',
      description: '50 compute tokens for Voice Cloning and Audio-to-SRT jobs',
      tokens: 50,
      priceBdt: 149,
      enabled: true
    },
    {
      id: 'tok_creator_150',
      title: 'Creator 150 Tokens',
      description: '150 compute tokens for regular studio creators',
      tokens: 150,
      priceBdt: 349,
      enabled: true
    },
    {
      id: 'tok_studio_500',
      title: 'Studio 500 Tokens',
      description: '500 compute tokens for high-volume production pipelines',
      tokens: 500,
      priceBdt: 899,
      enabled: true
    }
  ];

  let serverPaymentSettings: ServerPaymentSettings = {
    bkashEnabled: true,
    bkashNumber: '',
    bkashAccountType: 'Personal',
    bkashInstructions:
      'bKash App বা *247# ডায়াল করে উপরের নম্বরে Send Money / Payment করুন। এরপর Transaction ID (TrxID) ও আপনার বিকাশ নম্বর দিয়ে সাবমিট করুন।',
    nagadEnabled: true,
    nagadNumber: '',
    nagadAccountType: 'Personal',
    nagadInstructions:
      'Nagad App বা *167# ডায়াল করে উপরের নম্বরে Send Money / Payment করুন। এরপর Transaction ID (TrxID) ও আপনার নগদ নম্বর দিয়ে সাবমিট করুন।'
  };

  let serverUsersRegistry: Record<string, ServerUserRecord> = {};
  let serverUserEntitlements: Record<string, ServerUserEntitlement> = {};
  let serverPaymentRequests: ServerPaymentRequest[] = [];
  let serverTokenLedger: ServerTokenLedgerEntry[] = [];
  let serverAuditLogs: ServerAuditLogEntry[] = [];
  let usedPaymentTrxIds: Record<
    string,
    { orderId: string; paymentMethod: PaymentMethodId; userUid: string; status: string }
  > = {};

  try {
    if (fs.existsSync(GOVERNANCE_STATE_FILE)) {
      const saved = JSON.parse(fs.readFileSync(GOVERNANCE_STATE_FILE, 'utf8'));
      if (saved?.governance) {
        serverGovernancePolicy = { ...serverGovernancePolicy, ...saved.governance };
      }
      if (saved?.ownership) {
        serverOwnershipState = {
          ...serverOwnershipState,
          primaryOwnerUid: saved.ownership.primaryOwnerUid || '',
          primaryOwnerEmail:
            OWNER_EMAIL_ALLOWLIST.size > 0 && process.env.OWNER_EMAIL
              ? PRIMARY_DEFAULT_OWNER_EMAIL
              : saved.ownership.primaryOwnerEmail || PRIMARY_DEFAULT_OWNER_EMAIL,
          organizationName: saved.ownership.organizationName || 'VoiceLab AI',
          ownershipLocked:
            typeof saved.ownership.ownershipLocked === 'boolean'
              ? saved.ownership.ownershipLocked
              : true
        };
      }
      if (saved?.featureConfigs) {
        serverFeatureConfigs = {
          voice_cloning: { ...serverFeatureConfigs.voice_cloning, ...saved.featureConfigs.voice_cloning },
          long_script: { ...serverFeatureConfigs.long_script, ...saved.featureConfigs.long_script },
          srt_generator: { ...serverFeatureConfigs.srt_generator, ...saved.featureConfigs.srt_generator }
        };
      }
      if (Array.isArray(saved?.featurePackages) && saved.featurePackages.length > 0) {
        serverFeaturePackages = saved.featurePackages;
      }
      if (Array.isArray(saved?.tokenPackages) && saved.tokenPackages.length > 0) {
        serverTokenPackages = saved.tokenPackages;
      }
      if (saved?.paymentSettings) {
        serverPaymentSettings = { ...serverPaymentSettings, ...saved.paymentSettings };
      } else if (saved?.monetization) {
        // Migrate legacy bKash settings cleanly if present
        serverPaymentSettings.bkashEnabled = Boolean(saved.monetization.bkashEnabled ?? true);
        serverPaymentSettings.bkashNumber = String(saved.monetization.bkashNumber || '');
        serverPaymentSettings.bkashAccountType = saved.monetization.bkashAccountType || 'Personal';
        serverPaymentSettings.bkashInstructions =
          saved.monetization.bkashInstructions || serverPaymentSettings.bkashInstructions;
      }
      if (saved?.usersRegistry && typeof saved.usersRegistry === 'object') {
        serverUsersRegistry = saved.usersRegistry;
      }
      if (saved?.entitlements && typeof saved.entitlements === 'object') {
        serverUserEntitlements = saved.entitlements;
      }
      if (Array.isArray(saved?.paymentRequests)) {
        serverPaymentRequests = saved.paymentRequests;
      }
      if (Array.isArray(saved?.tokenLedger)) {
        serverTokenLedger = saved.tokenLedger;
      }
      if (Array.isArray(saved?.auditLogs)) {
        serverAuditLogs = saved.auditLogs;
      }
      if (saved?.usedPaymentTrxIds && typeof saved.usedPaymentTrxIds === 'object') {
        usedPaymentTrxIds = saved.usedPaymentTrxIds;
      }
    }
  } catch {
    // Use defaults cleanly
  }

  async function persistGovernanceState(): Promise<void> {
    await fs.promises
      .writeFile(
        GOVERNANCE_STATE_FILE,
        JSON.stringify(
          {
            governance: serverGovernancePolicy,
            ownership: serverOwnershipState,
            featureConfigs: serverFeatureConfigs,
            featurePackages: serverFeaturePackages,
            tokenPackages: serverTokenPackages,
            paymentSettings: serverPaymentSettings,
            usersRegistry: serverUsersRegistry,
            entitlements: serverUserEntitlements,
            paymentRequests: serverPaymentRequests.slice(0, 300),
            tokenLedger: serverTokenLedger.slice(0, 500),
            auditLogs: serverAuditLogs.slice(0, 200),
            usedPaymentTrxIds
          },
          null,
          2
        ),
        'utf8'
      )
      .catch(() => {});
  }

  // Persistent User-Specific Saved Voice Profiles Registry & Secure UID-Isolated Vault
  interface ServerSavedVoiceRecord extends SavedVoiceProfile {
    referenceAnalysis: AudioSignalAnalysis;
  }

  const savedVoicesRegistry = new Map<string, ServerSavedVoiceRecord>();

  function resolveVaultVoicePaths(
    ownerUid: string,
    voiceId: string
  ): {
    dir: string;
    refWavPath: string;
    promptWavPath: string;
    profileJsonPath: string;
    storageKey: string;
  } | null {
    const cleanUid = path.basename(String(ownerUid || '').trim());
    const cleanVoiceId = path.basename(String(voiceId || '').trim());
    if (
      !/^[a-zA-Z0-9_-]{3,128}$/.test(cleanUid) ||
      !/^[a-zA-Z0-9_-]{3,128}$/.test(cleanVoiceId)
    ) {
      return null;
    }
    const vaultBase = path.resolve(VOICE_VAULT_DIR);
    const dir = path.resolve(vaultBase, cleanUid, cleanVoiceId);
    if (!dir.startsWith(vaultBase + path.sep)) {
      return null;
    }
    return {
      dir,
      refWavPath: path.join(dir, 'reference.wav'),
      promptWavPath: path.join(dir, 'prompt_5s.wav'),
      profileJsonPath: path.join(dir, 'profile.json'),
      storageKey: `${cleanUid}/${cleanVoiceId}`
    };
  }

  function hasVaultConditioningAudio(rec: ServerSavedVoiceRecord): boolean {
    const paths = resolveVaultVoicePaths(rec.ownerUid, rec.voiceId);
    if (!paths) return false;
    return fs.existsSync(paths.refWavPath) && fs.existsSync(paths.promptWavPath);
  }

  function serializeSavedVoiceProfile(
    rec: ServerSavedVoiceRecord,
    overrideHasConditioningAudio?: boolean
  ): SavedVoiceProfile {
    return {
      voiceId: rec.voiceId,
      ownerUid: rec.ownerUid,
      ownerEmail: rec.ownerEmail,
      name: rec.name,
      language: rec.language,
      languageName: rec.languageName,
      conditioningMethod: rec.conditioningMethod,
      modelCheckpoint: rec.modelCheckpoint,
      speakerConditioningSha256: rec.speakerConditioningSha256,
      promptConditioningSha256: rec.promptConditioningSha256,
      referenceTranscript: rec.referenceTranscript,
      referenceDurationSec: rec.referenceDurationSec,
      promptDurationSec: rec.promptDurationSec,
      fundamentalFreqHz: rec.fundamentalFreqHz,
      pitchMinHz: rec.pitchMinHz,
      pitchMaxHz: rec.pitchMaxHz,
      spectralWarmthRatio: rec.spectralWarmthRatio,
      estimatedSnrDb: rec.estimatedSnrDb,
      qualityGrade: rec.qualityGrade,
      sourceType: rec.sourceType,
      sourceFileName: rec.sourceFileName,
      storageKey: rec.storageKey,
      consentConfirmed: rec.consentConfirmed,
      usageCount: rec.usageCount,
      hasConditioningAudio:
        typeof overrideHasConditioningAudio === 'boolean'
          ? overrideHasConditioningAudio
          : hasVaultConditioningAudio(rec),
      lastUsedAt: rec.lastUsedAt,
      createdAt: rec.createdAt,
      updatedAt: rec.updatedAt
    };
  }

  function buildSyntheticAnalysisFromProfile(profile: SavedVoiceProfile): AudioSignalAnalysis {
    return {
      fileId: profile.voiceId,
      fileName: profile.sourceFileName || `${profile.voiceId}.wav`,
      mimeType: 'audio/wav',
      formatName: 'wav',
      codecName: 'pcm_s16le',
      fileSizeBytes: 0,
      durationSeconds: Number(profile.referenceDurationSec) || 5,
      sampleRateHz: 24000,
      channels: 1,
      bitRateKbps: 384,
      rmsLevelDb: -20,
      peakLevelDb: -3,
      noiseFloorDb: -45,
      estimatedSnrDb: Number(profile.estimatedSnrDb) || 25,
      silenceRatio: 0.1,
      fundamentalFreqHz: Number(profile.fundamentalFreqHz) || 160,
      pitchRangeHz: [Number(profile.pitchMinHz) || 100, Number(profile.pitchMaxHz) || 240],
      spectralWarmthRatio: Number(profile.spectralWarmthRatio) || 1,
      qualityGrade: profile.qualityGrade || 'Good',
      qualityWarnings: [],
      isValid: true,
      previewUrl: `/api/voices/${encodeURIComponent(profile.voiceId)}/preview`
    };
  }

  function loadSavedVoicesFromVaultDisk(): void {
    try {
      if (fs.existsSync(SAVED_VOICES_INDEX_FILE)) {
        const parsed = JSON.parse(fs.readFileSync(SAVED_VOICES_INDEX_FILE, 'utf8'));
        if (Array.isArray(parsed?.voices)) {
          for (const rec of parsed.voices) {
            if (rec?.voiceId && rec?.ownerUid && rec?.referenceAnalysis) {
              savedVoicesRegistry.set(rec.voiceId, rec as ServerSavedVoiceRecord);
            }
          }
        }
      }
      // Also scan vault user folders to guarantee zero lost records after restarts
      const uidDirs = fs.readdirSync(VOICE_VAULT_DIR, { withFileTypes: true });
      for (const uidEntry of uidDirs) {
        if (!uidEntry.isDirectory() || !/^[a-zA-Z0-9_-]{3,128}$/.test(uidEntry.name)) continue;
        const userDir = path.join(VOICE_VAULT_DIR, uidEntry.name);
        const voiceDirs = fs.readdirSync(userDir, { withFileTypes: true });
        for (const vEntry of voiceDirs) {
          if (!vEntry.isDirectory() || !/^[a-zA-Z0-9_-]{3,128}$/.test(vEntry.name)) continue;
          const profilePath = path.join(userDir, vEntry.name, 'profile.json');
          if (fs.existsSync(profilePath) && !savedVoicesRegistry.has(vEntry.name)) {
            try {
              const rec = JSON.parse(fs.readFileSync(profilePath, 'utf8'));
              if (rec?.voiceId === vEntry.name && rec?.ownerUid === uidEntry.name) {
                savedVoicesRegistry.set(rec.voiceId, rec as ServerSavedVoiceRecord);
              }
            } catch {
              // Ignore corrupted individual profile file
            }
          }
        }
      }
    } catch {
      // Keep clean empty map if disk vault uninitialized
    }
  }
  loadSavedVoicesFromVaultDisk();

  async function persistSavedVoicesIndex(): Promise<void> {
    const tmpFile = `${SAVED_VOICES_INDEX_FILE}.tmp`;
    const payload = JSON.stringify(
      {
        version: 1,
        updatedAt: new Date().toISOString(),
        voices: Array.from(savedVoicesRegistry.values())
      },
      null,
      2
    );
    await fs.promises.writeFile(tmpFile, payload, 'utf8');
    await fs.promises.rename(tmpFile, SAVED_VOICES_INDEX_FILE);
  }

  async function persistSavedVoiceProfileRecord(rec: ServerSavedVoiceRecord): Promise<void> {
    const paths = resolveVaultVoicePaths(rec.ownerUid, rec.voiceId);
    if (paths) {
      await fs.promises.mkdir(paths.dir, { recursive: true });
      await fs.promises.writeFile(paths.profileJsonPath, JSON.stringify(rec, null, 2), 'utf8');
    }
    savedVoicesRegistry.set(rec.voiceId, rec);
    await persistSavedVoicesIndex().catch(() => {});
  }

  const voiceHydrationInFlight = new Map<string, Promise<ServerSavedVoiceRecord | null>>();

  async function ensureSavedVoiceHydratedOnServer(
    voiceId: string,
    identity: VerifiedFirebaseIdentity,
    bearerToken: string | null
  ): Promise<ServerSavedVoiceRecord | null> {
    const existing = savedVoicesRegistry.get(voiceId);
    if (existing) {
      if (existing.ownerUid !== identity.uid) {
        return existing; // Caller checks ownership and returns 403
      }
      if (hasVaultConditioningAudio(existing)) {
        return existing;
      }
    }

    if (!bearerToken) {
      return existing || null;
    }

    const lockKey = `${identity.uid}:${voiceId}`;
    const pending = voiceHydrationInFlight.get(lockKey);
    if (pending) {
      return pending;
    }

    const work = (async (): Promise<ServerSavedVoiceRecord | null> => {
      try {
        const vaultPaths = resolveVaultVoicePaths(identity.uid, voiceId);
        if (!vaultPaths) return existing || null;

        const hydrated = await hydrateSavedVoiceFilesFromFirestore({
          bearerToken,
          voiceId,
          ownerUid: identity.uid,
          vaultDir: vaultPaths.dir,
          refWavPath: vaultPaths.refWavPath,
          promptWavPath: vaultPaths.promptWavPath
        });

        if (!hydrated.profile) {
          return existing || null;
        }

        let refAnalysis = existing?.referenceAnalysis;
        if (hydrated.hydrated && fs.existsSync(vaultPaths.refWavPath)) {
          refAnalysis = await inspectAndValidateAudioFile(
            vaultPaths.refWavPath,
            voiceId,
            hydrated.profile.sourceFileName || `${voiceId}.wav`,
            'audio/wav',
            `/api/voices/${encodeURIComponent(voiceId)}/preview`
          ).catch(() => buildSyntheticAnalysisFromProfile(hydrated.profile!));
        } else if (!refAnalysis) {
          refAnalysis = buildSyntheticAnalysisFromProfile(hydrated.profile);
        }

        const record: ServerSavedVoiceRecord = {
          ...(existing || {}),
          ...hydrated.profile,
          hasConditioningAudio:
            fs.existsSync(vaultPaths.refWavPath) && fs.existsSync(vaultPaths.promptWavPath),
          referenceAnalysis: {
            ...refAnalysis,
            fileId: voiceId,
            previewUrl: `/api/voices/${encodeURIComponent(voiceId)}/preview`
          }
        };

        await persistSavedVoiceProfileRecord(record);
        return record;
      } catch (err) {
        console.warn(`[VoiceLab VaultHydrate] Failed hydrating ${voiceId}:`, err);
        return existing || null;
      } finally {
        voiceHydrationInFlight.delete(lockKey);
      }
    })();

    voiceHydrationInFlight.set(lockKey, work);
    return work;
  }

  function ensureUserEntitlementRecord(uid: string, email: string): ServerUserEntitlement {
    const existing = serverUserEntitlements[uid];
    if (existing) {
      if (typeof existing.tokenBalance !== 'number' || !Number.isFinite(existing.tokenBalance)) {
        existing.tokenBalance = 0;
      }
      if (!existing.email && email) existing.email = email;
      return existing;
    }
    const created: ServerUserEntitlement = {
      uid,
      email,
      tokenBalance: 0,
      voiceCloningUnlocked: false,
      voiceCloningExpiresAtMs: 0,
      longScriptUnlocked: false,
      longScriptExpiresAtMs: 0,
      srtGeneratorUnlocked: false,
      srtGeneratorExpiresAtMs: 0,
      proBundleUnlocked: false,
      proBundleExpiresAtMs: 0,
      updatedAt: new Date().toISOString()
    };
    serverUserEntitlements[uid] = created;
    return created;
  }

  function recordServerAuditLog(
    actorUid: string,
    actorEmail: string,
    action: string,
    target: string,
    details: string
  ) {
    serverAuditLogs.unshift({
      logId: `log_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`,
      actorUid,
      actorEmail,
      action: action.slice(0, 120),
      target: target.slice(0, 254),
      details: details.slice(0, 500),
      createdAt: new Date().toISOString()
    });
    if (serverAuditLogs.length > 200) {
      serverAuditLogs.length = 200;
    }
  }

  interface VerifiedFirebaseIdentity {
    uid: string;
    email: string;
    displayName: string;
    emailVerified: boolean;
    role: 'owner' | 'user';
  }

  const recentSecurityEvents: Array<{
    timestamp: string;
    event: string;
    actor: string;
    detail: string;
  }> = [];
  function recordServerSecurityEvent(event: string, actor: string, detail: string) {
    recentSecurityEvents.unshift({
      timestamp: new Date().toISOString(),
      event,
      actor,
      detail
    });
    if (recentSecurityEvents.length > 50) {
      recentSecurityEvents.length = 50;
    }
  }

  function isOwnerIdentityEmail(emailLower: string): boolean {
    if (!emailLower) return false;
    return (
      OWNER_EMAIL_ALLOWLIST.has(emailLower) ||
      emailLower === serverOwnershipState.primaryOwnerEmail.toLowerCase()
    );
  }

  async function extractVerifiedFirebaseIdentity(
    req: express.Request
  ): Promise<VerifiedFirebaseIdentity | null> {
    const projectId = getFirebaseProjectId();
    if (!projectId) return null;
    const authHeader = String(req.headers.authorization || '').trim();
    if (!authHeader.toLowerCase().startsWith('bearer ')) return null;
    const token = authHeader.slice(7).trim();
    if (!token || token.length < 32) return null;

    try {
      const { payload } = await jwtVerify(token, FIREBASE_JWKS, {
        issuer: `https://securetoken.google.com/${projectId}`,
        audience: projectId
      });
      const uid = typeof payload.sub === 'string' ? payload.sub : '';
      const email = typeof payload.email === 'string' ? payload.email.toLowerCase() : '';
      const emailVerified = payload.email_verified === true;
      const displayName =
        typeof payload.name === 'string' && payload.name.trim()
          ? payload.name.trim().slice(0, 120)
          : email.split('@')[0] || 'User';
      if (!uid || !email || !emailVerified) return null;

      // Owner-only role assignment: verified exclusively against server-side Owner allowlist
      const role: 'owner' | 'user' = isOwnerIdentityEmail(email) ? 'owner' : 'user';
      if (role === 'owner' && !serverOwnershipState.primaryOwnerUid) {
        serverOwnershipState.primaryOwnerUid = uid;
        serverOwnershipState.primaryOwnerEmail = email;
      }

      return { uid, email, displayName, emailVerified, role };
    } catch {
      return null;
    }
  }

  async function requireVerifiedUser(
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
  ) {
    const identity = await extractVerifiedFirebaseIdentity(req);
    if (!identity) {
      res.status(401).type('application/json').json({
        error: 'Verified Google Sign-In required.',
        code: 'AUTH_REQUIRED'
      });
      return;
    }
    const existingUser = serverUsersRegistry[identity.uid];
    if (existingUser && existingUser.accountStatus === 'suspended' && identity.role !== 'owner') {
      res.status(423).type('application/json').json({
        error: 'Your account has been suspended by the workspace owner.',
        code: 'ACCOUNT_SUSPENDED'
      });
      return;
    }
    (req as any).firebaseIdentity = identity;
    next();
  }

  async function requirePrimaryOwner(
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
  ) {
    const identity = await extractVerifiedFirebaseIdentity(req);
    if (!identity) {
      res.status(401).type('application/json').json({
        error: 'Verified Owner authentication required.',
        code: 'OWNER_AUTH_REQUIRED'
      });
      return;
    }
    if (identity.role !== 'owner') {
      recordServerSecurityEvent('UNAUTHORIZED_ADMIN_ATTEMPT', identity.email, req.originalUrl);
      res.status(401).type('application/json').json({
        error: 'Access denied. Administrator controls are restricted exclusively to the Owner.',
        code: 'OWNER_FORBIDDEN'
      });
      return;
    }
    (req as any).firebaseAdmin = identity;
    next();
  }

  async function enforceGovernanceComputeGuard(
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
  ) {
    const identity = await extractVerifiedFirebaseIdentity(req);
    const isOwnerActor = identity?.role === 'owner';

    if (identity && !isOwnerActor) {
      const userRec = serverUsersRegistry[identity.uid];
      if (userRec?.accountStatus === 'suspended') {
        res.status(423).type('application/json').json({
          error: 'Your account is suspended. Contact workspace support to restore access.',
          code: 'ACCOUNT_SUSPENDED'
        });
        return;
      }
    }

    if (serverGovernancePolicy.maintenanceMode && !isOwnerActor) {
      res.status(423).type('application/json').json({
        error:
          serverGovernancePolicy.maintenanceNotice ||
          'VoiceLab AI is currently undergoing scheduled maintenance.',
        code: 'MAINTENANCE_MODE'
      });
      return;
    }

    if (serverGovernancePolicy.requireAuthForCompute && !identity) {
      res.status(401).type('application/json').json({
        error: 'Verified Google Sign-In is required before running audio processing jobs.',
        code: 'COMPUTE_AUTH_REQUIRED'
      });
      return;
    }

    (req as any).firebaseIdentity = identity;
    next();
  }

  function isEntitlementActive(unlocked: boolean | undefined, expiresAtMs: number | undefined): boolean {
    if (!unlocked) return false;
    if (!expiresAtMs || expiresAtMs === 0) return true;
    return expiresAtMs > Date.now();
  }

  function checkFeatureAccessGate(
    fbIdentity: VerifiedFirebaseIdentity | null,
    featureKey: FeatureGateKey
  ): {
    allowed: boolean;
    statusCode: number;
    code?: string;
    error?: string;
    tokenCost: number;
  } {
    const cfg = serverFeatureConfigs[featureKey];
    const isOwnerActor = fbIdentity?.role === 'owner';

    if (!cfg.enabled && !isOwnerActor) {
      return {
        allowed: false,
        statusCode: 423,
        code: 'FEATURE_DISABLED_BY_ADMIN',
        error: `${cfg.title} is currently disabled by administrator policy.`,
        tokenCost: cfg.tokenCost
      };
    }

    if (isOwnerActor) {
      return { allowed: true, statusCode: 200, tokenCost: 0 };
    }

    if (cfg.isPaid) {
      if (!fbIdentity) {
        return {
          allowed: false,
          statusCode: 401,
          code: 'AUTH_REQUIRED_FOR_PAID_FEATURE',
          error: `${cfg.title} is a premium feature. Sign in with Google to upgrade or request access.`,
          tokenCost: cfg.tokenCost
        };
      }
      const ent = serverUserEntitlements[fbIdentity.uid];
      const hasBundle = ent ? isEntitlementActive(ent.proBundleUnlocked, ent.proBundleExpiresAtMs) : false;
      const hasSpecific = ent
        ? featureKey === 'voice_cloning'
          ? isEntitlementActive(ent.voiceCloningUnlocked, ent.voiceCloningExpiresAtMs)
          : featureKey === 'long_script'
          ? isEntitlementActive(ent.longScriptUnlocked, ent.longScriptExpiresAtMs)
          : isEntitlementActive(ent.srtGeneratorUnlocked, ent.srtGeneratorExpiresAtMs)
        : false;
      if (!hasBundle && !hasSpecific) {
        return {
          allowed: false,
          statusCode: 402,
          code: 'PAYMENT_REQUIRED',
          error: `${cfg.title} requires an active premium entitlement. Open Upgrade / Request Access to purchase via bKash or Nagad.`,
          tokenCost: cfg.tokenCost
        };
      }
    }

    return { allowed: true, statusCode: 200, tokenCost: cfg.tokenCost };
  }

  function applyAtomicTokenDelta(params: {
    uid: string;
    email: string;
    delta: number;
    type: ServerTokenLedgerEntry['type'];
    reason: string;
    actorUid: string;
  }): { ok: boolean; balanceAfter: number; entry?: ServerTokenLedgerEntry; error?: string } {
    const ent = ensureUserEntitlementRecord(params.uid, params.email);
    const nextBalance = ent.tokenBalance + params.delta;
    if (nextBalance < 0) {
      return {
        ok: false,
        balanceAfter: ent.tokenBalance,
        error: `Insufficient token balance (required ${Math.abs(params.delta)}, available ${ent.tokenBalance}).`
      };
    }
    ent.tokenBalance = nextBalance;
    ent.updatedAt = new Date().toISOString();

    const entry: ServerTokenLedgerEntry = {
      entryId: `tok_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`,
      userUid: params.uid,
      userEmail: params.email || ent.email,
      delta: params.delta,
      balanceAfter: nextBalance,
      type: params.type,
      reason: params.reason.slice(0, 240),
      actorUid: params.actorUid,
      createdAt: new Date().toISOString()
    };
    serverTokenLedger.unshift(entry);
    if (serverTokenLedger.length > 500) {
      serverTokenLedger.length = 500;
    }
    return { ok: true, balanceAfter: nextBalance, entry };
  }

  // Heavy AI compute rate-limiter middleware (governed dynamically by Owner policy)
  function enforceComputeRateLimit(
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
  ) {
    const clientIp = String(req.ip || req.socket.remoteAddress || '127.0.0.1');
    const limitPerMin = Math.max(1, Math.min(60, serverGovernancePolicy.rateLimitPerMinute || 25));
    const status = checkSlidingRateLimit(computeRateBuckets, clientIp, limitPerMin, 60_000);
    if (!status.allowed) {
      res.status(429).type('application/json').json({
        error: `AI synthesis/transcription rate limit reached (${limitPerMin} requests/minute). Please wait before submitting another job.`,
        retryAfterSeconds: status.resetSeconds
      });
      return;
    }
    next();
  }

  // 1. System Health & Environment Info
  app.get('/api/health', (_req, res) => {
    const resolvedKey = resolveRuntimeGeminiApiKey();
    const hasGeminiKey = Boolean(
      resolvedKey && resolvedKey !== 'MY_GEMINI_API_KEY' && resolvedKey.length > 16
    );
    res.status(200).type('application/json').json({
      status: 'operational',
      application: 'VoiceLab AI',
      version: '3.3.0',
      environment: IS_PRODUCTION ? 'production' : 'development',
      uptimeSeconds: Math.round(process.uptime()),
      ffmpegAvailable: ffmpegBinaryVerified,
      ffprobeAvailable: ffprobeBinaryVerified,
      ffmpegVersion: ffmpegVersionSummary,
      capabilities: {
        cloudGeminiConfigured: hasGeminiKey,
        sessionSecretConfigured: true,
        sessionSecretSource: SESSION_CONFIG.source,
        sessionSecretEntropyBits: SESSION_CONFIG.entropyBits,
        apiRoutingMode: 'same-origin-relative (/api/*)',
        localNeuralNodeMode: 'loopback-or-https-tunnel (127.0.0.1 / Colab T4 / Kaggle GPU)',
        sameLanguageCloningLanguages: ['bn', 'hi', 'en'],
        multilingualSrtTranscription: true
      },
      storagePolicy: 'Permanent Cloud FLAC Vault (Firestore) + Persistent Workspace Storage (no auto-expiry)',
      activeTemporaryFiles: uploadedFilesRegistry.size
    });
  });

  // 1b. Session & Security Status Endpoint (Initializes or refreshes session)
  app.get('/api/auth/session', (req, res) => {
    const sess = (req as any).voicelabSession as VerifiedSessionState | undefined;
    const isHttps = isHttpsPublicRequest(req);
    res.status(200).type('application/json').json({
      authenticated: Boolean(sess?.valid),
      sessionId: sess?.sessionId || null,
      issuedAt: sess?.issuedAtMs ? new Date(sess.issuedAtMs).toISOString() : null,
      csrfProtected: true,
      sessionSecretSource: SESSION_CONFIG.source,
      sessionSecretEntropyBits: SESSION_CONFIG.entropyBits,
      cookieSecurity: {
        httpOnly: true,
        sameSite: isHttps ? 'None' : 'Lax',
        secure: isHttps,
        partitioned: isHttps
      }
    });
  });

  // 1c. Live Operation Progress Endpoints (SSE stream & JSON snapshot)
  app.get('/api/operations/:operationId', (req, res) => {
    const opId = String(req.params.operationId || '').trim();
    if (!opId || !SAFE_IDENTIFIER_REGEX.test(opId)) {
      res.status(400).type('application/json').json({ error: 'Invalid operationId.' });
      return;
    }
    const op = operationProgressRegistry.get(opId) || null;
    res.status(200).type('application/json').json({ operation: op });
  });

  app.get('/api/operations/:operationId/events', (req, res) => {
    const opId = String(req.params.operationId || '').trim();
    if (!opId || !SAFE_IDENTIFIER_REGEX.test(opId)) {
      res.status(400).type('application/json').json({ error: 'Invalid operationId.' });
      return;
    }

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders?.();

    const current = operationProgressRegistry.get(opId);
    if (current) {
      res.write(`data: ${JSON.stringify(current)}\n\n`);
      if (current.status === 'succeeded' || current.status === 'failed') {
        res.end();
        return;
      }
    }

    let subs = operationSseSubscribers.get(opId);
    if (!subs) {
      subs = new Set();
      operationSseSubscribers.set(opId, subs);
    }
    subs.add(res);

    req.on('close', () => {
      const set = operationSseSubscribers.get(opId);
      if (set) {
        set.delete(res);
        if (set.size === 0) {
          operationSseSubscribers.delete(opId);
        }
      }
    });
  });

  // 2. Load or Synthesize Real Authorized Reference Sample Audio (Workspace A: Voice Cloning only)
  app.get('/api/sample-reference/:lang', async (req, res) => {
    const operationId = extractOperationIdFromReq(req);
    try {
      const lang = req.params.lang as CloningLanguageCode;
      if (!['bn', 'hi', 'en'].includes(lang)) {
        res.status(400).type('application/json').json({
          error: 'Sample language must be bn, hi, or en.'
        });
        return;
      }

      emitServerOperationProgress(operationId, {
        stageIndex: 1,
        stageLabel: `Preparing & inspecting ${lang.toUpperCase()} studio reference audio...`,
        progressPct: null,
        progressSource: 'indeterminate-inference',
        status: 'running'
      });

      const shimMap = new Map<string, { filePath: string; analysis: AudioSignalAnalysis }>();
      for (const [k, v] of uploadedFilesRegistry.entries()) {
        shimMap.set(k, { filePath: v.filePath, analysis: v.analysis });
      }

      const analysis = await getOrCreateAuthorizedSampleAudio(lang, SAMPLES_DIR, shimMap);
      const updated = shimMap.get(`sample_ref_${lang}`);
      if (updated) {
        uploadedFilesRegistry.set(`sample_ref_${lang}`, {
          filePath: updated.filePath,
          analysis: updated.analysis,
          createdAtMs: Date.now()
        });
      }

      emitServerOperationProgress(operationId, {
        stageIndex: 2,
        stageLabel: 'Studio reference voice loaded',
        progressPct: 100,
        progressSource: 'completed',
        status: 'succeeded'
      });

      res.status(200).type('application/json').json({ analysis });
    } catch (err: any) {
      const quotaInfo = extractQuotaExhaustedInfo(err);
      const status =
        err instanceof QuotaExhaustedError || quotaInfo.isQuotaExhausted ? 429 : 400;
      emitServerOperationProgress(operationId, {
        stageIndex: 1,
        stageLabel: err?.message || 'Failed to load sample reference',
        progressPct: null,
        progressSource: 'indeterminate-inference',
        status: 'failed'
      });
      res.status(status).type('application/json').json({
        error: err?.message || 'Failed to prepare authorized sample reference recording.',
        code: status === 429 ? 'RESOURCE_EXHAUSTED' : 'SAMPLE_LOAD_FAILED',
        retryAfterText: err?.retryAfterText || quotaInfo.retryAfterText
      });
    }
  });

  // 3. Stream Sample Audio
  app.get('/api/audio/sample/:lang', async (req, res) => {
    const lang = req.params.lang as CloningLanguageCode;
    if (!['bn', 'hi', 'en'].includes(lang)) {
      res.status(400).type('application/json').json({ error: 'Invalid sample language.' });
      return;
    }
    const entry = uploadedFilesRegistry.get(`sample_ref_${lang}`);
    if (!entry || !fs.existsSync(entry.filePath)) {
      res.status(404).type('application/json').json({ error: 'Sample audio not yet initialized.' });
      return;
    }
    res.setHeader('Content-Type', 'audio/wav');
    fs.createReadStream(entry.filePath).pipe(res);
  });

  // 4. Upload & Validate Audio File (Used by Workspace A and Workspace B)
  app.post(
    '/api/audio/upload',
    requireAuthenticatedSession,
    enforceGovernanceComputeGuard,
    handleAudioUploadMiddleware,
    async (req, res) => {
      const operationId = extractOperationIdFromReq(req);
      try {
        if (!req.file || !req.file.path) {
          res.status(400).type('application/json').json({ error: 'No audio file provided.' });
          return;
        }

        emitServerOperationProgress(operationId, {
          stageIndex: 1,
          stageLabel: 'Verifying audio container signature & running FFprobe/FFmpeg DSP...',
          progressPct: null,
          progressSource: 'indeterminate-inference',
          status: 'running'
        });

        // Verify binary magic bytes before running FFprobe/FFmpeg
        const hasValidSignature = await verifyAudioFileMagicBytes(req.file.path);
        if (!hasValidSignature) {
          await fs.promises.unlink(req.file.path).catch(() => {});
          const sigErr =
            'Security validation failed: File binary header does not match a valid audio container (WAV, MP3, FLAC, OGG, M4A, AAC, WEBM).';
          emitServerOperationProgress(operationId, {
            stageIndex: 1,
            stageLabel: sigErr,
            progressPct: null,
            progressSource: 'indeterminate-inference',
            status: 'failed'
          });
          res.status(400).type('application/json').json({
            error: sigErr
          });
          return;
        }

        const workspace = String(req.body.workspace || 'voice-cloning');
        const fileId = path.basename(req.file.filename, path.extname(req.file.filename));
        const previewUrl = `/api/audio/uploaded/${fileId}`;

        let activeAudioPath = req.file.path;
        let effectiveMimeType = req.file.mimetype || 'audio/wav';
        let autoTrimMeta: {
          originalDurationSec: number;
          windowStartSec: number;
          windowEndSec: number;
        } | null = null;

        if (workspace === 'voice-cloning') {
          const trimmed10sPath = path.join(UPLOAD_DIR, `${fileId}_10s.wav`);
          const autoExtract = await extractAuto10SecVoiceCloneReference(
            req.file.path,
            trimmed10sPath,
            10
          );
          if ((autoExtract.trimmed || autoExtract.normalized) && fs.existsSync(autoExtract.outputPath)) {
            activeAudioPath = autoExtract.outputPath;
            effectiveMimeType = 'audio/wav';
            if (autoExtract.trimmed) {
              autoTrimMeta = {
                originalDurationSec: autoExtract.originalDurationSec,
                windowStartSec: autoExtract.windowStartSec,
                windowEndSec: autoExtract.windowEndSec
              };
            }
            if (autoExtract.outputPath !== req.file.path) {
              await fs.promises.unlink(req.file.path).catch(() => {});
            }
            emitServerOperationProgress(operationId, {
              stageIndex: 1,
              stageLabel: autoExtract.trimmed
                ? `Auto-cut cleanest 10.0s speech clip from ${autoExtract.originalDurationSec}s reference audio...`
                : 'Auto-normalized quiet reference audio for clean speaker conditioning...',
              progressPct: null,
              progressSource: 'indeterminate-inference',
              status: 'running'
            });
          }
        }

        const analysis = await inspectAndValidateAudioFile(
          activeAudioPath,
          fileId,
          req.file.originalname,
          effectiveMimeType,
          previewUrl
        );

        if (autoTrimMeta) {
          analysis.autoTrimmedFromSec = autoTrimMeta.originalDurationSec;
          analysis.autoTrimWindowStartSec = autoTrimMeta.windowStartSec;
          analysis.autoTrimWindowEndSec = autoTrimMeta.windowEndSec;
        }

        if (!analysis.isValid) {
          await fs.promises.unlink(activeAudioPath).catch(() => {});
          if (activeAudioPath !== req.file.path) {
            await fs.promises.unlink(req.file.path).catch(() => {});
          }
          const valErr =
            analysis.qualityWarnings.join(' ') ||
            'Uploaded audio is empty, corrupted, or contains only silence.';
          emitServerOperationProgress(operationId, {
            stageIndex: 1,
            stageLabel: valErr,
            progressPct: null,
            progressSource: 'indeterminate-inference',
            status: 'failed'
          });
          res.status(422).type('application/json').json({
            error: valErr
          });
          return;
        }

        // If uploaded in Voice Cloning Studio, also run spoken language & vocal identity extraction
        if (workspace === 'voice-cloning') {
          emitServerOperationProgress(operationId, {
            stageIndex: 2,
            stageLabel: 'Analyzing spoken language & 10s reference vocal identity...',
            progressPct: null,
            progressSource: 'indeterminate-inference',
            status: 'running'
          });
          analysis.detectedSpokenLanguage = await analyzeReferenceVoiceLanguageAndIdentity(
            activeAudioPath,
            analysis
          );
        }

        await saveUploadedFileEntry(fileId, activeAudioPath, analysis);

        emitServerOperationProgress(operationId, {
          stageIndex: workspace === 'voice-cloning' ? 2 : 1,
          stageLabel: autoTrimMeta
            ? `Auto-cut 10.0s from ${autoTrimMeta.originalDurationSec}s audio — ready for cloning`
            : 'Audio validated and ready',
          progressPct: 100,
          progressSource: 'completed',
          status: 'succeeded'
        });

        res.status(200).type('application/json').json({ analysis });
      } catch (err: any) {
        if (req.file?.path) {
          const fallbackFileId = path.basename(req.file.filename, path.extname(req.file.filename));
          await fs.promises.unlink(req.file.path).catch(() => {});
          await fs.promises.unlink(path.join(UPLOAD_DIR, `${fallbackFileId}_10s.wav`)).catch(() => {});
        }
        emitServerOperationProgress(operationId, {
          stageIndex: 1,
          stageLabel: err?.message || 'Audio validation failed',
          progressPct: null,
          progressSource: 'indeterminate-inference',
          status: 'failed'
        });
        res.status(400).type('application/json').json({
          error: err?.message || 'Failed to validate uploaded audio file.'
        });
      }
    }
  );

  // 5. Stream Uploaded Audio File
  app.get('/api/audio/uploaded/:fileId', async (req, res) => {
    const rawId = String(req.params.fileId || '');
    if (!SAFE_IDENTIFIER_REGEX.test(rawId) || rawId.includes('..')) {
      res.status(400).type('application/json').json({ error: 'Invalid audio file identifier.' });
      return;
    }
    const entry = await resolveUploadedFileEntry(rawId);
    if (!entry || !fs.existsSync(entry.filePath)) {
      res.status(404).type('application/json').json({
        error: 'Audio recording not found or expired.'
      });
      return;
    }
    res.setHeader('Content-Type', entry.analysis.mimeType || 'audio/wav');
    fs.createReadStream(entry.filePath).pipe(res);
  });

  // 6. Stream / Download Generated Cloned Audio (WAV or MP3)
  app.get('/api/audio/generated/:filename', (req, res) => {
    const fullPath = resolveSafeChildPath(GENERATED_DIR, String(req.params.filename || ''));
    if (!fullPath || !fs.existsSync(fullPath)) {
      res.status(404).type('application/json').json({
        error: 'Generated audio file not found.'
      });
      return;
    }
    const safeName = path.basename(fullPath);
    const isMp3 = safeName.endsWith('.mp3');
    res.setHeader('Content-Type', isMp3 ? 'audio/mpeg' : 'audio/wav');
    if (req.query.download === '1') {
      res.setHeader('Content-Disposition', `attachment; filename="voicelab_${safeName}"`);
    }
    fs.createReadStream(fullPath).pipe(res);
  });

  // 6b. Check Mode 2 Zero-Shot Neural Node Readiness (Loopback or SSRF-guarded HTTPS GPU tunnel)
  app.post(
    '/api/voice-clone/node-health',
    requireAuthenticatedSession,
    async (req, res) => {
      try {
        const rawUrl = String(
          req.body?.localEndpointUrl || process.env.LOCAL_NEURAL_CLONE_URL || ''
        ).trim();
        if (!rawUrl) {
          res.status(200).type('application/json').json({
            configured: false,
            reachable: false,
            error: 'No Mode 2 Neural Node URL configured.'
          });
          return;
        }
        const origin = await assertSafeNeuralNodeUrl(rawUrl);
        const rawToken =
          typeof req.body?.localNodeToken === 'string' ? req.body.localNodeToken.trim() : '';
        const token = rawToken || process.env.LOCAL_NODE_AUTH_TOKEN || 'voicelab-local-token';
        const resp = await fetch(`${origin}/health`, {
          method: 'GET',
          headers: {
            Accept: 'application/json',
            'X-VoiceLab-Node-Token': token,
            'ngrok-skip-browser-warning': 'true'
          },
          signal: AbortSignal.timeout(7000)
        });
        const bodyText = await resp.text();
        let parsed: any = null;
        try {
          parsed = bodyText ? JSON.parse(bodyText) : null;
        } catch {
          parsed = null;
        }
        res.status(200).type('application/json').json({
          configured: true,
          origin,
          reachable: resp.ok,
          httpStatus: resp.status,
          health: parsed,
          error: resp.ok
            ? undefined
            : parsed?.detail ||
              parsed?.error ||
              bodyText ||
              `Node responded with HTTP ${resp.status}`
        });
      } catch (err: any) {
        res.status(422).type('application/json').json({
          configured: true,
          reachable: false,
          error: err?.message || 'Mode 2 Neural Node unreachable.'
        });
      }
    }
  );

  interface VoiceCloneJobRecord {
    jobId: string;
    operationId: string | null;
    sessionId: string;
    uid: string | null;
    status: 'running' | 'succeeded' | 'failed' | 'cancelled';
    result?: VoiceCloneResult;
    error?: string;
    code?: string;
    httpStatus?: number;
    retryAfterText?: string;
    deductedTokens: number;
    deductedUser: VerifiedFirebaseIdentity | null;
    refunded: boolean;
    abortController: AbortController;
    createdAtMs: number;
    updatedAtMs: number;
    latestProgress?: ServerOperationProgress;
  }

  const voiceCloneJobsRegistry = new Map<string, VoiceCloneJobRecord>();

  function pruneOldVoiceCloneJobs(): void {
    const cutoff = Date.now() - 24 * 60 * 60 * 1000;
    for (const [id, rec] of voiceCloneJobsRegistry.entries()) {
      if (rec.status !== 'running' && rec.updatedAtMs < cutoff) {
        voiceCloneJobsRegistry.delete(id);
      }
    }
    if (voiceCloneJobsRegistry.size > 300) {
      const oldest = Array.from(voiceCloneJobsRegistry.values())
        .filter((r) => r.status !== 'running')
        .sort((a, b) => a.updatedAtMs - b.updatedAtMs);
      for (const r of oldest.slice(0, voiceCloneJobsRegistry.size - 220)) {
        voiceCloneJobsRegistry.delete(r.jobId);
      }
    }
  }

  async function refundVoiceCloneJobTokensOnce(job: VoiceCloneJobRecord, reason: string): Promise<void> {
    if (job.refunded || job.deductedTokens <= 0 || !job.deductedUser) return;
    job.refunded = true;
    applyAtomicTokenDelta({
      uid: job.deductedUser.uid,
      email: job.deductedUser.email,
      delta: job.deductedTokens,
      type: 'refund',
      reason,
      actorUid: 'system'
    });
    await persistGovernanceState();
  }

  // 7. WORKSPACE A — Same-Language AI Voice Cloning Generation
  app.post(
    '/api/voice-clone/generate',
    requireAuthenticatedSession,
    enforceGovernanceComputeGuard,
    enforceComputeRateLimit,
    async (req, res) => {
      req.setTimeout?.(20 * 60 * 1000);
      res.setTimeout?.(20 * 60 * 1000);
      let deductedTokens = 0;
      let deductedUser: VerifiedFirebaseIdentity | null = null;
      try {
        const fbIdentity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity | null;
        const sess = (req as any).voicelabSession as VerifiedSessionState | undefined;
        const isOwnerActor = fbIdentity?.role === 'owner';
        if (!serverGovernancePolicy.allowVoiceCloning && !isOwnerActor) {
          res.status(423).type('application/json').json({
            error: 'Voice Cloning is currently paused by the Administrator.',
            code: 'FEATURE_DISABLED_BY_ADMIN'
          });
          return;
        }

        const {
          referenceFileId,
          savedVoiceId,
          language,
          text,
          referenceTranscript,
          audioQuality = 'studio',
          preservePitchContour = true,
          preserveSpectralTimbre = true,
          reuseCompletedChunks = true,
          consentConfirmed = false,
          localEndpointUrl,
          localNodeToken,
          enableCloudVerification = false,
          asyncJob = false
        } = req.body || {};

        const validatedAudioQuality: VoiceAudioQualityMode =
          audioQuality === 'standard' || audioQuality === 'studio_plus' ? audioQuality : 'studio';

        if (!consentConfirmed) {
          res.status(400).type('application/json').json({
            error:
              'Consent confirmation required: You must confirm that you own or have explicit authorization to clone this reference voice.'
          });
          return;
        }

        if (String(text || '').length > serverGovernancePolicy.maxScriptChars) {
          res.status(400).type('application/json').json({
            error: `Script exceeds the administrator limit of ${serverGovernancePolicy.maxScriptChars} characters.`
          });
          return;
        }

        if (!['bn', 'hi', 'en'].includes(language)) {
          res.status(400).type('application/json').json({
            error: 'Invalid language selection. Supported same-language cloning options: bn, hi, en.'
          });
          return;
        }

        const vcGate = checkFeatureAccessGate(fbIdentity, 'voice_cloning');
        if (!vcGate.allowed) {
          res.status(vcGate.statusCode).type('application/json').json({
            error: vcGate.error,
            code: vcGate.code,
            featureKey: 'voice_cloning'
          });
          return;
        }

        const spokenCharsCount = String(text || '').replace(/\[[^\]]+\]/g, '').trim().length;
        let longScriptTokenCost = 0;
        if (spokenCharsCount > 1000) {
          const longGate = checkFeatureAccessGate(fbIdentity, 'long_script');
          if (!longGate.allowed) {
            res.status(longGate.statusCode).type('application/json').json({
              error: longGate.error,
              code: longGate.code,
              featureKey: 'long_script'
            });
            return;
          }
          longScriptTokenCost = longGate.tokenCost;
        }

        let refEntry: { filePath: string; analysis: AudioSignalAnalysis; createdAtMs: number } | null = null;
        let preConditionedPromptPath: string | undefined;
        let effectiveSavedVoiceId: string | undefined;
        let effectiveReferenceTranscript =
          typeof referenceTranscript === 'string' ? referenceTranscript : undefined;

        const bearerToken = extractBearerTokenFromAuthHeader(req.headers.authorization);
        const cleanSavedVoiceId = typeof savedVoiceId === 'string' ? savedVoiceId.trim() : '';
        if (cleanSavedVoiceId) {
          if (!SAFE_IDENTIFIER_REGEX.test(cleanSavedVoiceId)) {
            res.status(400).type('application/json').json({
              error: 'Invalid saved voice profile identifier.'
            });
            return;
          }
          if (!fbIdentity) {
            res.status(401).type('application/json').json({
              error: 'Verified Google Sign-In required to generate using a saved voice profile.',
              code: 'AUTH_REQUIRED_FOR_SAVED_VOICE'
            });
            return;
          }
          const savedRec = await ensureSavedVoiceHydratedOnServer(
            cleanSavedVoiceId,
            fbIdentity,
            bearerToken
          );
          if (!savedRec) {
            res.status(404).type('application/json').json({
              error: 'Saved voice profile not found.',
              code: 'SAVED_VOICE_NOT_FOUND'
            });
            return;
          }
          if (savedRec.ownerUid !== fbIdentity.uid) {
            recordServerSecurityEvent(
              'UNAUTHORIZED_SAVED_VOICE_USE',
              fbIdentity.email,
              `Attempted to clone using savedVoiceId=${cleanSavedVoiceId} owned by another UID`
            );
            res.status(403).type('application/json').json({
              error: 'Access denied: You can only generate audio using your own saved voice profiles.',
              code: 'VOICE_FORBIDDEN'
            });
            return;
          }
          if (!savedRec.consentConfirmed) {
            res.status(400).type('application/json').json({
              error: 'Saved voice profile lacks valid authorization consent.',
              code: 'CONSENT_REQUIRED'
            });
            return;
          }
          if (savedRec.language !== language) {
            res.status(400).type('application/json').json({
              error: `Cross-Language Blocked: Saved voice "${savedRec.name}" is locked to ${savedRec.languageName} (${savedRec.language.toUpperCase()}), not ${String(language).toUpperCase()}.`,
              code: 'CROSS_LANGUAGE_CLONING_BLOCKED'
            });
            return;
          }
          const vaultPaths = resolveVaultVoicePaths(savedRec.ownerUid, savedRec.voiceId);
          if (
            !vaultPaths ||
            !fs.existsSync(vaultPaths.refWavPath) ||
            !fs.existsSync(vaultPaths.promptWavPath)
          ) {
            res.status(422).type('application/json').json({
              error:
                'Saved voice speaker-conditioning audio is unavailable on server storage and cloud vault. VoiceLab AI refuses to substitute a generic TTS voice — please re-upload or record this voice again.',
              code: 'SAVED_VOICE_AUDIO_MISSING'
            });
            return;
          }
          refEntry = {
            filePath: vaultPaths.refWavPath,
            analysis: {
              ...savedRec.referenceAnalysis,
              fileId: savedRec.voiceId,
              previewUrl: `/api/voices/${encodeURIComponent(savedRec.voiceId)}/preview`
            },
            createdAtMs: Date.now()
          };
          preConditionedPromptPath = vaultPaths.promptWavPath;
          effectiveSavedVoiceId = savedRec.voiceId;
          if (!effectiveReferenceTranscript || !effectiveReferenceTranscript.trim()) {
            effectiveReferenceTranscript = savedRec.referenceTranscript || undefined;
          }
        } else {
          refEntry =
            uploadedFilesRegistry.get(referenceFileId) ||
            (await resolveUploadedFileEntry(referenceFileId));
          if (!refEntry || !fs.existsSync(refEntry.filePath)) {
            res.status(404).type('application/json').json({
              error: 'Reference audio recording not found. Please upload or select a reference voice.'
            });
            return;
          }
        }

        const totalRequiredTokens = isOwnerActor ? 0 : vcGate.tokenCost + longScriptTokenCost;
        if (totalRequiredTokens > 0) {
          if (!fbIdentity) {
            res.status(401).type('application/json').json({
              error: 'Sign in with Google required to use token-gated Voice Cloning.',
              code: 'AUTH_REQUIRED_FOR_TOKEN_FEATURE',
              requiredTokens: totalRequiredTokens
            });
            return;
          }
          const debit = applyAtomicTokenDelta({
            uid: fbIdentity.uid,
            email: fbIdentity.email,
            delta: -totalRequiredTokens,
            type: 'feature_debit',
            reason: `Voice Cloning synthesis (${spokenCharsCount} chars)`,
            actorUid: fbIdentity.uid
          });
          if (!debit.ok) {
            res.status(402).type('application/json').json({
              error: debit.error,
              code: 'INSUFFICIENT_TOKENS',
              requiredTokens: totalRequiredTokens,
              tokenBalance: debit.balanceAfter
            });
            return;
          }
          deductedTokens = totalRequiredTokens;
          deductedUser = fbIdentity;
          await persistGovernanceState();
        }

        const effectiveCloudVerification = serverGovernancePolicy.enforceZeroGeminiQuota
          ? false
          : Boolean(enableCloudVerification);

        const operationId = extractOperationIdFromReq(req);
        const jobId = `clone_${language}_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`;
        const abortController = new AbortController();

        if (asyncJob === true) {
          pruneOldVoiceCloneJobs();
          const jobRecord: VoiceCloneJobRecord = {
            jobId,
            operationId,
            sessionId: sess?.sessionId || '',
            uid: fbIdentity?.uid || null,
            status: 'running',
            deductedTokens,
            deductedUser,
            refunded: false,
            abortController,
            createdAtMs: Date.now(),
            updatedAtMs: Date.now()
          };
          voiceCloneJobsRegistry.set(jobId, jobRecord);

          // Prevent duplicate refund in outer catch once background runner owns the lifecycle
          deductedTokens = 0;
          deductedUser = null;

          res.status(202).type('application/json').json({
            status: 'queued',
            jobId,
            operationId
          });

          void (async () => {
            try {
              const result = await executeSameLanguageVoiceCloning({
                jobId,
                referenceFilePath: refEntry.filePath,
                preConditionedPromptPath,
                savedVoiceId: effectiveSavedVoiceId,
                referenceAnalysis: refEntry.analysis,
                language: language as CloningLanguageCode,
                text: String(text || ''),
                referenceTranscript: effectiveReferenceTranscript,
                audioQuality: validatedAudioQuality,
                preservePitchContour: Boolean(preservePitchContour),
                preserveSpectralTimbre: Boolean(preserveSpectralTimbre),
                reuseCompletedChunks: Boolean(reuseCompletedChunks),
                outputDir: GENERATED_DIR,
                localEndpointUrl: typeof localEndpointUrl === 'string' ? localEndpointUrl : undefined,
                localNodeToken: typeof localNodeToken === 'string' ? localNodeToken : undefined,
                enableCloudVerification: effectiveCloudVerification,
                abortSignal: abortController.signal,
                onProgress: (ev) => {
                  const snap: ServerOperationProgress = {
                    operationId: operationId || jobId,
                    ...ev,
                    status: 'running',
                    updatedAtMs: Date.now()
                  };
                  jobRecord.latestProgress = snap;
                  jobRecord.updatedAtMs = snap.updatedAtMs;
                  emitServerOperationProgress(operationId, {
                    ...ev,
                    status: 'running'
                  });
                }
              });

              if (jobRecord.status === 'cancelled' || abortController.signal.aborted) {
                await refundVoiceCloneJobTokensOnce(
                  jobRecord,
                  'Automatic refund: Voice Cloning job was cancelled'
                );
                return;
              }

              if (effectiveSavedVoiceId) {
                const sv = savedVoicesRegistry.get(effectiveSavedVoiceId);
                if (sv) {
                  const usedIso = new Date().toISOString();
                  sv.usageCount = (sv.usageCount || 0) + 1;
                  sv.lastUsedAt = usedIso;
                  sv.updatedAt = usedIso;
                  await persistSavedVoiceProfileRecord(sv);
                  if (bearerToken) {
                    void incrementSavedVoiceUsageInFirestore(bearerToken, effectiveSavedVoiceId);
                  }
                }
              }

              jobRecord.status = 'succeeded';
              jobRecord.result = result;
              jobRecord.updatedAtMs = Date.now();
              emitServerOperationProgress(operationId, {
                stageIndex: 4,
                stageLabel: 'Voice cloning completed',
                progressPct: 100,
                progressSource: 'completed',
                status: 'succeeded'
              });
            } catch (err: any) {
              const wasCancelled =
                jobRecord.status === 'cancelled' ||
                abortController.signal.aborted ||
                err?.code === 'JOB_CANCELLED';
              await refundVoiceCloneJobTokensOnce(
                jobRecord,
                wasCancelled
                  ? 'Automatic refund: Voice Cloning job cancelled'
                  : 'Automatic refund: Voice Cloning pipeline failed before completion'
              );
              const quotaInfo = extractQuotaExhaustedInfo(err);
              const rawStatus =
                typeof err?.httpStatus === 'number'
                  ? err.httpStatus
                  : quotaInfo.isQuotaExhausted
                  ? 429
                  : 400;
              const safeStatus = rawStatus === 403 ? 401 : rawStatus;
              jobRecord.status = wasCancelled ? 'cancelled' : 'failed';
              jobRecord.error = err?.message || 'Voice cloning pipeline failed.';
              jobRecord.code =
                err?.code || (safeStatus === 429 ? 'RESOURCE_EXHAUSTED' : 'VOICE_CLONE_FAILED');
              jobRecord.httpStatus = safeStatus;
              jobRecord.retryAfterText = err?.retryAfterText || quotaInfo.retryAfterText;
              jobRecord.updatedAtMs = Date.now();
              emitServerOperationProgress(operationId, {
                stageIndex: 1,
                stageLabel: jobRecord.error || 'Voice cloning failed',
                progressPct: null,
                progressSource: 'indeterminate-inference',
                status: 'failed'
              });
            }
          })();
          return;
        }

        res.on('close', () => {
          if (!res.writableEnded) {
            abortController.abort();
          }
        });

        const result = await executeSameLanguageVoiceCloning({
          jobId,
          referenceFilePath: refEntry.filePath,
          preConditionedPromptPath,
          savedVoiceId: effectiveSavedVoiceId,
          referenceAnalysis: refEntry.analysis,
          language: language as CloningLanguageCode,
          text: String(text || ''),
          referenceTranscript: effectiveReferenceTranscript,
          audioQuality: validatedAudioQuality,
          preservePitchContour: Boolean(preservePitchContour),
          preserveSpectralTimbre: Boolean(preserveSpectralTimbre),
          reuseCompletedChunks: Boolean(reuseCompletedChunks),
          outputDir: GENERATED_DIR,
          localEndpointUrl: typeof localEndpointUrl === 'string' ? localEndpointUrl : undefined,
          localNodeToken: typeof localNodeToken === 'string' ? localNodeToken : undefined,
          enableCloudVerification: effectiveCloudVerification,
          abortSignal: abortController.signal,
          onProgress: (ev) => {
            emitServerOperationProgress(operationId, {
              ...ev,
              status: 'running'
            });
          }
        });

        if (effectiveSavedVoiceId) {
          const sv = savedVoicesRegistry.get(effectiveSavedVoiceId);
          if (sv) {
            const usedIso = new Date().toISOString();
            sv.usageCount = (sv.usageCount || 0) + 1;
            sv.lastUsedAt = usedIso;
            sv.updatedAt = usedIso;
            await persistSavedVoiceProfileRecord(sv);
            if (bearerToken) {
              void incrementSavedVoiceUsageInFirestore(bearerToken, effectiveSavedVoiceId);
            }
          }
        }

        emitServerOperationProgress(operationId, {
          stageIndex: 4,
          stageLabel: 'Voice cloning completed',
          progressPct: 100,
          progressSource: 'completed',
          status: 'succeeded'
        });

        if (!res.headersSent) {
          res.status(200).type('application/json').json({ result });
        }
      } catch (err: any) {
        if (deductedTokens > 0 && deductedUser) {
          applyAtomicTokenDelta({
            uid: deductedUser.uid,
            email: deductedUser.email,
            delta: deductedTokens,
            type: 'refund',
            reason: 'Automatic refund: Voice Cloning pipeline failed before completion',
            actorUid: 'system'
          });
          await persistGovernanceState();
        }
        const operationId = extractOperationIdFromReq(req);
        const quotaInfo = extractQuotaExhaustedInfo(err);
        const rawStatus =
          typeof err?.httpStatus === 'number'
            ? err.httpStatus
            : quotaInfo.isQuotaExhausted
            ? 429
            : 400;
        const safeStatus = rawStatus === 403 ? 401 : rawStatus;

        emitServerOperationProgress(operationId, {
          stageIndex: 1,
          stageLabel: err?.message || 'Voice cloning failed',
          progressPct: null,
          progressSource: 'indeterminate-inference',
          status: 'failed'
        });
        if (!res.headersSent) {
          res.status(safeStatus).type('application/json').json({
            error: err?.message || 'Voice cloning pipeline failed.',
            code: err?.code || (safeStatus === 429 ? 'RESOURCE_EXHAUSTED' : 'VOICE_CLONE_FAILED'),
            retryAfterText: err?.retryAfterText || quotaInfo.retryAfterText
          });
        }
      }
    }
  );

  // 7b. Poll status / result of an async Voice Cloning job (session/user scoped)
  app.get('/api/voice-clone/jobs/:jobId', requireAuthenticatedSession, (req, res) => {
    const jobId = String(req.params.jobId || '').trim();
    if (!SAFE_IDENTIFIER_REGEX.test(jobId)) {
      res.status(400).type('application/json').json({ error: 'Invalid job identifier.' });
      return;
    }
    const job = voiceCloneJobsRegistry.get(jobId);
    if (!job) {
      res.status(404).type('application/json').json({
        error: 'Voice cloning job not found or expired.',
        code: 'JOB_NOT_FOUND'
      });
      return;
    }
    const sess = (req as any).voicelabSession as VerifiedSessionState | undefined;
    const fbIdentity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity | null;
    const ownsBySession = Boolean(sess?.sessionId && job.sessionId && sess.sessionId === job.sessionId);
    const ownsByUid = Boolean(fbIdentity?.uid && job.uid && fbIdentity.uid === job.uid);
    if (!ownsBySession && !ownsByUid) {
      res.status(403).type('application/json').json({
        error: 'Access denied for this voice cloning job.',
        code: 'JOB_FORBIDDEN'
      });
      return;
    }

    res.status(200).type('application/json').json({
      jobId: job.jobId,
      status: job.status,
      operation: job.latestProgress || null,
      result: job.status === 'succeeded' ? job.result : undefined,
      error: job.status === 'failed' || job.status === 'cancelled' ? job.error : undefined,
      code: job.code,
      httpStatus: job.httpStatus,
      retryAfterText: job.retryAfterText
    });
  });

  // 7c. Cancel a running async Voice Cloning job cleanly
  app.post('/api/voice-clone/jobs/:jobId/cancel', requireAuthenticatedSession, async (req, res) => {
    const jobId = String(req.params.jobId || '').trim();
    if (!SAFE_IDENTIFIER_REGEX.test(jobId)) {
      res.status(400).type('application/json').json({ error: 'Invalid job identifier.' });
      return;
    }
    const job = voiceCloneJobsRegistry.get(jobId);
    if (!job) {
      res.status(404).type('application/json').json({ error: 'Voice cloning job not found.' });
      return;
    }
    const sess = (req as any).voicelabSession as VerifiedSessionState | undefined;
    const fbIdentity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity | null;
    const ownsBySession = Boolean(sess?.sessionId && job.sessionId && sess.sessionId === job.sessionId);
    const ownsByUid = Boolean(fbIdentity?.uid && job.uid && fbIdentity.uid === job.uid);
    if (!ownsBySession && !ownsByUid) {
      res.status(403).type('application/json').json({ error: 'Access denied for this voice cloning job.' });
      return;
    }

    if (job.status === 'running') {
      job.status = 'cancelled';
      job.error = 'Cancelled by user';
      job.code = 'JOB_CANCELLED';
      job.updatedAtMs = Date.now();
      job.abortController.abort();
      await refundVoiceCloneJobTokensOnce(
        job,
        'Automatic refund: Voice Cloning job cancelled by user'
      );
      emitServerOperationProgress(job.operationId, {
        stageIndex: 1,
        stageLabel: 'Voice cloning cancelled',
        progressPct: null,
        progressSource: 'indeterminate-inference',
        status: 'failed'
      });
    }
    res.status(200).type('application/json').json({ status: job.status, jobId: job.jobId });
  });

  // 8. WORKSPACE B — Universal Multilingual Audio-to-SRT Transcription
  // Supports both direct multipart File/Blob upload ('audio' field) AND uploaded fileId JSON reference.
  // Strictly prohibits demo/sample audio substitution.
  const handleSrtTranscriptionRequest = async (req: express.Request, res: express.Response) => {
    const operationId = extractOperationIdFromReq(req);
    try {
      const requestedLanguage = String(
        req.body?.requestedLanguage || req.body?.language || 'auto'
      );
      const incomingFileId = req.body?.fileId ? String(req.body.fileId).trim() : '';

      if (incomingFileId.startsWith('sample_ref_')) {
        res.status(400).type('application/json').json({
          error:
            'Demo and sample audio files are disabled in Audio to SRT. Please upload an actual audio file.'
        });
        return;
      }

      let targetFilePath = '';
      let targetFileName = '';
      let targetDurationSeconds = 0;
      let targetPreviewUrl = '';

      // Case 1: Actual uploaded File/Blob sent directly in multipart/form-data request
      if (req.file && req.file.path && fs.existsSync(req.file.path)) {
        const hasValidSignature = await verifyAudioFileMagicBytes(req.file.path);
        if (!hasValidSignature) {
          await fs.promises.unlink(req.file.path).catch(() => {});
          res.status(400).type('application/json').json({
            error:
              'Security validation failed: Uploaded file binary header does not match a valid audio container.'
          });
          return;
        }

        const uploadedFileId = path.basename(req.file.filename, path.extname(req.file.filename));
        const previewUrl = `/api/audio/uploaded/${uploadedFileId}`;
        const analysis = await inspectAndValidateAudioFile(
          req.file.path,
          uploadedFileId,
          req.file.originalname,
          req.file.mimetype || 'audio/wav',
          previewUrl
        );

        if (!analysis.isValid) {
          await fs.promises.unlink(req.file.path).catch(() => {});
          res.status(422).type('application/json').json({
            error:
              analysis.qualityWarnings.join(' ') ||
              'Uploaded audio is empty, corrupted, or contains only silence.'
          });
          return;
        }

        await saveUploadedFileEntry(uploadedFileId, req.file.path, analysis);
        targetFilePath = req.file.path;
        targetFileName = analysis.fileName;
        targetDurationSeconds = analysis.durationSeconds;
        targetPreviewUrl = previewUrl;
      } else if (incomingFileId) {
        // Case 2: Resolve previously uploaded fileId from memory or disk
        const entry = await resolveUploadedFileEntry(incomingFileId);
        if (!entry || !fs.existsSync(entry.filePath)) {
          res.status(404).type('application/json').json({
            error:
              'Uploaded audio recording not found or expired. Please re-upload your audio file and try again.'
          });
          return;
        }
        targetFilePath = entry.filePath;
        targetFileName = entry.analysis.fileName;
        targetDurationSeconds = entry.analysis.durationSeconds;
        targetPreviewUrl = entry.analysis.previewUrl;
      } else {
        res.status(400).type('application/json').json({
          error:
            'No uploaded audio file provided. Please upload an audio file before generating SRT.'
        });
        return;
      }

      res.setTimeout?.(10 * 60 * 1000);
      const abortController = new AbortController();
      const handleResponseClose = () => {
        if (!res.writableEnded) {
          abortController.abort();
        }
      };
      res.on('close', handleResponseClose);

      const jobId = `srt_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`;
      try {
        const result = await executeMultilingualSrtPipeline({
          jobId,
          filePath: targetFilePath,
          originalFileName: targetFileName,
          durationSeconds: targetDurationSeconds,
          requestedLanguage,
          audioPreviewUrl: targetPreviewUrl,
          signal: abortController.signal,
          onProgress: (ev) => {
            emitServerOperationProgress(operationId, {
              ...ev,
              status: 'running'
            });
          }
        });

        emitServerOperationProgress(operationId, {
          stageIndex: 2,
          stageLabel: `Generated ${result.segments.length} SRT subtitle blocks`,
          progressPct: 100,
          progressSource: 'completed',
          status: 'succeeded'
        });

        if (!res.headersSent && !res.writableEnded) {
          res.status(200).type('application/json').json({ result });
        }
      } finally {
        res.off('close', handleResponseClose);
      }
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        emitServerOperationProgress(operationId, {
          stageIndex: 1,
          stageLabel: 'SRT transcription cancelled',
          progressPct: null,
          progressSource: 'indeterminate-inference',
          status: 'failed'
        });
        return;
      }
      emitServerOperationProgress(operationId, {
        stageIndex: 1,
        stageLabel: err?.message || 'SRT transcription failed',
        progressPct: null,
        progressSource: 'indeterminate-inference',
        status: 'failed'
      });
      const errMsg = err?.message || 'Multilingual SRT transcription failed for the uploaded audio.';
      const statusCode =
        typeof err?.httpStatus === 'number' && [400, 422, 429, 503].includes(err.httpStatus)
          ? err.httpStatus
          : /HTTP 503|UNAVAILABLE/i.test(errMsg)
          ? 503
          : 422;
      if (!res.headersSent && !res.writableEnded) {
        res.status(statusCode).type('application/json').json({
          error: errMsg
        });
      }
    }
  };

  const guardedSrtHandler = async (req: express.Request, res: express.Response) => {
    const fbIdentity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity | null;
    const isOwnerActor = fbIdentity?.role === 'owner';
    if (!serverGovernancePolicy.allowSrtGeneration && !isOwnerActor) {
      res.status(423).type('application/json').json({
        error: 'Audio-to-SRT is currently paused by the Administrator.',
        code: 'FEATURE_DISABLED_BY_ADMIN'
      });
      return;
    }
    const srtGate = checkFeatureAccessGate(fbIdentity, 'srt_generator');
    if (!srtGate.allowed) {
      res.status(srtGate.statusCode).type('application/json').json({
        error: srtGate.error,
        code: srtGate.code,
        featureKey: 'srt_generator'
      });
      return;
    }

    let deductedTokens = 0;
    if (!isOwnerActor && srtGate.tokenCost > 0) {
      if (!fbIdentity) {
        res.status(401).type('application/json').json({
          error: 'Sign in with Google required to use token-gated Audio-to-SRT.',
          code: 'AUTH_REQUIRED_FOR_TOKEN_FEATURE',
          requiredTokens: srtGate.tokenCost
        });
        return;
      }
      const debit = applyAtomicTokenDelta({
        uid: fbIdentity.uid,
        email: fbIdentity.email,
        delta: -srtGate.tokenCost,
        type: 'feature_debit',
        reason: 'Audio-to-SRT transcription job',
        actorUid: fbIdentity.uid
      });
      if (!debit.ok) {
        res.status(402).type('application/json').json({
          error: debit.error,
          code: 'INSUFFICIENT_TOKENS',
          requiredTokens: srtGate.tokenCost,
          tokenBalance: debit.balanceAfter
        });
        return;
      }
      deductedTokens = srtGate.tokenCost;
      await persistGovernanceState();
    }

    // Track response outcome so unexpected pre-completion errors refund deducted tokens
    const origStatus = res.status.bind(res);
    let capturedStatus = 200;
    (res as any).status = (code: number) => {
      capturedStatus = code;
      return origStatus(code);
    };

    await handleSrtTranscriptionRequest(req, res);

    if (deductedTokens > 0 && fbIdentity && capturedStatus >= 400) {
      applyAtomicTokenDelta({
        uid: fbIdentity.uid,
        email: fbIdentity.email,
        delta: deductedTokens,
        type: 'refund',
        reason: 'Automatic refund: Audio-to-SRT request did not complete',
        actorUid: 'system'
      });
      await persistGovernanceState();
    }
  };

  app.post(
    '/api/srt/transcribe',
    requireAuthenticatedSession,
    enforceGovernanceComputeGuard,
    enforceComputeRateLimit,
    handleAudioUploadMiddleware,
    guardedSrtHandler
  );
  app.post(
    '/api/srt/generate',
    requireAuthenticatedSession,
    enforceGovernanceComputeGuard,
    enforceComputeRateLimit,
    handleAudioUploadMiddleware,
    guardedSrtHandler
  );

  // 9. Ephemeral Storage Manual Purge Endpoint
  app.delete('/api/storage/cleanup', requireAuthenticatedSession, async (_req, res) => {
    let deletedCount = 0;
    for (const [fileId, entry] of uploadedFilesRegistry.entries()) {
      if (!fileId.startsWith('sample_ref_')) {
        await fs.promises.unlink(entry.filePath).catch(() => {});
        const metaPath = resolveSafeChildPath(UPLOAD_DIR, `${fileId}.meta.json`);
        if (metaPath) {
          await fs.promises.unlink(metaPath).catch(() => {});
        }
        uploadedFilesRegistry.delete(fileId);
        deletedCount++;
      }
    }
    const uploadFiles = await fs.promises.readdir(UPLOAD_DIR).catch(() => []);
    for (const f of uploadFiles) {
      const p = resolveSafeChildPath(UPLOAD_DIR, f);
      if (p) await fs.promises.unlink(p).catch(() => {});
    }
    const genFiles = await fs.promises.readdir(GENERATED_DIR).catch(() => []);
    for (const f of genFiles) {
      const p = resolveSafeChildPath(GENERATED_DIR, f);
      if (p) {
        await fs.promises.unlink(p).catch(() => {});
        deletedCount++;
      }
    }
    res.status(200).type('application/json').json({
      status: 'cleaned',
      deletedFilesCount: deletedCount,
      message: 'All temporary user uploads and generated audio files have been permanently removed. Persistent saved voice profiles remain intact.'
    });
  });

  // 9b. Persistent User-Specific Saved Voice Profiles Endpoints (Strictly Owner-Isolated + Cloud Permanent)
  app.get('/api/voices', requireAuthenticatedSession, requireVerifiedUser, async (req, res) => {
    const identity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity;
    const bearerToken = extractBearerTokenFromAuthHeader(req.headers.authorization);

    const localForUser = Array.from(savedVoicesRegistry.values()).filter(
      (v) => v.ownerUid === identity.uid
    );
    const mergedByVoiceId = new Map<string, SavedVoiceProfile>();
    for (const rec of localForUser) {
      mergedByVoiceId.set(rec.voiceId, serializeSavedVoiceProfile(rec));
    }

    if (bearerToken) {
      const cloudProfiles = await listUserSavedVoicesFromFirestore(bearerToken, identity.uid);
      await Promise.all(
        cloudProfiles.map(async (cp) => {
          const localRec = savedVoicesRegistry.get(cp.voiceId);
          if (localRec && hasVaultConditioningAudio(localRec)) {
            mergedByVoiceId.set(cp.voiceId, serializeSavedVoiceProfile(localRec, true));
            // Automatically backfill cloud lossless FLAC chunks for any older voice saved before cloud audio sync
            const vp = resolveVaultVoicePaths(identity.uid, cp.voiceId);
            if (vp) {
              void checkCloudConditioningExists(bearerToken, cp.voiceId, identity.uid).then(
                (hasCloud) => {
                  if (!hasCloud) {
                    void persistSavedVoiceAndAudioToFirestore({
                      bearerToken,
                      profile: serializeSavedVoiceProfile(localRec, true),
                      refWavPath: vp.refWavPath,
                      promptWavPath: vp.promptWavPath,
                      vaultDir: vp.dir
                    });
                  }
                }
              );
            }
            return;
          }

          // Cold server restart or cross-device login: verify cloud conditioning chunks and warm disk vault
          const hasCloudChunks = await checkCloudConditioningExists(
            bearerToken,
            cp.voiceId,
            identity.uid
          );
          const syntheticRec: ServerSavedVoiceRecord = {
            ...cp,
            hasConditioningAudio: hasCloudChunks,
            referenceAnalysis: buildSyntheticAnalysisFromProfile(cp)
          };
          savedVoicesRegistry.set(cp.voiceId, syntheticRec);
          mergedByVoiceId.set(cp.voiceId, serializeSavedVoiceProfile(syntheticRec, hasCloudChunks));

          if (hasCloudChunks) {
            void ensureSavedVoiceHydratedOnServer(cp.voiceId, identity, bearerToken);
          }
        })
      );
    }

    const voices = Array.from(mergedByVoiceId.values()).sort(
      (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );

    res.status(200).type('application/json').json({ voices });
  });

  app.post('/api/voices', requireAuthenticatedSession, requireVerifiedUser, async (req, res) => {
    const identity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity;
    const bearerToken = extractBearerTokenFromAuthHeader(req.headers.authorization);
    try {
      const {
        referenceFileId,
        name,
        language,
        referenceTranscript,
        modelCheckpoint,
        sourceType = 'cloned_output',
        consentConfirmed = false,
        deduplicateByPromptHash = false
      } = req.body || {};

      if (!consentConfirmed) {
        res.status(400).type('application/json').json({
          error:
            'Consent confirmation required: You must confirm ownership or explicit authorization before saving a voice profile.'
        });
        return;
      }

      if (!['bn', 'hi', 'en'].includes(language)) {
        res.status(400).type('application/json').json({
          error: 'Invalid language for saved voice profile. Supported: bn, hi, en.'
        });
        return;
      }

      const cleanRefId = typeof referenceFileId === 'string' ? referenceFileId.trim() : '';
      if (!cleanRefId || !SAFE_IDENTIFIER_REGEX.test(cleanRefId)) {
        res.status(400).type('application/json').json({
          error: 'Valid reference audio identifier is required to save a voice profile.'
        });
        return;
      }

      const refEntry =
        uploadedFilesRegistry.get(cleanRefId) || (await resolveUploadedFileEntry(cleanRefId));
      if (!refEntry || !fs.existsSync(refEntry.filePath)) {
        res.status(404).type('application/json').json({
          error:
            'Reference audio recording was not found. Please upload or record reference audio before saving.'
        });
        return;
      }

      const detectedRefCode = refEntry.analysis.detectedSpokenLanguage?.code;
      if (
        detectedRefCode &&
        ['bn', 'hi', 'en'].includes(detectedRefCode) &&
        detectedRefCode !== language
      ) {
        res.status(400).type('application/json').json({
          error: `Cross-Language Blocked: Reference audio language (${detectedRefCode.toUpperCase()}) does not match selected language (${String(language).toUpperCase()}).`
        });
        return;
      }

      const existingCountForUser = Array.from(savedVoicesRegistry.values()).filter(
        (v) => v.ownerUid === identity.uid
      ).length;
      if (existingCountForUser >= 50) {
        res.status(400).type('application/json').json({
          error: 'Maximum limit of 50 saved voice profiles per account reached. Delete an unused profile first.'
        });
        return;
      }

      const langLabels: Record<CloningLanguageCode, string> = {
        bn: 'বাংলা (Bengali)',
        hi: 'हिन्दी (Hindi)',
        en: 'English'
      };
      const voiceId = `vp_${language}_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`;
      const vaultPaths = resolveVaultVoicePaths(identity.uid, voiceId);
      if (!vaultPaths) {
        res.status(400).type('application/json').json({ error: 'Invalid user or voice profile identifier.' });
        return;
      }

      await fs.promises.mkdir(vaultPaths.dir, { recursive: true });

      // Canonical 24kHz mono PCM WAV conversion
      await execFileAsync(
        'ffmpeg',
        [
          '-y',
          '-i',
          refEntry.filePath,
          '-ar',
          '24000',
          '-ac',
          '1',
          '-c:a',
          'pcm_s16le',
          vaultPaths.refWavPath
        ],
        { timeout: 15_000 }
      );

      const refBytes = await fs.promises.readFile(vaultPaths.refWavPath);
      const speakerConditioningSha256 = crypto.createHash('sha256').update(refBytes).digest('hex');

      // Extract & persist exact canonical silence-trimmed 5.5s conditioning prompt WAV
      const promptBytes = await prepareCanonicalSpeakerPromptWav(
        refBytes,
        vaultPaths.dir,
        speakerConditioningSha256
      );
      await fs.promises.writeFile(vaultPaths.promptWavPath, promptBytes);
      const promptConditioningSha256 = crypto
        .createHash('sha256')
        .update(promptBytes)
        .digest('hex');

      // Check deterministic speaker deduplication if auto-saving after generation
      if (deduplicateByPromptHash) {
        const duplicate = Array.from(savedVoicesRegistry.values()).find(
          (v) =>
            v.ownerUid === identity.uid &&
            v.language === language &&
            (v.promptConditioningSha256 === promptConditioningSha256 ||
              v.speakerConditioningSha256 === speakerConditioningSha256)
        );
        if (duplicate && hasVaultConditioningAudio(duplicate)) {
          await fs.promises.rm(vaultPaths.dir, { recursive: true, force: true }).catch(() => {});
          const nowIso = new Date().toISOString();
          duplicate.usageCount = (duplicate.usageCount || 0) + 1;
          duplicate.lastUsedAt = nowIso;
          duplicate.updatedAt = nowIso;
          await persistSavedVoiceProfileRecord(duplicate);

          const dupVaultPaths = resolveVaultVoicePaths(duplicate.ownerUid, duplicate.voiceId);
          if (bearerToken && dupVaultPaths) {
            void persistSavedVoiceAndAudioToFirestore({
              bearerToken,
              profile: serializeSavedVoiceProfile(duplicate, true),
              refWavPath: dupVaultPaths.refWavPath,
              promptWavPath: dupVaultPaths.promptWavPath,
              vaultDir: dupVaultPaths.dir
            });
            void incrementSavedVoiceUsageInFirestore(bearerToken, duplicate.voiceId);
          }

          res.status(200).type('application/json').json({
            voice: serializeSavedVoiceProfile(duplicate, true),
            deduplicated: true,
            cloudSynced: true
          });
          return;
        }
      }

      const canonicalAnalysis = await inspectAndValidateAudioFile(
        vaultPaths.refWavPath,
        voiceId,
        refEntry.analysis.fileName || `${voiceId}.wav`,
        'audio/wav',
        `/api/voices/${encodeURIComponent(voiceId)}/preview`
      );
      if (refEntry.analysis.detectedSpokenLanguage) {
        canonicalAnalysis.detectedSpokenLanguage = refEntry.analysis.detectedSpokenLanguage;
      }

      const promptDurationSec = Number(
        Math.max(0.3, (promptBytes.length - 44) / (24000 * 2)).toFixed(2)
      );

      const cleanNameRaw = String(name || '').trim().slice(0, 80);
      const defaultName = `${langLabels[language as CloningLanguageCode]} Voice #${existingCountForUser + 1}`;
      const finalName = cleanNameRaw.length >= 1 ? cleanNameRaw : defaultName;

      const cleanTranscript = String(
        referenceTranscript ||
          refEntry.analysis.detectedSpokenLanguage?.transcriptExcerpt ||
          ''
      )
        .trim()
        .slice(0, 1000);

      const validSourceTypes: SavedVoiceSourceType[] = [
        'cloned_output',
        'reference_upload',
        'microphone',
        'studio_sample'
      ];
      const safeSourceType: SavedVoiceSourceType = validSourceTypes.includes(sourceType)
        ? sourceType
        : cleanRefId.startsWith('sample_ref_')
        ? 'studio_sample'
        : String(refEntry.analysis.fileName || '').startsWith('mic_reference_')
        ? 'microphone'
        : 'reference_upload';

      const defaultModelCheckpoint =
        language === 'en'
          ? 'SWivid/F5-TTS · ChatterboxTTS Zero-Shot'
          : language === 'hi'
          ? 'ai4bharat/IndicF5 (Zero-Shot Flow-Matching DiT)'
          : 'ai4bharat/IndicF5 · Fun-CosyVoice3 Bengali Zero-Shot';

      const nowIso = new Date().toISOString();
      const record: ServerSavedVoiceRecord = {
        voiceId,
        ownerUid: identity.uid,
        ownerEmail: identity.email,
        name: finalName,
        language: language as CloningLanguageCode,
        languageName: langLabels[language as CloningLanguageCode],
        conditioningMethod: 'zero-shot-canonical-wav-v1',
        modelCheckpoint:
          String(modelCheckpoint || defaultModelCheckpoint).trim().slice(0, 160) ||
          defaultModelCheckpoint,
        speakerConditioningSha256,
        promptConditioningSha256,
        referenceTranscript: cleanTranscript,
        referenceDurationSec: Number(canonicalAnalysis.durationSeconds || refEntry.analysis.durationSeconds || 1),
        promptDurationSec,
        fundamentalFreqHz: Number(canonicalAnalysis.fundamentalFreqHz || refEntry.analysis.fundamentalFreqHz || 0),
        pitchMinHz: Number(canonicalAnalysis.pitchRangeHz?.[0] || 0),
        pitchMaxHz: Number(canonicalAnalysis.pitchRangeHz?.[1] || 0),
        spectralWarmthRatio: Number(canonicalAnalysis.spectralWarmthRatio || 0),
        estimatedSnrDb: Number(canonicalAnalysis.estimatedSnrDb || 0),
        qualityGrade: canonicalAnalysis.qualityGrade || 'Good',
        sourceType: safeSourceType,
        sourceFileName: String(refEntry.analysis.fileName || `${voiceId}.wav`).slice(0, 160),
        storageKey: vaultPaths.storageKey,
        consentConfirmed: true,
        usageCount: 0,
        hasConditioningAudio: true,
        lastUsedAt: nowIso,
        createdAt: nowIso,
        updatedAt: nowIso,
        referenceAnalysis: {
          ...canonicalAnalysis,
          fileId: voiceId,
          previewUrl: `/api/voices/${encodeURIComponent(voiceId)}/preview`
        }
      };

      await persistSavedVoiceProfileRecord(record);
      const serialized = serializeSavedVoiceProfile(record, true);

      // Persist BOTH metadata and lossless FLAC conditioning audio chunks to Firestore immediately
      const cloudPersist = await persistSavedVoiceAndAudioToFirestore({
        bearerToken,
        profile: serialized,
        refWavPath: vaultPaths.refWavPath,
        promptWavPath: vaultPaths.promptWavPath,
        vaultDir: vaultPaths.dir
      });

      if (!cloudPersist.synced) {
        console.warn(
          `[VoiceLab VaultSync] Cloud persistence notice for ${record.voiceId}:`,
          cloudPersist.error
        );
      }

      recordServerSecurityEvent(
        'VOICE_PROFILE_SAVED',
        identity.email,
        `Saved ${record.voiceId} (${record.language}, promptSha=${promptConditioningSha256.slice(0, 10)}, cloudSynced=${cloudPersist.synced})`
      );

      res.status(201).type('application/json').json({
        voice: serialized,
        cloudSynced: cloudPersist.synced,
        cloudSyncWarning: cloudPersist.synced ? undefined : cloudPersist.error
      });
    } catch (err: any) {
      res.status(400).type('application/json').json({
        error: err?.message || 'Failed to save persistent voice profile.'
      });
    }
  });

  app.post('/api/voices/:voiceId/restore', requireAuthenticatedSession, requireVerifiedUser, async (req, res) => {
    const identity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity;
    const bearerToken = extractBearerTokenFromAuthHeader(req.headers.authorization);
    const voiceId = String(req.params.voiceId || '').trim();
    if (!SAFE_IDENTIFIER_REGEX.test(voiceId)) {
      res.status(400).type('application/json').json({ error: 'Invalid voice profile identifier.' });
      return;
    }

    const cleanRefId = typeof req.body?.referenceFileId === 'string' ? req.body.referenceFileId.trim() : '';
    if (!cleanRefId || !SAFE_IDENTIFIER_REGEX.test(cleanRefId)) {
      res.status(400).type('application/json').json({
        error: 'Valid reference audio identifier is required to restore cloud speaker conditioning.'
      });
      return;
    }

    const refEntry =
      uploadedFilesRegistry.get(cleanRefId) || (await resolveUploadedFileEntry(cleanRefId));
    if (!refEntry || !fs.existsSync(refEntry.filePath)) {
      res.status(404).type('application/json').json({
        error: 'Reference audio recording was not found. Please upload reference audio first.'
      });
      return;
    }

    let existing: ServerSavedVoiceRecord | undefined = savedVoicesRegistry.get(voiceId);
    if (!existing && bearerToken) {
      const cloudProfile = await fetchSingleSavedVoiceFromFirestore(bearerToken, voiceId);
      if (cloudProfile) {
        existing = {
          ...cloudProfile,
          hasConditioningAudio: false,
          referenceAnalysis: buildSyntheticAnalysisFromProfile(cloudProfile)
        };
      }
    }
    if (!existing) {
      res.status(404).type('application/json').json({ error: 'Saved voice profile not found.' });
      return;
    }
    if (existing.ownerUid !== identity.uid) {
      res.status(403).type('application/json').json({
        error: 'Access denied: You can only restore your own saved voice profiles.',
        code: 'VOICE_FORBIDDEN'
      });
      return;
    }

    const vaultPaths = resolveVaultVoicePaths(identity.uid, voiceId);
    if (!vaultPaths) {
      res.status(400).type('application/json').json({ error: 'Invalid user or voice profile identifier.' });
      return;
    }

    try {
      await fs.promises.mkdir(vaultPaths.dir, { recursive: true });
      await execFileAsync(
        'ffmpeg',
        ['-y', '-i', refEntry.filePath, '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', vaultPaths.refWavPath],
        { timeout: 20_000 }
      );

      const refBytes = await fs.promises.readFile(vaultPaths.refWavPath);
      const speakerConditioningSha256 = crypto.createHash('sha256').update(refBytes).digest('hex');
      const promptBytes = await prepareCanonicalSpeakerPromptWav(
        refBytes,
        vaultPaths.dir,
        speakerConditioningSha256
      );
      await fs.promises.writeFile(vaultPaths.promptWavPath, promptBytes);
      const promptConditioningSha256 = crypto.createHash('sha256').update(promptBytes).digest('hex');

      const canonicalAnalysis = await inspectAndValidateAudioFile(
        vaultPaths.refWavPath,
        voiceId,
        refEntry.analysis.fileName || existing.sourceFileName || `${voiceId}.wav`,
        'audio/wav',
        `/api/voices/${encodeURIComponent(voiceId)}/preview`
      );
      if (refEntry.analysis.detectedSpokenLanguage) {
        canonicalAnalysis.detectedSpokenLanguage = refEntry.analysis.detectedSpokenLanguage;
      }

      const promptDurationSec = Number(Math.max(0.3, (promptBytes.length - 44) / (24000 * 2)).toFixed(2));
      const updatedRecord: ServerSavedVoiceRecord = {
        ...existing,
        speakerConditioningSha256,
        promptConditioningSha256,
        referenceDurationSec: Number(canonicalAnalysis.durationSeconds || existing.referenceDurationSec || 1),
        promptDurationSec,
        fundamentalFreqHz: Number(canonicalAnalysis.fundamentalFreqHz || existing.fundamentalFreqHz || 0),
        pitchMinHz: Number(canonicalAnalysis.pitchRangeHz?.[0] || existing.pitchMinHz || 0),
        pitchMaxHz: Number(canonicalAnalysis.pitchRangeHz?.[1] || existing.pitchMaxHz || 0),
        spectralWarmthRatio: Number(canonicalAnalysis.spectralWarmthRatio || existing.spectralWarmthRatio || 0),
        estimatedSnrDb: Number(canonicalAnalysis.estimatedSnrDb || existing.estimatedSnrDb || 0),
        qualityGrade: canonicalAnalysis.qualityGrade || existing.qualityGrade || 'Good',
        hasConditioningAudio: true,
        updatedAt: new Date().toISOString(),
        referenceAnalysis: {
          ...canonicalAnalysis,
          fileId: voiceId,
          previewUrl: `/api/voices/${encodeURIComponent(voiceId)}/preview`
        }
      };

      await persistSavedVoiceProfileRecord(updatedRecord);
      const serialized = serializeSavedVoiceProfile(updatedRecord, true);

      const cloudPersist = await persistSavedVoiceAndAudioToFirestore({
        bearerToken,
        profile: serialized,
        refWavPath: vaultPaths.refWavPath,
        promptWavPath: vaultPaths.promptWavPath,
        vaultDir: vaultPaths.dir
      });

      res.status(200).type('application/json').json({
        voice: serialized,
        cloudSynced: cloudPersist.synced,
        cloudSyncWarning: cloudPersist.synced ? undefined : cloudPersist.error
      });
    } catch (err: any) {
      res.status(400).type('application/json').json({
        error: err?.message || 'Failed to restore speaker conditioning audio.'
      });
    }
  });

  app.patch('/api/voices/:voiceId', requireAuthenticatedSession, requireVerifiedUser, async (req, res) => {
    const identity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity;
    const bearerToken = extractBearerTokenFromAuthHeader(req.headers.authorization);
    const voiceId = String(req.params.voiceId || '').trim();
    if (!SAFE_IDENTIFIER_REGEX.test(voiceId)) {
      res.status(400).type('application/json').json({ error: 'Invalid voice profile identifier.' });
      return;
    }

    const existing =
      savedVoicesRegistry.get(voiceId) ||
      (await ensureSavedVoiceHydratedOnServer(voiceId, identity, bearerToken));
    if (!existing) {
      res.status(404).type('application/json').json({ error: 'Saved voice profile not found.' });
      return;
    }
    if (existing.ownerUid !== identity.uid) {
      recordServerSecurityEvent(
        'UNAUTHORIZED_VOICE_RENAME',
        identity.email,
        `Blocked rename attempt on ${voiceId}`
      );
      res.status(403).type('application/json').json({
        error: 'Access denied: You can only rename your own saved voice profiles.',
        code: 'VOICE_FORBIDDEN'
      });
      return;
    }

    const nextName = String(req.body?.name || '').trim().slice(0, 80);
    if (!nextName) {
      res.status(400).type('application/json').json({ error: 'Voice profile name cannot be empty.' });
      return;
    }

    existing.name = nextName;
    existing.updatedAt = new Date().toISOString();
    await persistSavedVoiceProfileRecord(existing);
    if (bearerToken) {
      await renameSavedVoiceInFirestore(bearerToken, voiceId, nextName).catch(() => {});
    }

    res.status(200).type('application/json').json({
      voice: serializeSavedVoiceProfile(existing)
    });
  });

  app.delete('/api/voices/:voiceId', requireAuthenticatedSession, requireVerifiedUser, async (req, res) => {
    const identity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity;
    const bearerToken = extractBearerTokenFromAuthHeader(req.headers.authorization);
    const voiceId = String(req.params.voiceId || '').trim();
    if (!SAFE_IDENTIFIER_REGEX.test(voiceId)) {
      res.status(400).type('application/json').json({ error: 'Invalid voice profile identifier.' });
      return;
    }

    let existing: ServerSavedVoiceRecord | SavedVoiceProfile | undefined =
      savedVoicesRegistry.get(voiceId);
    if (!existing && bearerToken) {
      const cloudDoc = await fetchSingleSavedVoiceFromFirestore(bearerToken, voiceId);
      if (cloudDoc) existing = cloudDoc;
    }
    if (!existing) {
      res.status(404).type('application/json').json({ error: 'Saved voice profile not found.' });
      return;
    }
    if (existing.ownerUid !== identity.uid) {
      recordServerSecurityEvent(
        'UNAUTHORIZED_VOICE_DELETE',
        identity.email,
        `Blocked delete attempt on ${voiceId}`
      );
      res.status(403).type('application/json').json({
        error: 'Access denied: You can only delete your own saved voice profiles.',
        code: 'VOICE_FORBIDDEN'
      });
      return;
    }

    const vaultPaths = resolveVaultVoicePaths(existing.ownerUid, existing.voiceId);
    if (vaultPaths) {
      await fs.promises.rm(vaultPaths.dir, { recursive: true, force: true }).catch(() => {});
    }
    savedVoicesRegistry.delete(voiceId);
    await persistSavedVoicesIndex().catch(() => {});
    await deleteSavedVoiceFromFirestore(bearerToken, voiceId, identity.uid);

    recordServerSecurityEvent(
      'VOICE_PROFILE_DELETED',
      identity.email,
      `Deleted ${voiceId} and associated conditioning audio`
    );

    res.status(200).type('application/json').json({
      ok: true,
      deletedVoiceId: voiceId
    });
  });

  // Private owner-only audio preview stream (hydrates automatically from cloud vault on cold server restarts)
  app.get('/api/voices/:voiceId/preview', requireAuthenticatedSession, requireVerifiedUser, async (req, res) => {
    const identity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity;
    const bearerToken = extractBearerTokenFromAuthHeader(req.headers.authorization);
    const voiceId = String(req.params.voiceId || '').trim();
    if (!SAFE_IDENTIFIER_REGEX.test(voiceId)) {
      res.status(400).type('application/json').json({ error: 'Invalid voice profile identifier.' });
      return;
    }

    const existing = await ensureSavedVoiceHydratedOnServer(voiceId, identity, bearerToken);
    if (!existing) {
      res.status(404).type('application/json').json({ error: 'Saved voice profile not found.' });
      return;
    }
    if (existing.ownerUid !== identity.uid) {
      recordServerSecurityEvent(
        'UNAUTHORIZED_VOICE_PREVIEW',
        identity.email,
        `Blocked preview stream on ${voiceId}`
      );
      res.status(403).type('application/json').json({
        error: 'Access denied: Reference audio previews are strictly private to the profile owner.',
        code: 'VOICE_FORBIDDEN'
      });
      return;
    }

    const vaultPaths = resolveVaultVoicePaths(existing.ownerUid, existing.voiceId);
    if (!vaultPaths || !fs.existsSync(vaultPaths.refWavPath)) {
      res.status(404).type('application/json').json({
        error: 'Stored speaker reference audio file was not found.'
      });
      return;
    }

    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader('Cache-Control', 'private, no-store');
    fs.createReadStream(vaultPaths.refWavPath).pipe(res);
  });

  // 10. Verified User Identity, Public Catalog, and Purchase Endpoints (Zero Owner/Payment Secrets Leaked)
  app.get('/api/auth/me', requireVerifiedUser, async (req, res) => {
    const identity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity;
    const nowIso = new Date().toISOString();
    const existing = serverUsersRegistry[identity.uid];
    serverUsersRegistry[identity.uid] = {
      uid: identity.uid,
      email: identity.email,
      displayName: identity.displayName || existing?.displayName || identity.email.split('@')[0],
      role: identity.role,
      accountStatus: existing?.accountStatus || 'active',
      createdAt: existing?.createdAt || nowIso,
      lastSeenAt: nowIso
    };
    const ent = ensureUserEntitlementRecord(identity.uid, identity.email);
    await persistGovernanceState();

    const myRequests = serverPaymentRequests
      .filter((r) => r.userUid === identity.uid)
      .slice(0, 25);
    const myLedger = serverTokenLedger
      .filter((l) => l.userUid === identity.uid)
      .slice(0, 25);

    res.status(200).type('application/json').json({
      authenticated: true,
      uid: identity.uid,
      email: identity.email,
      displayName: serverUsersRegistry[identity.uid].displayName,
      isOwner: identity.role === 'owner',
      role: identity.role,
      accountStatus: serverUsersRegistry[identity.uid].accountStatus,
      entitlement: ent,
      tokenBalance: ent.tokenBalance,
      tokenLedger: myLedger,
      paymentRequests: myRequests
    });
  });

  app.get('/api/governance/public', (_req, res) => {
    res.status(200).type('application/json').json({
      governance: serverGovernancePolicy,
      organizationName: serverOwnershipState.organizationName
    });
  });

  app.get('/api/catalog/public', async (_req, res) => {
    const availableMethods: PaymentMethodId[] = [];
    if (serverPaymentSettings.bkashEnabled && serverPaymentSettings.bkashNumber.trim().length >= 11) {
      availableMethods.push('bkash');
    }
    if (serverPaymentSettings.nagadEnabled && serverPaymentSettings.nagadNumber.trim().length >= 11) {
      availableMethods.push('nagad');
    }

    res.status(200).type('application/json').json({
      governance: serverGovernancePolicy,
      features: serverFeatureConfigs,
      featurePackages: serverFeaturePackages.filter((p) => p.enabled),
      tokenPackages: serverTokenPackages.filter((p) => p.enabled),
      availablePaymentMethods: availableMethods
    });
  });

  app.get('/api/payments/my', requireAuthenticatedSession, requireVerifiedUser, (req, res) => {
    const identity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity;
    const ent = ensureUserEntitlementRecord(identity.uid, identity.email);
    res.status(200).type('application/json').json({
      uid: identity.uid,
      tokenBalance: ent.tokenBalance,
      entitlement: ent,
      paymentRequests: serverPaymentRequests.filter((r) => r.userUid === identity.uid).slice(0, 30),
      tokenLedger: serverTokenLedger.filter((l) => l.userUid === identity.uid).slice(0, 30)
    });
  });

  // Returns payment instructions strictly for the specific chosen package and method
  app.post('/api/payments/instructions', requireAuthenticatedSession, requireVerifiedUser, (req, res) => {
    const { itemType, itemId, paymentMethod } = req.body || {};
    if (!['feature_package', 'token_package'].includes(itemType)) {
      res.status(400).type('application/json').json({ error: 'Invalid purchase type.' });
      return;
    }
    if (!['bkash', 'nagad'].includes(paymentMethod)) {
      res.status(400).type('application/json').json({ error: 'Select bKash or Nagad payment method.' });
      return;
    }

    const method = paymentMethod as PaymentMethodId;
    const methodEnabled =
      method === 'bkash' ? serverPaymentSettings.bkashEnabled : serverPaymentSettings.nagadEnabled;
    const receiverNumber = (
      method === 'bkash' ? serverPaymentSettings.bkashNumber : serverPaymentSettings.nagadNumber
    ).trim();
    const accountType =
      method === 'bkash' ? serverPaymentSettings.bkashAccountType : serverPaymentSettings.nagadAccountType;
    const instructions =
      method === 'bkash'
        ? serverPaymentSettings.bkashInstructions
        : serverPaymentSettings.nagadInstructions;

    if (!methodEnabled || receiverNumber.length < 11) {
      res.status(423).type('application/json').json({
        error: `${method === 'bkash' ? 'bKash' : 'Nagad'} payment is currently not configured or disabled by the administrator.`
      });
      return;
    }

    if (itemType === 'feature_package') {
      const pkg = serverFeaturePackages.find((p) => p.id === itemId && p.enabled);
      if (!pkg) {
        res.status(404).type('application/json').json({ error: 'Feature package not found or disabled.' });
        return;
      }
      res.status(200).type('application/json').json({
        itemType,
        itemId: pkg.id,
        itemTitle: pkg.title,
        amountBdt: pkg.priceBdt,
        tokensGranted: 0,
        durationDays: pkg.durationDays,
        paymentMethod: method,
        accountType,
        receiverNumber,
        instructions
      });
      return;
    }

    const tokPkg = serverTokenPackages.find((p) => p.id === itemId && p.enabled);
    if (!tokPkg) {
      res.status(404).type('application/json').json({ error: 'Token package not found or disabled.' });
      return;
    }
    res.status(200).type('application/json').json({
      itemType,
      itemId: tokPkg.id,
      itemTitle: tokPkg.title,
      amountBdt: tokPkg.priceBdt,
      tokensGranted: tokPkg.tokens,
      durationDays: 0,
      paymentMethod: method,
      accountType,
      receiverNumber,
      instructions
    });
  });

  // Creates a PENDING payment request — NEVER grants access or tokens before Owner approval!
  app.post('/api/payments/submit', requireAuthenticatedSession, requireVerifiedUser, async (req, res) => {
    const identity = (req as any).firebaseIdentity as VerifiedFirebaseIdentity;
    const { itemType, itemId, paymentMethod, senderNumber, trxId, note } = req.body || {};

    if (!['feature_package', 'token_package'].includes(itemType)) {
      res.status(400).type('application/json').json({ error: 'Invalid purchase type.' });
      return;
    }
    if (!['bkash', 'nagad'].includes(paymentMethod)) {
      res.status(400).type('application/json').json({ error: 'Select bKash or Nagad payment method.' });
      return;
    }

    const method = paymentMethod as PaymentMethodId;
    const methodEnabled =
      method === 'bkash' ? serverPaymentSettings.bkashEnabled : serverPaymentSettings.nagadEnabled;
    const receiverNumber = (
      method === 'bkash' ? serverPaymentSettings.bkashNumber : serverPaymentSettings.nagadNumber
    ).trim();

    if (!methodEnabled || receiverNumber.length < 11) {
      res.status(423).type('application/json').json({
        error: `${method === 'bkash' ? 'bKash' : 'Nagad'} payment is currently unavailable.`
      });
      return;
    }

    const cleanSender = String(senderNumber || '').trim();
    if (!/^01[3-9][0-9]{8}$/.test(cleanSender)) {
      res.status(400).type('application/json').json({
        error: 'সঠিক ১১ সংখ্যার বাংলাদেশী মোবাইল নম্বর দিন (যেমন: 017XXXXXXXX)।'
      });
      return;
    }

    const cleanTrx = String(trxId || '').trim().toUpperCase();
    if (!/^[A-Z0-9]{8,16}$/.test(cleanTrx)) {
      res.status(400).type('application/json').json({
        error: 'সঠিক Transaction ID (TrxID) দিন (৮–১৬ অক্ষরের ইংরেজি অক্ষর ও সংখ্যা)।'
      });
      return;
    }

    const trxCompositeKey = `${method}:${cleanTrx}`;
    const existingTrx = usedPaymentTrxIds[trxCompositeKey] || usedPaymentTrxIds[cleanTrx];
    if (existingTrx && existingTrx.status !== 'rejected') {
      res.status(409).type('application/json').json({
        error: 'এই Transaction ID আগে থেকেই ব্যবহৃত বা যাচাইয়ের জন্য জমা দেওয়া হয়েছে।'
      });
      return;
    }

    let itemTitle = '';
    let amountBdt = 0;
    let tokensGranted = 0;
    let durationDays = 0;
    let featureKey: PackageFeatureKey | undefined;

    if (itemType === 'feature_package') {
      const pkg = serverFeaturePackages.find((p) => p.id === itemId && p.enabled);
      if (!pkg) {
        res.status(404).type('application/json').json({ error: 'Feature package not found or disabled.' });
        return;
      }
      itemTitle = pkg.title;
      amountBdt = pkg.priceBdt;
      durationDays = pkg.durationDays;
      featureKey = pkg.featureKey;
    } else {
      const tokPkg = serverTokenPackages.find((p) => p.id === itemId && p.enabled);
      if (!tokPkg) {
        res.status(404).type('application/json').json({ error: 'Token package not found or disabled.' });
        return;
      }
      itemTitle = tokPkg.title;
      amountBdt = tokPkg.priceBdt;
      tokensGranted = tokPkg.tokens;
    }

    if (amountBdt < 1) {
      res.status(400).type('application/json').json({ error: 'Selected package amount must be at least ৳1.' });
      return;
    }

    const nowIso = new Date().toISOString();
    const orderId = `${method}_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
    const newRequest: ServerPaymentRequest = {
      orderId,
      userUid: identity.uid,
      userEmail: identity.email,
      userName: identity.displayName,
      itemType,
      itemId,
      itemTitle,
      featureKey,
      amountBdt,
      tokensGranted,
      durationDays,
      paymentMethod: method,
      senderNumber: cleanSender,
      trxId: cleanTrx,
      receiverNumber,
      note: String(note || '').trim().slice(0, 200),
      status: 'pending',
      credited: false,
      reviewedByUid: '',
      rejectionReason: '',
      createdAt: nowIso,
      updatedAt: nowIso
    };

    serverPaymentRequests.unshift(newRequest);
    usedPaymentTrxIds[trxCompositeKey] = {
      orderId,
      paymentMethod: method,
      userUid: identity.uid,
      status: 'pending'
    };
    if (!serverUsersRegistry[identity.uid]) {
      serverUsersRegistry[identity.uid] = {
        uid: identity.uid,
        email: identity.email,
        displayName: identity.displayName,
        role: identity.role,
        accountStatus: 'active',
        createdAt: nowIso,
        lastSeenAt: nowIso
      };
    }

    await persistGovernanceState();
    recordServerSecurityEvent(
      'PAYMENT_REQUEST_SUBMITTED',
      identity.email,
      `${method.toUpperCase()} ৳${amountBdt} · ${itemTitle} (${cleanTrx})`
    );

    res.status(200).type('application/json').json({
      ok: true,
      request: newRequest
    });
  });

  // 11. Owner-Only Admin Panel Endpoints (Strictly requirePrimaryOwner on every endpoint)
  app.get('/api/admin/overview', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    const uploadFiles = await fs.promises.readdir(UPLOAD_DIR).catch(() => []);
    const genFiles = await fs.promises.readdir(GENERATED_DIR).catch(() => []);
    const cacheFiles = await fs.promises
      .readdir(path.join(GENERATED_DIR, 'chunk_cache'))
      .catch(() => []);

    let totalTempBytes = 0;
    for (const f of uploadFiles) {
      const st = await fs.promises.stat(path.join(UPLOAD_DIR, f)).catch(() => null);
      if (st?.isFile()) totalTempBytes += st.size;
    }
    for (const f of genFiles) {
      const st = await fs.promises.stat(path.join(GENERATED_DIR, f)).catch(() => null);
      if (st?.isFile()) totalTempBytes += st.size;
    }

    res.status(200).type('application/json').json({
      actor: {
        uid: owner.uid,
        email: owner.email,
        role: owner.role
      },
      ownership: {
        primaryOwnerEmail: serverOwnershipState.primaryOwnerEmail,
        primaryOwnerUid: serverOwnershipState.primaryOwnerUid || owner.uid,
        organizationName: serverOwnershipState.organizationName,
        ownershipLocked: serverOwnershipState.ownershipLocked
      },
      governance: serverGovernancePolicy,
      features: serverFeatureConfigs,
      featurePackages: serverFeaturePackages,
      tokenPackages: serverTokenPackages,
      paymentSettings: serverPaymentSettings,
      users: Object.values(serverUsersRegistry),
      entitlements: Object.values(serverUserEntitlements),
      paymentRequests: serverPaymentRequests.slice(0, 150),
      tokenLedger: serverTokenLedger.slice(0, 150),
      auditLogs: serverAuditLogs.slice(0, 100),
      runtime: {
        uptimeSeconds: Math.round(process.uptime()),
        ffmpegAvailable: ffmpegBinaryVerified,
        ffprobeAvailable: ffprobeBinaryVerified,
        sessionSecretSource: SESSION_CONFIG.source,
        sessionSecretEntropyBits: SESSION_CONFIG.entropyBits,
        uploadedFileCount: uploadFiles.length,
        generatedFileCount: genFiles.length,
        cachedChunkCount: cacheFiles.length,
        totalTempStorageMb: Number((totalTempBytes / (1024 * 1024)).toFixed(2)),
        activeRateBuckets: computeRateBuckets.size
      },
      recentSecurityEvents
    });
  });

  app.post('/api/admin/features', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    const b = req.body || {};
    const clampCost = (v: unknown, fb: number) =>
      typeof v === 'number' && Number.isFinite(v) ? Math.max(0, Math.min(1000, Math.round(v))) : fb;

    for (const key of ['voice_cloning', 'long_script', 'srt_generator'] as FeatureGateKey[]) {
      const incomingCfg = b[key] || b?.features?.[key];
      if (incomingCfg && typeof incomingCfg === 'object') {
        serverFeatureConfigs[key] = {
          ...serverFeatureConfigs[key],
          enabled:
            typeof incomingCfg.enabled === 'boolean'
              ? incomingCfg.enabled
              : serverFeatureConfigs[key].enabled,
          isPaid:
            typeof incomingCfg.isPaid === 'boolean'
              ? incomingCfg.isPaid
              : serverFeatureConfigs[key].isPaid,
          tokenCost: clampCost(incomingCfg.tokenCost, serverFeatureConfigs[key].tokenCost)
        };
      }
    }

    await persistGovernanceState();
    recordServerAuditLog(
      owner.uid,
      owner.email,
      'UPDATE_PREMIUM_FEATURES',
      'system/catalog',
      `vcPaid=${serverFeatureConfigs.voice_cloning.isPaid}(${serverFeatureConfigs.voice_cloning.tokenCost}t), longPaid=${serverFeatureConfigs.long_script.isPaid}(${serverFeatureConfigs.long_script.tokenCost}t), srtPaid=${serverFeatureConfigs.srt_generator.isPaid}(${serverFeatureConfigs.srt_generator.tokenCost}t)`
    );
    recordServerSecurityEvent('PREMIUM_FEATURES_UPDATED', owner.email, 'Feature paywall & token costs saved');
    res.status(200).type('application/json').json({ ok: true, features: serverFeatureConfigs });
  });

  app.post('/api/admin/plans', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    const { featurePackages, tokenPackages } = req.body || {};

    if (Array.isArray(featurePackages)) {
      const sanitized: ServerFeaturePackage[] = [];
      for (const raw of featurePackages.slice(0, 20)) {
        if (!raw || typeof raw !== 'object') continue;
        const fk = raw.featureKey;
        if (!['voice_cloning', 'long_script', 'srt_generator', 'pro_bundle'].includes(fk)) continue;
        const rawId = typeof raw.id === 'string' && /^[a-zA-Z0-9_-]{3,64}$/.test(raw.id.trim())
          ? raw.id.trim()
          : `pkg_${fk}_${crypto.randomBytes(3).toString('hex')}`;
        sanitized.push({
          id: rawId,
          featureKey: fk,
          title: String(raw.title || 'Feature Plan').trim().slice(0, 80) || 'Feature Plan',
          description: String(raw.description || '').trim().slice(0, 200),
          priceBdt: Math.max(1, Math.min(50000, Math.round(Number(raw.priceBdt) || 199))),
          durationDays: Math.max(0, Math.min(3650, Math.round(Number(raw.durationDays) || 0))),
          enabled: raw.enabled !== false
        });
      }
      if (sanitized.length > 0) {
        serverFeaturePackages = sanitized;
      }
    }

    if (Array.isArray(tokenPackages)) {
      const sanitizedTok: ServerTokenPackage[] = [];
      for (const raw of tokenPackages.slice(0, 20)) {
        if (!raw || typeof raw !== 'object') continue;
        const rawId = typeof raw.id === 'string' && /^[a-zA-Z0-9_-]{3,64}$/.test(raw.id.trim())
          ? raw.id.trim()
          : `tok_${Date.now()}_${crypto.randomBytes(2).toString('hex')}`;
        sanitizedTok.push({
          id: rawId,
          title: String(raw.title || 'Token Pack').trim().slice(0, 80) || 'Token Pack',
          description: String(raw.description || '').trim().slice(0, 200),
          tokens: Math.max(1, Math.min(100000, Math.round(Number(raw.tokens) || 50))),
          priceBdt: Math.max(1, Math.min(50000, Math.round(Number(raw.priceBdt) || 99))),
          enabled: raw.enabled !== false
        });
      }
      serverTokenPackages = sanitizedTok;
    }

    await persistGovernanceState();
    recordServerAuditLog(
      owner.uid,
      owner.email,
      'UPDATE_PLANS_PRICING',
      'system/plans',
      `featurePlans=${serverFeaturePackages.length}, tokenPackages=${serverTokenPackages.length}`
    );
    res.status(200).type('application/json').json({
      ok: true,
      featurePackages: serverFeaturePackages,
      tokenPackages: serverTokenPackages
    });
  });

  app.post('/api/admin/payment-settings', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    const b = req.body || {};
    serverPaymentSettings = {
      bkashEnabled:
        typeof b.bkashEnabled === 'boolean' ? b.bkashEnabled : serverPaymentSettings.bkashEnabled,
      bkashNumber:
        typeof b.bkashNumber === 'string'
          ? b.bkashNumber.trim().slice(0, 20)
          : serverPaymentSettings.bkashNumber,
      bkashAccountType: ['Personal', 'Merchant', 'Agent'].includes(b.bkashAccountType)
        ? b.bkashAccountType
        : serverPaymentSettings.bkashAccountType,
      bkashInstructions:
        typeof b.bkashInstructions === 'string'
          ? b.bkashInstructions.trim().slice(0, 500)
          : serverPaymentSettings.bkashInstructions,
      nagadEnabled:
        typeof b.nagadEnabled === 'boolean' ? b.nagadEnabled : serverPaymentSettings.nagadEnabled,
      nagadNumber:
        typeof b.nagadNumber === 'string'
          ? b.nagadNumber.trim().slice(0, 20)
          : serverPaymentSettings.nagadNumber,
      nagadAccountType: ['Personal', 'Merchant', 'Agent'].includes(b.nagadAccountType)
        ? b.nagadAccountType
        : serverPaymentSettings.nagadAccountType,
      nagadInstructions:
        typeof b.nagadInstructions === 'string'
          ? b.nagadInstructions.trim().slice(0, 500)
          : serverPaymentSettings.nagadInstructions
    };

    await persistGovernanceState();
    recordServerAuditLog(
      owner.uid,
      owner.email,
      'UPDATE_PAYMENT_SETTINGS',
      'system/paymentSettings',
      `bkash=${serverPaymentSettings.bkashEnabled}, nagad=${serverPaymentSettings.nagadEnabled}`
    );
    recordServerSecurityEvent('PAYMENT_SETTINGS_UPDATED', owner.email, 'bKash/Nagad settings saved');
    res.status(200).type('application/json').json({ ok: true, paymentSettings: serverPaymentSettings });
  });

  // Atomic Owner Payment Verification (Approve / Reject — strictly prevents duplicate approval or double credit)
  app.post('/api/admin/payments/review', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    const { orderId, decision, rejectionReason } = req.body || {};

    if (!orderId || typeof orderId !== 'string') {
      res.status(400).type('application/json').json({ error: 'Valid orderId is required.' });
      return;
    }
    if (!['approved', 'rejected'].includes(decision)) {
      res.status(400).type('application/json').json({ error: 'Decision must be approved or rejected.' });
      return;
    }

    const reqRecord = serverPaymentRequests.find((r) => r.orderId === orderId.trim());
    if (!reqRecord) {
      res.status(404).type('application/json').json({ error: 'Payment request not found.' });
      return;
    }

    // Strict synchronous idempotency guard: prevent duplicate approvals or re-opening terminal requests
    if (reqRecord.status !== 'pending' || reqRecord.credited) {
      res.status(409).type('application/json').json({
        error: `Payment request ${reqRecord.orderId} has already been ${reqRecord.status} and cannot be processed again.`,
        code: 'PAYMENT_ALREADY_REVIEWED'
      });
      return;
    }

    const nowIso = new Date().toISOString();
    reqRecord.status = decision;
    reqRecord.reviewedByUid = owner.uid;
    reqRecord.updatedAt = nowIso;

    const trxKey = `${reqRecord.paymentMethod}:${reqRecord.trxId}`;
    if (usedPaymentTrxIds[trxKey]) {
      usedPaymentTrxIds[trxKey].status = decision;
    }

    let updatedEntitlement: ServerUserEntitlement | null = null;
    if (decision === 'approved') {
      reqRecord.credited = true;
      reqRecord.rejectionReason = '';
      const ent = ensureUserEntitlementRecord(reqRecord.userUid, reqRecord.userEmail);

      if (reqRecord.itemType === 'token_package' && reqRecord.tokensGranted > 0) {
        applyAtomicTokenDelta({
          uid: reqRecord.userUid,
          email: reqRecord.userEmail,
          delta: reqRecord.tokensGranted,
          type: 'purchase_credit',
          reason: `${reqRecord.paymentMethod.toUpperCase()} TrxID ${reqRecord.trxId} (${reqRecord.itemTitle})`,
          actorUid: owner.uid
        });
      } else if (reqRecord.itemType === 'feature_package' && reqRecord.featureKey) {
        const expMs =
          reqRecord.durationDays > 0
            ? Date.now() + reqRecord.durationDays * 24 * 60 * 60 * 1000
            : 0;
        if (reqRecord.featureKey === 'voice_cloning') {
          ent.voiceCloningUnlocked = true;
          ent.voiceCloningExpiresAtMs = expMs;
        } else if (reqRecord.featureKey === 'long_script') {
          ent.longScriptUnlocked = true;
          ent.longScriptExpiresAtMs = expMs;
        } else if (reqRecord.featureKey === 'srt_generator') {
          ent.srtGeneratorUnlocked = true;
          ent.srtGeneratorExpiresAtMs = expMs;
        } else if (reqRecord.featureKey === 'pro_bundle') {
          ent.proBundleUnlocked = true;
          ent.proBundleExpiresAtMs = expMs;
        }
        ent.updatedAt = nowIso;
      }

      updatedEntitlement = ent;
      recordServerAuditLog(
        owner.uid,
        owner.email,
        'APPROVE_PAYMENT',
        `${reqRecord.userEmail} (${reqRecord.trxId})`,
        `${reqRecord.paymentMethod.toUpperCase()} ৳${reqRecord.amountBdt} -> ${reqRecord.itemTitle}`
      );
      recordServerSecurityEvent(
        'PAYMENT_APPROVED',
        owner.email,
        `${reqRecord.orderId} (${reqRecord.trxId}) -> ${reqRecord.userEmail}`
      );
    } else {
      reqRecord.credited = false;
      reqRecord.rejectionReason = String(
        rejectionReason || 'Transaction verification failed or amount mismatch.'
      )
        .trim()
        .slice(0, 250);
      recordServerAuditLog(
        owner.uid,
        owner.email,
        'REJECT_PAYMENT',
        `${reqRecord.userEmail} (${reqRecord.trxId})`,
        `reason=${reqRecord.rejectionReason}`
      );
      recordServerSecurityEvent(
        'PAYMENT_REJECTED',
        owner.email,
        `${reqRecord.orderId} (${reqRecord.trxId})`
      );
    }

    await persistGovernanceState();
    res.status(200).type('application/json').json({
      ok: true,
      request: reqRecord,
      entitlement: updatedEntitlement
    });
  });

  // Owner Token Management (Grant / Revoke Tokens + Immutable Ledger)
  app.post('/api/admin/tokens/adjust', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    const { targetUid, targetEmail, delta, reason } = req.body || {};
    const cleanUid = typeof targetUid === 'string' ? targetUid.trim() : '';
    const cleanEmail = typeof targetEmail === 'string' ? targetEmail.trim().toLowerCase() : '';
    const numericDelta = Math.round(Number(delta));

    if (!cleanUid || !SAFE_IDENTIFIER_REGEX.test(cleanUid) || !cleanEmail.includes('@')) {
      res.status(400).type('application/json').json({ error: 'Valid user UID and email are required.' });
      return;
    }
    if (!Number.isFinite(numericDelta) || numericDelta === 0 || Math.abs(numericDelta) > 1000000) {
      res.status(400).type('application/json').json({ error: 'Token adjustment delta must be non-zero (-1M to +1M).' });
      return;
    }

    const result = applyAtomicTokenDelta({
      uid: cleanUid,
      email: cleanEmail,
      delta: numericDelta,
      type: numericDelta > 0 ? 'admin_grant' : 'admin_revoke',
      reason: String(reason || (numericDelta > 0 ? 'Owner token grant' : 'Owner token revocation')).trim(),
      actorUid: owner.uid
    });
    if (!result.ok) {
      res.status(400).type('application/json').json({ error: result.error });
      return;
    }

    await persistGovernanceState();
    recordServerAuditLog(
      owner.uid,
      owner.email,
      numericDelta > 0 ? 'GRANT_TOKENS' : 'REVOKE_TOKENS',
      cleanEmail,
      `delta=${numericDelta}, balanceAfter=${result.balanceAfter}`
    );
    res.status(200).type('application/json').json({
      ok: true,
      tokenBalance: result.balanceAfter,
      ledgerEntry: result.entry
    });
  });

  // Owner Manual Feature Entitlement Grant / Revoke
  app.post('/api/admin/entitlements/grant', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    const b = req.body || {};
    const targetUid = typeof b.uid === 'string' ? b.uid.trim() : '';
    const targetEmail = typeof b.email === 'string' ? b.email.trim().toLowerCase() : '';
    if (!targetUid || !SAFE_IDENTIFIER_REGEX.test(targetUid) || !targetEmail.includes('@')) {
      res.status(400).type('application/json').json({ error: 'Valid target UID and email required.' });
      return;
    }

    const durationDays = Math.max(0, Math.min(3650, Math.round(Number(b.durationDays) || 0)));
    const expMs = durationDays > 0 ? Date.now() + durationDays * 24 * 60 * 60 * 1000 : 0;
    const existing = ensureUserEntitlementRecord(targetUid, targetEmail);

    serverUserEntitlements[targetUid] = {
      ...existing,
      uid: targetUid,
      email: targetEmail,
      voiceCloningUnlocked: Boolean(b.voiceCloningUnlocked),
      voiceCloningExpiresAtMs: b.voiceCloningUnlocked ? expMs : 0,
      longScriptUnlocked: Boolean(b.longScriptUnlocked),
      longScriptExpiresAtMs: b.longScriptUnlocked ? expMs : 0,
      srtGeneratorUnlocked: Boolean(b.srtGeneratorUnlocked),
      srtGeneratorExpiresAtMs: b.srtGeneratorUnlocked ? expMs : 0,
      proBundleUnlocked: Boolean(b.proBundleUnlocked),
      proBundleExpiresAtMs: b.proBundleUnlocked ? expMs : 0,
      updatedAt: new Date().toISOString()
    };

    await persistGovernanceState();
    recordServerAuditLog(
      owner.uid,
      owner.email,
      'UPDATE_USER_ENTITLEMENT',
      targetEmail,
      `vc=${Boolean(b.voiceCloningUnlocked)}, long=${Boolean(b.longScriptUnlocked)}, srt=${Boolean(b.srtGeneratorUnlocked)}, bundle=${Boolean(b.proBundleUnlocked)}, days=${durationDays}`
    );
    res.status(200).type('application/json').json({
      ok: true,
      entitlement: serverUserEntitlements[targetUid]
    });
  });

  app.post('/api/admin/users/status', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    const { targetUid, accountStatus } = req.body || {};
    const cleanUid = typeof targetUid === 'string' ? targetUid.trim() : '';
    if (!cleanUid || !['active', 'suspended'].includes(accountStatus)) {
      res.status(400).type('application/json').json({ error: 'Valid targetUid and accountStatus required.' });
      return;
    }
    if (cleanUid === owner.uid) {
      res.status(400).type('application/json').json({ error: 'Cannot suspend the active Owner account.' });
      return;
    }
    if (!serverUsersRegistry[cleanUid]) {
      res.status(404).type('application/json').json({ error: 'User record not found.' });
      return;
    }
    serverUsersRegistry[cleanUid].accountStatus = accountStatus;
    await persistGovernanceState();
    recordServerAuditLog(
      owner.uid,
      owner.email,
      accountStatus === 'suspended' ? 'SUSPEND_USER' : 'ACTIVATE_USER',
      serverUsersRegistry[cleanUid].email,
      `accountStatus=${accountStatus}`
    );
    res.status(200).type('application/json').json({ ok: true, user: serverUsersRegistry[cleanUid] });
  });

  app.post('/api/admin/governance', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    const b = req.body || {};
    serverGovernancePolicy = {
      maintenanceMode:
        typeof b.maintenanceMode === 'boolean'
          ? b.maintenanceMode
          : serverGovernancePolicy.maintenanceMode,
      maintenanceNotice:
        typeof b.maintenanceNotice === 'string'
          ? b.maintenanceNotice.trim().slice(0, 300)
          : serverGovernancePolicy.maintenanceNotice,
      allowVoiceCloning:
        typeof b.allowVoiceCloning === 'boolean'
          ? b.allowVoiceCloning
          : serverGovernancePolicy.allowVoiceCloning,
      allowSrtGeneration:
        typeof b.allowSrtGeneration === 'boolean'
          ? b.allowSrtGeneration
          : serverGovernancePolicy.allowSrtGeneration,
      requireAuthForCompute:
        typeof b.requireAuthForCompute === 'boolean'
          ? b.requireAuthForCompute
          : serverGovernancePolicy.requireAuthForCompute,
      maxScriptChars:
        typeof b.maxScriptChars === 'number'
          ? Math.max(1000, Math.min(25000, Math.round(b.maxScriptChars)))
          : serverGovernancePolicy.maxScriptChars,
      rateLimitPerMinute:
        typeof b.rateLimitPerMinute === 'number'
          ? Math.max(1, Math.min(60, Math.round(b.rateLimitPerMinute)))
          : serverGovernancePolicy.rateLimitPerMinute,
      enforceZeroGeminiQuota:
        typeof b.enforceZeroGeminiQuota === 'boolean'
          ? b.enforceZeroGeminiQuota
          : serverGovernancePolicy.enforceZeroGeminiQuota
    };
    await persistGovernanceState();
    recordServerAuditLog(
      owner.uid,
      owner.email,
      'UPDATE_GOVERNANCE',
      'system/governance',
      `maintenance=${serverGovernancePolicy.maintenanceMode}, cloning=${serverGovernancePolicy.allowVoiceCloning}, srt=${serverGovernancePolicy.allowSrtGeneration}`
    );
    res.status(200).type('application/json').json({
      ok: true,
      governance: serverGovernancePolicy
    });
  });

  app.post('/api/admin/ownership', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    const b = req.body || {};
    if (!serverOwnershipState.primaryOwnerUid) {
      serverOwnershipState.primaryOwnerUid = owner.uid;
      serverOwnershipState.primaryOwnerEmail = owner.email;
    }
    const nextOrg =
      typeof b.organizationName === 'string' && b.organizationName.trim().length >= 2
        ? b.organizationName.trim().slice(0, 120)
        : serverOwnershipState.organizationName;
    const nextLocked =
      typeof b.ownershipLocked === 'boolean'
        ? b.ownershipLocked
        : serverOwnershipState.ownershipLocked;
    serverOwnershipState.organizationName = nextOrg;
    serverOwnershipState.ownershipLocked = nextLocked;
    await persistGovernanceState();
    recordServerAuditLog(
      owner.uid,
      owner.email,
      'UPDATE_OWNERSHIP',
      'system/ownership',
      `org=${nextOrg}, locked=${nextLocked}`
    );
    res.status(200).type('application/json').json({
      ok: true,
      ownership: serverOwnershipState
    });
  });

  app.post('/api/admin/storage/purge', requireAuthenticatedSession, requirePrimaryOwner, async (req, res) => {
    const owner = (req as any).firebaseAdmin as VerifiedFirebaseIdentity;
    let deletedCount = 0;
    for (const [fileId, entry] of uploadedFilesRegistry.entries()) {
      if (!fileId.startsWith('sample_ref_')) {
        await fs.promises.unlink(entry.filePath).catch(() => {});
        const metaPath = resolveSafeChildPath(UPLOAD_DIR, `${fileId}.meta.json`);
        if (metaPath) await fs.promises.unlink(metaPath).catch(() => {});
        uploadedFilesRegistry.delete(fileId);
        deletedCount++;
      }
    }
    for (const dir of [UPLOAD_DIR, GENERATED_DIR, path.join(GENERATED_DIR, 'chunk_cache')]) {
      const files = await fs.promises.readdir(dir).catch(() => []);
      for (const f of files) {
        const p = resolveSafeChildPath(dir, f);
        if (p) {
          const st = await fs.promises.stat(p).catch(() => null);
          if (st?.isFile()) {
            await fs.promises.unlink(p).catch(() => {});
            deletedCount++;
          }
        }
      }
    }
    recordServerAuditLog(owner.uid, owner.email, 'ADMIN_STORAGE_PURGE', 'server/temp_storage', `deleted=${deletedCount}`);
    res.status(200).type('application/json').json({
      ok: true,
      deletedFilesCount: deletedCount
    });
  });

  // Strict JSON 404 barrier for any unmatched /api/* route so Vite/SPA fallback never returns index.html
  app.all('/api/*', (req, res) => {
    res.status(404).type('application/json').json({
      error: `API endpoint not found: ${req.method} ${req.originalUrl}`
    });
  });

  // Mount Vite dev server (initialized in parallel so /api/* routes listen on port 3000 immediately)
  // or static production assets
  if (!IS_PRODUCTION) {
    let resolvedViteMiddlewares: express.RequestHandler | null = null;
    const viteReadyPromise = (async () => {
      const { createServer: createViteServer } = await import('vite');
      const vite = await createViteServer({
        server: {
          middlewareMode: true,
          hmr: process.env.DISABLE_HMR === 'true' ? false : undefined
        },
        appType: 'spa'
      });
      resolvedViteMiddlewares = vite.middlewares;
      return vite.middlewares;
    })();
    viteReadyPromise.catch((viteErr) => {
      console.error('[VoiceLab AI] Vite middleware initialization error:', viteErr);
    });

    app.use(async (req, res, next) => {
      try {
        const handler = resolvedViteMiddlewares || (await viteReadyPromise);
        handler(req, res, next);
      } catch (err) {
        next(err);
      }
    });
  } else {
    const distPath = path.resolve(process.cwd(), 'dist');
    app.use(
      express.static(distPath, {
        index: false,
        maxAge: '1d',
        setHeaders: (res, filePath) => {
          if (filePath.endsWith('.js')) {
            res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
          } else if (filePath.endsWith('.css')) {
            res.setHeader('Content-Type', 'text/css; charset=utf-8');
          } else if (filePath.endsWith('.svg')) {
            res.setHeader('Content-Type', 'image/svg+xml');
          }
          if (filePath.includes(`${path.sep}assets${path.sep}`)) {
            res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
          }
        }
      })
    );

    // SPA Fallback for client-side routes (strictly after /api/* barrier)
    app.get('*', (req, res) => {
      if (req.path.startsWith('/api/')) {
        res.status(404).type('application/json').json({
          error: `API endpoint not found: ${req.method} ${req.originalUrl}`
        });
        return;
      }
      const indexFile = path.join(distPath, 'index.html');
      if (!fs.existsSync(indexFile)) {
        res.status(500).type('application/json').json({
          error: 'Production build assets not found in dist/. Run npm run build first.'
        });
        return;
      }
      res.setHeader('Cache-Control', 'no-cache');
      res.sendFile(indexFile);
    });
  }

  // Global Express Error Middleware — guarantees valid JSON instead of HTML error pages
  app.use(
    (
      err: any,
      _req: express.Request,
      res: express.Response,
      _next: express.NextFunction
    ) => {
      const status =
        typeof err?.status === 'number' &&
        err.status >= 400 &&
        err.status < 600 &&
        ![403, 502, 503, 504].includes(err.status)
          ? err.status
          : 400;
      res.status(status).type('application/json').json({
        error: err?.message || 'Unexpected server error while processing request.'
      });
    }
  );

  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(
      `VoiceLab AI server (${IS_PRODUCTION ? 'production' : 'development'}) listening on http://0.0.0.0:${PORT}`
    );
  });

  // Extended timeout configuration (30 minutes for long-form multi-chunk CPU/GPU audio processing)
  server.requestTimeout = 30 * 60 * 1000;
  server.headersTimeout = 65 * 1000;
  server.keepAliveTimeout = 60 * 1000;

  // Graceful shutdown handlers
  const handleShutdown = (signal: string) => {
    console.log(`[VoiceLab AI] Received ${signal}. Shutting down gracefully...`);
    clearInterval(cleanupIntervalTimer);
    server.close(() => {
      process.exit(0);
    });
    setTimeout(() => {
      process.exit(0);
    }, 5000).unref();
  };

  process.once('SIGTERM', () => handleShutdown('SIGTERM'));
  process.once('SIGINT', () => handleShutdown('SIGINT'));
}

startServer();
