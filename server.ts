import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import os from 'os';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { promisify } from 'util';
import type { AudioSignalAnalysis, CloningLanguageCode } from './src/types/voicelab.ts';
import { inspectAndValidateAudioFile } from './src/server/audioDsp.ts';
import {
  analyzeReferenceVoiceLanguageAndIdentity,
  executeMultilingualSrtPipeline,
  executeSameLanguageVoiceCloning,
  getOrCreateAuthorizedSampleAudio,
  resolveRuntimeGeminiApiKey
} from './src/server/aiPipelines.ts';

const execFileAsync = promisify(execFile);

const PORT =
  process.env.PORT && process.env.PORT !== process.env.NGINX_PORT
    ? Number(process.env.PORT)
    : Number(process.env.DEFAULT_APP_PORT) || 3000;
const IS_PRODUCTION =
  process.env.NODE_ENV === 'production' || process.env.SERVE_STATIC_DIST === 'true';
const TEMP_ROOT = path.join(os.tmpdir(), 'voicelab_ai_storage');
const UPLOAD_DIR = path.join(TEMP_ROOT, 'uploads');
const GENERATED_DIR = path.join(TEMP_ROOT, 'generated');
const SAMPLES_DIR = path.join(TEMP_ROOT, 'samples');

for (const dir of [TEMP_ROOT, UPLOAD_DIR, GENERATED_DIR, SAMPLES_DIR]) {
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
    fileSize: 50 * 1024 * 1024, // 50 MB max upload size
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
          ? 'Audio file exceeds the 50 MB maximum upload limit.'
          : err?.message || 'Audio file upload failed validation.';
      res.status(400).type('application/json').json({
        error: message
      });
      return;
    }
    next();
  });
}

// Automatic temporary file cleanup every 15 minutes (removes uploads/generated older than 30 minutes)
const cleanupIntervalTimer = setInterval(async () => {
  const now = Date.now();
  const ttlMs = 30 * 60 * 1000;
  for (const [fileId, entry] of uploadedFilesRegistry.entries()) {
    if (!fileId.startsWith('sample_ref_') && now - entry.createdAtMs > ttlMs) {
      await fs.promises.unlink(entry.filePath).catch(() => {});
      const metaPath = resolveSafeChildPath(UPLOAD_DIR, `${fileId}.meta.json`);
      if (metaPath) {
        await fs.promises.unlink(metaPath).catch(() => {});
      }
      uploadedFilesRegistry.delete(fileId);
    }
  }
  // Prune stale rate limit buckets
  for (const store of [generalRateBuckets, computeRateBuckets]) {
    for (const [key, bucket] of store.entries()) {
      if (bucket.timestamps.every((t) => now - t > 120_000)) {
        store.delete(key);
      }
    }
  }
}, 15 * 60 * 1000);
cleanupIntervalTimer.unref?.();

async function startServer() {
  await verifyMediaBinaries();

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
        res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
        res.setHeader(
          'Access-Control-Allow-Headers',
          'Content-Type, Accept, X-Requested-With, X-VoiceLab-Session'
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

    const clientIp = String(req.ip || req.socket.remoteAddress || '127.0.0.1');
    const rateStatus = checkSlidingRateLimit(generalRateBuckets, clientIp, 120, 60_000);
    res.setHeader('X-RateLimit-Limit', '120');
    res.setHeader('X-RateLimit-Remaining', String(rateStatus.remaining));
    res.setHeader('X-RateLimit-Reset', String(rateStatus.resetSeconds));

    if (!rateStatus.allowed) {
      res.status(429).type('application/json').json({
        error: 'Rate limit exceeded. Please wait before sending additional API requests.',
        retryAfterSeconds: rateStatus.resetSeconds
      });
      return;
    }

    res.on('finish', () => {
      const elapsedMs = Date.now() - startMs;
      const safePath = req.originalUrl.split('?')[0];
      console.log(
        `[VoiceLab API] ${req.method} ${safePath} -> HTTP ${res.statusCode} (${elapsedMs}ms)`
      );
    });

    next();
  });

  // Heavy AI compute rate-limiter middleware (25 requests / 60s per IP)
  function enforceComputeRateLimit(
    req: express.Request,
    res: express.Response,
    next: express.NextFunction
  ) {
    const clientIp = String(req.ip || req.socket.remoteAddress || '127.0.0.1');
    const status = checkSlidingRateLimit(computeRateBuckets, clientIp, 25, 60_000);
    if (!status.allowed) {
      res.status(429).type('application/json').json({
        error:
          'AI synthesis/transcription rate limit reached (25 requests/minute). Please wait before submitting another job.',
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
        localNeuralNodeMode: 'loopback-on-demand (127.0.0.1 / localhost)',
        sameLanguageCloningLanguages: ['bn', 'hi', 'en'],
        multilingualSrtTranscription: true
      },
      storagePolicy: 'Ephemeral temporary storage (auto-purged after 30 minutes)',
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

  // 2. Load or Synthesize Real Authorized Reference Sample Audio (Workspace A: Voice Cloning only)
  app.get('/api/sample-reference/:lang', async (req, res) => {
    try {
      const lang = req.params.lang as CloningLanguageCode;
      if (!['bn', 'hi', 'en'].includes(lang)) {
        res.status(400).type('application/json').json({
          error: 'Sample language must be bn, hi, or en.'
        });
        return;
      }

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

      res.status(200).type('application/json').json({ analysis });
    } catch (err: any) {
      res.status(400).type('application/json').json({
        error: err?.message || 'Failed to prepare authorized sample reference recording.'
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
    handleAudioUploadMiddleware,
    async (req, res) => {
      try {
        if (!req.file || !req.file.path) {
          res.status(400).type('application/json').json({ error: 'No audio file provided.' });
          return;
        }

        // Verify binary magic bytes before running FFprobe/FFmpeg
        const hasValidSignature = await verifyAudioFileMagicBytes(req.file.path);
        if (!hasValidSignature) {
          await fs.promises.unlink(req.file.path).catch(() => {});
          res.status(400).type('application/json').json({
            error:
              'Security validation failed: File binary header does not match a valid audio container (WAV, MP3, FLAC, OGG, M4A, AAC, WEBM).'
          });
          return;
        }

        const workspace = String(req.body.workspace || 'voice-cloning');
        const fileId = path.basename(req.file.filename, path.extname(req.file.filename));
        const previewUrl = `/api/audio/uploaded/${fileId}`;

        const analysis = await inspectAndValidateAudioFile(
          req.file.path,
          fileId,
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

        // If uploaded in Voice Cloning Studio, also run spoken language & vocal identity extraction
        if (workspace === 'voice-cloning') {
          analysis.detectedSpokenLanguage = await analyzeReferenceVoiceLanguageAndIdentity(
            req.file.path,
            analysis
          );
        }

        await saveUploadedFileEntry(fileId, req.file.path, analysis);

        res.status(200).type('application/json').json({ analysis });
      } catch (err: any) {
        if (req.file?.path) {
          await fs.promises.unlink(req.file.path).catch(() => {});
        }
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

  // 7. WORKSPACE A — Same-Language AI Voice Cloning Generation
  app.post(
    '/api/voice-clone/generate',
    requireAuthenticatedSession,
    enforceComputeRateLimit,
    async (req, res) => {
      try {
        const {
          referenceFileId,
          language,
          text,
          preservePitchContour = false,
          preserveSpectralTimbre = false,
          consentConfirmed = false,
          localEndpointUrl
        } = req.body || {};

        if (!consentConfirmed) {
          // Use 400 instead of 403 so Nginx proxy_intercept_errors does not replace JSON with /forbidden.html
          res.status(400).type('application/json').json({
            error:
              'Consent confirmation required: You must confirm that you own or have explicit authorization to clone this reference voice.'
          });
          return;
        }

        if (!['bn', 'hi', 'en'].includes(language)) {
          res.status(400).type('application/json').json({
            error: 'Invalid language selection. Supported same-language cloning options: bn, hi, en.'
          });
          return;
        }

        const refEntry =
          uploadedFilesRegistry.get(referenceFileId) ||
          (await resolveUploadedFileEntry(referenceFileId));
        if (!refEntry || !fs.existsSync(refEntry.filePath)) {
          res.status(404).type('application/json').json({
            error: 'Reference audio recording not found. Please upload or select a reference voice.'
          });
          return;
        }

        const jobId = `clone_${language}_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`;
        const result = await executeSameLanguageVoiceCloning({
          jobId,
          referenceFilePath: refEntry.filePath,
          referenceAnalysis: refEntry.analysis,
          language: language as CloningLanguageCode,
          text: String(text || ''),
          preservePitchContour: Boolean(preservePitchContour),
          preserveSpectralTimbre: Boolean(preserveSpectralTimbre),
          outputDir: GENERATED_DIR,
          localEndpointUrl: typeof localEndpointUrl === 'string' ? localEndpointUrl : undefined
        });

        res.status(200).type('application/json').json({ result });
      } catch (err: any) {
        res.status(400).type('application/json').json({
          error: err?.message || 'Voice cloning pipeline failed.'
        });
      }
    }
  );

  // 8. WORKSPACE B — Universal Multilingual Audio-to-SRT Transcription
  // Supports both direct multipart File/Blob upload ('audio' field) AND uploaded fileId JSON reference.
  // Strictly prohibits demo/sample audio substitution.
  const handleSrtTranscriptionRequest = async (req: express.Request, res: express.Response) => {
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

      const jobId = `srt_${Date.now()}_${crypto.randomBytes(3).toString('hex')}`;
      const result = await executeMultilingualSrtPipeline({
        jobId,
        filePath: targetFilePath,
        originalFileName: targetFileName,
        durationSeconds: targetDurationSeconds,
        requestedLanguage,
        audioPreviewUrl: targetPreviewUrl
      });

      res.status(200).type('application/json').json({ result });
    } catch (err: any) {
      res.status(422).type('application/json').json({
        error: err?.message || 'Multilingual SRT transcription failed for the uploaded audio.'
      });
    }
  };

  app.post(
    '/api/srt/transcribe',
    requireAuthenticatedSession,
    enforceComputeRateLimit,
    handleAudioUploadMiddleware,
    handleSrtTranscriptionRequest
  );
  app.post(
    '/api/srt/generate',
    requireAuthenticatedSession,
    enforceComputeRateLimit,
    handleAudioUploadMiddleware,
    handleSrtTranscriptionRequest
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
      message: 'All temporary user uploads and generated audio files have been permanently removed.'
    });
  });

  // Strict JSON 404 barrier for any unmatched /api/* route so Vite/SPA fallback never returns index.html
  app.all('/api/*', (req, res) => {
    res.status(404).type('application/json').json({
      error: `API endpoint not found: ${req.method} ${req.originalUrl}`
    });
  });

  // Mount Vite dev server or static production assets
  if (!IS_PRODUCTION) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa'
    });
    app.use(vite.middlewares);
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

  // Production timeout configuration (5 minutes for long-form audio processing)
  server.requestTimeout = 5 * 60 * 1000;
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
