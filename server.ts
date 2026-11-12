import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import os from 'os';
import crypto from 'crypto';
import { createServer as createViteServer } from 'vite';
import { AudioSignalAnalysis, CloningLanguageCode } from './src/types/voicelab.ts';
import { inspectAndValidateAudioFile } from './src/server/audioDsp.ts';
import {
  analyzeReferenceVoiceLanguageAndIdentity,
  executeMultilingualSrtPipeline,
  executeSameLanguageVoiceCloning,
  getOrCreateAuthorizedSampleAudio
} from './src/server/aiPipelines.ts';

const PORT = 3000;
const TEMP_ROOT = path.join(os.tmpdir(), 'voicelab_ai_storage');
const UPLOAD_DIR = path.join(TEMP_ROOT, 'uploads');
const GENERATED_DIR = path.join(TEMP_ROOT, 'generated');
const SAMPLES_DIR = path.join(TEMP_ROOT, 'samples');

for (const dir of [TEMP_ROOT, UPLOAD_DIR, GENERATED_DIR, SAMPLES_DIR]) {
  fs.mkdirSync(dir, { recursive: true });
}

// In-memory registry of temporary uploaded/sample audio files with timestamps for automatic TTL cleanup
const uploadedFilesRegistry = new Map<
  string,
  { filePath: string; analysis: AudioSignalAnalysis; createdAtMs: number }
>();

const ALLOWED_EXTENSIONS = new Set(['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.aac']);

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, UPLOAD_DIR);
  },
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase() || '.wav';
    const safeExt = ALLOWED_EXTENSIONS.has(ext) ? ext : '.wav';
    const uniqueId = `aud_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
    cb(null, `${uniqueId}${safeExt}`);
  }
});

const upload = multer({
  storage,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50 MB max upload size
  },
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const mimeOk =
      file.mimetype.startsWith('audio/') ||
      file.mimetype === 'video/webm' ||
      file.mimetype === 'video/mp4' ||
      file.mimetype === 'application/octet-stream';
    if (ALLOWED_EXTENSIONS.has(ext) || mimeOk) {
      cb(null, true);
    } else {
      cb(new Error('Unsupported audio format. Allowed formats: WAV, MP3, M4A, FLAC, OGG.'));
    }
  }
});

// Sidecar persistence helper so server restarts never lose uploaded audio references
async function saveUploadedFileEntry(
  fileId: string,
  filePath: string,
  analysis: AudioSignalAnalysis
): Promise<void> {
  const createdAtMs = Date.now();
  uploadedFilesRegistry.set(fileId, { filePath, analysis, createdAtMs });
  const metaPath = path.join(UPLOAD_DIR, `${fileId}.meta.json`);
  await fs.promises
    .writeFile(metaPath, JSON.stringify({ filePath, analysis, createdAtMs }), 'utf8')
    .catch(() => {});
}

async function resolveUploadedFileEntry(
  fileId: string | undefined
): Promise<{ filePath: string; analysis: AudioSignalAnalysis; createdAtMs: number } | null> {
  if (!fileId || typeof fileId !== 'string') return null;
  const cleanId = path.basename(fileId.trim());
  const inMem = uploadedFilesRegistry.get(cleanId);
  if (inMem && fs.existsSync(inMem.filePath)) {
    return inMem;
  }

  // Check disk sidecar metadata if server restarted after upload
  const metaPath = path.join(UPLOAD_DIR, `${cleanId}.meta.json`);
  if (fs.existsSync(metaPath)) {
    try {
      const raw = await fs.promises.readFile(metaPath, 'utf8');
      const parsed = JSON.parse(raw);
      if (parsed?.filePath && fs.existsSync(parsed.filePath) && parsed?.analysis) {
        const entry = {
          filePath: String(parsed.filePath),
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
      const fullPath = path.join(UPLOAD_DIR, matchFile);
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
      res.status(400).type('application/json').json({
        error: err?.message || 'Audio file upload failed validation.'
      });
      return;
    }
    next();
  });
}

// Automatic temporary file cleanup every 15 minutes (removes uploads/generated older than 30 minutes)
setInterval(async () => {
  const now = Date.now();
  const ttlMs = 30 * 60 * 1000;
  for (const [fileId, entry] of uploadedFilesRegistry.entries()) {
    if (!fileId.startsWith('sample_ref_') && now - entry.createdAtMs > ttlMs) {
      await fs.promises.unlink(entry.filePath).catch(() => {});
      await fs.promises.unlink(path.join(UPLOAD_DIR, `${fileId}.meta.json`)).catch(() => {});
      uploadedFilesRegistry.delete(fileId);
    }
  }
}, 15 * 60 * 1000);

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '15mb' }));

  // Ensure all /api/* responses disable proxy HTML interception and set JSON content type by default
  app.use('/api', (_req, res, next) => {
    res.setHeader('Cache-Control', 'no-store');
    next();
  });

  // 1. System Health & Environment Info
  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'operational',
      application: 'VoiceLab AI',
      ffmpegAvailable: true,
      storagePolicy: 'Ephemeral temporary storage (auto-purged after 30 minutes)',
      activeTemporaryFiles: uploadedFilesRegistry.size
    });
  });

  // 2. Load or Synthesize Real Authorized Reference Sample Audio (Workspace A: Voice Cloning only)
  app.get('/api/sample-reference/:lang', async (req, res) => {
    try {
      const lang = req.params.lang as CloningLanguageCode;
      if (!['bn', 'hi', 'en'].includes(lang)) {
        res.status(400).json({ error: 'Sample language must be bn, hi, or en.' });
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

      res.json({ analysis });
    } catch (err: any) {
      res.status(500).json({
        error: err?.message || 'Failed to prepare authorized sample reference recording.'
      });
    }
  });

  // 3. Stream Sample Audio
  app.get('/api/audio/sample/:lang', async (req, res) => {
    const lang = req.params.lang as CloningLanguageCode;
    const entry = uploadedFilesRegistry.get(`sample_ref_${lang}`);
    if (!entry || !fs.existsSync(entry.filePath)) {
      res.status(404).json({ error: 'Sample audio not yet initialized.' });
      return;
    }
    res.setHeader('Content-Type', 'audio/wav');
    fs.createReadStream(entry.filePath).pipe(res);
  });

  // 4. Upload & Validate Audio File (Used by Workspace A and Workspace B)
  app.post('/api/audio/upload', handleAudioUploadMiddleware, async (req, res) => {
    try {
      if (!req.file) {
        res.status(400).json({ error: 'No audio file provided.' });
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
        res.status(422).json({
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
  });

  // 5. Stream Uploaded Audio File
  app.get('/api/audio/uploaded/:fileId', async (req, res) => {
    const entry = await resolveUploadedFileEntry(req.params.fileId);
    if (!entry || !fs.existsSync(entry.filePath)) {
      res.status(404).json({ error: 'Audio recording not found or expired.' });
      return;
    }
    res.setHeader('Content-Type', entry.analysis.mimeType || 'audio/wav');
    fs.createReadStream(entry.filePath).pipe(res);
  });

  // 6. Stream / Download Generated Cloned Audio (WAV or MP3)
  app.get('/api/audio/generated/:filename', (req, res) => {
    const safeName = path.basename(req.params.filename);
    const fullPath = path.join(GENERATED_DIR, safeName);
    if (!fs.existsSync(fullPath)) {
      res.status(404).json({ error: 'Generated audio file not found.' });
      return;
    }
    const isMp3 = safeName.endsWith('.mp3');
    res.setHeader('Content-Type', isMp3 ? 'audio/mpeg' : 'audio/wav');
    if (req.query.download === '1') {
      res.setHeader('Content-Disposition', `attachment; filename="voicelab_${safeName}"`);
    }
    fs.createReadStream(fullPath).pipe(res);
  });

  // 7. WORKSPACE A — Same-Language AI Voice Cloning Generation
  app.post('/api/voice-clone/generate', async (req, res) => {
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
  });

  // 8. WORKSPACE B — Universal Multilingual Audio-to-SRT Transcription
  // Supports both direct multipart File/Blob upload ('audio' field) AND uploaded fileId JSON reference.
  // Strictly prohibits demo/sample audio substitution.
  app.post('/api/srt/transcribe', handleAudioUploadMiddleware, async (req, res) => {
    try {
      const requestedLanguage = String(req.body?.requestedLanguage || 'auto');
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
          error: 'No uploaded audio file provided. Please upload an audio file before generating SRT.'
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
  });

  // 9. Ephemeral Storage Manual Purge Endpoint
  app.delete('/api/storage/cleanup', async (_req, res) => {
    let deletedCount = 0;
    for (const [fileId, entry] of uploadedFilesRegistry.entries()) {
      if (!fileId.startsWith('sample_ref_')) {
        await fs.promises.unlink(entry.filePath).catch(() => {});
        await fs.promises.unlink(path.join(UPLOAD_DIR, `${fileId}.meta.json`)).catch(() => {});
        uploadedFilesRegistry.delete(fileId);
        deletedCount++;
      }
    }
    const uploadFiles = await fs.promises.readdir(UPLOAD_DIR).catch(() => []);
    for (const f of uploadFiles) {
      await fs.promises.unlink(path.join(UPLOAD_DIR, f)).catch(() => {});
    }
    const genFiles = await fs.promises.readdir(GENERATED_DIR).catch(() => []);
    for (const f of genFiles) {
      await fs.promises.unlink(path.join(GENERATED_DIR, f)).catch(() => {});
      deletedCount++;
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
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa'
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
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
        typeof err?.status === 'number' && err.status >= 400 && err.status < 600 && ![403, 502, 503, 504].includes(err.status)
          ? err.status
          : 400;
      res.status(status).type('application/json').json({
        error: err?.message || 'Unexpected server error while processing request.'
      });
    }
  );

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`VoiceLab AI server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();

