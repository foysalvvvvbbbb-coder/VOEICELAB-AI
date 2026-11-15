import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { execFile } from 'child_process';
import { promisify } from 'util';
import type { CloningLanguageCode, SavedVoiceProfile, SavedVoiceSourceType } from '../types/voicelab.ts';

const execFileAsync = promisify(execFile);

const MAX_CHUNK_BYTES = 380 * 1024; // 380 KB raw FLAC binary per Firestore doc (well below 512 KB rule cap & 1 MiB doc cap)
const MAX_CHUNKS_PER_ASSET = 28; // Safely below 32 limit in firestore.rules

interface FirebaseAppletConfig {
  projectId?: string;
  firestoreDatabaseId?: string;
}

function loadFirebaseTarget(): { projectId: string; databaseId: string; docsBaseUrl: string } | null {
  try {
    const cfgPath = path.resolve(process.cwd(), 'firebase-applet-config.json');
    if (!fs.existsSync(cfgPath)) return null;
    const parsed = JSON.parse(fs.readFileSync(cfgPath, 'utf8')) as FirebaseAppletConfig;
    const projectId = String(parsed?.projectId || '').trim();
    if (!projectId) return null;
    const databaseId = String(parsed?.firestoreDatabaseId || '(default)').trim() || '(default)';
    const docsBaseUrl = `https://firestore.googleapis.com/v1/projects/${encodeURIComponent(
      projectId
    )}/databases/${encodeURIComponent(databaseId)}/documents`;
    return { projectId, databaseId, docsBaseUrl };
  } catch {
    return null;
  }
}

const FIREBASE_TARGET = loadFirebaseTarget();

export function extractBearerTokenFromAuthHeader(authHeader: string | undefined): string | null {
  const raw = String(authHeader || '').trim();
  if (!raw.toLowerCase().startsWith('bearer ')) return null;
  const tok = raw.slice(7).trim();
  return tok.length >= 32 ? tok : null;
}

function docResourceName(relativePath: string): string {
  if (!FIREBASE_TARGET) {
    throw new Error('Firebase configuration unavailable.');
  }
  const cleanRel = relativePath.replace(/^\/+/, '');
  return `projects/${FIREBASE_TARGET.projectId}/databases/${FIREBASE_TARGET.databaseId}/documents/${cleanRel}`;
}

function decodeFsValue(val: any): any {
  if (!val || typeof val !== 'object') return undefined;
  if ('stringValue' in val) return String(val.stringValue);
  if ('integerValue' in val) return Number(val.integerValue);
  if ('doubleValue' in val) return Number(val.doubleValue);
  if ('booleanValue' in val) return Boolean(val.booleanValue);
  if ('timestampValue' in val) return String(val.timestampValue);
  if ('bytesValue' in val) {
    return Buffer.from(String(val.bytesValue || ''), 'base64');
  }
  if ('nullValue' in val) return null;
  return undefined;
}

function decodeFsDocument(doc: any): Record<string, any> | null {
  if (!doc || typeof doc !== 'object' || !doc.fields) return null;
  const out: Record<string, any> = {};
  for (const [k, v] of Object.entries(doc.fields)) {
    out[k] = decodeFsValue(v);
  }
  if (typeof doc.name === 'string') {
    out.__docId = path.posix.basename(doc.name);
  }
  return out;
}

function mapDecodedProfile(raw: Record<string, any>): SavedVoiceProfile | null {
  const voiceId = String(raw.voiceId || raw.__docId || '').trim();
  const ownerUid = String(raw.ownerUid || '').trim();
  const language = String(raw.language || '').trim() as CloningLanguageCode;
  if (!voiceId || !ownerUid || !['bn', 'hi', 'en'].includes(language)) return null;

  const nowIso = new Date().toISOString();
  return {
    voiceId,
    ownerUid,
    ownerEmail: String(raw.ownerEmail || ''),
    name: String(raw.name || 'Saved Voice'),
    language,
    languageName: String(raw.languageName || language),
    conditioningMethod: String(raw.conditioningMethod || 'zero-shot-canonical-wav-v1'),
    modelCheckpoint: String(raw.modelCheckpoint || 'Mode 2 Zero-Shot Neural'),
    speakerConditioningSha256: String(raw.speakerConditioningSha256 || ''),
    promptConditioningSha256: String(raw.promptConditioningSha256 || ''),
    referenceTranscript: String(raw.referenceTranscript || ''),
    referenceDurationSec: Number(raw.referenceDurationSec) || 1,
    promptDurationSec: Number(raw.promptDurationSec) || 1,
    fundamentalFreqHz: Number(raw.fundamentalFreqHz) || 0,
    pitchMinHz: Number(raw.pitchMinHz) || 0,
    pitchMaxHz: Number(raw.pitchMaxHz) || 0,
    spectralWarmthRatio: Number(raw.spectralWarmthRatio) || 0,
    estimatedSnrDb: Number(raw.estimatedSnrDb) || 0,
    qualityGrade: (raw.qualityGrade as SavedVoiceProfile['qualityGrade']) || 'Good',
    sourceType: (raw.sourceType as SavedVoiceSourceType) || 'reference_upload',
    sourceFileName: String(raw.sourceFileName || `${voiceId}.wav`),
    storageKey: String(raw.storageKey || `${ownerUid}/${voiceId}`),
    consentConfirmed: Boolean(raw.consentConfirmed),
    usageCount: Math.max(0, Math.round(Number(raw.usageCount) || 0)),
    hasConditioningAudio: true,
    lastUsedAt: String(raw.lastUsedAt || nowIso),
    createdAt: String(raw.createdAt || nowIso),
    updatedAt: String(raw.updatedAt || nowIso)
  };
}

async function firestoreFetchJson(
  url: string,
  bearerToken: string,
  init: RequestInit = {}
): Promise<any> {
  const resp = await fetch(url, {
    ...init,
    headers: {
      Authorization: `Bearer ${bearerToken}`,
      'Content-Type': 'application/json',
      ...(init.headers || {})
    },
    signal: AbortSignal.timeout(25_000)
  });
  const text = await resp.text();
  let parsed: any = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    parsed = null;
  }
  if (!resp.ok) {
    const errMsg =
      parsed?.error?.message ||
      parsed?.[0]?.error?.message ||
      text ||
      `Firestore REST HTTP ${resp.status}`;
    const err = new Error(errMsg) as Error & { httpStatus?: number };
    err.httpStatus = resp.status;
    throw err;
  }
  return parsed;
}

async function commitWritesInBatches(
  bearerToken: string,
  writes: any[],
  batchSize = 6
): Promise<void> {
  if (!FIREBASE_TARGET || writes.length === 0) return;
  const commitUrl = `${FIREBASE_TARGET.docsBaseUrl}:commit`;
  for (let i = 0; i < writes.length; i += batchSize) {
    const slice = writes.slice(i, i + batchSize);
    await firestoreFetchJson(commitUrl, bearerToken, {
      method: 'POST',
      body: JSON.stringify({ writes: slice })
    });
  }
}

async function encodeCanonicalFlacFromWav(
  wavPath: string,
  flacPath: string,
  maxSeconds?: number
): Promise<Buffer> {
  const args = ['-y', '-i', wavPath];
  if (maxSeconds && maxSeconds > 0) {
    args.push('-t', String(maxSeconds));
  }
  args.push('-ar', '24000', '-ac', '1', '-c:a', 'flac', '-compression_level', '8', flacPath);
  await execFileAsync('ffmpeg', args, { timeout: 25_000 });
  return fs.promises.readFile(flacPath);
}

function buildProfileDocumentFields(profile: SavedVoiceProfile): Record<string, any> {
  const refDur = Math.max(0.1, Math.min(1800, Number(profile.referenceDurationSec) || 1));
  const promptDur = Math.max(0.1, Math.min(60, Number(profile.promptDurationSec) || 1));
  const allowedGrades = ['Studio Grade', 'Good', 'Acceptable (Noisy)', 'Poor / Corrupted'];
  const qualityGrade = allowedGrades.includes(profile.qualityGrade) ? profile.qualityGrade : 'Good';
  const allowedSources = ['cloned_output', 'reference_upload', 'microphone', 'studio_sample'];
  const sourceType = allowedSources.includes(profile.sourceType)
    ? profile.sourceType
    : 'reference_upload';

  return {
    voiceId: { stringValue: String(profile.voiceId).slice(0, 128) },
    ownerUid: { stringValue: String(profile.ownerUid).slice(0, 128) },
    ownerEmail: { stringValue: String(profile.ownerEmail).slice(0, 254) },
    name: { stringValue: (String(profile.name || 'Saved Voice').trim() || 'Saved Voice').slice(0, 80) },
    language: { stringValue: profile.language },
    languageName: { stringValue: (String(profile.languageName || profile.language).trim() || 'Voice').slice(0, 64) },
    conditioningMethod: {
      stringValue: (String(profile.conditioningMethod || 'zero-shot-canonical-wav-v1').trim() || 'zero-shot-canonical-wav-v1').slice(0, 80)
    },
    modelCheckpoint: {
      stringValue: (String(profile.modelCheckpoint || 'Mode 2 Zero-Shot Neural').trim() || 'Mode 2 Zero-Shot Neural').slice(0, 160)
    },
    speakerConditioningSha256: { stringValue: profile.speakerConditioningSha256 },
    promptConditioningSha256: { stringValue: profile.promptConditioningSha256 },
    referenceTranscript: { stringValue: String(profile.referenceTranscript || '').slice(0, 1000) },
    referenceDurationSec: { doubleValue: refDur },
    promptDurationSec: { doubleValue: promptDur },
    fundamentalFreqHz: { doubleValue: Math.max(0, Math.min(1000, Number(profile.fundamentalFreqHz) || 0)) },
    pitchMinHz: { doubleValue: Math.max(0, Math.min(1000, Number(profile.pitchMinHz) || 0)) },
    pitchMaxHz: { doubleValue: Math.max(0, Math.min(2000, Number(profile.pitchMaxHz) || 0)) },
    spectralWarmthRatio: { doubleValue: Math.max(0, Math.min(1000, Number(profile.spectralWarmthRatio) || 0)) },
    estimatedSnrDb: { doubleValue: Math.max(-100, Math.min(200, Number(profile.estimatedSnrDb) || 0)) },
    qualityGrade: { stringValue: qualityGrade },
    sourceType: { stringValue: sourceType },
    sourceFileName: {
      stringValue: (String(profile.sourceFileName || `${profile.voiceId}.wav`).trim() || `${profile.voiceId}.wav`).slice(0, 160)
    },
    storageKey: {
      stringValue: String(profile.storageKey || `${profile.ownerUid}/${profile.voiceId}`).slice(0, 200)
    },
    consentConfirmed: { booleanValue: true },
    usageCount: { integerValue: '0' }
  };
}

function buildConditioningChunkWrites(params: {
  voiceId: string;
  ownerUid: string;
  assetKind: 'prompt_flac' | 'ref_flac';
  flacBytes: Buffer;
  targetWavSha256: string;
}): any[] {
  const assetSha256 = crypto.createHash('sha256').update(params.flacBytes).digest('hex');
  const slices: Buffer[] = [];
  for (let offset = 0; offset < params.flacBytes.length; offset += MAX_CHUNK_BYTES) {
    slices.push(params.flacBytes.subarray(offset, offset + MAX_CHUNK_BYTES));
  }
  if (slices.length === 0 || slices.length > MAX_CHUNKS_PER_ASSET) {
    throw new Error(
      `Conditioning audio ${params.assetKind} produced invalid chunk count (${slices.length}).`
    );
  }

  const prefix = params.assetKind === 'prompt_flac' ? 'prompt' : 'ref';
  const totalChunks = slices.length;

  return slices.map((slice, idx) => {
    const chunkId = `${prefix}_${String(idx).padStart(3, '0')}`;
    return {
      update: {
        name: docResourceName(`savedVoices/${params.voiceId}/conditioningChunks/${chunkId}`),
        fields: {
          chunkId: { stringValue: chunkId },
          voiceId: { stringValue: params.voiceId },
          ownerUid: { stringValue: params.ownerUid },
          assetKind: { stringValue: params.assetKind },
          codec: { stringValue: 'flac' },
          chunkIndex: { integerValue: String(idx) },
          totalChunks: { integerValue: String(totalChunks) },
          assetSha256: { stringValue: assetSha256 },
          targetWavSha256: { stringValue: params.targetWavSha256 },
          payload: { bytesValue: slice.toString('base64') }
        }
      },
      updateTransforms: [{ fieldPath: 'createdAt', setToServerValue: 'REQUEST_TIME' }]
    };
  });
}

export async function persistSavedVoiceAndAudioToFirestore(params: {
  bearerToken: string | null;
  profile: SavedVoiceProfile;
  refWavPath: string;
  promptWavPath: string;
  vaultDir: string;
}): Promise<{ synced: boolean; uploadedChunks: number; error?: string }> {
  if (!FIREBASE_TARGET || !params.bearerToken) {
    return { synced: false, uploadedChunks: 0, error: 'Missing Firebase config or user token.' };
  }

  try {
    const promptFlacPath = path.join(params.vaultDir, 'prompt_5s.flac');
    const refFlacPath = path.join(params.vaultDir, 'reference.flac');

    const promptFlacBytes = await encodeCanonicalFlacFromWav(params.promptWavPath, promptFlacPath);
    let refFlacBytes = await encodeCanonicalFlacFromWav(params.refWavPath, refFlacPath);
    // If a full uploaded reference exceeds ~10.5 MB FLAC (very long recording), keep first 60s lossless for cloud vault
    if (refFlacBytes.length > MAX_CHUNK_BYTES * MAX_CHUNKS_PER_ASSET) {
      refFlacBytes = await encodeCanonicalFlacFromWav(params.refWavPath, refFlacPath, 60);
    }

    // Ensure parent profile doc exists FIRST (required by firestore.rules Master Gate on subcollection create)
    const profileUrl = `${FIREBASE_TARGET.docsBaseUrl}/savedVoices/${encodeURIComponent(params.profile.voiceId)}`;
    let parentExists = false;
    try {
      await firestoreFetchJson(profileUrl, params.bearerToken, { method: 'GET' });
      parentExists = true;
    } catch (err: any) {
      if (err?.httpStatus !== 404) {
        throw err;
      }
    }

    if (!parentExists) {
      await commitWritesInBatches(params.bearerToken, [
        {
          update: {
            name: docResourceName(`savedVoices/${params.profile.voiceId}`),
            fields: buildProfileDocumentFields(params.profile)
          },
          currentDocument: { exists: false },
          updateTransforms: [
            { fieldPath: 'createdAt', setToServerValue: 'REQUEST_TIME' },
            { fieldPath: 'updatedAt', setToServerValue: 'REQUEST_TIME' },
            { fieldPath: 'lastUsedAt', setToServerValue: 'REQUEST_TIME' }
          ]
        }
      ]);
    }

    // Check which chunks already exist so backfills/retries are completely idempotent
    const existingChunks = await queryVoiceConditioningChunks(
      params.bearerToken,
      params.profile.voiceId,
      params.profile.ownerUid
    ).catch(() => []);
    const existingChunkIds = new Set(existingChunks.map((c) => c.chunkId));

    const promptWrites = buildConditioningChunkWrites({
      voiceId: params.profile.voiceId,
      ownerUid: params.profile.ownerUid,
      assetKind: 'prompt_flac',
      flacBytes: promptFlacBytes,
      targetWavSha256: params.profile.promptConditioningSha256
    });
    const refWrites = buildConditioningChunkWrites({
      voiceId: params.profile.voiceId,
      ownerUid: params.profile.ownerUid,
      assetKind: 'ref_flac',
      flacBytes: refFlacBytes,
      targetWavSha256: params.profile.speakerConditioningSha256
    });

    const pendingWrites = [...promptWrites, ...refWrites].filter((w) => {
      const cid = String(w?.update?.fields?.chunkId?.stringValue || '');
      return cid && !existingChunkIds.has(cid);
    });

    if (pendingWrites.length > 0) {
      // Subcollection rule only allows create (update: if false), so enforce exists: false on new chunks
      const strictCreateWrites = pendingWrites.map((w) => ({
        ...w,
        currentDocument: { exists: false }
      }));
      await commitWritesInBatches(params.bearerToken, strictCreateWrites, 5);
    }

    return { synced: true, uploadedChunks: pendingWrites.length };
  } catch (err: any) {
    return {
      synced: false,
      uploadedChunks: 0,
      error: err?.message || 'Firestore voice persistence failed.'
    };
  }
}

interface RawConditioningChunk {
  chunkId: string;
  assetKind: 'prompt_flac' | 'ref_flac';
  chunkIndex: number;
  totalChunks: number;
  assetSha256: string;
  targetWavSha256: string;
  payload: Buffer;
}

async function queryVoiceConditioningChunks(
  bearerToken: string,
  voiceId: string,
  ownerUid: string
): Promise<RawConditioningChunk[]> {
  if (!FIREBASE_TARGET) return [];
  const queryUrl = `${FIREBASE_TARGET.docsBaseUrl}/savedVoices/${encodeURIComponent(voiceId)}:runQuery`;
  const rows = await firestoreFetchJson(queryUrl, bearerToken, {
    method: 'POST',
    body: JSON.stringify({
      structuredQuery: {
        from: [{ collectionId: 'conditioningChunks' }],
        where: {
          fieldFilter: {
            field: { fieldPath: 'ownerUid' },
            op: 'EQUAL',
            value: { stringValue: ownerUid }
          }
        }
      }
    })
  });

  if (!Array.isArray(rows)) return [];
  const out: RawConditioningChunk[] = [];
  for (const row of rows) {
    const decoded = decodeFsDocument(row?.document);
    if (!decoded || !(decoded.payload instanceof Buffer)) continue;
    const assetKind = decoded.assetKind === 'prompt_flac' ? 'prompt_flac' : 'ref_flac';
    out.push({
      chunkId: String(decoded.chunkId || decoded.__docId || ''),
      assetKind,
      chunkIndex: Number(decoded.chunkIndex) || 0,
      totalChunks: Number(decoded.totalChunks) || 1,
      assetSha256: String(decoded.assetSha256 || ''),
      targetWavSha256: String(decoded.targetWavSha256 || ''),
      payload: decoded.payload
    });
  }
  return out;
}

export async function listUserSavedVoicesFromFirestore(
  bearerToken: string | null,
  ownerUid: string
): Promise<SavedVoiceProfile[]> {
  if (!FIREBASE_TARGET || !bearerToken || !ownerUid) return [];
  try {
    const queryUrl = `${FIREBASE_TARGET.docsBaseUrl}:runQuery`;
    const rows = await firestoreFetchJson(queryUrl, bearerToken, {
      method: 'POST',
      body: JSON.stringify({
        structuredQuery: {
          from: [{ collectionId: 'savedVoices' }],
          where: {
            fieldFilter: {
              field: { fieldPath: 'ownerUid' },
              op: 'EQUAL',
              value: { stringValue: ownerUid }
            }
          }
        }
      })
    });
    if (!Array.isArray(rows)) return [];
    const profiles: SavedVoiceProfile[] = [];
    for (const row of rows) {
      const decoded = decodeFsDocument(row?.document);
      if (!decoded) continue;
      const mapped = mapDecodedProfile(decoded);
      if (mapped && mapped.ownerUid === ownerUid) {
        profiles.push(mapped);
      }
    }
    return profiles;
  } catch (err) {
    console.warn('[VoiceLab FirestoreVault] Failed listing cloud savedVoices:', err);
    return [];
  }
}

export async function fetchSingleSavedVoiceFromFirestore(
  bearerToken: string | null,
  voiceId: string
): Promise<SavedVoiceProfile | null> {
  if (!FIREBASE_TARGET || !bearerToken || !voiceId) return null;
  try {
    const url = `${FIREBASE_TARGET.docsBaseUrl}/savedVoices/${encodeURIComponent(voiceId)}`;
    const doc = await firestoreFetchJson(url, bearerToken, { method: 'GET' });
    const decoded = decodeFsDocument(doc);
    return decoded ? mapDecodedProfile(decoded) : null;
  } catch {
    return null;
  }
}

export async function checkCloudConditioningExists(
  bearerToken: string | null,
  voiceId: string,
  ownerUid: string
): Promise<boolean> {
  if (!FIREBASE_TARGET || !bearerToken || !voiceId || !ownerUid) return false;
  try {
    const queryUrl = `${FIREBASE_TARGET.docsBaseUrl}/savedVoices/${encodeURIComponent(voiceId)}:runQuery`;
    const rows = await firestoreFetchJson(queryUrl, bearerToken, {
      method: 'POST',
      body: JSON.stringify({
        structuredQuery: {
          from: [{ collectionId: 'conditioningChunks' }],
          where: {
            fieldFilter: {
              field: { fieldPath: 'ownerUid' },
              op: 'EQUAL',
              value: { stringValue: ownerUid }
            }
          },
          limit: 1
        }
      })
    });
    return Array.isArray(rows) && rows.some((r) => Boolean(r?.document));
  } catch {
    return false;
  }
}

function reassembleFlacChunks(chunks: RawConditioningChunk[], expectedSha?: string): Buffer | null {
  if (chunks.length === 0) return null;
  const sorted = [...chunks].sort((a, b) => a.chunkIndex - b.chunkIndex);
  const totalExpected = sorted[0].totalChunks;
  if (sorted.length < totalExpected) return null;
  const deduped: Buffer[] = [];
  for (let i = 0; i < totalExpected; i++) {
    const match = sorted.find((c) => c.chunkIndex === i);
    if (!match) return null;
    deduped.push(match.payload);
  }
  const combined = Buffer.concat(deduped);
  const actualSha = crypto.createHash('sha256').update(combined).digest('hex');
  const targetSha = sorted[0].assetSha256 || expectedSha;
  if (targetSha && actualSha !== targetSha) {
    throw new Error('Cloud conditioning audio SHA-256 verification failed.');
  }
  return combined;
}

export async function hydrateSavedVoiceFilesFromFirestore(params: {
  bearerToken: string | null;
  voiceId: string;
  ownerUid: string;
  vaultDir: string;
  refWavPath: string;
  promptWavPath: string;
}): Promise<{
  hydrated: boolean;
  profile: SavedVoiceProfile | null;
}> {
  if (!FIREBASE_TARGET || !params.bearerToken) {
    return { hydrated: false, profile: null };
  }

  const profile = await fetchSingleSavedVoiceFromFirestore(params.bearerToken, params.voiceId);
  if (!profile || profile.ownerUid !== params.ownerUid) {
    return { hydrated: false, profile: null };
  }

  const chunks = await queryVoiceConditioningChunks(
    params.bearerToken,
    params.voiceId,
    params.ownerUid
  );
  if (chunks.length === 0) {
    return { hydrated: false, profile };
  }

  const promptChunks = chunks.filter((c) => c.assetKind === 'prompt_flac');
  const refChunks = chunks.filter((c) => c.assetKind === 'ref_flac');

  const promptFlac = reassembleFlacChunks(promptChunks);
  const refFlac = reassembleFlacChunks(refChunks);

  if (!promptFlac && !refFlac) {
    return { hydrated: false, profile };
  }

  await fs.promises.mkdir(params.vaultDir, { recursive: true });

  if (promptFlac) {
    const promptFlacPath = path.join(params.vaultDir, 'prompt_5s.flac');
    await fs.promises.writeFile(promptFlacPath, promptFlac);
    await execFileAsync(
      'ffmpeg',
      ['-y', '-i', promptFlacPath, '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', params.promptWavPath],
      { timeout: 20_000 }
    );
  }

  if (refFlac) {
    const refFlacPath = path.join(params.vaultDir, 'reference.flac');
    await fs.promises.writeFile(refFlacPath, refFlac);
    await execFileAsync(
      'ffmpeg',
      ['-y', '-i', refFlacPath, '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', params.refWavPath],
      { timeout: 25_000 }
    );
  } else if (fs.existsSync(params.promptWavPath)) {
    await fs.promises.copyFile(params.promptWavPath, params.refWavPath);
  }

  if (!fs.existsSync(params.promptWavPath) && fs.existsSync(params.refWavPath)) {
    await fs.promises.copyFile(params.refWavPath, params.promptWavPath);
  }

  const ok = fs.existsSync(params.refWavPath) && fs.existsSync(params.promptWavPath);
  return { hydrated: ok, profile };
}

export async function renameSavedVoiceInFirestore(
  bearerToken: string | null,
  voiceId: string,
  nextName: string
): Promise<void> {
  if (!FIREBASE_TARGET || !bearerToken) return;
  await commitWritesInBatches(bearerToken, [
    {
      update: {
        name: docResourceName(`savedVoices/${voiceId}`),
        fields: {
          name: { stringValue: nextName.trim().slice(0, 80) }
        }
      },
      updateMask: { fieldPaths: ['name', 'updatedAt'] },
      currentDocument: { exists: true },
      updateTransforms: [{ fieldPath: 'updatedAt', setToServerValue: 'REQUEST_TIME' }]
    }
  ]);
}

export async function incrementSavedVoiceUsageInFirestore(
  bearerToken: string | null,
  voiceId: string
): Promise<void> {
  if (!FIREBASE_TARGET || !bearerToken) return;
  await commitWritesInBatches(bearerToken, [
    {
      update: {
        name: docResourceName(`savedVoices/${voiceId}`),
        fields: {}
      },
      updateMask: { fieldPaths: ['usageCount', 'lastUsedAt', 'updatedAt'] },
      currentDocument: { exists: true },
      updateTransforms: [
        { fieldPath: 'usageCount', increment: { integerValue: '1' } },
        { fieldPath: 'lastUsedAt', setToServerValue: 'REQUEST_TIME' },
        { fieldPath: 'updatedAt', setToServerValue: 'REQUEST_TIME' }
      ]
    }
  ]).catch(() => {});
}

export async function deleteSavedVoiceFromFirestore(
  bearerToken: string | null,
  voiceId: string,
  ownerUid: string
): Promise<void> {
  if (!FIREBASE_TARGET || !bearerToken) return;
  const chunks = await queryVoiceConditioningChunks(bearerToken, voiceId, ownerUid).catch(() => []);
  const deletes: any[] = chunks.map((c) => ({
    delete: docResourceName(`savedVoices/${voiceId}/conditioningChunks/${c.chunkId}`)
  }));
  deletes.push({
    delete: docResourceName(`savedVoices/${voiceId}`)
  });
  await commitWritesInBatches(bearerToken, deletes, 10).catch((err) => {
    console.warn('[VoiceLab FirestoreVault] Cloud delete warning:', err);
  });
}
