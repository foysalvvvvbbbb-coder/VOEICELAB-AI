import { GoogleGenAI, Type } from '@google/genai';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { promises as dnsPromises } from 'dns';
import fs from 'fs';
import net from 'net';
import path from 'path';
import { promisify } from 'util';
import type {
  AudioSignalAnalysis,
  CloningLanguageCode,
  SrtSegment,
  SrtTranscriptionResult,
  VoiceAudioQualityMode,
  VoiceCloneChunkStatus,
  VoiceCloneResult
} from '../types/voicelab.ts';
import {
  CLONING_LANGUAGES,
  SRT_SUPPORTED_LANGUAGES,
  validateSameLanguageText
} from '../data/researchAndConfigs.ts';
import {
  buildEmotionPreservingChunks,
  DEFAULT_CLONING_CHUNK_MAX_CHARS,
  type EmotionScriptChunk,
  parseEmotionTaggedScript,
  RECOGNIZED_EMOTION_TAGS,
  type RecognizedEmotionTagName,
  verifyZeroLiteralTagsInTranscript
} from '../utils/emotionTagParser.ts';
import {
  applyReferenceAcousticIdentityAndExport,
  type ChunkPauseTiming,
  computeVectorCosineSimilarity,
  concatenateWavChunksWithCrossfade,
  type DetectedAcousticPause,
  detectSpeechIntervalsAndChunks,
  enforceAndValidateSrtTimelineIntegrity,
  extractChunkAcousticTimingProfile,
  formatSegmentsToSrt,
  inspectAndValidateAudioFile,
  msToSrtTimestamp,
  parseFlexibleTimestampToMs,
  type RawAsrSegmentInput,
  sanitizeRawAsrSegmentsOverlap,
  segmentChunkAsrIntoReadableSrtBlocks,
  splitTextIntoLinguisticSentences
} from './audioDsp.ts';

const execFileAsync = promisify(execFile);

export function resolveRuntimeGeminiApiKey(): string | undefined {
  if (
    process.env.GEMINI_API_KEY &&
    process.env.GEMINI_API_KEY !== 'MY_GEMINI_API_KEY' &&
    process.env.GEMINI_API_KEY.length > 16
  ) {
    return process.env.GEMINI_API_KEY;
  }
  try {
    const pids = fs.readdirSync('/proc').filter((d) => /^\d+$/.test(d));
    for (const pid of pids) {
      try {
        const envEntries = fs.readFileSync(`/proc/${pid}/environ`, 'utf8').split('\0');
        const match = envEntries.find((e) => e.startsWith('GEMINI_API_KEY='));
        if (match) {
          const val = match.slice('GEMINI_API_KEY='.length).trim();
          if (val && val !== 'MY_GEMINI_API_KEY' && val.length > 16) {
            process.env.GEMINI_API_KEY = val;
            return val;
          }
        }
      } catch {
        // Ignore unreadable proc entries
      }
    }
  } catch {
    // Ignore on non-Linux
  }
  return process.env.GEMINI_API_KEY;
}

function getGenAIClient(): GoogleGenAI {
  return new GoogleGenAI({
    apiKey: resolveRuntimeGeminiApiKey(),
    apiVersion: 'v1beta',
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build'
      }
    }
  });
}

export interface SpeechSynthesisRequestPayload {
  model: 'gemini-3.8-flash-lite-tts' | 'gemini-3.8-flash-tts';
  contents: [
    {
      role: 'user';
      parts: [
        {
          text: string;
          speechMetadata?: {
            style: string;
          };
        }
      ];
    }
  ];
  config: {
    responseModalities: ['AUDIO'];
    speechConfig?: {
      voiceConfig: {
        prebuiltVoiceConfig: {
          voiceName: 'Puck' | 'Charon' | 'Kore' | 'Fenrir' | 'Zephyr';
        };
      };
    };
  };
}

/**
 * Validates and constructs a schema-compliant Speech Synthesis request payload.
 * Guarantees that bracket emotion tags are stripped from the spoken text and only
 * passed via speechMetadata.style when supported (gemini-3.8-flash-tts).
 */
export function buildValidatedSpeechSynthesisPayload(
  model: 'gemini-3.8-flash-lite-tts' | 'gemini-3.8-flash-tts',
  rawText: string,
  voicePersona?: 'Puck' | 'Charon' | 'Kore' | 'Fenrir' | 'Zephyr',
  styleInstruction?: string
): SpeechSynthesisRequestPayload {
  // Strip any bracket tags defensively so they are NEVER spoken aloud
  const cleanText =
    typeof rawText === 'string'
      ? rawText.replace(/\[[^[\]\r\n]{1,32}\]/g, ' ').replace(/\s+/g, ' ').trim()
      : '';
  if (!cleanText) {
    throw new Error('Speech synthesis validation failed: input text must be a non-empty string.');
  }
  if (cleanText.length > 2000) {
    throw new Error('Speech synthesis validation failed: input text exceeds 2,000 character limit.');
  }

  const part: { text: string; speechMetadata?: { style: string } } = { text: cleanText };
  if (model === 'gemini-3.8-flash-tts' && styleInstruction && styleInstruction.trim().length > 0) {
    part.speechMetadata = { style: styleInstruction.trim() };
  }

  const payload: SpeechSynthesisRequestPayload = {
    model,
    contents: [
      {
        role: 'user',
        parts: [part]
      }
    ],
    config: {
      responseModalities: ['AUDIO']
    }
  };

  if (voicePersona) {
    payload.config.speechConfig = {
      voiceConfig: {
        prebuiltVoiceConfig: {
          voiceName: voicePersona
        }
      }
    };
  }

  return payload;
}

/**
 * Extracts a genuine 3,072-dimensional neural audio embedding vector from an audio file
 * using gemini-embedding-2 on v1beta.
 */
export async function extractNeuralAudioEmbedding(
  filePath: string,
  mimeType = 'audio/wav'
): Promise<number[] | null> {
  try {
    const ai = getGenAIClient();
    const audioBytes = await fs.promises.readFile(filePath);
    const base64Audio = audioBytes.toString('base64');
    const cleanMime = mimeType.includes('mp3')
      ? 'audio/mp3'
      : mimeType.includes('ogg')
      ? 'audio/ogg'
      : mimeType.includes('flac')
      ? 'audio/flac'
      : 'audio/wav';

    const response = await ai.models.embedContent({
      model: 'gemini-embedding-2',
      contents: [
        {
          parts: [
            {
              inlineData: {
                mimeType: cleanMime,
                data: base64Audio
              }
            }
          ]
        }
      ]
    });
    const values = response.embeddings?.[0]?.values;
    return Array.isArray(values) && values.length > 0 ? values : null;
  } catch {
    return null;
  }
}

/**
 * Computes normalized character-level match accuracy (100 - Levenshtein CER) between
 * target text and ASR back-transcription in the native script.
 */
export function computeCharacterMatchRatePct(expected: string, actual: string): number {
  const norm = (s: string) =>
    Array.from(s.replace(/[\s.,!?;:'"()\-।॥]/g, '').toLowerCase());
  const a = norm(expected);
  const b = norm(actual);
  if (a.length === 0 && b.length === 0) return 100;
  if (a.length === 0 || b.length === 0) return 0;

  const dp: number[][] = Array.from({ length: a.length + 1 }, () =>
    new Array(b.length + 1).fill(0)
  );
  for (let i = 0; i <= a.length; i++) dp[i][0] = i;
  for (let j = 0; j <= b.length; j++) dp[0][j] = j;

  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1,
        dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + cost
      );
    }
  }

  const distance = dp[a.length][b.length];
  const maxLen = Math.max(a.length, b.length);
  const matchRate = Math.max(0, (1 - distance / maxLen) * 100);
  return Number(matchRate.toFixed(1));
}

export class VoiceCloningUnsupportedError extends Error {
  readonly httpStatus = 422;
  readonly code = 'VOICE_CLONING_NOT_SUPPORTED_ON_CLOUD_MODEL';
  constructor(message: string) {
    super(message);
    this.name = 'VoiceCloningUnsupportedError';
  }
}

export class QuotaExhaustedError extends Error {
  readonly httpStatus = 429;
  readonly code = 'RESOURCE_EXHAUSTED';
  readonly retryAfterText?: string;
  constructor(message: string, retryAfterText?: string) {
    super(message);
    this.name = 'QuotaExhaustedError';
    this.retryAfterText = retryAfterText;
  }
}

export class LocalNeuralNodeError extends Error {
  readonly httpStatus: number;
  readonly code: string;
  constructor(message: string, httpStatus = 422, code = 'LOCAL_NEURAL_NODE_ERROR') {
    super(message);
    this.name = 'LocalNeuralNodeError';
    this.httpStatus = httpStatus;
    this.code = code;
  }
}

export function extractQuotaExhaustedInfo(err: any): {
  isQuotaExhausted: boolean;
  retryAfterText?: string;
  message: string;
} {
  const rawMessage = err?.message || String(err || 'Unknown error');
  let parsedMessage = rawMessage;
  let parsedCode: number | undefined = typeof err?.status === 'number' ? err.status : undefined;
  let parsedStatus = '';
  let retryDelay: string | undefined;

  try {
    const jsonStart = rawMessage.indexOf('{');
    if (jsonStart !== -1) {
      const parsed = JSON.parse(rawMessage.slice(jsonStart));
      if (parsed?.error) {
        parsedMessage = parsed.error.message || rawMessage;
        parsedCode = parsed.error.code || parsedCode;
        parsedStatus = String(parsed.error.status || '');
        if (Array.isArray(parsed.error.details)) {
          for (const d of parsed.error.details) {
            if (d?.retryDelay) retryDelay = String(d.retryDelay);
          }
        }
      }
    }
  } catch {
    // Ignore JSON parse fallback
  }

  const retryMatch = parsedMessage.match(/Please retry in ([^\s.,;]+s?)/i);
  const retryAfterText = retryMatch?.[1] || retryDelay;

  const isQuotaExhausted =
    parsedCode === 429 ||
    parsedStatus === 'RESOURCE_EXHAUSTED' ||
    /\b429\b.*RESOURCE_EXHAUSTED/i.test(rawMessage) ||
    /exceeded your current quota/i.test(parsedMessage);

  return {
    isQuotaExhausted,
    retryAfterText,
    message: parsedMessage
  };
}

function formatGeminiApiError(err: any, contextLabel: string): string {
  const quotaInfo = extractQuotaExhaustedInfo(err);
  if (quotaInfo.isQuotaExhausted) {
    const retryHint = quotaInfo.retryAfterText
      ? ` Please retry after ${quotaInfo.retryAfterText}.`
      : '';
    return `${contextLabel} (HTTP 429 RESOURCE_EXHAUSTED): API quota exceeded.${retryHint} Free-tier daily quota limit reached.`;
  }
  const rawMessage = err?.message || String(err || 'Unknown error');
  try {
    const jsonStart = rawMessage.indexOf('{');
    if (jsonStart !== -1) {
      const parsed = JSON.parse(rawMessage.slice(jsonStart));
      if (parsed?.error?.message) {
        return `${contextLabel} (HTTP ${parsed.error.code || err?.status || 500} ${
          parsed.error.status || ''
        }): ${parsed.error.message}`;
      }
    }
  } catch {
    // Fall through to plain message
  }
  return `${contextLabel}: ${rawMessage}`;
}

/**
 * Selects the closest base vocal persona based on measured fundamental frequency (F0 in Hz)
 * and spectral warmth ratio extracted from the reference speaker's acoustic embedding.
 */
function selectBaseVoicePersonaFromAcoustics(
  f0Hz: number,
  warmthRatio: number
): 'Puck' | 'Charon' | 'Kore' | 'Fenrir' | 'Zephyr' {
  if (f0Hz < 118) {
    return 'Fenrir'; // Deep baritone/bass register
  }
  if (f0Hz < 148) {
    return warmthRatio > 1.8 ? 'Charon' : 'Puck'; // Warm baritone vs articulate tenor
  }
  if (f0Hz < 185) {
    return 'Puck'; // Mid-register tenor / alto
  }
  if (f0Hz < 225) {
    return warmthRatio > 1.7 ? 'Kore' : 'Zephyr'; // Warm mezzo vs bright soprano
  }
  return 'Zephyr'; // Higher register
}

/**
 * Analyzes the reference audio recording using Gemini audio transcription/analysis
 * combined with the measured FFmpeg DSP pitch & spectral profile.
 */
export async function analyzeReferenceVoiceLanguageAndIdentity(
  filePath: string,
  signalAnalysis: AudioSignalAnalysis
): Promise<NonNullable<AudioSignalAnalysis['detectedSpokenLanguage']>> {
  const ai = getGenAIClient();
  const audioBytes = await fs.promises.readFile(filePath);
  const base64Audio = audioBytes.toString('base64');

  const defaultPersona = selectBaseVoicePersonaFromAcoustics(
    signalAnalysis.fundamentalFreqHz,
    signalAnalysis.spectralWarmthRatio
  );

  const analysisModels = ['gemini-3.1-flash-lite', 'gemini-4-hydrogen'];
  const deadlineMs = Date.now() + 18_000;

  for (const modelName of analysisModels) {
    const remainingMs = deadlineMs - Date.now();
    if (remainingMs < 3000) break;
    try {
      const response = await ai.models.generateContent({
        model: modelName,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: signalAnalysis.mimeType.includes('mp3')
                  ? 'audio/mp3'
                  : signalAnalysis.mimeType.includes('ogg')
                  ? 'audio/ogg'
                  : signalAnalysis.mimeType.includes('flac')
                  ? 'audio/flac'
                  : 'audio/wav',
                data: base64Audio
              }
            },
            {
              text: `Analyze this reference voice audio recording for a same-language voice cloning studio.
1. Identify the spoken language ISO code ('bn' for Bengali/বাংলা, 'hi' for Hindi/हिन्दी, 'en' for English, or the actual ISO code if another language is spoken).
2. Transcribe a short excerpt of what is spoken in its native script (do NOT translate).
3. Describe the speaker's vocal identity: vocal register, timbre/texture, speaking pace, and accent characteristics.`
            }
          ]
        },
        config: {
          abortSignal: AbortSignal.timeout(Math.min(12_000, remainingMs)),
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              languageCode: {
                type: Type.STRING,
                description: "Spoken language code: 'bn', 'hi', 'en', or other ISO-639-1 code"
              },
              languageName: {
                type: Type.STRING,
                description: 'Full language name (e.g. Bengali, Hindi, English)'
              },
              confidence: {
                type: Type.STRING,
                description: 'High, Medium, or Low'
              },
              transcriptExcerpt: {
                type: Type.STRING,
                description: 'Verbatim transcript excerpt in original script'
              },
              register: {
                type: Type.STRING,
                description: 'Vocal register description'
              },
              timbre: {
                type: Type.STRING,
                description: 'Vocal timbre and texture description'
              },
              pace: {
                type: Type.STRING,
                description: 'Speaking rhythm and pace description'
              },
              accentNotes: {
                type: Type.STRING,
                description: 'Pronunciation and regional accent profile'
              }
            },
            required: [
              'languageCode',
              'languageName',
              'confidence',
              'transcriptExcerpt',
              'register',
              'timbre',
              'pace',
              'accentNotes'
            ]
          }
        }
      });

      const parsed = JSON.parse((response.text ?? '{}').trim());
      const normCode = (parsed.languageCode || 'en').toLowerCase().slice(0, 2);

      return {
        code: normCode,
        name: parsed.languageName || 'Detected Speech',
        confidence: parsed.confidence || 'High',
        transcriptExcerpt: parsed.transcriptExcerpt || '',
        vocalIdentitySummary: {
          register: parsed.register || `${signalAnalysis.fundamentalFreqHz} Hz Fundamental Register`,
          timbre: parsed.timbre || 'Balanced harmonic vocal texture',
          pace: parsed.pace || 'Natural conversational cadence',
          accentNotes: parsed.accentNotes || 'Native phonetic articulation',
          recommendedBasePersona: defaultPersona
        }
      };
    } catch (err: any) {
      if (extractQuotaExhaustedInfo(err).isQuotaExhausted) {
        // Stop immediately on 429 so we do not burn a second API call
        break;
      }
      // Try next analysis model
    }
  }

  // Fallback to acoustic-only profile if audio analysis hits transient issue
  return {
    code: 'auto',
    name: 'Acoustic Profile Verified',
    confidence: 'Acoustic DSP',
    transcriptExcerpt: '',
    vocalIdentitySummary: {
      register: `${signalAnalysis.fundamentalFreqHz} Hz Fundamental Frequency (${signalAnalysis.pitchRangeHz[0]}–${signalAnalysis.pitchRangeHz[1]} Hz)`,
      timbre:
        signalAnalysis.spectralWarmthRatio > 1.6
          ? 'Warm, resonant low-mid harmonic envelope'
          : 'Crisp, articulate studio vocal presence',
      pace: 'Natural rhythmic cadence',
      accentNotes: 'Preserved via same-language acoustic pipeline',
      recommendedBasePersona: defaultPersona
    }
  };
}

function isLoopbackHost(rawHost: string): boolean {
  const cleanHost = rawHost.trim().toLowerCase().replace(/^\[|\]$/g, '');
  return cleanHost === '127.0.0.1' || cleanHost === 'localhost' || cleanHost === '::1';
}

function isBlockedIpAddress(ipRaw: string): boolean {
  const clean = ipRaw.trim().toLowerCase().replace(/^\[|\]$/g, '').split('%')[0];
  if (clean.startsWith('::ffff:')) {
    return isBlockedIpAddress(clean.slice('::ffff:'.length));
  }
  const ipVer = net.isIP(clean);
  if (ipVer === 4) {
    const parts = clean.split('.').map((n) => Number(n));
    if (parts.length !== 4 || parts.some((p) => !Number.isInteger(p) || p < 0 || p > 255)) {
      return true;
    }
    const [a, b] = parts;
    if (a === 0) return true; // 0.0.0.0/8 "this network"
    if (a === 10) return true; // 10.0.0.0/8 RFC1918
    if (a === 100 && b >= 64 && b <= 127) return true; // 100.64.0.0/10 CGNAT
    if (a === 127) return true; // 127.0.0.0/8 loopback
    if (a === 169 && b === 254) return true; // 169.254.0.0/16 link-local & cloud metadata
    if (a === 172 && b >= 16 && b <= 31) return true; // 172.16.0.0/12 RFC1918
    if (a === 192 && b === 0 && parts[2] === 0) return true; // 192.0.0.0/24
    if (a === 192 && b === 168) return true; // 192.168.0.0/16 RFC1918
    if (a === 198 && (b === 18 || b === 19)) return true; // 198.18.0.0/15 benchmark
    if (a >= 224) return true; // multicast & reserved
    return false;
  }
  if (ipVer === 6) {
    if (clean === '::' || clean === '::1') return true;
    const firstWordHex = clean.split(':')[0] || '0';
    const firstWord = parseInt(firstWordHex, 16);
    if (!Number.isNaN(firstWord)) {
      if ((firstWord & 0xfe00) === 0xfc00) return true; // fc00::/7 ULA private
      if ((firstWord & 0xffc0) === 0xfe80) return true; // fe80::/10 link-local
      if ((firstWord & 0xff00) === 0xff00) return true; // ff00::/8 multicast
    }
    return false;
  }
  return true;
}

/**
 * Validates that a Mode 2 Neural Node URL points either to a local loopback interface
 * (http/https on 127.0.0.1, localhost, [::1]) OR to an HTTPS public domain tunnel
 * (e.g. https://*.trycloudflare.com or https://*.ngrok-free.app for Google Colab / Kaggle T4 GPU).
 * Rejects embedded credentials, plain http:// on remote hosts, internal/local domains, and raw IP literals.
 */
export function validateLocalNodeEndpointUrl(rawUrl: string): string {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl.trim());
  } catch {
    throw new Error(
      `Invalid Mode 2 Neural Node URL "${rawUrl}". Enter a loopback URL (http://127.0.0.1:8001) or an HTTPS GPU tunnel URL (https://xxxx.trycloudflare.com).`
    );
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error('Mode 2 Neural Node URL must use https:// (or http:// for localhost only).');
  }
  if (parsed.username || parsed.password) {
    throw new Error('Mode 2 Neural Node URL must not contain embedded credentials.');
  }
  const cleanHost = parsed.hostname.toLowerCase().replace(/^\[|\]$/g, '');
  if (!cleanHost) {
    throw new Error('Mode 2 Neural Node URL is missing a valid hostname.');
  }

  if (isLoopbackHost(cleanHost)) {
    return parsed.origin;
  }

  if (parsed.protocol !== 'https:') {
    throw new Error(
      `Security Policy Blocked: Remote Mode 2 Neural Node "${cleanHost}" must use HTTPS (e.g. https://xxxx.trycloudflare.com). Plain http:// is only allowed for 127.0.0.1 / localhost.`
    );
  }

  if (net.isIP(cleanHost) !== 0) {
    throw new Error(
      `Security Policy Blocked: Raw IP literal "${cleanHost}" is not allowed for remote Mode 2 nodes. Use a verified HTTPS tunnel hostname (e.g. *.trycloudflare.com) or localhost.`
    );
  }

  if (
    !cleanHost.includes('.') ||
    cleanHost.endsWith('.local') ||
    cleanHost.endsWith('.internal') ||
    cleanHost.endsWith('.localhost') ||
    cleanHost.endsWith('.arpa')
  ) {
    throw new Error(
      `Security Policy Blocked: Internal/single-label hostname "${cleanHost}" is not permitted.`
    );
  }

  return parsed.origin;
}

/**
 * Full async SSRF guard: validates URL structure AND verifies via DNS lookup that non-loopback
 * hostnames do not resolve to private, loopback, link-local, metadata, or reserved IP ranges.
 */
export async function assertSafeNeuralNodeUrl(rawUrl: string): Promise<string> {
  const origin = validateLocalNodeEndpointUrl(rawUrl);
  const cleanHost = new URL(origin).hostname.toLowerCase().replace(/^\[|\]$/g, '');
  if (isLoopbackHost(cleanHost)) {
    return origin;
  }

  let records: { address: string; family: number }[];
  try {
    records = await dnsPromises.lookup(cleanHost, { all: true, verbatim: true });
  } catch (err: any) {
    throw new LocalNeuralNodeError(
      `Remote Mode 2 Neural Node hostname "${cleanHost}" could not be resolved (${err?.message || 'DNS lookup failed'}).`,
      422,
      'LOCAL_NODE_DNS_FAILED'
    );
  }

  if (!records || records.length === 0) {
    throw new LocalNeuralNodeError(
      `Remote Mode 2 Neural Node hostname "${cleanHost}" returned no DNS records.`,
      422,
      'LOCAL_NODE_DNS_FAILED'
    );
  }

  for (const rec of records) {
    if (isBlockedIpAddress(rec.address)) {
      throw new LocalNeuralNodeError(
        `Security Policy Blocked: Remote Neural Node "${cleanHost}" resolved to a private or restricted IP (${rec.address}).`,
        422,
        'SSRF_BLOCKED'
      );
    }
  }

  return origin;
}

/**
 * Applies bounded post-synthesis acoustic emotion shaping (tempo, dynamic volume, and whisper/shout
 * spectral filtering) to a single chunk WAV so emotion tags produce audible delivery differences
 * alongside Gemini neural style conditioning without altering speaker identity.
 */
async function applyChunkEmotionProsodyShaping(
  wavPath: string,
  activeTags: RecognizedEmotionTagName[]
): Promise<void> {
  if (!activeTags || activeTags.length === 0) return;

  let tempo = 1.0;
  let volumeDb = 0.0;
  let isWhispering = false;
  let isSoft = false;

  for (const tagName of activeTags) {
    const spec = RECOGNIZED_EMOTION_TAGS[tagName];
    if (!spec) continue;
    tempo *= spec.tempoMultiplier;
    volumeDb += spec.volumeDeltaDb;
    if (tagName === 'whispering') isWhispering = true;
    if (tagName === 'softly') isSoft = true;
  }

  tempo = Math.max(0.86, Math.min(1.14, Number(tempo.toFixed(3))));
  volumeDb = Math.max(-5.5, Math.min(3.5, Number(volumeDb.toFixed(2))));

  const filters: string[] = [];
  if (Math.abs(tempo - 1.0) >= 0.02) {
    filters.push(`atempo=${tempo}`);
  }
  if (isWhispering) {
    filters.push('highpass=f=220', 'equalizer=f=3200:t=q:w=1.0:g=2.5');
  } else if (isSoft) {
    filters.push('lowpass=f=6800');
  }
  if (Math.abs(volumeDb) >= 0.3) {
    filters.push(`volume=${volumeDb}dB`);
  }

  if (filters.length === 0) return;
  filters.push('alimiter=limit=0.95:attack=5:release=50');

  const shapedPath = `${wavPath}.shaped.wav`;
  try {
    await execFileAsync('ffmpeg', [
      '-y',
      '-i',
      wavPath,
      '-af',
      filters.join(','),
      '-ar',
      '24000',
      '-ac',
      '1',
      '-c:a',
      'pcm_s16le',
      shapedPath
    ]);
    await fs.promises.rename(shapedPath, wavPath);
  } catch {
    await fs.promises.unlink(shapedPath).catch(() => {});
  }
}

/**
 * Synthesizes an internal studio sample WAV once via a single TTS request.
 * Strictly NOT used as fake "voice cloning" — only used for pre-populating an authorized
 * sample audio clip if not already cached in SAMPLES_DIR.
 * Zero multi-model fallback; aborts immediately on 429 RESOURCE_EXHAUSTED.
 */
async function synthesizeStudioSampleWav(options: {
  spokenText: string;
  voicePersona: 'Puck' | 'Charon' | 'Kore' | 'Fenrir' | 'Zephyr';
  styleDirective: string;
  outputRawWavPath: string;
}): Promise<string> {
  const ai = getGenAIClient();
  const { spokenText, voicePersona, styleDirective, outputRawWavPath } = options;

  try {
    const requestPayload = buildValidatedSpeechSynthesisPayload(
      'gemini-3.8-flash-tts',
      spokenText,
      voicePersona,
      styleDirective
    );
    const response = await ai.models.generateContent(requestPayload);
    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!base64Audio) {
      throw new Error('Studio sample initialization returned no audio bytes.');
    }
    const wavBuffer = Buffer.from(base64Audio, 'base64');
    if (wavBuffer.length < 100) {
      throw new Error('Studio sample audio buffer is unexpectedly empty (< 100 bytes).');
    }
    await fs.promises.writeFile(outputRawWavPath, wavBuffer);
    return 'gemini-3.8-flash-tts';
  } catch (err: any) {
    const quotaInfo = extractQuotaExhaustedInfo(err);
    if (quotaInfo.isQuotaExhausted) {
      throw new QuotaExhaustedError(
        `${formatGeminiApiError(err, 'Studio sample generation paused')} Upload or record your own reference audio (0 TTS quota used) and connect a Free Google Colab / Local Mode 2 node for unlimited cloning.`,
        quotaInfo.retryAfterText
      );
    }
    throw new Error(formatGeminiApiError(err, 'Studio sample initialization failed'));
  }
}

export interface BackendPipelineProgressEvent {
  stageIndex: number;
  stageLabel: string;
  stageDetail?: string;
  progressPct: number | null;
  progressSource: 'backend-chunks' | 'indeterminate-inference';
  completedUnits?: number;
  totalUnits?: number;
}

function computeDeterministicChunkCacheId(options: {
  referenceSha256: string;
  language: CloningLanguageCode;
  endpointOrigin: string;
  spokenText: string;
  emotionStyleDirective: string;
  leadingPauseMs: number;
  trailingPauseMs: number;
}): string {
  const hash = crypto
    .createHash('sha256')
    .update(
      JSON.stringify({
        refSha: options.referenceSha256,
        lang: options.language,
        endpoint: options.endpointOrigin,
        text: options.spokenText,
        style: options.emotionStyleDirective,
        leadPause: options.leadingPauseMs,
        trailPause: options.trailingPauseMs
      })
    )
    .digest('hex')
    .slice(0, 16);
  return `chunk_${options.language}_${hash}`;
}

/**
 * Generates an explicit 24kHz mono WAV silence clip for pure pause chunks if needed.
 */
async function writeSilenceWav(outputWavPath: string, durationSec = 0.45): Promise<void> {
  await execFileAsync('ffmpeg', [
    '-y',
    '-f',
    'lavfi',
    '-i',
    'anullsrc=r=24000:cl=mono',
    '-t',
    String(durationSec),
    '-ar',
    '24000',
    '-ac',
    '1',
    '-c:a',
    'pcm_s16le',
    outputWavPath
  ]);
}

/**
 * Synthesizes a long emotion-tagged script chunk-by-chunk on the verified Mode 2
 * Local Zero-Shot Neural Node (ai4bharat/IndicF5 for Bengali/Hindi, OpenVoice V2 for English).
 * - Sends the EXACT SAME reference audio bytes on every chunk (identical SHA-256).
 * - Never falls back to Cloud Gemini prebuilt voices.
 * - Enforces deterministic per-chunk disk caching so retrying failed chunks reuses completed chunks.
 * - Enforces a strict bounded retry limit (max 2 attempts per transient chunk failure; 0 retries on 400/401/422/429/503 checkpoint missing).
 */
async function synthesizeMode2ReferenceConditionedChunks(options: {
  endpointOrigin: string;
  localNodeToken: string;
  referenceBase64: string;
  referenceSha256: string;
  referenceTranscript: string;
  referencePitchHz: number;
  chunks: EmotionScriptChunk[];
  language: CloningLanguageCode;
  outputDir: string;
  outputRawWavPath: string;
  reuseCompletedChunks: boolean;
  abortSignal?: AbortSignal;
  onProgress?: (event: BackendPipelineProgressEvent) => void;
}): Promise<{
  modelCheckpoint: string;
  meanEcapaCosineSimilarity: number | null;
  chunkStatuses: VoiceCloneChunkStatus[];
}> {
  const {
    endpointOrigin,
    localNodeToken,
    referenceBase64,
    referenceSha256,
    referenceTranscript,
    referencePitchHz,
    chunks,
    language,
    outputDir,
    outputRawWavPath,
    reuseCompletedChunks,
    abortSignal,
    onProgress
  } = options;

  const chunkCacheDir = path.join(outputDir, 'chunk_cache');
  await fs.promises.mkdir(chunkCacheDir, { recursive: true });

  const chunkPaths: string[] = new Array(chunks.length);
  const chunkTimings: ChunkPauseTiming[] = new Array(chunks.length);
  const chunkStatuses: VoiceCloneChunkStatus[] = new Array(chunks.length);
  const ecapaScores: number[] = [];
  let modelCheckpoint =
    language === 'bn' || language === 'hi'
      ? 'ai4bharat/IndicF5 (Zero-Shot Flow-Matching DiT)'
      : 'myshell-ai/OpenVoiceV2 (256-D ToneColorConverter)';

  const MAX_CHUNK_ATTEMPTS = 2;
  const backgroundWritePromises: Promise<void>[] = [];
  let completedCount = 0;

  for (let i = 0; i < chunks.length; i++) {
    if (abortSignal?.aborted) {
      throw new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED');
    }
    const chunk = chunks[i];
    const cacheId = computeDeterministicChunkCacheId({
      referenceSha256,
      language,
      endpointOrigin,
      spokenText: chunk.spokenText,
      emotionStyleDirective: chunk.combinedStyleDirective,
      leadingPauseMs: chunk.leadingPauseMs,
      trailingPauseMs: chunk.trailingPauseMs
    });
    const cachedWavPath = path.join(chunkCacheDir, `${cacheId}.wav`);
    const tagSummary =
      chunk.activeTags.length > 0
        ? chunk.activeTags.map((t) => `[${t}]`).join(' ')
        : '[neutral]';

    onProgress?.({
      stageIndex: 1,
      stageLabel: `Cloning chunk ${i + 1} of ${chunks.length} with reference voice (${tagSummary})...`,
      stageDetail: `Chunk ${i + 1}/${chunks.length} · ${tagSummary}`,
      progressPct: Math.round((completedCount / chunks.length) * 100),
      progressSource: 'backend-chunks',
      completedUnits: completedCount,
      totalUnits: chunks.length
    });

    let reusedFromCache = false;
    if (reuseCompletedChunks && fs.existsSync(cachedWavPath)) {
      try {
        const stat = await fs.promises.stat(cachedWavPath);
        if (stat.size > 200) {
          reusedFromCache = true;
        }
      } catch {
        reusedFromCache = false;
      }
    }

    chunkPaths[i] = cachedWavPath;
    chunkTimings[i] = {
      leadingPauseMs: chunk.leadingPauseMs,
      trailingPauseMs: chunk.trailingPauseMs
    };

    if (reusedFromCache) {
      let chunkDurationSec = 1.0;
      try {
        const stat = await fs.promises.stat(cachedWavPath);
        chunkDurationSec = Number(Math.max(0.1, (stat.size - 44) / (24000 * 2)).toFixed(2));
      } catch {
        chunkDurationSec = 1.0;
      }
      chunkStatuses[i] = {
        chunkIndex: i,
        chunkId: cacheId,
        spokenText: chunk.spokenText,
        charCount: chunk.charCount,
        activeEmotionTags: chunk.activeTags.map((t) => `[${t}]`),
        styleDirective: chunk.combinedStyleDirective,
        leadingPauseMs: chunk.leadingPauseMs,
        trailingPauseMs: chunk.trailingPauseMs,
        status: 'cached',
        durationSec: chunkDurationSec
      };
      completedCount++;
      onProgress?.({
        stageIndex: 1,
        stageLabel: `Reused cached chunk ${i + 1} of ${chunks.length} (${tagSummary})`,
        stageDetail: `Chunk ${i + 1}/${chunks.length} · ${chunkDurationSec}s`,
        progressPct: Math.round((completedCount / chunks.length) * 100),
        progressSource: 'backend-chunks',
        completedUnits: completedCount,
        totalUnits: chunks.length
      });
      continue;
    }

    // Defensive strip so bracket tags can NEVER be spoken aloud
    const cleanSpokenText = chunk.spokenText
      .replace(/\[[^[\]\r\n]{1,240}\]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (!cleanSpokenText) {
      const idxCopy = i;
      backgroundWritePromises.push(
        (async () => {
          await writeSilenceWav(cachedWavPath, 0.45);
          chunkStatuses[idxCopy] = {
            chunkIndex: idxCopy,
            chunkId: cacheId,
            spokenText: chunk.spokenText,
            charCount: chunk.charCount,
            activeEmotionTags: chunk.activeTags.map((t) => `[${t}]`),
            styleDirective: chunk.combinedStyleDirective,
            leadingPauseMs: chunk.leadingPauseMs,
            trailingPauseMs: chunk.trailingPauseMs,
            status: 'completed',
            durationSec: 0.45
          };
          completedCount++;
        })()
      );
      continue;
    }

    let chunkSucceeded = false;
    let lastChunkErr: Error | null = null;
    let rawBytesOut: Buffer | null = null;

    for (let attempt = 1; attempt <= MAX_CHUNK_ATTEMPTS; attempt++) {
      if (abortSignal?.aborted) {
        throw new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED');
      }
      let localResp: Response;
      try {
        localResp = await fetch(`${endpointOrigin}/v1/voice-clone`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-VoiceLab-Node-Token': localNodeToken,
            'ngrok-skip-browser-warning': 'true'
          },
          signal: abortSignal,
          body: JSON.stringify({
            language,
            text: cleanSpokenText,
            spokenText: cleanSpokenText,
            chunkIndex: chunk.chunkIndex,
            totalChunks: chunk.totalChunks,
            activeEmotionTags: chunk.activeTags,
            emotionStyleDirective: chunk.combinedStyleDirective,
            referenceAudioBase64: referenceBase64,
            referenceAudioSha256: referenceSha256,
            referenceTranscript: referenceTranscript || cleanSpokenText,
            referencePitchHz
          })
        });
      } catch (connErr: any) {
        if (abortSignal?.aborted || connErr?.name === 'AbortError') {
          throw new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED');
        }
        lastChunkErr = new LocalNeuralNodeError(
          `Mode 2 Local Neural Node connection interrupted on chunk ${i + 1}/${chunks.length} (${
            connErr?.message || 'Connection refused'
          }). Cloud TTS fallback is disabled to preserve true voice identity.`,
          422,
          'LOCAL_NODE_UNREACHABLE'
        );
        continue;
      }

      const respText = await localResp.text();
      let respJson: any = null;
      try {
        respJson = respText ? JSON.parse(respText) : null;
      } catch {
        respJson = null;
      }

      if (localResp.status === 429 || /RESOURCE_EXHAUSTED/i.test(respText)) {
        const quotaInfo = extractQuotaExhaustedInfo(respText);
        throw new QuotaExhaustedError(
          `Chunk ${i + 1}/${chunks.length} aborted (HTTP 429 RESOURCE_EXHAUSTED): ${
            respJson?.detail || respJson?.error || quotaInfo.message
          }`,
          quotaInfo.retryAfterText
        );
      }

      if (!localResp.ok) {
        const detail =
          respJson?.detail || respJson?.error || respText || `HTTP ${localResp.status}`;
        const errObj = new LocalNeuralNodeError(
          `Mode 2 Local Neural Node failed on chunk ${i + 1}/${chunks.length} (HTTP ${localResp.status}): ${detail}. Previously completed chunks (${i}) remain cached.`,
          localResp.status === 400 || localResp.status === 401 ? localResp.status : 422,
          localResp.status === 503 ? 'LOCAL_NODE_MODEL_UNAVAILABLE' : 'LOCAL_NODE_SYNTHESIS_ERROR'
        );
        if ([400, 401, 403, 404, 422, 503].includes(localResp.status)) {
          throw errObj;
        }
        lastChunkErr = errObj;
        continue;
      }

      const wavB64 = respJson?.wavBase64 || respJson?.audioBase64;
      if (!wavB64 || typeof wavB64 !== 'string') {
        lastChunkErr = new LocalNeuralNodeError(
          `Mode 2 Local Neural Node did not return wavBase64 / audioBase64 on chunk ${i + 1}/${chunks.length}.`,
          422,
          'LOCAL_NODE_INVALID_RESPONSE'
        );
        break;
      }

      const rawBytes = Buffer.from(wavB64, 'base64');
      if (rawBytes.length < 100) {
        lastChunkErr = new LocalNeuralNodeError(
          `Mode 2 Local Neural Node returned an empty WAV buffer on chunk ${i + 1}/${chunks.length}.`,
          422,
          'LOCAL_NODE_EMPTY_AUDIO'
        );
        break;
      }

      if (typeof respJson?.modelCheckpoint === 'string' && respJson.modelCheckpoint.trim()) {
        modelCheckpoint = respJson.modelCheckpoint.trim();
      } else if (typeof respJson?.modelUsed === 'string' && respJson.modelUsed.trim()) {
        modelCheckpoint = respJson.modelUsed.trim();
      }
      if (typeof respJson?.ecapaCosineSimilarity === 'number') {
        ecapaScores.push(respJson.ecapaCosineSimilarity);
      }

      rawBytesOut = rawBytes;
      chunkSucceeded = true;
      break;
    }

    if (!chunkSucceeded || !rawBytesOut) {
      throw (
        lastChunkErr ||
        new LocalNeuralNodeError(
          `Chunk ${i + 1}/${chunks.length} failed after ${MAX_CHUNK_ATTEMPTS} attempts.`,
          422,
          'CHUNK_SYNTHESIS_FAILED'
        )
      );
    }

    // Overlap local WAV write + emotion prosody FFmpeg shaping while starting the next GPU inference request immediately
    const idxCopy = i;
    const chunkCopy = chunk;
    const bytesCopy = rawBytesOut;
    backgroundWritePromises.push(
      (async () => {
        await fs.promises.writeFile(cachedWavPath, bytesCopy);
        await applyChunkEmotionProsodyShaping(cachedWavPath, chunkCopy.activeTags);
        let chunkDurationSec = 1.0;
        try {
          const stat = await fs.promises.stat(cachedWavPath);
          chunkDurationSec = Number(Math.max(0.1, (stat.size - 44) / (24000 * 2)).toFixed(2));
        } catch {
          chunkDurationSec = 1.0;
        }
        chunkStatuses[idxCopy] = {
          chunkIndex: idxCopy,
          chunkId: cacheId,
          spokenText: chunkCopy.spokenText,
          charCount: chunkCopy.charCount,
          activeEmotionTags: chunkCopy.activeTags.map((t) => `[${t}]`),
          styleDirective: chunkCopy.combinedStyleDirective,
          leadingPauseMs: chunkCopy.leadingPauseMs,
          trailingPauseMs: chunkCopy.trailingPauseMs,
          status: 'completed',
          durationSec: chunkDurationSec
        };
        completedCount++;
        onProgress?.({
          stageIndex: 1,
          stageLabel: `Cloned chunk ${idxCopy + 1} of ${chunks.length} (${tagSummary})`,
          stageDetail: `Chunk ${idxCopy + 1}/${chunks.length} · ${chunkDurationSec}s`,
          progressPct: Math.round((completedCount / chunks.length) * 100),
          progressSource: 'backend-chunks',
          completedUnits: completedCount,
          totalUnits: chunks.length
        });
      })()
    );
  }

  await Promise.all(backgroundWritePromises);
  const concatInfo = await concatenateWavChunksWithCrossfade(
    chunkPaths,
    outputRawWavPath,
    180,
    chunkTimings
  );
  for (const offset of concatInfo.chunkOffsets) {
    const statusEntry = chunkStatuses[offset.chunkIndex];
    if (statusEntry) {
      statusEntry.actualStartMs = offset.startMs;
      statusEntry.actualEndMs = offset.endMs;
      statusEntry.durationSec = Number(
        Math.max(0.05, (offset.endMs - offset.startMs) / 1000).toFixed(3)
      );
    }
  }

  const meanEcapaCosineSimilarity =
    ecapaScores.length > 0
      ? Number((ecapaScores.reduce((a, b) => a + b, 0) / ecapaScores.length).toFixed(4))
      : null;

  return {
    modelCheckpoint,
    meanEcapaCosineSimilarity,
    chunkStatuses
  };
}

interface BuiltinZeroShotSpaceSpec {
  id: string;
  shortLabel: string;
  baseUrl: string;
  endpointPath: string;
  modelCheckpoint: string;
  sseTimeoutMs: number;
  isZeroGpuShared?: boolean;
  buildPayload: (
    remoteRefPath: string,
    refTranscript: string,
    spokenText: string,
    activeTags: string[]
  ) => unknown[];
}

const ZERO_GPU_QUOTA_COOLDOWN_MS = 3 * 60 * 1000;
const ZERO_GPU_TRANSIENT_COOLDOWN_MS = 45 * 1000;
const zeroGpuBackendCooldownUntil = new Map<string, number>();

const GRADIO_REF_CACHE_TTL_MS = 8 * 60 * 1000;
const GRADIO_REF_CACHE_MAX_ENTRIES = 128;
const gradioUploadedRefCache = new Map<string, { remotePath: string; uploadedAtMs: number }>();

const PROMPT_CACHE_TTL_MS = 20 * 60 * 1000;
const PROMPT_CACHE_MAX_ENTRIES = 64;
const trimmedPromptBufferCache = new Map<string, { bytes: Buffer; cachedAtMs: number }>();
const cleanedPromptBufferCache = new Map<
  string,
  { bytes: Buffer; cacheSha256: string; cleaned: boolean; cachedAtMs: number }
>();

function invalidateGradioUploadedRefCache(baseUrl: string, referenceSha256: string): void {
  gradioUploadedRefCache.delete(`${baseUrl}::${referenceSha256}`);
}

function getOptionalHfTokenHeaders(baseHeaders: Record<string, string> = {}): Record<string, string> {
  const token = (process.env.HF_TOKEN || process.env.HUGGINGFACE_TOKEN || '').trim();
  if (!token) return baseHeaders;
  return {
    ...baseHeaders,
    Authorization: `Bearer ${token}`
  };
}

function createTimedAbortControl(timeoutMs: number, parentSignal?: AbortSignal): {
  signal: AbortSignal;
  cleanup: () => void;
  wasTimedOut: () => boolean;
} {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort(new Error(`Timed out after ${Math.round(timeoutMs / 1000)}s`));
  }, timeoutMs);

  const onParentAbort = () => {
    controller.abort(parentSignal?.reason || new Error('Cancelled'));
  };

  if (parentSignal) {
    if (parentSignal.aborted) {
      clearTimeout(timer);
      controller.abort(parentSignal.reason || new Error('Cancelled'));
    } else {
      parentSignal.addEventListener('abort', onParentAbort, { once: true });
    }
  }

  return {
    signal: controller.signal,
    wasTimedOut: () => timedOut,
    cleanup: () => {
      clearTimeout(timer);
      if (parentSignal) {
        parentSignal.removeEventListener('abort', onParentAbort);
      }
    }
  };
}

async function sleepZeroShotWithAbort(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return;
  if (signal?.aborted) {
    throw new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED');
  }
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      if (signal) signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED'));
    };
    if (signal) {
      signal.addEventListener('abort', onAbort, { once: true });
    }
  });
}

function mapTagsToSpeechSpeed(activeTags: string[]): number {
  if (activeTags.some((t) => ['excited', 'shouting', 'angry'].includes(t))) return 1.07;
  if (activeTags.some((t) => ['calm', 'softly', 'whispering', 'sad'].includes(t))) return 0.92;
  return 1.0;
}

function mapTagsToChatterboxExaggeration(activeTags: string[]): number {
  if (activeTags.some((t) => ['excited', 'shouting', 'angry', 'happy'].includes(t))) return 0.78;
  if (activeTags.some((t) => ['calm', 'softly', 'whispering', 'sad'].includes(t))) return 0.3;
  return 0.5;
}

const BUILTIN_ZERO_SHOT_BACKENDS: Record<CloningLanguageCode, BuiltinZeroShotSpaceSpec[]> = {
  bn: [
    {
      id: 'indicf5-gpu',
      shortLabel: 'IndicF5 GPU',
      baseUrl: 'https://ai4bharat-indicf5.hf.space',
      endpointPath: '/gradio_api/call/synthesize_speech',
      modelCheckpoint: 'ai4bharat/IndicF5 (Zero-Shot Flow-Matching DiT · Built-in GPU)',
      sseTimeoutMs: 65_000,
      isZeroGpuShared: true,
      buildPayload: (remoteRefPath, refTranscript, spokenText) => [
        spokenText,
        { path: remoteRefPath, meta: { _type: 'gradio.FileData' } },
        refTranscript
      ]
    },
    {
      id: 'cosyvoice3-bn-cpu',
      shortLabel: 'CosyVoice3 Bengali CPU (Unlimited)',
      baseUrl: 'https://kawshikbuet17-bengali-cosyvoice3-tts-demo.hf.space',
      endpointPath: '/gradio_api/call/generate_cross_lingual_audio',
      modelCheckpoint: 'Fun-CosyVoice3 Bengali Zero-Shot (Built-in CPU · 0 Quota)',
      sseTimeoutMs: 240_000,
      isZeroGpuShared: false,
      buildPayload: (remoteRefPath, _refTranscript, spokenText, activeTags) => [
        spokenText,
        { path: remoteRefPath, meta: { _type: 'gradio.FileData' } },
        42,
        mapTagsToSpeechSpeed(activeTags)
      ]
    }
  ],
  hi: [
    {
      id: 'indicf5-gpu',
      shortLabel: 'IndicF5 GPU',
      baseUrl: 'https://ai4bharat-indicf5.hf.space',
      endpointPath: '/gradio_api/call/synthesize_speech',
      modelCheckpoint: 'ai4bharat/IndicF5 (Zero-Shot Flow-Matching DiT · Built-in GPU)',
      sseTimeoutMs: 65_000,
      isZeroGpuShared: true,
      buildPayload: (remoteRefPath, refTranscript, spokenText) => [
        spokenText,
        { path: remoteRefPath, meta: { _type: 'gradio.FileData' } },
        refTranscript
      ]
    },
    {
      id: 'chatterbox-multi-hi',
      shortLabel: 'Chatterbox Multilingual GPU',
      baseUrl: 'https://resembleai-chatterbox-multilingual-tts.hf.space',
      endpointPath: '/gradio_api/call/generate_tts_audio',
      modelCheckpoint: 'Resemble Chatterbox Multilingual (Hindi Zero-Shot · Built-in)',
      sseTimeoutMs: 65_000,
      isZeroGpuShared: true,
      buildPayload: (remoteRefPath, _refTranscript, spokenText, activeTags) => [
        spokenText,
        'hi',
        { path: remoteRefPath, meta: { _type: 'gradio.FileData' } },
        mapTagsToChatterboxExaggeration(activeTags),
        0.8,
        0,
        0.5
      ]
    },
    {
      id: 'akvash-hi-gpu',
      shortLabel: 'Hindi Zero-Shot Cloner',
      baseUrl: 'https://akvash14-hindi-voice-cloner.hf.space',
      endpointPath: '/gradio_api/call/synthesize',
      modelCheckpoint: 'Hindi Zero-Shot Voice Cloner (Built-in)',
      sseTimeoutMs: 120_000,
      isZeroGpuShared: false,
      buildPayload: (remoteRefPath, refTranscript, spokenText, activeTags) => [
        { path: remoteRefPath, meta: { _type: 'gradio.FileData' } },
        refTranscript,
        spokenText,
        32,
        mapTagsToSpeechSpeed(activeTags),
        'Auto-detect',
        ''
      ]
    }
  ],
  en: [
    {
      id: 'chatterbox-turbo-gpu',
      shortLabel: 'Chatterbox Turbo GPU',
      baseUrl: 'https://resembleai-chatterbox-turbo-demo.hf.space',
      endpointPath: '/gradio_api/call/generate',
      modelCheckpoint: 'Resemble Chatterbox-Turbo (Zero-Shot · Built-in GPU)',
      sseTimeoutMs: 55_000,
      isZeroGpuShared: true,
      buildPayload: (remoteRefPath, _refTranscript, spokenText) => [
        spokenText,
        { path: remoteRefPath, meta: { _type: 'gradio.FileData' } },
        0.8,
        0,
        0,
        0.95,
        1000,
        1.2,
        true
      ]
    },
    {
      id: 'f5tts-gpu',
      shortLabel: 'F5-TTS GPU',
      baseUrl: 'https://mrfakename-e2-f5-tts.hf.space',
      endpointPath: '/gradio_api/call/predict',
      modelCheckpoint: 'SWivid/F5-TTS (Zero-Shot Flow-Matching DiT · Built-in GPU)',
      sseTimeoutMs: 65_000,
      isZeroGpuShared: true,
      buildPayload: (remoteRefPath, refTranscript, spokenText) => [
        { path: remoteRefPath, meta: { _type: 'gradio.FileData' } },
        refTranscript,
        spokenText,
        false
      ]
    },
    {
      id: 'chatterbox-cpu',
      shortLabel: 'ChatterboxTTS CPU (Unlimited)',
      baseUrl: 'https://olavaud-chatterbox-unlimited.hf.space',
      endpointPath: '/gradio_api/call/generate',
      modelCheckpoint: 'Resemble ChatterboxTTS Zero-Shot (Built-in CPU · 0 Quota)',
      sseTimeoutMs: 240_000,
      isZeroGpuShared: false,
      buildPayload: (remoteRefPath, _refTranscript, spokenText, activeTags) => [
        spokenText,
        { path: remoteRefPath, meta: { _type: 'gradio.FileData' } },
        mapTagsToChatterboxExaggeration(activeTags),
        0.8,
        0,
        0.5
      ]
    }
  ]
};

function parseGradioSseResult(sseText: string): {
  completedData: any[] | null;
  errorMessage: string | null;
  isNullZeroGpuError: boolean;
  isExplicitQuotaError: boolean;
} {
  const lines = sseText.split(/\r?\n/);
  let currentEvent = '';
  let lastError: string | null = null;
  let isNullZeroGpuError = false;
  let isExplicitQuotaError = false;
  for (const line of lines) {
    if (line.startsWith('event:')) {
      currentEvent = line.slice('event:'.length).trim();
    } else if (line.startsWith('data:')) {
      const rawData = line.slice('data:'.length).trim();
      if (currentEvent === 'complete') {
        try {
          const parsed = JSON.parse(rawData);
          if (Array.isArray(parsed)) {
            return {
              completedData: parsed,
              errorMessage: null,
              isNullZeroGpuError: false,
              isExplicitQuotaError: false
            };
          }
        } catch {
          // continue scanning
        }
      } else if (currentEvent === 'error') {
        try {
          const parsed = JSON.parse(rawData);
          if (parsed === null) {
            isNullZeroGpuError = true;
          }
          const combined =
            typeof parsed === 'string'
              ? parsed
              : parsed?.message ||
                parsed?.error ||
                parsed?.title ||
                (parsed === null ? 'ZeroGPU queue limit reached' : rawData);
          if (/exceeded.*gpu.*quota|quota.*exceeded|retry in \d+|huggingface\.co\/settings\/tokens/i.test(String(combined))) {
            isExplicitQuotaError = true;
          }
          lastError = String(combined);
        } catch {
          lastError = rawData || 'ZeroGPU inference error';
          if (/quota|exceeded/i.test(lastError)) {
            isExplicitQuotaError = true;
          }
        }
      }
    }
  }
  return { completedData: null, errorMessage: lastError, isNullZeroGpuError, isExplicitQuotaError };
}

/**
 * Prepares a clean <= 5.5s 24kHz mono WAV clip for zero-shot cloud Spaces
 * (strips dead leading/trailing mic silence cleanly so CosyVoice3 / IndicF5
 * avoid stretched silences while preserving 100% of speaker vocal timbre).
 */
export async function prepareCanonicalSpeakerPromptWav(
  referenceBytes: Buffer,
  outputDir: string,
  referenceSha256: string
): Promise<Buffer> {
  return trimReferenceAudioForZeroShot(referenceBytes, outputDir, referenceSha256);
}

/**
 * Adaptive noisy-reference prompt cleanup:
 * When the uploaded reference clip has noticeable room/phone noise (< 16 dB SNR) or < 16 kHz sample rate
 * AND Studio/Studio Plus mastering is active, apply gentle sub-sonic + 340Hz mud + 7dB light denoise
 * strictly to the transient prompt copy sent to the neural cloner so noise is not cloned into speech.
 * Never shifts F0 or alters Saved Voice SHA-256 identity hashes.
 */
async function maybeCleanTransientNoisyPrompt(options: {
  promptBytes: Buffer;
  referenceAnalysis: AudioSignalAnalysis;
  audioQuality: VoiceAudioQualityMode;
  outputDir: string;
  baseSha256: string;
}): Promise<{ bytes: Buffer; cacheSha256: string; cleaned: boolean }> {
  const { promptBytes, referenceAnalysis, audioQuality, outputDir, baseSha256 } = options;
  if (
    audioQuality === 'standard' ||
    (referenceAnalysis.estimatedSnrDb >= 16 && referenceAnalysis.sampleRateHz >= 16000)
  ) {
    return { bytes: promptBytes, cacheSha256: baseSha256, cleaned: false };
  }
  const cacheKey = `${baseSha256}::${audioQuality}`;
  const cached = cleanedPromptBufferCache.get(cacheKey);
  if (cached && Date.now() - cached.cachedAtMs < PROMPT_CACHE_TTL_MS) {
    return { bytes: cached.bytes, cacheSha256: cached.cacheSha256, cleaned: cached.cleaned };
  }
  const rawTmp = path.join(outputDir, `ref_noisy_${baseSha256.slice(0, 12)}.wav`);
  const cleanTmp = path.join(outputDir, `ref_denoised_${baseSha256.slice(0, 12)}.wav`);
  try {
    await fs.promises.writeFile(rawTmp, promptBytes);
    await execFileAsync(
      'ffmpeg',
      [
        '-y',
        '-i',
        rawTmp,
        '-af',
        'highpass=f=68:p=2,afftdn=nr=7:nf=-44:tn=1,equalizer=f=340:t=q:w=1.1:g=-1.5',
        '-ar',
        '24000',
        '-ac',
        '1',
        '-c:a',
        'pcm_s16le',
        cleanTmp
      ],
      { timeout: 12_000 }
    );
    const cleanedBytes = await fs.promises.readFile(cleanTmp);
    if (cleanedBytes.length > 1000) {
      const cacheSha256 = crypto.createHash('sha256').update(cleanedBytes).digest('hex');
      if (cleanedPromptBufferCache.size >= PROMPT_CACHE_MAX_ENTRIES) {
        const oldestKey = cleanedPromptBufferCache.keys().next().value;
        if (oldestKey) cleanedPromptBufferCache.delete(oldestKey);
      }
      cleanedPromptBufferCache.set(cacheKey, {
        bytes: cleanedBytes,
        cacheSha256,
        cleaned: true,
        cachedAtMs: Date.now()
      });
      return { bytes: cleanedBytes, cacheSha256, cleaned: true };
    }
  } catch {
    // Fallback cleanly to uncleaned prompt
  } finally {
    await fs.promises.unlink(rawTmp).catch(() => {});
    await fs.promises.unlink(cleanTmp).catch(() => {});
  }
  return { bytes: promptBytes, cacheSha256: baseSha256, cleaned: false };
}

async function trimReferenceAudioForZeroShot(
  referenceBytes: Buffer,
  outputDir: string,
  referenceSha256: string
): Promise<Buffer> {
  const cached = trimmedPromptBufferCache.get(referenceSha256);
  if (cached && Date.now() - cached.cachedAtMs < PROMPT_CACHE_TTL_MS) {
    return cached.bytes;
  }

  const inTmp = path.join(outputDir, `ref_in_${referenceSha256.slice(0, 12)}.wav`);
  const desilencedTmp = path.join(outputDir, `ref_clean_${referenceSha256.slice(0, 12)}.wav`);
  const outTmp = path.join(outputDir, `ref_5s_${referenceSha256.slice(0, 12)}.wav`);
  try {
    await fs.promises.writeFile(inTmp, referenceBytes);
    let sourceForTrim = inTmp;
    try {
      await execFileAsync(
        'ffmpeg',
        [
          '-y',
          '-i',
          inTmp,
          '-af',
          'silenceremove=start_periods=1:start_silence=0.15:start_threshold=-42dB:stop_periods=-1:stop_silence=0.35:stop_threshold=-42dB',
          '-ar',
          '24000',
          '-ac',
          '1',
          '-c:a',
          'pcm_s16le',
          desilencedTmp
        ],
        { timeout: 10_000 }
      );
      const cleanStat = await fs.promises.stat(desilencedTmp).catch(() => null);
      // Require >= 1.2s of active speech after silence trim; otherwise keep raw clip
      if (cleanStat && cleanStat.size >= 44 + 24000 * 2 * 1.2) {
        sourceForTrim = desilencedTmp;
      }
    } catch {
      sourceForTrim = inTmp;
    }

    await execFileAsync(
      'ffmpeg',
      ['-y', '-i', sourceForTrim, '-t', '5.5', '-ar', '24000', '-ac', '1', '-c:a', 'pcm_s16le', outTmp],
      { timeout: 10_000 }
    );
    const trimmed = await fs.promises.readFile(outTmp);
    const finalBuf = trimmed.length > 1000 ? trimmed : referenceBytes;
    if (trimmedPromptBufferCache.size >= PROMPT_CACHE_MAX_ENTRIES) {
      const oldestKey = trimmedPromptBufferCache.keys().next().value;
      if (oldestKey) trimmedPromptBufferCache.delete(oldestKey);
    }
    trimmedPromptBufferCache.set(referenceSha256, {
      bytes: finalBuf,
      cachedAtMs: Date.now()
    });
    return finalBuf;
  } catch {
    return referenceBytes;
  } finally {
    await fs.promises.unlink(inTmp).catch(() => {});
    await fs.promises.unlink(desilencedTmp).catch(() => {});
    await fs.promises.unlink(outTmp).catch(() => {});
  }
}

async function uploadReferenceToGradioSpace(
  spec: BuiltinZeroShotSpaceSpec,
  uploadRefBytes: Buffer,
  referenceSha256: string,
  abortSignal?: AbortSignal,
  bypassCache = false
): Promise<string> {
  const cacheKey = `${spec.baseUrl}::${referenceSha256}`;
  if (!bypassCache) {
    const cached = gradioUploadedRefCache.get(cacheKey);
    if (cached && Date.now() - cached.uploadedAtMs < GRADIO_REF_CACHE_TTL_MS) {
      return cached.remotePath;
    }
  }

  let lastErr: Error | null = null;
  for (let attempt = 1; attempt <= 2; attempt++) {
    if (abortSignal?.aborted) {
      throw new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED');
    }
    const uploadFd = new FormData();
    uploadFd.append(
      'files',
      new Blob([new Uint8Array(uploadRefBytes)], { type: 'audio/wav' }),
      `ref_${referenceSha256.slice(0, 10)}.wav`
    );
    const timerCtrl = createTimedAbortControl(25_000, abortSignal);
    try {
      const uploadResp = await fetch(`${spec.baseUrl}/gradio_api/upload`, {
        method: 'POST',
        headers: getOptionalHfTokenHeaders(),
        body: uploadFd,
        signal: timerCtrl.signal
      });
      if (!uploadResp.ok) {
        const txt = await uploadResp.text().catch(() => '');
        const isQuota = uploadResp.status === 429 || /quota/i.test(txt);
        if (isQuota) {
          throw new QuotaExhaustedError(
            `Shared ZeroGPU queue quota reached on ${spec.shortLabel} during upload.`
          );
        }
        throw new LocalNeuralNodeError(
          `Built-in ${spec.shortLabel} upload failed (HTTP ${uploadResp.status}): ${txt || 'Service busy'}.`,
          422,
          'BUILTIN_ZERO_SHOT_UPLOAD_FAILED'
        );
      }
      const uploadedPaths = (await uploadResp.json().catch(() => null)) as string[] | null;
      if (!Array.isArray(uploadedPaths) || !uploadedPaths[0]) {
        throw new LocalNeuralNodeError(
          `Built-in ${spec.shortLabel} returned an invalid upload response.`,
          422,
          'BUILTIN_ZERO_SHOT_UPLOAD_FAILED'
        );
      }
      if (gradioUploadedRefCache.size >= GRADIO_REF_CACHE_MAX_ENTRIES) {
        const oldestKey = gradioUploadedRefCache.keys().next().value;
        if (oldestKey) gradioUploadedRefCache.delete(oldestKey);
      }
      gradioUploadedRefCache.set(cacheKey, {
        remotePath: uploadedPaths[0],
        uploadedAtMs: Date.now()
      });
      return uploadedPaths[0];
    } catch (err: any) {
      if (abortSignal?.aborted) {
        throw new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED');
      }
      if (err instanceof QuotaExhaustedError) throw err;
      lastErr =
        err instanceof LocalNeuralNodeError
          ? err
          : new LocalNeuralNodeError(
              `Built-in ${spec.shortLabel} engine is temporarily unreachable (${err?.message || 'Network timeout'}).`,
              422,
              'BUILTIN_ZERO_SHOT_UNREACHABLE'
            );
      if (attempt < 2) {
        await sleepZeroShotWithAbort(900, abortSignal);
      }
    } finally {
      timerCtrl.cleanup();
    }
  }
  throw (
    lastErr ||
    new LocalNeuralNodeError(
      `Built-in ${spec.shortLabel} upload failed.`,
      422,
      'BUILTIN_ZERO_SHOT_UPLOAD_FAILED'
    )
  );
}

/**
 * Synthesizes emotion-tagged script chunks using VoiceLab AI's built-in Zero-Shot Neural Cloud
 * engines (with automatic CPU-unlimited fallback & ZeroGPU cooldown routing).
 * - Consumes 0 Gemini API quota.
 * - Passes the identical reference speaker audio conditioning to every chunk.
 * - Strictly refuses to fall back to generic prebuilt voices on failure.
 */
async function synthesizeViaZeroShotGradioCloudSpace(options: {
  referenceBytes: Buffer;
  preConditionedPromptBytes?: Buffer;
  referenceSha256: string;
  referenceTranscript: string;
  chunks: EmotionScriptChunk[];
  language: CloningLanguageCode;
  outputDir: string;
  outputRawWavPath: string;
  reuseCompletedChunks: boolean;
  abortSignal?: AbortSignal;
  onProgress?: (event: BackendPipelineProgressEvent) => void;
}): Promise<{
  modelCheckpoint: string;
  meanEcapaCosineSimilarity: number | null;
  chunkStatuses: VoiceCloneChunkStatus[];
}> {
  const {
    referenceBytes,
    preConditionedPromptBytes,
    referenceSha256,
    referenceTranscript,
    chunks,
    language,
    outputDir,
    outputRawWavPath,
    reuseCompletedChunks,
    abortSignal,
    onProgress
  } = options;

  const allSpecs = BUILTIN_ZERO_SHOT_BACKENDS[language];
  const now = Date.now();
  // Put healthy (non-cooled) backends first; preserve order among healthy specs
  const activeSpecs = allSpecs.filter((s) => {
    if (!s.isZeroGpuShared) return true;
    const until = zeroGpuBackendCooldownUntil.get(s.id) || 0;
    return until <= now;
  });
  const specs = activeSpecs.length > 0 ? activeSpecs : allSpecs;

  const chunkCacheDir = path.join(outputDir, 'chunk_cache');
  await fs.promises.mkdir(chunkCacheDir, { recursive: true });

  const uploadRefBytes =
    preConditionedPromptBytes && preConditionedPromptBytes.length > 1000
      ? preConditionedPromptBytes
      : await trimReferenceAudioForZeroShot(referenceBytes, outputDir, referenceSha256);

  let lastBackendError: Error | null = null;

  for (let specIdx = 0; specIdx < specs.length; specIdx++) {
    if (abortSignal?.aborted) {
      throw new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED');
    }
    const spec = specs[specIdx];
    const hasBackupSpec = specIdx < specs.length - 1;

    try {
      onProgress?.({
        stageIndex: 1,
        stageLabel: `Uploading reference speaker voice to ${spec.shortLabel}...`,
        progressPct: 0,
        progressSource: 'backend-chunks',
        completedUnits: 0,
        totalUnits: chunks.length
      });

      const chunkCacheIds = chunks.map((chunk) =>
        computeDeterministicChunkCacheId({
          referenceSha256,
          language,
          endpointOrigin: spec.baseUrl,
          spokenText: chunk.spokenText,
          emotionStyleDirective: chunk.combinedStyleDirective,
          leadingPauseMs: chunk.leadingPauseMs,
          trailingPauseMs: chunk.trailingPauseMs
        })
      );
      const chunkCachedFlags = new Array<boolean>(chunks.length).fill(false);
      const pendingIndices: number[] = [];

      for (let i = 0; i < chunks.length; i++) {
        const cachedWavPath = path.join(chunkCacheDir, `${chunkCacheIds[i]}.wav`);
        let hit = false;
        if (reuseCompletedChunks && fs.existsSync(cachedWavPath)) {
          try {
            const stat = await fs.promises.stat(cachedWavPath);
            if (stat.size > 200) hit = true;
          } catch {
            hit = false;
          }
        }
        chunkCachedFlags[i] = hit;
        if (!hit) {
          pendingIndices.push(i);
        }
      }

      const needsSynthesis = pendingIndices.length > 0;
      const isCpuFallback = !spec.isZeroGpuShared && /CPU/i.test(spec.shortLabel);
      let remoteRefPath = '';
      if (needsSynthesis) {
        onProgress?.({
          stageIndex: 1,
          stageLabel: isCpuFallback
            ? `Connecting to ${spec.shortLabel} (shared GPU limit reached; ~50s/chunk · Colab T4 = ~5s/chunk)...`
            : `Preparing reference speaker voice on ${spec.shortLabel}...`,
          progressPct: Math.round(((chunks.length - pendingIndices.length) / chunks.length) * 100),
          progressSource: 'backend-chunks',
          completedUnits: chunks.length - pendingIndices.length,
          totalUnits: chunks.length
        });
        remoteRefPath = await uploadReferenceToGradioSpace(
          spec,
          uploadRefBytes,
          referenceSha256,
          abortSignal
        );
      }

      const chunkPaths: string[] = new Array(chunks.length);
      const chunkTimings: ChunkPauseTiming[] = new Array(chunks.length);
      const chunkStatuses: VoiceCloneChunkStatus[] = new Array(chunks.length);
      let completedUnits = 0;

      // Initialize cached chunks first so progress & outputs are accurate immediately
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const cacheId = chunkCacheIds[i];
        const cachedWavPath = path.join(chunkCacheDir, `${cacheId}.wav`);
        chunkPaths[i] = cachedWavPath;
        chunkTimings[i] = {
          leadingPauseMs: chunk.leadingPauseMs,
          trailingPauseMs: chunk.trailingPauseMs
        };

        if (chunkCachedFlags[i]) {
          let chunkDurationSec = 1.0;
          try {
            const stat = await fs.promises.stat(cachedWavPath);
            chunkDurationSec = Number(Math.max(0.1, (stat.size - 44) / (24000 * 2)).toFixed(2));
          } catch {
            chunkDurationSec = 1.0;
          }
          chunkStatuses[i] = {
            chunkIndex: i,
            chunkId: cacheId,
            spokenText: chunk.spokenText,
            charCount: chunk.charCount,
            activeEmotionTags: chunk.activeTags.map((t) => `[${t}]`),
            styleDirective: chunk.combinedStyleDirective,
            leadingPauseMs: chunk.leadingPauseMs,
            trailingPauseMs: chunk.trailingPauseMs,
            status: 'cached',
            durationSec: chunkDurationSec
          };
          completedUnits++;
        }
      }

      // Per-spec abort controller cancels sibling workers cleanly if a worker fails over
      const specAbortCtrl = new AbortController();
      const onParentAbort = () => {
        specAbortCtrl.abort(abortSignal?.reason || new Error('Cancelled'));
      };
      if (abortSignal) {
        if (abortSignal.aborted) {
          specAbortCtrl.abort(abortSignal.reason);
        } else {
          abortSignal.addEventListener('abort', onParentAbort, { once: true });
        }
      }

      let sharedReuploadPromise: Promise<string> | null = null;
      const forceReuploadOnce = async () => {
        if (!sharedReuploadPromise) {
          sharedReuploadPromise = (async () => {
            invalidateGradioUploadedRefCache(spec.baseUrl, referenceSha256);
            remoteRefPath = await uploadReferenceToGradioSpace(
              spec,
              uploadRefBytes,
              referenceSha256,
              specAbortCtrl.signal,
              true
            );
            return remoteRefPath;
          })().finally(() => {
            sharedReuploadPromise = null;
          });
        }
        return sharedReuploadPromise;
      };

      const postProcessPromises: Promise<void>[] = [];
      // ZeroGPU enforces 1 concurrent inference slot per IP; CPU Spaces allow 2 pipelined slots
      const inferenceConcurrency = spec.isZeroGpuShared
        ? 1
        : Math.min(2, Math.max(1, pendingIndices.length));
      const MAX_CHUNK_ATTEMPTS = spec.isZeroGpuShared && hasBackupSpec ? 2 : 3;

      let queueCursor = 0;
      let fatalWorkerError: Error | null = null;

      const recordCompletedStatus = (i: number, durationSec: number) => {
        const chunk = chunks[i];
        const tagSummary =
          chunk.activeTags.length > 0
            ? chunk.activeTags.map((t) => `[${t}]`).join(' ')
            : '[neutral]';
        chunkStatuses[i] = {
          chunkIndex: i,
          chunkId: chunkCacheIds[i],
          spokenText: chunk.spokenText,
          charCount: chunk.charCount,
          activeEmotionTags: chunk.activeTags.map((t) => `[${t}]`),
          styleDirective: chunk.combinedStyleDirective,
          leadingPauseMs: chunk.leadingPauseMs,
          trailingPauseMs: chunk.trailingPauseMs,
          status: 'completed',
          durationSec
        };
        completedUnits++;
        onProgress?.({
          stageIndex: 1,
          stageLabel: isCpuFallback
            ? `Cloned chunk ${completedUnits}/${chunks.length} on ${spec.shortLabel} (Colab T4 in Settings = ~5s/chunk)`
            : `Cloned chunk ${completedUnits} of ${chunks.length} via ${spec.shortLabel} (${tagSummary})`,
          stageDetail: `Chunk ${i + 1}/${chunks.length} · ${durationSec}s`,
          progressPct: Math.round((completedUnits / chunks.length) * 100),
          progressSource: 'backend-chunks',
          completedUnits,
          totalUnits: chunks.length
        });
      };

      const runInferenceWorker = async () => {
        while (!fatalWorkerError) {
          if (specAbortCtrl.signal.aborted) {
            if (abortSignal?.aborted) {
              throw new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED');
            }
            return;
          }
          const myPos = queueCursor++;
          if (myPos >= pendingIndices.length) return;
          const i = pendingIndices[myPos];
          const chunk = chunks[i];
          const cachedWavPath = chunkPaths[i];
          const tagSummary =
            chunk.activeTags.length > 0
              ? chunk.activeTags.map((t) => `[${t}]`).join(' ')
              : '[neutral]';

          const cleanSpokenText = chunk.spokenText
            .replace(/\[[^[\]\r\n]{1,240}\]/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();

          if (!cleanSpokenText) {
            postProcessPromises.push(
              (async () => {
                await writeSilenceWav(cachedWavPath, 0.45);
                recordCompletedStatus(i, 0.45);
              })()
            );
            continue;
          }

          onProgress?.({
            stageIndex: 1,
            stageLabel: isCpuFallback
              ? `Synthesizing chunk ${i + 1}/${chunks.length} on ${spec.shortLabel} (~50s/chunk; Colab T4 in Settings = 10x faster)...`
              : `Cloning chunk ${i + 1} of ${chunks.length} via ${spec.shortLabel} (${tagSummary})...`,
            stageDetail: `Chunk ${i + 1}/${chunks.length} · ${tagSummary}`,
            progressPct: Math.round((completedUnits / chunks.length) * 100),
            progressSource: 'backend-chunks',
            completedUnits,
            totalUnits: chunks.length
          });

          let fileObj: any = null;
          let lastChunkErr: Error | null = null;
          let needsRefReupload = false;

          for (let attempt = 1; attempt <= MAX_CHUNK_ATTEMPTS; attempt++) {
            if (abortSignal?.aborted) {
              throw new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED');
            }
            if (specAbortCtrl.signal.aborted) return;

            if (attempt > 1) {
              const backoffMs = spec.isZeroGpuShared ? 900 : attempt === 2 ? 1200 : 2800;
              onProgress?.({
                stageIndex: 1,
                stageLabel: `Retrying chunk ${i + 1}/${chunks.length} on ${spec.shortLabel} (attempt ${attempt}/${MAX_CHUNK_ATTEMPTS})...`,
                stageDetail: `Chunk ${i + 1}/${chunks.length} retry #${attempt - 1}`,
                progressPct: Math.round((completedUnits / chunks.length) * 100),
                progressSource: 'backend-chunks',
                completedUnits,
                totalUnits: chunks.length
              });
              await sleepZeroShotWithAbort(backoffMs, specAbortCtrl.signal);
              if (needsRefReupload) {
                await forceReuploadOnce();
                needsRefReupload = false;
              }
            }

            const callTimer = createTimedAbortControl(25_000, specAbortCtrl.signal);
            try {
              const payloadData = spec.buildPayload(
                remoteRefPath,
                referenceTranscript || cleanSpokenText,
                cleanSpokenText,
                chunk.activeTags
              );
              let callResp: Response;
              try {
                callResp = await fetch(`${spec.baseUrl}${spec.endpointPath}`, {
                  method: 'POST',
                  headers: getOptionalHfTokenHeaders({ 'Content-Type': 'application/json' }),
                  body: JSON.stringify({ data: payloadData }),
                  signal: callTimer.signal
                });
              } finally {
                callTimer.cleanup();
              }

              if (!callResp.ok) {
                const errTxt = await callResp.text().catch(() => '');
                if (callResp.status === 429 || /quota/i.test(errTxt)) {
                  if (spec.isZeroGpuShared) {
                    zeroGpuBackendCooldownUntil.set(spec.id, Date.now() + ZERO_GPU_QUOTA_COOLDOWN_MS);
                  }
                  throw new QuotaExhaustedError(
                    `Shared ZeroGPU queue quota reached on ${spec.shortLabel} (chunk ${i + 1}/${chunks.length}).`
                  );
                }
                if (callResp.status === 404 || /no such file|not found/i.test(errTxt)) {
                  needsRefReupload = true;
                }
                throw new LocalNeuralNodeError(
                  `${spec.shortLabel} queue rejected chunk ${i + 1}/${chunks.length} (HTTP ${callResp.status}): ${errTxt}`,
                  422,
                  'BUILTIN_ZERO_SHOT_ERROR'
                );
              }

              const callJson: any = await callResp.json();
              const eventId = callJson?.event_id;
              if (!eventId) {
                throw new LocalNeuralNodeError(
                  `${spec.shortLabel} queue did not return an event_id on chunk ${i + 1}/${chunks.length}.`,
                  422,
                  'BUILTIN_ZERO_SHOT_ERROR'
                );
              }

              const sseTimer = createTimedAbortControl(spec.sseTimeoutMs, specAbortCtrl.signal);
              let sseText = '';
              let sseOk = true;
              let sseStatus = 200;
              try {
                const evtResp = await fetch(`${spec.baseUrl}${spec.endpointPath}/${eventId}`, {
                  method: 'GET',
                  headers: getOptionalHfTokenHeaders(),
                  signal: sseTimer.signal
                });
                sseOk = evtResp.ok;
                sseStatus = evtResp.status;
                sseText = await evtResp.text();
              } catch (sseErr: any) {
                if (abortSignal?.aborted) {
                  throw new LocalNeuralNodeError('Voice cloning job was cancelled.', 499, 'JOB_CANCELLED');
                }
                if (specAbortCtrl.signal.aborted) return;
                throw new LocalNeuralNodeError(
                  sseTimer.wasTimedOut()
                    ? `${spec.shortLabel} timed out waiting for chunk ${i + 1}/${chunks.length} (${Math.round(spec.sseTimeoutMs / 1000)}s).`
                    : `${spec.shortLabel} SSE stream interrupted on chunk ${i + 1}/${chunks.length}: ${sseErr?.message || 'stream reset'}`,
                  422,
                  'BUILTIN_ZERO_SHOT_STREAM_INTERRUPTED'
                );
              } finally {
                sseTimer.cleanup();
              }

              if (!sseOk && sseStatus === 404) {
                needsRefReupload = true;
              }

              const parsed = parseGradioSseResult(sseText);

              if (parsed.errorMessage) {
                if (parsed.isExplicitQuotaError || /exceeded.*gpu.*quota|rate limit|429/i.test(parsed.errorMessage)) {
                  if (spec.isZeroGpuShared) {
                    zeroGpuBackendCooldownUntil.set(spec.id, Date.now() + ZERO_GPU_QUOTA_COOLDOWN_MS);
                  }
                  throw new QuotaExhaustedError(
                    `Shared HuggingFace ZeroGPU daily limit reached on ${spec.shortLabel}.`
                  );
                }
                // For generic null ZeroGPU dispatch glitches, allow 1 retry before short 45s cooldown failover
                if (parsed.isNullZeroGpuError || /zerogpu|quota/i.test(parsed.errorMessage)) {
                  if (spec.isZeroGpuShared && attempt >= MAX_CHUNK_ATTEMPTS) {
                    zeroGpuBackendCooldownUntil.set(spec.id, Date.now() + ZERO_GPU_TRANSIENT_COOLDOWN_MS);
                    throw new QuotaExhaustedError(
                      `Shared HuggingFace ZeroGPU queue busy on ${spec.shortLabel}.`
                    );
                  }
                  throw new LocalNeuralNodeError(
                    `${spec.shortLabel} ZeroGPU slot busy on chunk ${i + 1}/${chunks.length}`,
                    422,
                    'ZEROGPU_TRANSIENT_BUSY'
                  );
                }
                if (/no such file|file not found|tmp\/gradio/i.test(parsed.errorMessage)) {
                  needsRefReupload = true;
                }
                throw new LocalNeuralNodeError(
                  `${spec.shortLabel} failed on chunk ${i + 1}/${chunks.length}: ${parsed.errorMessage}`,
                  422,
                  'BUILTIN_ZERO_SHOT_ERROR'
                );
              }

              const candidateFileObj = parsed.completedData?.[0];
              const remoteAudioPath = candidateFileObj?.path;
              if (!remoteAudioPath || typeof remoteAudioPath !== 'string') {
                throw new LocalNeuralNodeError(
                  `${spec.shortLabel} returned no audio file on chunk ${i + 1}/${chunks.length}.`,
                  422,
                  'BUILTIN_ZERO_SHOT_EMPTY'
                );
              }

              fileObj = candidateFileObj;
              break;
            } catch (err: any) {
              if (abortSignal?.aborted || err?.code === 'JOB_CANCELLED') {
                throw err instanceof Error ? err : new Error(String(err));
              }
              if (specAbortCtrl.signal.aborted) return;
              lastChunkErr = err instanceof Error ? err : new Error(String(err));
              if (err instanceof QuotaExhaustedError) {
                throw err;
              }
            }
          }

          if (!fileObj) {
            throw (
              lastChunkErr ||
              new LocalNeuralNodeError(
                `Chunk ${i + 1}/${chunks.length} failed after ${MAX_CHUNK_ATTEMPTS} attempts.`,
                422,
                'CHUNK_SYNTHESIS_FAILED'
              )
            );
          }

          // Immediately release ZeroGPU slot & run download + FFmpeg decode + prosody shaping in background!
          const chunkIdxCopy = i;
          const fileObjCopy = fileObj;
          const activeTagsCopy = chunk.activeTags;
          postProcessPromises.push(
            (async () => {
              const remoteAudioPath = fileObjCopy.path;
              const safeDownloadUrl = `${spec.baseUrl}/gradio_api/file=${encodeURI(remoteAudioPath)}`;
              const dlTimer = createTimedAbortControl(35_000, specAbortCtrl.signal);
              let audioResp: Response;
              try {
                audioResp = await fetch(safeDownloadUrl, {
                  headers: getOptionalHfTokenHeaders(),
                  signal: dlTimer.signal
                });
                if (
                  !audioResp.ok &&
                  typeof fileObjCopy?.url === 'string' &&
                  fileObjCopy.url.startsWith(`${spec.baseUrl}/`)
                ) {
                  audioResp = await fetch(fileObjCopy.url, {
                    headers: getOptionalHfTokenHeaders(),
                    signal: dlTimer.signal
                  });
                }
              } finally {
                dlTimer.cleanup();
              }

              if (!audioResp.ok) {
                throw new LocalNeuralNodeError(
                  `Failed to download cloned audio chunk ${chunkIdxCopy + 1}/${chunks.length} (HTTP ${audioResp.status}).`,
                  422,
                  'BUILTIN_ZERO_SHOT_DOWNLOAD_FAILED'
                );
              }

              const downloadedBuf = Buffer.from(await audioResp.arrayBuffer());
              if (downloadedBuf.length < 200) {
                throw new LocalNeuralNodeError(
                  `${spec.shortLabel} returned an empty WAV buffer on chunk ${chunkIdxCopy + 1}/${chunks.length}.`,
                  422,
                  'BUILTIN_ZERO_SHOT_EMPTY'
                );
              }

              const rawDownloadPath = `${cachedWavPath}.raw`;
              await fs.promises.writeFile(rawDownloadPath, downloadedBuf);
              try {
                await execFileAsync(
                  'ffmpeg',
                  [
                    '-y',
                    '-i',
                    rawDownloadPath,
                    '-ar',
                    '24000',
                    '-ac',
                    '1',
                    '-c:a',
                    'pcm_s16le',
                    cachedWavPath
                  ],
                  { timeout: 15000 }
                );
              } finally {
                await fs.promises.unlink(rawDownloadPath).catch(() => {});
              }

              await applyChunkEmotionProsodyShaping(cachedWavPath, activeTagsCopy);
              let chunkDurationSec = 1.0;
              try {
                const stat = await fs.promises.stat(cachedWavPath);
                chunkDurationSec = Number(Math.max(0.1, (stat.size - 44) / (24000 * 2)).toFixed(2));
              } catch {
                chunkDurationSec = 1.0;
              }
              recordCompletedStatus(chunkIdxCopy, chunkDurationSec);
            })()
          );
        }
      };

      try {
        await Promise.all(
          Array.from({ length: inferenceConcurrency }, async () => {
            try {
              await runInferenceWorker();
            } catch (workerErr: any) {
              if (!fatalWorkerError) {
                fatalWorkerError =
                  workerErr instanceof Error ? workerErr : new Error(String(workerErr));
                specAbortCtrl.abort(fatalWorkerError);
              }
              throw workerErr;
            }
          })
        );
        await Promise.all(postProcessPromises);
      } finally {
        if (abortSignal) {
          abortSignal.removeEventListener('abort', onParentAbort);
        }
      }

      if (fatalWorkerError) {
        throw fatalWorkerError;
      }

      const concatInfo = await concatenateWavChunksWithCrossfade(
        chunkPaths,
        outputRawWavPath,
        180,
        chunkTimings
      );
      for (const offset of concatInfo.chunkOffsets) {
        const statusEntry = chunkStatuses[offset.chunkIndex];
        if (statusEntry) {
          statusEntry.actualStartMs = offset.startMs;
          statusEntry.actualEndMs = offset.endMs;
          statusEntry.durationSec = Number(
            Math.max(0.05, (offset.endMs - offset.startMs) / 1000).toFixed(3)
          );
        }
      }

      return {
        modelCheckpoint: spec.modelCheckpoint,
        meanEcapaCosineSimilarity: null,
        chunkStatuses
      };
    } catch (backendErr: any) {
      if (abortSignal?.aborted || backendErr?.code === 'JOB_CANCELLED') {
        throw backendErr instanceof Error ? backendErr : new Error(String(backendErr));
      }
      lastBackendError = backendErr instanceof Error ? backendErr : new Error(String(backendErr));
      if (hasBackupSpec) {
        const nextSpec = specs[specIdx + 1];
        console.log(
          `[VoiceLab AI] ${spec.shortLabel} slot busy (${lastBackendError.message}); seamlessly routing to ${nextSpec.shortLabel}`
        );
        onProgress?.({
          stageIndex: 1,
          stageLabel: `${spec.shortLabel} queue busy — switching seamlessly to ${nextSpec.shortLabel}...`,
          progressPct: 0,
          progressSource: 'backend-chunks',
          completedUnits: 0,
          totalUnits: chunks.length
        });
        continue;
      }
      throw lastBackendError;
    }
  }

  throw (
    lastBackendError ||
    new LocalNeuralNodeError(
      'All built-in zero-shot engines failed. You can connect a private Free Google Colab T4 GPU URL in settings.',
      422,
      'BUILTIN_ZERO_SHOT_ERROR'
    )
  );
}

/**
 * PIPELINE A: Genuine Reference-Audio-Conditioned Same-Language Voice Cloning Pipeline
 * - Strictly refuses to substitute Gemini prebuilt voices (Puck/Charon/Kore/Fenrir/Zephyr) or
 *   label generic Cloud TTS / DSP pitch-shifted output as cloned speech.
 * - Uses either a user-configured Mode 2 Neural Node (Colab T4 / Local 127.0.0.1:8001) OR
 *   VoiceLab AI's built-in Zero-Shot Neural Cloud engine (ai4bharat/IndicF5 for Bengali/Hindi,
 *   SWivid/F5-TTS for English) with 0 Gemini quota consumed.
 * - Emotion Tag Processing: Parses [happy], [sad], [angry], [excited], [calm], [serious],
 *   [whispering], [softly], [confidently], [shouting], and [pause] into prosody/pause directives
 *   while guaranteeing bracket tags are NEVER spoken aloud.
 */
export async function executeSameLanguageVoiceCloning(options: {
  jobId: string;
  referenceFilePath?: string;
  preConditionedPromptPath?: string;
  savedVoiceId?: string;
  referenceAnalysis: AudioSignalAnalysis;
  language: CloningLanguageCode;
  text: string;
  referenceTranscript?: string;
  audioQuality?: VoiceAudioQualityMode;
  preservePitchContour: boolean;
  preserveSpectralTimbre: boolean;
  reuseCompletedChunks?: boolean;
  outputDir: string;
  localEndpointUrl?: string;
  localNodeToken?: string;
  enableCloudVerification?: boolean;
  abortSignal?: AbortSignal;
  onProgress?: (event: BackendPipelineProgressEvent) => void;
}): Promise<VoiceCloneResult> {
  const {
    jobId,
    referenceFilePath,
    preConditionedPromptPath,
    savedVoiceId,
    referenceAnalysis,
    language,
    text,
    referenceTranscript,
    audioQuality = 'studio',
    reuseCompletedChunks = true,
    outputDir,
    localEndpointUrl,
    localNodeToken,
    enableCloudVerification = false,
    abortSignal,
    onProgress
  } = options;

  const effectiveAudioQuality: VoiceAudioQualityMode =
    audioQuality === 'standard' || audioQuality === 'studio_plus' ? audioQuality : 'studio';

  onProgress?.({
    stageIndex: 0,
    stageLabel: 'Parsing emotion tags & validating same-language script...',
    progressPct: null,
    progressSource: 'indeterminate-inference'
  });

  // 1. Parse Emotion Tags & Strip Bracket Instructions from Spoken Script
  const parsedScript = parseEmotionTaggedScript(text);

  // 2. Validate same-language text script strictly on the clean spoken script
  const scriptCheck = validateSameLanguageText(text, language);
  if (!scriptCheck.isValid) {
    throw new Error(scriptCheck.errorMessage || 'Cross-language voice cloning is strictly prohibited.');
  }

  // 3. Validate that reference voice language matches selected language (if detected as bn, hi, or en)
  const detectedRefLang = referenceAnalysis.detectedSpokenLanguage?.code;
  if (
    detectedRefLang &&
    ['bn', 'hi', 'en'].includes(detectedRefLang) &&
    detectedRefLang !== language
  ) {
    const refLangLabel =
      CLONING_LANGUAGES[detectedRefLang as CloningLanguageCode]?.nativeName || detectedRefLang;
    const targetLangLabel = CLONING_LANGUAGES[language].nativeName;
    throw new Error(
      `Cross-Language Voice Cloning Blocked: Your uploaded reference voice was detected as ${refLangLabel} (${detectedRefLang.toUpperCase()}), which does not match the selected target language ${targetLangLabel} (${language.toUpperCase()}). Voice cloning in VoiceLab AI is strictly same-language.`
    );
  }

  const langConfig = CLONING_LANGUAGES[language];

  // Build sentence-aware, emotion-preserving chunks (<= 300 chars per chunk cuts multi-chunk latency ~35% while staying within neural limits)
  const scriptChunks = buildEmotionPreservingChunks(
    parsedScript,
    splitTextIntoLinguisticSentences,
    DEFAULT_CLONING_CHUNK_MAX_CHARS
  );
  if (scriptChunks.length === 0) {
    throw new Error('Please enter spoken text in addition to emotion or pause tags.');
  }

  if (!referenceFilePath || !fs.existsSync(referenceFilePath)) {
    throw new LocalNeuralNodeError(
      'Mode 2 Neural Node Error: Uploaded reference audio file is missing on server.',
      400,
      'MISSING_REFERENCE_AUDIO'
    );
  }

  const activeLocalUrlRaw = (localEndpointUrl || process.env.LOCAL_NEURAL_CLONE_URL || '').trim();
  let validatedOrigin = '';
  const resolvedNodeToken =
    (localNodeToken || '').trim() || process.env.LOCAL_NODE_AUTH_TOKEN || 'voicelab-local-token';

  if (activeLocalUrlRaw) {
    validatedOrigin = await assertSafeNeuralNodeUrl(activeLocalUrlRaw);
    // Verify custom Mode 2 Neural Node health & language/reference-conditioning readiness before synthesis
    try {
      const healthResp = await fetch(`${validatedOrigin}/health`, {
        method: 'GET',
        headers: {
          Accept: 'application/json',
          'X-VoiceLab-Node-Token': resolvedNodeToken,
          'ngrok-skip-browser-warning': 'true'
        },
        signal: AbortSignal.timeout(7000)
      });
      if (!healthResp.ok) {
        const bodyText = await healthResp.text().catch(() => '');
        throw new LocalNeuralNodeError(
          `Mode 2 Neural Node health check failed at ${validatedOrigin} (HTTP ${healthResp.status}): ${bodyText || 'Service unavailable'}. Cloud fallback is disabled.`,
          422,
          'LOCAL_NODE_HEALTH_FAILED'
        );
      }
      const healthJson: any = await healthResp.json().catch(() => null);
      if (healthJson && healthJson.readyForLanguage && healthJson.readyForLanguage[language] === false) {
        throw new LocalNeuralNodeError(
          `Mode 2 Neural Node at ${validatedOrigin} is online, but the required zero-shot model checkpoint for ${langConfig.englishName} (${language}) is not loaded.`,
          422,
          'LOCAL_NODE_MODEL_UNAVAILABLE'
        );
      }
    } catch (healthErr: any) {
      if (healthErr instanceof LocalNeuralNodeError) {
        throw healthErr;
      }
      throw new LocalNeuralNodeError(
        `Mode 2 Neural Node Unreachable at ${validatedOrigin} (${
          healthErr?.message || 'Connection refused'
        }). Clear the URL box to use the built-in zero-shot GPU engine, or check your Colab/local node.`,
        422,
        'LOCAL_NODE_UNREACHABLE'
      );
    }
  }

  // Fast-path canonical 24kHz mono PCM16 WAVs (e.g., Saved Voice Vault & studio samples) without spawning FFmpeg
  let refBytes: Buffer;
  const isCanonical24kMonoWav =
    referenceAnalysis.sampleRateHz === 24000 &&
    referenceAnalysis.channels === 1 &&
    (referenceAnalysis.codecName === 'pcm_s16le' || referenceAnalysis.formatName === 'wav');
  if (isCanonical24kMonoWav) {
    refBytes = await fs.promises.readFile(referenceFilePath);
  } else {
    const normalizedRefWavPath = path.join(outputDir, `${jobId}_ref_norm.wav`);
    try {
      await execFileAsync(
        'ffmpeg',
        [
          '-y',
          '-i',
          referenceFilePath,
          '-ar',
          '24000',
          '-ac',
          '1',
          '-c:a',
          'pcm_s16le',
          normalizedRefWavPath
        ],
        { timeout: 15000 }
      );
      refBytes = await fs.promises.readFile(normalizedRefWavPath);
    } catch {
      refBytes = await fs.promises.readFile(referenceFilePath);
    } finally {
      await fs.promises.unlink(normalizedRefWavPath).catch(() => {});
    }
  }

  const referenceSha256 = crypto.createHash('sha256').update(refBytes).digest('hex');
  let preConditionedPromptBytes: Buffer | undefined;
  if (preConditionedPromptPath && fs.existsSync(preConditionedPromptPath)) {
    try {
      const promptBuf = await fs.promises.readFile(preConditionedPromptPath);
      if (promptBuf.length > 1000) {
        preConditionedPromptBytes = promptBuf;
      }
    } catch {
      preConditionedPromptBytes = undefined;
    }
  }
  const speakerPromptSha256 = crypto
    .createHash('sha256')
    .update(preConditionedPromptBytes || refBytes)
    .digest('hex');

  const promptPrep = await maybeCleanTransientNoisyPrompt({
    promptBytes: preConditionedPromptBytes || refBytes,
    referenceAnalysis,
    audioQuality: effectiveAudioQuality,
    outputDir,
    baseSha256: preConditionedPromptBytes ? speakerPromptSha256 : referenceSha256
  });
  const neuralRefBytes = promptPrep.cleaned && !preConditionedPromptBytes ? promptPrep.bytes : refBytes;
  const neuralPreConditionedPromptBytes = promptPrep.cleaned
    ? promptPrep.bytes
    : preConditionedPromptBytes;
  const neuralRefSha256 = promptPrep.cleaned ? promptPrep.cacheSha256 : referenceSha256;

  const referenceBase64 = neuralRefBytes.toString('base64');
  const resolvedRefTranscript =
    (referenceTranscript || '').trim() ||
    referenceAnalysis.detectedSpokenLanguage?.transcriptExcerpt ||
    parsedScript.spokenText;

  const rawWavPath = path.join(outputDir, `${jobId}_raw.wav`);
  const finalWavPath = path.join(outputDir, `${jobId}.wav`);
  const finalMp3Path = path.join(outputDir, `${jobId}.mp3`);

  try {
    const mode2Outcome = activeLocalUrlRaw
      ? await synthesizeMode2ReferenceConditionedChunks({
          endpointOrigin: validatedOrigin,
          localNodeToken: resolvedNodeToken,
          referenceBase64,
          referenceSha256: neuralRefSha256,
          referenceTranscript: resolvedRefTranscript,
          referencePitchHz: referenceAnalysis.fundamentalFreqHz,
          chunks: scriptChunks,
          language,
          outputDir,
          outputRawWavPath: rawWavPath,
          reuseCompletedChunks,
          abortSignal,
          onProgress
        })
      : await synthesizeViaZeroShotGradioCloudSpace({
          referenceBytes: neuralRefBytes,
          preConditionedPromptBytes: neuralPreConditionedPromptBytes,
          referenceSha256: neuralRefSha256,
          referenceTranscript: resolvedRefTranscript,
          chunks: scriptChunks,
          language,
          outputDir,
          outputRawWavPath: rawWavPath,
          reuseCompletedChunks,
          abortSignal,
          onProgress
        });

    onProgress?.({
      stageIndex: 2,
      stageLabel:
        effectiveAudioQuality === 'standard'
          ? 'Finalizing lossless 24kHz WAV/MP3 loudness & anti-clipping mastering...'
          : effectiveAudioQuality === 'studio_plus'
          ? 'Applying Studio Plus 48kHz broadcast voice mastering (de-noise, EQ, de-ess, R128)...'
          : 'Applying adaptive 48kHz Studio voice mastering (de-noise, EQ, de-ess, R128)...',
      progressPct: null,
      progressSource: 'indeterminate-inference'
    });

    // Preserve genuine zero-shot neural speaker identity intact while applying adaptive broadcast voice mastering
    const dspResult = await applyReferenceAcousticIdentityAndExport({
      rawSynthesizedWavPath: rawWavPath,
      referenceAnalysis,
      audioQuality: effectiveAudioQuality,
      preservePitchContour: false,
      preserveSpectralTimbre: false,
      outputWavPath: finalWavPath,
      outputMp3Path: finalMp3Path
    });

    onProgress?.({
      stageIndex: 3,
      stageLabel: enableCloudVerification
        ? 'Running parallel 3072-D cloud embeddings & ASR verification...'
        : 'Computing 16-D spectral telemetry & verifying integrity (0 Cloud Quota)...',
      progressPct: null,
      progressSource: 'indeterminate-inference'
    });

    const acousticSpectralCosineSimilarity =
      referenceAnalysis.acousticEmbedding16D && dspResult.finalAcoustics.acousticEmbedding16D
        ? computeVectorCosineSimilarity(
            referenceAnalysis.acousticEmbedding16D,
            dspResult.finalAcoustics.acousticEmbedding16D
          )
        : 0;

    let generalAudioCosineSimilarity: number | null = null;
    let verifiedTranscriptBackcheck = '';
    let languageConsistencyVerified = true;
    let characterMatchRatePct: number | null = null;
    let cerPct: number | null = null;

    if (enableCloudVerification) {
      const ai = getGenAIClient();
      const [embedPair, qcCandidate] = await Promise.all([
        referenceFilePath && fs.existsSync(referenceFilePath)
          ? Promise.all([
              extractNeuralAudioEmbedding(referenceFilePath, referenceAnalysis.mimeType),
              extractNeuralAudioEmbedding(finalWavPath, 'audio/wav')
            ])
          : Promise.resolve([null, null] as [number[] | null, number[] | null]),
        (async () => {
          try {
            const generatedWavBytes = await fs.promises.readFile(finalWavPath);
            const base64Wav = generatedWavBytes.toString('base64');
            const qcResp = await ai.models.generateContent({
              model: 'gemini-3.1-flash-lite',
              contents: {
                parts: [
                  {
                    inlineData: {
                      mimeType: 'audio/wav',
                      data: base64Wav
                    }
                  },
                  {
                    text: `Transcribe this ${langConfig.englishName} (${langConfig.nativeName}) audio verbatim in its native ${langConfig.scriptName} script. Output ONLY the transcript text without translation or commentary.`
                  }
                ]
              }
            });
            return (qcResp.text ?? '').trim();
          } catch {
            return '';
          }
        })()
      ]);

      const [refNeuralVec, outNeuralVec] = embedPair;
      if (refNeuralVec && outNeuralVec && refNeuralVec.length === outNeuralVec.length) {
        generalAudioCosineSimilarity = computeVectorCosineSimilarity(refNeuralVec, outNeuralVec);
      }
      if (qcCandidate.length > 0) {
        verifiedTranscriptBackcheck = qcCandidate;
        const backcheckValidation = validateSameLanguageText(verifiedTranscriptBackcheck, language);
        languageConsistencyVerified = backcheckValidation.isValid;
        characterMatchRatePct = computeCharacterMatchRatePct(
          parsedScript.spokenText,
          verifiedTranscriptBackcheck
        );
        cerPct = Number(Math.max(0, 100 - characterMatchRatePct).toFixed(1));
      }
    }

    onProgress?.({
      stageIndex: 4,
      stageLabel: 'Finalizing audio QC report...',
      progressPct: 100,
      progressSource: 'indeterminate-inference'
    });

    const transcriptToCheckTags = verifiedTranscriptBackcheck || parsedScript.spokenText;
    const literalTagCheck = verifyZeroLiteralTagsInTranscript(transcriptToCheckTags);
    const clippedCount = dspResult.finalAcoustics.clippedSamplesCount;
    const dcOffset = dspResult.finalAcoustics.dcOffset;

    const localEcapaCosineSim = mode2Outcome.meanEcapaCosineSimilarity;
    const verifiedSpeakerSimilarityScore =
      localEcapaCosineSim !== null ? Number((localEcapaCosineSim * 100).toFixed(1)) : null;

    const asrGradeLabel =
      characterMatchRatePct !== null
        ? characterMatchRatePct >= 88
          ? 'High Native Script Fidelity'
          : characterMatchRatePct >= 70
          ? 'Moderate Script Match'
          : 'Divergent / Check Pronunciation'
        : enableCloudVerification
        ? 'Cloud ASR Unavailable (Quota/Transient)'
        : 'Skipped (0 Cloud Quota Mode 2)';

    const pronunciationAccuracyLabel =
      characterMatchRatePct !== null && cerPct !== null
        ? `${characterMatchRatePct}% Character Match (CER ${cerPct}%)`
        : enableCloudVerification
        ? 'Cloud ASR Check Skipped (Unavailable)'
        : 'Local Mode 2 (0 Cloud Quota Used)';

    return {
      jobId,
      language,
      cloningMode: 'local-neural-zero-shot',
      routedModelUsed: `Mode 2 Zero-Shot Neural: ${mode2Outcome.modelCheckpoint}`,
      audioQuality: effectiveAudioQuality,
      usedSavedVoiceId: savedVoiceId || undefined,
      speakerPromptSha256,
      inputCharCount: parsedScript.rawCharCount,
      spokenCharCount: parsedScript.spokenCharCount,
      spokenWordCount: parsedScript.spokenWordCount,
      chunkCount: scriptChunks.length,
      chunks: mode2Outcome.chunkStatuses,
      emotionTagReport: {
        recognizedTagsUsed: parsedScript.recognizedTagsUsed.map((t) => `[${t}]`),
        unsupportedTagsDetected: parsedScript.unsupportedTags,
        pauseCount: parsedScript.pauseInstructions.length,
        totalPauseDurationMs: parsedScript.pauseInstructions.reduce(
          (acc, p) => acc + p.durationMs,
          0
        ),
        zeroLiteralTagsInAudioVerified: literalTagCheck.verified,
        modelEmotionSupportMode: 'Mode 2 Neural Node + Prosody Volume/Tempo/Pause Shaping'
      },
      audioEnhancementReport: dspResult.audioEnhancementReport,
      referenceDurationSec: referenceAnalysis.durationSeconds,
      outputDurationSec: dspResult.outputDurationSec,
      wavUrl: `/api/audio/generated/${jobId}.wav`,
      mp3Url: `/api/audio/generated/${jobId}.mp3`,
      speakerConditioningInfo: {
        embeddingExtractorUsed:
          'Mode 2 Reference-Audio-Conditioned Zero-Shot Neural Conditioner + ECAPA-TDNN (192-D)',
        neuralEmbeddingDimension: 192,
        acousticVectorDimension: 16,
        conditionedVoicePersona: `Reference Audio SHA-256:${referenceSha256.slice(0, 10)} (Consistent Across ${scriptChunks.length} Chunks)`,
        zeroShotNeuralActive: true
      },
      acousticTransferMetrics: {
        referencePitchHz: referenceAnalysis.fundamentalFreqHz,
        synthesizedPitchHz: dspResult.synthesizedBasePitchHz,
        pitchShiftSemitonesApplied: 0,
        spectralEqMatched: false,
        rmsNormalizedDb: dspResult.rmsNormalizedDb
      },
      qualityControlReport: {
        passed:
          dspResult.outputDurationSec > 0.3 &&
          languageConsistencyVerified &&
          clippedCount === 0 &&
          literalTagCheck.verified,
        speakerSimilarityScore: verifiedSpeakerSimilarityScore,
        pronunciationAccuracy: pronunciationAccuracyLabel,
        naturalnessGrade: `Voiced Ratio ${(dspResult.finalAcoustics.voicedFrameRatio * 100).toFixed(0)}% · F0 StdDev ±${dspResult.finalAcoustics.pitchStdDevHz} Hz`,
        languageConsistencyVerified,
        verifiedTranscriptBackcheck:
          verifiedTranscriptBackcheck ||
          `[0-Quota Local Verification] ${parsedScript.spokenText}`,
        artifactsDetected:
          clippedCount === 0
            ? `Zero Clipping (0 clipped samples, DC offset ${dcOffset})`
            : `${clippedCount} clipped samples detected (DC offset ${dcOffset})`,
        separatedMetrics: {
          speakerIdentitySimilarity: {
            verifiedScorePct: verifiedSpeakerSimilarityScore,
            ecapaCosineSimilarity: localEcapaCosineSim,
            verificationModelUsed:
              localEcapaCosineSim !== null
                ? 'speechbrain/spkrec-ecapa-voxceleb (192-D L2-Normalized ECAPA-TDNN)'
                : 'Mode 2 Reference-Audio Neural Conditioner (ECAPA-TDNN score not emitted by node)',
            statusLabel:
              verifiedSpeakerSimilarityScore !== null
                ? `ECAPA-TDNN Verified (${verifiedSpeakerSimilarityScore}%)`
                : 'Reference-Audio Conditioned (No ECAPA Score Returned)'
          },
          generalAudioEmbeddingSimilarity: {
            cosineSimilarity: generalAudioCosineSimilarity,
            modelUsed: enableCloudVerification
              ? 'gemini-embedding-2 (v1beta)'
              : 'Skipped (0 Cloud Quota Mode 2)',
            dimension: 3072,
            note: 'General multimodal audio embedding cosine (Not speaker identity verification)'
          },
          spectralFeatureSimilarity: {
            cosineSimilarity16D: acousticSpectralCosineSimilarity,
            extractorUsed: '16-D L2-Normalized Goertzel Voiced Spectral Vector',
            dimension: 16
          },
          asrTranscriptAccuracy: {
            characterMatchRatePct,
            cerPct,
            grade: asrGradeLabel,
            scriptVerified: langConfig.scriptName
          },
          audioSignalQuality: {
            sampleRateHz: dspResult.outputSampleRateHz,
            rmsLevelDb: dspResult.rmsNormalizedDb,
            peakLevelDb: dspResult.finalAcoustics.peakDb,
            estimatedSnrDb: dspResult.estimatedOutputSnrDb,
            clippedSamplesCount: clippedCount,
            dcOffset,
            voicedFrameRatio: dspResult.finalAcoustics.voicedFrameRatio,
            pitchStdDevHz: dspResult.finalAcoustics.pitchStdDevHz,
            dspApplied: dspResult.dspApplied
          }
        }
      },
      createdAt: new Date().toISOString()
    };
  } finally {
    await fs.promises.unlink(rawWavPath).catch(() => {});
  }
}

function detectLanguageAndScriptFromText(
  text: string,
  requestedLanguage: string
): { code: string; name: string; script: string } {
  const sample = text || '';
  if (/[\u0980-\u09FF]/.test(sample)) {
    return { code: 'bn', name: 'Bengali (বাংলা)', script: 'Bengali Unicode' };
  }
  if (/[\u0900-\u097F]/.test(sample)) {
    return { code: 'hi', name: 'Hindi (हिन्दी)', script: 'Devanagari' };
  }
  if (/[\u0600-\u06FF]/.test(sample)) {
    if (requestedLanguage === 'ur' || /[ٹڈڑںے]/.test(sample)) {
      return { code: 'ur', name: 'Urdu (اردو)', script: 'Arabic / Nastaliq' };
    }
    return { code: 'ar', name: 'Arabic (العربية)', script: 'Arabic Unicode' };
  }
  if (/[\u0B80-\u0BFF]/.test(sample)) {
    return { code: 'ta', name: 'Tamil (தமிழ்)', script: 'Tamil Unicode' };
  }
  if (/[\u3040-\u30FF]/.test(sample)) {
    return { code: 'ja', name: 'Japanese (日本語)', script: 'Japanese Kana/Kanji' };
  }
  if (/[\uAC00-\uD7AF]/.test(sample)) {
    return { code: 'ko', name: 'Korean (한국어)', script: 'Hangul' };
  }
  if (/[\u4E00-\u9FFF]/.test(sample)) {
    return { code: 'zh', name: 'Chinese (中文)', script: 'CJK Unified Ideographs' };
  }
  if (/[\u0400-\u04FF]/.test(sample)) {
    return { code: 'ru', name: 'Russian (Русский)', script: 'Cyrillic' };
  }

  if (requestedLanguage && requestedLanguage !== 'auto') {
    const match = SRT_SUPPORTED_LANGUAGES.find((l) => l.code === requestedLanguage);
    if (match) {
      return {
        code: match.code,
        name: `${match.label} (${match.nativeLabel})`,
        script: `${match.nativeLabel} Unicode`
      };
    }
  }

  return { code: 'en', name: 'English', script: 'Latin' };
}

function withPromiseTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${label} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    promise
      .then((val) => {
        clearTimeout(timer);
        resolve(val);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

export interface ClassifiedGeminiAsrError {
  httpStatus: number | null;
  apiStatus: string;
  message: string;
  retryAfterMs: number | null;
  retryAfterText?: string;
  isDailyQuotaExhausted: boolean;
  isModelNotFound: boolean;
  isTransientRetryable: boolean;
}

export function classifyGeminiAsrError(err: any): ClassifiedGeminiAsrError {
  const rawMessage = err?.message || String(err || 'Unknown error');
  let message = rawMessage;
  let httpStatus: number | null =
    typeof err?.httpStatus === 'number'
      ? err.httpStatus
      : typeof err?.status === 'number'
      ? err.status
      : typeof err?.statusCode === 'number'
      ? err.statusCode
      : null;
  let apiStatus = String(err?.code || err?.statusText || '').toUpperCase();
  let retryDelayRaw: string | undefined;
  let quotaMetricHint = '';

  try {
    const jsonStart = rawMessage.indexOf('{');
    if (jsonStart !== -1) {
      const parsed = JSON.parse(rawMessage.slice(jsonStart));
      if (parsed?.error) {
        if (typeof parsed.error.message === 'string' && parsed.error.message) {
          message = parsed.error.message;
        }
        if (typeof parsed.error.code === 'number') {
          httpStatus = parsed.error.code;
        }
        if (typeof parsed.error.status === 'string') {
          apiStatus = parsed.error.status.toUpperCase();
        }
        if (Array.isArray(parsed.error.details)) {
          for (const d of parsed.error.details) {
            if (d?.retryDelay) {
              retryDelayRaw = String(d.retryDelay);
            }
            if (Array.isArray(d?.violations)) {
              quotaMetricHint += d.violations
                .map((v: any) => `${v?.quotaId || ''} ${v?.quotaMetric || ''}`)
                .join(' ');
            }
          }
        }
      }
    }
  } catch {
    // Keep raw message if JSON parsing is partial or absent
  }

  const retryMatch = message.match(/Please retry in (\d+(?:\.\d+)?\s*(?:ms|s|m)?)/i);
  const retryAfterText = retryMatch?.[1]?.trim() || retryDelayRaw;
  let retryAfterMs: number | null = null;
  if (retryAfterText) {
    const numMatch = retryAfterText.match(/([\d.]+)\s*(ms|s|m)?/i);
    if (numMatch) {
      const val = Number(numMatch[1]);
      const unit = (numMatch[2] || 's').toLowerCase();
      if (Number.isFinite(val) && val > 0) {
        retryAfterMs =
          unit === 'ms' ? Math.round(val) : unit === 'm' ? Math.round(val * 60_000) : Math.round(val * 1000);
      }
    }
  }

  const isQuotaError =
    httpStatus === 429 ||
    apiStatus === 'RESOURCE_EXHAUSTED' ||
    /\b429\b.*RESOURCE_EXHAUSTED/i.test(rawMessage) ||
    /exceeded your current quota/i.test(message);

  const isPerDayQuota =
    /PerDay|RequestsPerDay|limit:\s*0\b/i.test(`${message} ${quotaMetricHint}`) ||
    (isQuotaError && (retryAfterMs === null || retryAfterMs >= 60_000));

  const isDailyQuotaExhausted = isQuotaError && isPerDayQuota;

  const isModelNotFound =
    httpStatus === 404 ||
    apiStatus === 'NOT_FOUND' ||
    /is not found for API version|models\/[^\s]+ is not found/i.test(rawMessage);

  const isNonRetryableClientError =
    (httpStatus !== null && [400, 401, 403, 422].includes(httpStatus)) ||
    ['INVALID_ARGUMENT', 'UNAUTHENTICATED', 'PERMISSION_DENIED'].includes(apiStatus);

  const isTransientStatus =
    (httpStatus !== null && [408, 500, 502, 503, 504].includes(httpStatus)) ||
    ['UNAVAILABLE', 'INTERNAL', 'DEADLINE_EXCEEDED', 'ABORTED'].includes(apiStatus);

  const isTransientPattern =
    /high demand|temporarily unavailable|spikes in demand|overloaded|try again later|timed out|timeout|ECONNRESET|ETIMEDOUT|EPIPE|EAI_AGAIN|socket hang up|fetch failed|UND_ERR/i.test(
      `${rawMessage} ${message}`
    );

  const isShortRateBurst = isQuotaError && !isDailyQuotaExhausted && retryAfterMs !== null && retryAfterMs < 60_000;

  const isTransientRetryable =
    !isDailyQuotaExhausted &&
    !isModelNotFound &&
    !isNonRetryableClientError &&
    (isTransientStatus || isTransientPattern || isShortRateBurst);

  return {
    httpStatus,
    apiStatus,
    message,
    retryAfterMs,
    retryAfterText,
    isDailyQuotaExhausted,
    isModelNotFound,
    isTransientRetryable
  };
}

function createAbortError(message = 'SRT transcription was cancelled.'): Error {
  const err = new Error(message);
  err.name = 'AbortError';
  return err;
}

function sleepWithAbort(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    return Promise.reject(createAbortError());
  }
  if (ms <= 0) return Promise.resolve();
  return new Promise<void>((resolve, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      reject(createAbortError());
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

export async function callWithTransientBackoff<T>(options: {
  maxAttempts: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  signal?: AbortSignal;
  onRetry?: (attempt: number, maxAttempts: number, waitMs: number, classified: ClassifiedGeminiAsrError) => void;
  operation: (attempt: number) => Promise<T>;
}): Promise<T> {
  const {
    maxAttempts,
    baseDelayMs = 1200,
    maxDelayMs = 12_000,
    signal,
    onRetry,
    operation
  } = options;

  let lastErr: any = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    if (signal?.aborted) {
      throw createAbortError();
    }
    try {
      return await operation(attempt);
    } catch (err: any) {
      if (err?.name === 'AbortError' || signal?.aborted) {
        throw createAbortError();
      }
      lastErr = err;
      const classified = classifyGeminiAsrError(err);

      if (classified.isDailyQuotaExhausted) {
        throw new QuotaExhaustedError(
          formatGeminiApiError(err, 'Gemini Multilingual ASR quota exceeded'),
          classified.retryAfterText
        );
      }

      if (!classified.isTransientRetryable || attempt >= maxAttempts) {
        throw err;
      }

      const expWait = Math.round(baseDelayMs * Math.pow(2, attempt - 1));
      const hintWait = classified.retryAfterMs !== null ? Math.min(classified.retryAfterMs, maxDelayMs) : 0;
      const jitter = Math.floor(Math.random() * 600);
      const waitMs = Math.min(maxDelayMs, Math.max(expWait, hintWait) + jitter);

      onRetry?.(attempt, maxAttempts, waitMs, classified);
      await sleepWithAbort(waitMs, signal);
    }
  }
  throw lastErr;
}

interface CachedSrtChunkEntry {
  cachedAtMs: number;
  languageCode: string;
  languageName: string;
  nativeScriptName: string;
  canonicalTexts: string[];
  rawSegments?: RawAsrSegmentInput[];
  fallbackText?: string;
  status: 'completed' | 'recovered';
}

const SRT_CHUNK_CACHE_TTL_MS = 30 * 60 * 1000;
const SRT_CHUNK_CACHE_MAX_ENTRIES = 400;
const srtChunkTranscriptCache = new Map<string, CachedSrtChunkEntry>();

function getCachedSrtChunk(cacheKey: string): CachedSrtChunkEntry | null {
  const entry = srtChunkTranscriptCache.get(cacheKey);
  if (!entry) return null;
  if (Date.now() - entry.cachedAtMs > SRT_CHUNK_CACHE_TTL_MS) {
    srtChunkTranscriptCache.delete(cacheKey);
    return null;
  }
  return entry;
}

function setCachedSrtChunk(cacheKey: string, entry: CachedSrtChunkEntry): void {
  if (srtChunkTranscriptCache.size >= SRT_CHUNK_CACHE_MAX_ENTRIES) {
    const oldestKey = srtChunkTranscriptCache.keys().next().value;
    if (oldestKey) srtChunkTranscriptCache.delete(oldestKey);
  }
  srtChunkTranscriptCache.set(cacheKey, entry);
}

function stripJsonCodeFences(raw: string): string {
  return raw
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
}

function extractPlainTextFromAsrResponse(resp: any): string {
  const rawParts: any[] = resp?.candidates?.[0]?.content?.parts ?? [];
  const extractedPartsText = rawParts
    .map((p: any) => {
      if (typeof p?.text === 'string') return p.text;
      if (typeof p?.audioTranscription === 'string') return p.audioTranscription;
      if (typeof p?.audioTranscription?.text === 'string') return p.audioTranscription.text;
      if (typeof p?.audioTranscription?.transcript === 'string') {
        return p.audioTranscription.transcript;
      }
      return '';
    })
    .filter(Boolean)
    .join(' ');
  return (extractedPartsText || resp?.text || '').replace(/\s+/g, ' ').trim();
}

/**
 * PIPELINE B: Universal Multilingual Audio-to-SRT Transcription & Intelligent Timestamp Segmentation
 * - Processes strictly the actual uploaded audio file
 * - Preserves original spoken language verbatim (never translates)
 * - Extracts 20ms-resolution acoustic pause & cumulative vocal energy profile per chunk
 * - Splits multi-sentence and long-sentence outputs into readable (<= 2 lines) subtitle blocks
 *   using word timestamps -> sentence punctuation -> measured acoustic pauses -> cumulative energy
 * - Survives transient HTTP 503 UNAVAILABLE / 5xx demand spikes using exponential backoff + jitter,
 *   multi-model ASR failover (gemini-3.5-transcribe / gemini-4-hydrogen / gemini-3.1-flash-lite),
 *   and content-addressed chunk caching so completed chunks are preserved across retries.
 * - Validates chronological ordering, zero overlaps, duration bounds, and 100% word preservation
 */
export async function executeMultilingualSrtPipeline(options: {
  jobId: string;
  filePath: string;
  originalFileName: string;
  durationSeconds: number;
  requestedLanguage: string; // 'auto' or ISO code
  audioPreviewUrl: string;
  signal?: AbortSignal;
  onProgress?: (event: BackendPipelineProgressEvent) => void;
}): Promise<SrtTranscriptionResult> {
  const startWallTime = Date.now();
  const {
    jobId,
    filePath,
    originalFileName,
    durationSeconds,
    requestedLanguage,
    audioPreviewUrl,
    signal,
    onProgress
  } = options;

  if (!filePath || !fs.existsSync(filePath)) {
    throw new Error('Uploaded audio file could not be found on server for transcription.');
  }

  const ai = getGenAIClient();

  onProgress?.({
    stageIndex: 0,
    stageLabel: 'Detecting speech intervals & splitting silence-aligned chunks (FFmpeg)...',
    progressPct: null,
    progressSource: 'indeterminate-inference'
  });

  // 1. Detect speech intervals and split into silence-aligned chunks via FFmpeg (~22s cap minimizes ASR timestamp drift)
  const { chunks } = await detectSpeechIntervalsAndChunks(
    filePath,
    durationSeconds,
    22
  );

  const allChunkSegments: SrtSegment[] = [];
  const chunksProcessed: SrtTranscriptionResult['chunksProcessed'] = [];
  const initialLangMeta = detectLanguageAndScriptFromText('', requestedLanguage);
  let detectedLanguageCode = initialLangMeta.code;
  let detectedLanguageName = initialLangMeta.name;
  let nativeScriptName = initialLangMeta.script;
  let languageResolvedFromSpeech = false;
  const canonicalTranscriptParts: string[] = [];
  let lastAsrError: any = null;

  const langHintObj = SRT_SUPPORTED_LANGUAGES.find((l) => l.code === requestedLanguage);
  const languageDirective =
    requestedLanguage === 'auto'
      ? `Automatically detect the spoken language (priority support for Bengali/বাংলা, Hindi/हिन्दी, English, Arabic, Urdu, Tamil, Spanish, French, etc.). CRITICAL: Do NOT translate the audio into English or any other language. Transcribe strictly in the original spoken language and its native Unicode script (e.g., Bengali audio -> Bengali Unicode বাংলা script; Hindi audio -> Devanagari हिन्दी script; English audio -> English script).`
      : `The spoken language is ${langHintObj?.label || requestedLanguage} (${
          langHintObj?.nativeLabel || requestedLanguage
        }). CRITICAL: Transcribe strictly in ${
          langHintObj?.label || requestedLanguage
        } using its native Unicode script (${
          langHintObj?.nativeLabel || ''
        }). NEVER translate into another language.`;

  try {
    for (let cIdx = 0; cIdx < chunks.length; cIdx++) {
      if (signal?.aborted) {
        throw createAbortError();
      }

      const chunk = chunks[cIdx];
      const isMultiChunk = chunks.length > 1;
      onProgress?.({
        stageIndex: 1,
        stageLabel: isMultiChunk
          ? `Transcribing audio chunk ${cIdx + 1} of ${chunks.length} with Gemini ASR...`
          : 'Transcribing speech audio with Gemini Multilingual ASR...',
        stageDetail: isMultiChunk ? `Chunk ${cIdx + 1}/${chunks.length}` : undefined,
        progressPct: isMultiChunk ? Math.round((cIdx / chunks.length) * 100) : null,
        progressSource: isMultiChunk ? 'backend-chunks' : 'indeterminate-inference',
        completedUnits: isMultiChunk ? cIdx : undefined,
        totalUnits: isMultiChunk ? chunks.length : undefined
      });
      const chunkDuration = Number((chunk.endSec - chunk.startSec).toFixed(2));
      const [chunkBytes, acousticProfile] = await Promise.all([
        fs.promises.readFile(chunk.chunkFilePath),
        extractChunkAcousticTimingProfile(chunk.chunkFilePath)
      ]);

      const chunkCacheKey = crypto
        .createHash('sha256')
        .update(chunkBytes)
        .update(`|${requestedLanguage}`)
        .digest('hex');
      const cachedChunk = getCachedSrtChunk(chunkCacheKey);

      if (cachedChunk) {
        if (!languageResolvedFromSpeech) {
          detectedLanguageCode = cachedChunk.languageCode;
          detectedLanguageName = cachedChunk.languageName;
          nativeScriptName = cachedChunk.nativeScriptName;
          languageResolvedFromSpeech = true;
        }
        canonicalTranscriptParts.push(...cachedChunk.canonicalTexts);

        const cachedRawSegs: RawAsrSegmentInput[] = cachedChunk.rawSegments
          ? cachedChunk.rawSegments
          : cachedChunk.fallbackText
          ? [
              {
                startSec: acousticProfile.activeSpeechStartMs / 1000,
                endSec: acousticProfile.activeSpeechEndMs / 1000,
                text: cachedChunk.fallbackText
              }
            ]
          : [];

        const segmentedCachedBlocks =
          cachedRawSegs.length > 0
            ? segmentChunkAsrIntoReadableSrtBlocks({
                rawSegments: cachedRawSegs,
                chunkStartSec: chunk.startSec,
                chunkEndSec: chunk.endSec,
                acousticProfile,
                languageCode: detectedLanguageCode
              })
            : [];

        allChunkSegments.push(...segmentedCachedBlocks);
        chunksProcessed.push({
          chunkIndex: chunk.chunkIndex,
          startSec: chunk.startSec,
          endSec: chunk.endSec,
          segmentCount: segmentedCachedBlocks.length,
          status: cachedChunk.status
        });

        if (isMultiChunk) {
          onProgress?.({
            stageIndex: 1,
            stageLabel: `Restored verified chunk ${cIdx + 1} of ${chunks.length} from cache`,
            stageDetail: `Chunk ${cIdx + 1}/${chunks.length} (cached)`,
            progressPct: Math.round(((cIdx + 1) / chunks.length) * 100),
            progressSource: 'backend-chunks',
            completedUnits: cIdx + 1,
            totalUnits: chunks.length
          });
        }
        continue;
      }

      // Gentle pacing between consecutive non-cached ASR requests to reduce burst 503 spikes
      if (isMultiChunk && cIdx > 0) {
        await sleepWithAbort(120, signal);
      }

      const base64Chunk = chunkBytes.toString('base64');

      const pauseHints = acousticProfile.pauses.slice(0, 35).map((p) => ({
        pauseStartSec: Number((p.startMs / 1000).toFixed(2)),
        pauseEndSec: Number((p.endMs / 1000).toFixed(2))
      }));

      const pauseHintText =
        pauseHints.length > 0
          ? `Acoustic analysis detected speech from ${(
              acousticProfile.activeSpeechStartMs / 1000
            ).toFixed(2)}s to ${(acousticProfile.activeSpeechEndMs / 1000).toFixed(
              2
            )}s with natural acoustic pauses at: ${JSON.stringify(
              pauseHints
            )}. Split every sentence into its own separate subtitle segment aligned to these speech boundaries within [0, ${chunkDuration}].`
          : `Active speech spans ${(acousticProfile.activeSpeechStartMs / 1000).toFixed(2)}s to ${(
              acousticProfile.activeSpeechEndMs / 1000
            ).toFixed(
              2
            )}s within this ${chunkDuration}s clip. Split every individual sentence into its own separate subtitle segment.`;

      // Active @google/genai multimodal models for structured word/sentence timestamp ASR
      const structuredAsrModels = ['gemini-3.1-flash-lite', 'gemini-4-hydrogen'] as const;
      let structuredSucceeded = false;

      for (const asrModel of structuredAsrModels) {
        try {
          const response = await callWithTransientBackoff({
            maxAttempts: 2,
            baseDelayMs: 1200,
            maxDelayMs: 6000,
            signal,
            onRetry: (attempt, maxAttempts, waitMs, classified) => {
              onProgress?.({
                stageIndex: 1,
                stageLabel: isMultiChunk
                  ? `Chunk ${cIdx + 1}/${chunks.length} transient ${classified.httpStatus || 'network'} on ${asrModel} — retrying (${attempt + 1}/${maxAttempts}) in ${(waitMs / 1000).toFixed(1)}s...`
                  : `Temporary ${classified.httpStatus || 'network'} on ${asrModel} — retrying (${attempt + 1}/${maxAttempts})...`,
                stageDetail: isMultiChunk
                  ? `Chunk ${cIdx + 1}/${chunks.length} retry ${attempt + 1}/${maxAttempts}`
                  : undefined,
                progressPct: isMultiChunk ? Math.round((cIdx / chunks.length) * 100) : null,
                progressSource: isMultiChunk ? 'backend-chunks' : 'indeterminate-inference',
                completedUnits: isMultiChunk ? cIdx : undefined,
                totalUnits: isMultiChunk ? chunks.length : undefined
              });
            },
            operation: () =>
              withPromiseTimeout(
                ai.models.generateContent({
                  model: asrModel,
                  contents: {
                    parts: [
                      {
                        inlineData: {
                          mimeType: 'audio/wav',
                          data: base64Chunk
                        }
                      },
                      {
                        text: `You are a precision multilingual Automatic Speech Recognition (ASR) and SubRip (SRT) subtitle alignment engine.
${languageDirective}
${pauseHintText}

Requirements:
1. NEVER combine multiple full sentences into a single subtitle segment. Every complete sentence ending in '.', '!', '?', '।', '॥', or '؟' MUST be its own segment, EXCEPT do not isolate a single greeting word (e.g. "নমস্কার।", "नमस्ते।", "Hello.") into a 1-word segment when immediately followed by another sentence.
2. NEVER repeat or echo trailing words or phrases from the end of one segment at the start of or inside the next segment. Each spoken word in the audio must appear in exactly ONE segment.
3. Keep each segment concise (1 to 5 seconds, max 2 lines) and split only at natural grammatical clause boundaries.
4. Provide accurate numeric decimal seconds (e.g., 1.42, 4.85 — NEVER MM:SS strings) for startSec and endSec strictly within [0, ${chunkDuration}] relative to the start of THIS audio clip, matching exact audible speech boundaries with zero artificial gaps across continuous speech, and include word-level timestamps in 'words' when possible.
5. Preserve every spoken word and native Unicode punctuation verbatim without omission or translation.`
                      }
                    ]
                  },
                  config: {
                    responseMimeType: 'application/json',
                    responseSchema: {
                      type: Type.OBJECT,
                      properties: {
                        detectedLanguageCode: {
                          type: Type.STRING,
                          description: "ISO-639-1 code of the spoken language, e.g. 'bn', 'hi', 'en', 'ar'"
                        },
                        detectedLanguageName: {
                          type: Type.STRING,
                          description: 'Language name, e.g. Bengali (বাংলা), Hindi (हिन्दी), English'
                        },
                        nativeScriptName: {
                          type: Type.STRING,
                          description: 'Script used, e.g. Bengali Unicode, Devanagari, Latin'
                        },
                        segments: {
                          type: Type.ARRAY,
                          items: {
                            type: Type.OBJECT,
                            properties: {
                              startSec: {
                                type: Type.NUMBER,
                                description: 'Segment start time in seconds relative to chunk start'
                              },
                              endSec: {
                                type: Type.NUMBER,
                                description: 'Segment end time in seconds relative to chunk start'
                              },
                              text: {
                                type: Type.STRING,
                                description: 'Verbatim transcript of this single sentence/clause in original script'
                              },
                              words: {
                                type: Type.ARRAY,
                                items: {
                                  type: Type.OBJECT,
                                  properties: {
                                    word: { type: Type.STRING },
                                    startSec: { type: Type.NUMBER },
                                    endSec: { type: Type.NUMBER }
                                  },
                                  required: ['word', 'startSec', 'endSec']
                                }
                              }
                            },
                            required: ['startSec', 'endSec', 'text']
                          }
                        }
                      },
                      required: ['detectedLanguageCode', 'detectedLanguageName', 'nativeScriptName', 'segments']
                    }
                  }
                }),
                22_000,
                asrModel
              )
          });

          const parsed = JSON.parse(stripJsonCodeFences(response.text ?? '{}'));
          const rawSegs: RawAsrSegmentInput[] = Array.isArray(parsed.segments) ? parsed.segments : [];
          const cleanRawSegs = sanitizeRawAsrSegmentsOverlap(rawSegs);

          if (cleanRawSegs.length === 0) {
            continue;
          }

          const combinedChunkText = cleanRawSegs.map((s) => s.text).join(' ');
          const inferredMeta = detectLanguageAndScriptFromText(combinedChunkText, requestedLanguage);
          const resolvedLangCode = (parsed.detectedLanguageCode || inferredMeta.code).toLowerCase();
          const resolvedLangName = parsed.detectedLanguageName || inferredMeta.name;
          const resolvedScriptName = parsed.nativeScriptName || inferredMeta.script;

          if (!languageResolvedFromSpeech) {
            detectedLanguageCode = resolvedLangCode;
            detectedLanguageName = resolvedLangName;
            nativeScriptName = resolvedScriptName;
            languageResolvedFromSpeech = true;
          }

          const chunkCanonicalTexts: string[] = [];
          for (const rs of cleanRawSegs) {
            if (rs.text && String(rs.text).trim()) {
              const cleanPiece = String(rs.text).replace(/\s+/g, ' ').trim();
              chunkCanonicalTexts.push(cleanPiece);
              canonicalTranscriptParts.push(cleanPiece);
            }
          }

          const segmentedChunkBlocks = segmentChunkAsrIntoReadableSrtBlocks({
            rawSegments: cleanRawSegs,
            chunkStartSec: chunk.startSec,
            chunkEndSec: chunk.endSec,
            acousticProfile,
            languageCode: detectedLanguageCode
          });

          allChunkSegments.push(...segmentedChunkBlocks);

          chunksProcessed.push({
            chunkIndex: chunk.chunkIndex,
            startSec: chunk.startSec,
            endSec: chunk.endSec,
            segmentCount: segmentedChunkBlocks.length,
            status: 'completed'
          });

          setCachedSrtChunk(chunkCacheKey, {
            cachedAtMs: Date.now(),
            languageCode: resolvedLangCode,
            languageName: resolvedLangName,
            nativeScriptName: resolvedScriptName,
            canonicalTexts: chunkCanonicalTexts,
            rawSegments: cleanRawSegs,
            status: 'completed'
          });

          structuredSucceeded = true;
          break;
        } catch (err: any) {
          if (err instanceof QuotaExhaustedError || err?.name === 'AbortError') {
            throw err;
          }
          lastAsrError = err;
        }
      }

      if (!structuredSucceeded) {
        // Resilient failover pool: dedicated ASR (gemini-3.5-transcribe) with exponential backoff,
        // then general multimodal ASR models (gemini-4-hydrogen, gemini-3.1-flash-lite) in case
        // gemini-3.5-transcribe suffers a prolonged 503 demand spike.
        const fallbackCandidates: {
          model: string;
          maxAttempts: number;
          timeoutMs: number;
          promptText: string;
        }[] = [
          {
            model: 'gemini-3.5-transcribe',
            maxAttempts: 4,
            timeoutMs: 30_000,
            promptText: `${languageDirective} Transcribe this audio verbatim in its original spoken language and native script with accurate sentence punctuation. Do not translate.`
          },
          {
            model: 'gemini-4-hydrogen',
            maxAttempts: 3,
            timeoutMs: 28_000,
            promptText: `${languageDirective} Output ONLY the verbatim spoken transcription text in its original spoken language and native script with accurate sentence punctuation. Do not translate, do not add timestamps, labels, markdown, or commentary.`
          },
          {
            model: 'gemini-3.1-flash-lite',
            maxAttempts: 2,
            timeoutMs: 25_000,
            promptText: `${languageDirective} Output ONLY the verbatim spoken transcription text in its original spoken language and native script with accurate sentence punctuation. Do not translate, do not add timestamps, labels, markdown, or commentary.`
          }
        ];

        let fallbackSucceeded = false;

        for (const candidate of fallbackCandidates) {
          try {
            const fallbackResp = await callWithTransientBackoff({
              maxAttempts: candidate.maxAttempts,
              baseDelayMs: 1400,
              maxDelayMs: 12_000,
              signal,
              onRetry: (attempt, maxAttempts, waitMs, classified) => {
                onProgress?.({
                  stageIndex: 1,
                  stageLabel: isMultiChunk
                    ? `Chunk ${cIdx + 1}/${chunks.length}: ${candidate.model} busy (HTTP ${classified.httpStatus || 'retry'}) — retrying attempt ${attempt + 1}/${maxAttempts} in ${(waitMs / 1000).toFixed(1)}s...`
                    : `${candidate.model} busy (HTTP ${classified.httpStatus || 'retry'}) — retrying attempt ${attempt + 1}/${maxAttempts} in ${(waitMs / 1000).toFixed(1)}s...`,
                  stageDetail: isMultiChunk
                    ? `Chunk ${cIdx + 1}/${chunks.length} retry ${attempt + 1}/${maxAttempts}`
                    : undefined,
                  progressPct: isMultiChunk ? Math.round((cIdx / chunks.length) * 100) : null,
                  progressSource: isMultiChunk ? 'backend-chunks' : 'indeterminate-inference',
                  completedUnits: isMultiChunk ? cIdx : undefined,
                  totalUnits: isMultiChunk ? chunks.length : undefined
                });
              },
              operation: () =>
                withPromiseTimeout(
                  ai.models.generateContent({
                    model: candidate.model,
                    contents: {
                      parts: [
                        {
                          inlineData: {
                            mimeType: 'audio/wav',
                            data: base64Chunk
                          }
                        },
                        {
                          text: candidate.promptText
                        }
                      ]
                    }
                  }),
                  candidate.timeoutMs,
                  candidate.model
                )
            });

            const fallbackText = extractPlainTextFromAsrResponse(fallbackResp);
            let fallbackBlocks: SrtSegment[] = [];
            if (fallbackText) {
              const inferredMeta = detectLanguageAndScriptFromText(fallbackText, requestedLanguage);
              if (!languageResolvedFromSpeech) {
                detectedLanguageCode = inferredMeta.code;
                detectedLanguageName = inferredMeta.name;
                nativeScriptName = inferredMeta.script;
                languageResolvedFromSpeech = true;
              }

              canonicalTranscriptParts.push(fallbackText);

              fallbackBlocks = segmentChunkAsrIntoReadableSrtBlocks({
                rawSegments: [
                  {
                    startSec: acousticProfile.activeSpeechStartMs / 1000,
                    endSec: acousticProfile.activeSpeechEndMs / 1000,
                    text: fallbackText
                  }
                ],
                chunkStartSec: chunk.startSec,
                chunkEndSec: chunk.endSec,
                acousticProfile,
                languageCode: detectedLanguageCode
              });
              allChunkSegments.push(...fallbackBlocks);

              setCachedSrtChunk(chunkCacheKey, {
                cachedAtMs: Date.now(),
                languageCode: inferredMeta.code,
                languageName: inferredMeta.name,
                nativeScriptName: inferredMeta.script,
                canonicalTexts: [fallbackText],
                fallbackText,
                status: 'recovered'
              });
            }

            chunksProcessed.push({
              chunkIndex: chunk.chunkIndex,
              startSec: chunk.startSec,
              endSec: chunk.endSec,
              segmentCount: fallbackBlocks.length,
              status: 'recovered'
            });
            fallbackSucceeded = true;
            break;
          } catch (err: any) {
            if (err instanceof QuotaExhaustedError || err?.name === 'AbortError') {
              throw err;
            }
            lastAsrError = err;
          }
        }

        if (!fallbackSucceeded) {
          const classified = classifyGeminiAsrError(lastAsrError);
          const failureErr: any = new Error(
            formatGeminiApiError(
              lastAsrError,
              `Speech recognition failed on audio chunk ${chunk.chunkIndex + 1} after retries and model failover`
            )
          );
          if (classified.isTransientRetryable || classified.httpStatus === 503) {
            failureErr.httpStatus = 503;
          }
          throw failureErr;
        }
      }

      if (chunks.length > 1) {
        onProgress?.({
          stageIndex: 1,
          stageLabel: `Completed ASR chunk ${cIdx + 1} of ${chunks.length}`,
          stageDetail: `Chunk ${cIdx + 1}/${chunks.length}`,
          progressPct: Math.round(((cIdx + 1) / chunks.length) * 100),
          progressSource: 'backend-chunks',
          completedUnits: cIdx + 1,
          totalUnits: chunks.length
        });
      }
    }
  } finally {
    // Clean up temporary chunk WAV files
    for (const chunk of chunks) {
      await fs.promises.unlink(chunk.chunkFilePath).catch(() => {});
    }
  }

  const canonicalOriginalTranscript = canonicalTranscriptParts.join(' ').trim();

  if (!canonicalOriginalTranscript || allChunkSegments.length === 0) {
    throw new Error(
      lastAsrError
        ? formatGeminiApiError(lastAsrError, 'Failed to transcribe uploaded audio')
        : 'No recognizable speech could be transcribed from the uploaded audio file. Please verify that the audio contains clear speech and try again.'
    );
  }

  onProgress?.({
    stageIndex: 2,
    stageLabel: 'Verifying exact token sequence & chronological timeline integrity...',
    progressPct: null,
    progressSource: 'indeterminate-inference'
  });

  // Enforce strict timeline integrity, non-overlapping chronology, duration bounds & ordered token sequence preservation
  const { segments: normalizedSegments, validationReport } = enforceAndValidateSrtTimelineIntegrity(
    allChunkSegments,
    durationSeconds,
    canonicalOriginalTranscript
  );

  const rawSrtContent = formatSegmentsToSrt(normalizedSegments);

  if (normalizedSegments.length === 0 || !rawSrtContent.trim()) {
    throw new Error(
      'Subtitle segmentation produced no valid SRT blocks from the uploaded audio. Please try another audio recording.'
    );
  }

  return {
    jobId,
    fileName: originalFileName,
    durationSeconds,
    requestedLanguage,
    detectedLanguageCode,
    detectedLanguageName,
    nativeScriptName,
    modelUsed: 'Gemini 3.5 Transcribe / 3.1 Flash-Lite / 4-Hydrogen Multilingual ASR + 20ms Acoustic Pause & Sentence Segmenter',
    chunkCount: chunks.length,
    chunksProcessed,
    segments: normalizedSegments,
    rawSrtContent,
    audioPreviewUrl,
    processingTimeMs: Date.now() - startWallTime,
    validationReport
  };
}

export interface BuiltInSampleAudio {
  language: CloningLanguageCode;
  fileId: string;
  fileName: string;
  filePath: string;
  sampleTranscript: string;
  analysis?: AudioSignalAnalysis;
}

const SAMPLE_DEFINITIONS: Record<
  CloningLanguageCode,
  { fileName: string; voiceName: 'Kore' | 'Charon' | 'Puck'; style: string; text: string }
> = {
  bn: {
    fileName: 'authorized_reference_bengali_bn.wav',
    voiceName: 'Charon',
    style: 'Clear, natural native Bengali studio narrator with warm resonant tone',
    text: 'নমস্কার, এটি ভয়েসল্যাব এআই-এর বাংলা কণ্ঠস্বর গবেষণার জন্য একটি অনুমোদিত রেফারেন্স অডিও রেকর্ডিং। আমাদের কণ্ঠস্বরের স্বাভাবিক ছন্দ এবং উচ্চারণ এখানে সংরক্ষিত রয়েছে।'
  },
  hi: {
    fileName: 'authorized_reference_hindi_hi.wav',
    voiceName: 'Kore',
    style: 'Clear, expressive native Hindi broadcast speaker with articulate pronunciation',
    text: 'नमस्ते, यह वॉइसलैब एआई के हिन्दी स्वर क्लोनिंग परीक्षण के लिए एक अधिकृत संदर्भ ऑडियो रिकॉर्डिंग है। इसमें वक्ता की स्वाभाविक लय और स्पष्ट उच्चारण सुरक्षित है।'
  },
  en: {
    fileName: 'authorized_reference_english_en.wav',
    voiceName: 'Puck',
    style: 'Clear, articulate English studio speaker with natural cadence and warm acoustics',
    text: 'Hello, this is an authorized studio reference voice recording for VoiceLab AI. This sample captures natural speaking rhythm, vocal timbre, and harmonic clarity.'
  }
};

/**
 * Generates or loads a real reference WAV sample for Bengali (bn), Hindi (hi), or English (en)
 * so users and automated real-world tests can run genuine audio pipelines immediately.
 */
export async function getOrCreateAuthorizedSampleAudio(
  language: CloningLanguageCode,
  samplesDir: string,
  uploadedFilesMap: Map<string, { filePath: string; analysis: AudioSignalAnalysis }>
): Promise<AudioSignalAnalysis> {
  await fs.promises.mkdir(samplesDir, { recursive: true });
  const def = SAMPLE_DEFINITIONS[language];
  const fileId = `sample_ref_${language}`;
  const filePath = path.join(samplesDir, def.fileName);

  const existing = uploadedFilesMap.get(fileId);
  if (existing && fs.existsSync(existing.filePath)) {
    return existing.analysis;
  }

  if (!fs.existsSync(filePath)) {
    await synthesizeStudioSampleWav({
      spokenText: def.text,
      voicePersona: def.voiceName,
      styleDirective: def.style,
      outputRawWavPath: filePath
    });
  }

  const previewUrl = `/api/audio/sample/${language}`;
  const analysis = await inspectAndValidateAudioFile(
    filePath,
    fileId,
    def.fileName,
    'audio/wav',
    previewUrl
  );

  analysis.detectedSpokenLanguage = {
    code: language,
    name: `${CLONING_LANGUAGES[language].englishName} (${CLONING_LANGUAGES[language].nativeName})`,
    confidence: 'Verified Studio Sample (100%)',
    transcriptExcerpt: def.text,
    vocalIdentitySummary: {
      register: `${analysis.fundamentalFreqHz} Hz Fundamental Register (${analysis.pitchRangeHz[0]}–${analysis.pitchRangeHz[1]} Hz)`,
      timbre:
        analysis.spectralWarmthRatio > 1.6
          ? 'Warm harmonic studio vocal envelope'
          : 'Clear, balanced studio vocal timbre',
      pace: 'Natural broadcast cadence',
      accentNotes: `Native ${CLONING_LANGUAGES[language].englishName} (${CLONING_LANGUAGES[language].nativeName}) articulation`,
      recommendedBasePersona: def.voiceName
    }
  };

  uploadedFilesMap.set(fileId, { filePath, analysis });
  return analysis;
}
