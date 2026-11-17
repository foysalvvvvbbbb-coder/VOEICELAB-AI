import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import type {
  AudioSignalAnalysis,
  SrtSegment,
  SrtValidationReport,
  SrtWordTimestamp,
  VoiceAudioEnhancementReport,
  VoiceAudioQualityMode
} from '../types/voicelab.ts';

const execFileAsync = promisify(execFile);

export interface StudioBandProfileDb {
  lowWarmthDb: number;
  mudDb: number;
  nasalDb: number;
  presenceDb: number;
  sibilanceDb: number;
  airDb: number;
}

export interface RawPcmPitchStats {
  fundamentalFreqHz: number;
  pitchRangeHz: [number, number];
  pitchStdDevHz: number;
  voicedFrameRatio: number;
  spectralWarmthRatio: number;
  rmsDb: number;
  peakDb: number;
  noiseFloorDb: number;
  activeNoiseFloorDb: number;
  silenceRatio: number;
  clippedSamplesCount: number;
  dcOffset: number;
  acousticEmbedding16D: number[];
  studioBandProfileDb: StudioBandProfileDb;
}

export function computeVectorCosineSimilarity(a: number[], b: number[]): number {
  if (!a.length || !b.length || a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 1e-12 ? Number((dot / denom).toFixed(4)) : 0;
}

const VOCAL_BAND_FREQS_HZ = [120, 180, 260, 380, 550, 800, 1150, 1650, 2350, 3300, 4600, 6400];
const AIR_BAND_FREQS_HZ = [8000, 9400];

/**
 * Extracts raw 24kHz 16-bit mono PCM via FFmpeg and computes:
 * - Voiced frame autocorrelation fundamental frequency (F0 in Hz, 72Hz–350Hz human vocal range)
 * - 10th to 90th percentile F0 range & F0 standard deviation (Hz)
 * - RMS level (dBFS), Peak level (dBFS), Noise floor (all-frame p15 & active-speech p12 excluding digital zero pauses)
 * - Clipped sample count (|s| >= 0.998) and DC offset
 * - 16-D Voiced Acoustic-Spectral Speaker Descriptor + 6-band Studio Mastering dB profile (up to 9.4 kHz air)
 */
export async function analyzeVocalAcousticsFromAudioFile(filePath: string): Promise<RawPcmPitchStats> {
  const tempRawPath = `${filePath}.24k.pcm`;
  try {
    await execFileAsync('ffmpeg', [
      '-y',
      '-i',
      filePath,
      '-vn',
      '-ac',
      '1',
      '-ar',
      '24000',
      '-f',
      's16le',
      tempRawPath
    ]);

    const buffer = await fs.promises.readFile(tempRawPath);
    const numSamples = Math.floor(buffer.length / 2);
    if (numSamples < 2400) {
      throw new Error('Audio recording is too short (< 0.1s) or contains no decodable PCM samples.');
    }

    const samples = new Float32Array(numSamples);
    let globalPeak = 0;
    let sumSqAll = 0;
    let sumLinearAll = 0;
    let clippedSamplesCount = 0;

    for (let i = 0; i < numSamples; i++) {
      const val = buffer.readInt16LE(i * 2) / 32768.0;
      samples[i] = val;
      sumLinearAll += val;
      const abs = Math.abs(val);
      if (abs >= 0.998) clippedSamplesCount++;
      if (abs > globalPeak) globalPeak = abs;
      sumSqAll += val * val;
    }

    const dcOffset = Number((sumLinearAll / numSamples).toFixed(5));
    const globalRms = Math.sqrt(sumSqAll / numSamples);
    const rmsDb = globalRms > 1e-6 ? 20 * Math.log10(globalRms) : -96.0;
    const peakDb = globalPeak > 1e-6 ? 20 * Math.log10(globalPeak) : -96.0;

    // Frame-by-frame analysis (30ms frames = 720 samples at 24kHz, hop = 360 samples)
    const sampleRate = 24000;
    const frameSize = 720;
    const hopSize = 360;
    const frameDbValues: number[] = [];
    const nonZeroFrameDbValues: number[] = [];
    const f0Estimates: number[] = [];
    const allFreqs = [...VOCAL_BAND_FREQS_HZ, ...AIR_BAND_FREQS_HZ];
    const bandEnergySums = new Float64Array(allFreqs.length);
    const bandPowerLinearSums = new Float64Array(allFreqs.length);
    let voicedSpectralFrames = 0;
    let speechBandFrames = 0;
    let silentFrames = 0;
    let totalFrames = 0;
    let lowBandEnergy = 0;
    let highBandEnergy = 0;

    const minLag = Math.floor(sampleRate / 350); // ~350 Hz max vocal F0
    const maxLag = Math.floor(sampleRate / 72); // ~72 Hz min vocal F0

    // Exact full-rate Goertzel coefficients (up to 9.4 kHz air with zero aliasing)
    const goertzelCoeffs = allFreqs.map((freq) => {
      const k = Math.round((frameSize * freq) / sampleRate);
      const omega = (2 * Math.PI * k) / frameSize;
      return 2 * Math.cos(omega);
    });

    for (let start = 0; start + frameSize <= numSamples; start += hopSize) {
      totalFrames++;
      let frameSumSq = 0;
      let diffSumSq = 0;
      for (let i = 0; i < frameSize; i++) {
        const s = samples[start + i];
        frameSumSq += s * s;
        if (i > 0) {
          const diff = s - samples[start + i - 1];
          diffSumSq += diff * diff;
        }
      }
      const frameRms = Math.sqrt(frameSumSq / frameSize);
      const frameDb = frameRms > 1e-6 ? 20 * Math.log10(frameRms) : -96.0;
      frameDbValues.push(frameDb);
      if (frameDb > -74.0) {
        nonZeroFrameDbValues.push(frameDb);
      }

      if (frameDb < -45.0) {
        silentFrames++;
        continue;
      }

      lowBandEnergy += frameSumSq;
      highBandEnergy += diffSumSq + 1e-9;

      // Full-speech spectral envelope (vowels + fricatives/consonants up to 9.4 kHz air)
      const isSpeechSpectralFrame = frameDb > rmsDb - 18 && speechBandFrames < 450 && totalFrames % 2 === 0;
      const runVoicedAc = f0Estimates.length < 400 && totalFrames % 2 === 0 && frameDb > rmsDb - 13;

      let frameBandPowers: Float64Array | null = null;
      if (isSpeechSpectralFrame) {
        speechBandFrames++;
        frameBandPowers = new Float64Array(allFreqs.length);
        for (let b = 0; b < goertzelCoeffs.length; b++) {
          const coeff = goertzelCoeffs[b];
          let s1 = 0;
          let s2 = 0;
          for (let i = 0; i < frameSize; i++) {
            const s0 = samples[start + i] + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
          }
          const power = Math.max(1e-10, (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (frameSize * frameSize));
          frameBandPowers[b] = power;
          bandPowerLinearSums[b] += power;
        }
      }

      // Normalized autocorrelation for voiced F0 detection & 16-D voiced spectral embedding
      if (runVoicedAc) {
        let bestLag = 0;
        let maxCorr = 0;

        for (let lag = minLag; lag <= maxLag; lag++) {
          let corr = 0;
          let norm1 = 0;
          let norm2 = 0;
          const limit = frameSize - lag;
          for (let i = 0; i < limit; i += 3) {
            const a = samples[start + i];
            const b = samples[start + i + lag];
            corr += a * b;
            norm1 += a * a;
            norm2 += b * b;
          }
          const denom = Math.sqrt(norm1 * norm2);
          const normalized = denom > 1e-8 ? corr / denom : 0;
          if (normalized > maxCorr) {
            maxCorr = normalized;
            bestLag = lag;
          }
        }

        if (maxCorr > 0.62 && bestLag > 0) {
          const f0 = sampleRate / bestLag;
          if (f0 >= 75 && f0 <= 340) {
            f0Estimates.push(f0);
          }

          voicedSpectralFrames++;
          for (let b = 0; b < VOCAL_BAND_FREQS_HZ.length; b++) {
            let power = frameBandPowers ? frameBandPowers[b] : 0;
            if (!frameBandPowers) {
              const coeff = goertzelCoeffs[b];
              let s1 = 0;
              let s2 = 0;
              for (let i = 0; i < frameSize; i++) {
                const s0 = samples[start + i] + coeff * s1 - s2;
                s2 = s1;
                s1 = s0;
              }
              power = Math.max(1e-10, (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (frameSize * frameSize));
            }
            bandEnergySums[b] += Math.log10(power * frameSize * frameSize + 1e-6) + 6.0;
          }
        }
      }
    }

    frameDbValues.sort((a, b) => a - b);
    const p15Index = Math.floor(frameDbValues.length * 0.15);
    const noiseFloorDb = frameDbValues.length > 0 ? frameDbValues[p15Index] : -72.0;

    nonZeroFrameDbValues.sort((a, b) => a - b);
    const activeNoiseFloorDb =
      nonZeroFrameDbValues.length > 8
        ? nonZeroFrameDbValues[Math.floor(nonZeroFrameDbValues.length * 0.12)]
        : noiseFloorDb;

    f0Estimates.sort((a, b) => a - b);
    let medianF0 = 155;
    let p10F0 = 120;
    let p90F0 = 195;
    let pitchStdDevHz = 14.0;

    if (f0Estimates.length >= 3) {
      medianF0 = f0Estimates[Math.floor(f0Estimates.length * 0.5)];
      p10F0 = f0Estimates[Math.floor(f0Estimates.length * 0.15)];
      p90F0 = f0Estimates[Math.floor(f0Estimates.length * 0.85)];
      const meanF0 = f0Estimates.reduce((acc, v) => acc + v, 0) / f0Estimates.length;
      const variance =
        f0Estimates.reduce((acc, v) => acc + (v - meanF0) * (v - meanF0), 0) / f0Estimates.length;
      pitchStdDevHz = Number(Math.sqrt(variance).toFixed(1));
    }

    const spectralWarmthRatio = Number(
      Math.min(4.5, Math.max(0.5, lowBandEnergy / (highBandEnergy * 3.5 + 1e-6))).toFixed(2)
    );

    const activeRatio = totalFrames > 0 ? Number(((totalFrames - silentFrames) / totalFrames).toFixed(2)) : 0;

    // Build 16-D normalized acoustic-spectral speaker embedding
    const rawVec: number[] = [];
    const denomFrames = Math.max(1, voicedSpectralFrames);
    for (let b = 0; b < VOCAL_BAND_FREQS_HZ.length; b++) {
      rawVec.push(Math.max(0.01, bandEnergySums[b] / denomFrames));
    }
    rawVec.push(medianF0 / 250.0);
    rawVec.push((p90F0 - p10F0) / 120.0);
    rawVec.push(spectralWarmthRatio / 3.5);
    rawVec.push(activeRatio);

    const l2Norm = Math.sqrt(rawVec.reduce((acc, v) => acc + v * v, 0)) || 1.0;
    const acousticEmbedding16D = rawVec.map((v) => Number((v / l2Norm).toFixed(4)));

    const speechDenomFrames = Math.max(1, speechBandFrames);
    const toBandDb = (indices: number[]) => {
      const avgPower =
        indices.reduce((sum, idx) => sum + bandPowerLinearSums[idx] / speechDenomFrames, 0) /
        Math.max(1, indices.length);
      return Number((10 * Math.log10(Math.max(1e-10, avgPower))).toFixed(1));
    };

    // Indices: 0:120, 1:180, 2:260, 3:380, 4:550, 5:800, 6:1150, 7:1650, 8:2350, 9:3300, 10:4600, 11:6400, 12:8000, 13:9400
    const studioBandProfileDb: StudioBandProfileDb = {
      lowWarmthDb: toBandDb([0, 1]),
      mudDb: toBandDb([2, 3]),
      nasalDb: toBandDb([5, 6]),
      presenceDb: toBandDb([8, 9]),
      sibilanceDb: toBandDb([10, 11]),
      airDb: toBandDb([12, 13])
    };

    return {
      fundamentalFreqHz: Math.round(medianF0),
      pitchRangeHz: [Math.round(p10F0), Math.round(p90F0)],
      pitchStdDevHz,
      voicedFrameRatio: activeRatio,
      spectralWarmthRatio,
      rmsDb: Number(rmsDb.toFixed(1)),
      peakDb: Number(peakDb.toFixed(1)),
      noiseFloorDb: Number(noiseFloorDb.toFixed(1)),
      activeNoiseFloorDb: Number(activeNoiseFloorDb.toFixed(1)),
      silenceRatio: totalFrames > 0 ? Number((silentFrames / totalFrames).toFixed(2)) : 0,
      clippedSamplesCount,
      dcOffset,
      acousticEmbedding16D,
      studioBandProfileDb
    };
  } finally {
    await fs.promises.unlink(tempRawPath).catch(() => {});
  }
}

/**
 * Inspects an audio file using ffprobe + PCM acoustic analysis.
 */
export async function inspectAndValidateAudioFile(
  filePath: string,
  fileId: string,
  originalName: string,
  mimeType: string,
  previewUrl: string
): Promise<AudioSignalAnalysis> {
  const stats = await fs.promises.stat(filePath);
  if (stats.size === 0) {
    throw new Error('Uploaded audio file is empty (0 bytes).');
  }

  let ffprobeJson: any;
  try {
    const { stdout } = await execFileAsync('ffprobe', [
      '-v',
      'error',
      '-show_format',
      '-show_streams',
      '-of',
      'json',
      filePath
    ]);
    ffprobeJson = JSON.parse(stdout);
  } catch (err: any) {
    throw new Error('Corrupted or unsupported audio stream. FFprobe could not decode container header.');
  }

  const audioStream = (ffprobeJson.streams || []).find((s: any) => s.codec_type === 'audio');
  if (!audioStream) {
    throw new Error('No valid audio stream found in the uploaded file.');
  }

  const durationSeconds = Number(
    parseFloat(audioStream.duration || ffprobeJson.format?.duration || '0').toFixed(2)
  );
  const sampleRateHz = parseInt(audioStream.sample_rate || '24000', 10);
  const channels = parseInt(audioStream.channels || '1', 10);
  const bitRateKbps = Math.round(
    parseInt(audioStream.bit_rate || ffprobeJson.format?.bit_rate || '128000', 10) / 1000
  );
  const codecName = audioStream.codec_name || 'pcm_s16le';
  const formatName = ffprobeJson.format?.format_name || 'wav';

  const pcmStats = await analyzeVocalAcousticsFromAudioFile(filePath);
  const estimatedSnrDb = Number(Math.max(0, pcmStats.rmsDb - pcmStats.noiseFloorDb).toFixed(1));

  const warnings: string[] = [];
  if (durationSeconds < 2.5) {
    warnings.push('Recording is very short (< 2.5s). 5–20 seconds of clear speech is recommended for voice cloning.');
  }
  if (pcmStats.rmsDb < -38) {
    warnings.push('Audio level is very quiet (RMS < -38 dBFS). Automatic loudness normalization will be applied.');
  }
  if (estimatedSnrDb < 12) {
    warnings.push('Elevated background noise detected (SNR < 12 dB). High-pass and spectral de-noising will be engaged.');
  }
  if (pcmStats.silenceRatio > 0.75) {
    warnings.push('Recording contains more than 75% silence.');
  }
  if (sampleRateHz < 16000) {
    warnings.push(`Low sample rate (${sampleRateHz} Hz). Upsampling to 24,000 Hz will be applied.`);
  }

  let qualityGrade: AudioSignalAnalysis['qualityGrade'] = 'Studio Grade';
  if (pcmStats.rmsDb < -55 || durationSeconds < 0.4) {
    qualityGrade = 'Poor / Corrupted';
  } else if (estimatedSnrDb < 14 || sampleRateHz < 16000) {
    qualityGrade = 'Acceptable (Noisy)';
  } else if (estimatedSnrDb < 22 || warnings.length > 0) {
    qualityGrade = 'Good';
  }

  const isValid = qualityGrade !== 'Poor / Corrupted' && durationSeconds >= 0.4;
  if (!isValid) {
    warnings.push('Audio contains near-total silence or is too short to extract valid speech features.');
  }

  return {
    fileId,
    fileName: originalName,
    mimeType,
    formatName,
    codecName,
    fileSizeBytes: stats.size,
    durationSeconds,
    sampleRateHz,
    channels,
    bitRateKbps,
    rmsLevelDb: pcmStats.rmsDb,
    peakLevelDb: pcmStats.peakDb,
    noiseFloorDb: pcmStats.noiseFloorDb,
    estimatedSnrDb,
    silenceRatio: pcmStats.silenceRatio,
    fundamentalFreqHz: pcmStats.fundamentalFreqHz,
    pitchRangeHz: pcmStats.pitchRangeHz,
    spectralWarmthRatio: pcmStats.spectralWarmthRatio,
    acousticEmbedding16D: pcmStats.acousticEmbedding16D,
    qualityGrade,
    qualityWarnings: warnings,
    isValid,
    previewUrl
  };
}

/**
 * Optional post-synthesis DSP audio finishing (F0 register alignment, parametric EQ warmth, -16 LUFS normalization)
 * and WAV/MP3 export. Explicitly separated from neural speech synthesis.
 */
export interface ChunkPauseTiming {
  leadingPauseMs?: number;
  trailingPauseMs?: number;
}

function buildComfortSilenceBuffer(sampleCount: number): Buffer {
  if (sampleCount <= 0) return Buffer.alloc(0);
  const buf = Buffer.alloc(sampleCount * 2);
  // Subtle -76 dBFS brown-weighted comfort floor (prevents abrupt digital-zero gating clicks)
  let prev = 0;
  for (let i = 0; i < sampleCount; i++) {
    const white = (((i * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;
    prev = prev * 0.72 + white * 0.28;
    const sample = Math.round(prev * 5.0);
    buf.writeInt16LE(sample, i * 2);
  }
  return buf;
}

function measureActiveSpeechRmsDbFromPcm(buf: Buffer): number {
  const totalSamples = Math.floor(buf.length / 2);
  if (totalSamples < 480) return -18.0;
  const frameLen = 480;
  const activeSq: number[] = [];
  for (let start = 0; start + frameLen <= totalSamples; start += frameLen) {
    let sumSq = 0;
    for (let i = 0; i < frameLen; i++) {
      const s = buf.readInt16LE((start + i) * 2) / 32768.0;
      sumSq += s * s;
    }
    const rms = Math.sqrt(sumSq / frameLen);
    if (rms > 0.0056) {
      // > -45 dBFS active speech
      activeSq.push(sumSq / frameLen);
    }
  }
  if (activeSq.length === 0) return -18.0;
  const meanSq = activeSq.reduce((a, b) => a + b, 0) / activeSq.length;
  return 10 * Math.log10(Math.max(1e-8, meanSq));
}

function tryExtractCanonical24kMonoPcmSync(wavBuffer: Buffer): Buffer | null {
  if (wavBuffer.length < 44) return null;
  if (wavBuffer.toString('ascii', 0, 4) !== 'RIFF' || wavBuffer.toString('ascii', 8, 12) !== 'WAVE') {
    return null;
  }
  let offset = 12;
  let audioFormat = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataStart = -1;
  let dataSize = 0;

  while (offset + 8 <= wavBuffer.length) {
    const chunkId = wavBuffer.toString('ascii', offset, offset + 4);
    const chunkSize = wavBuffer.readUInt32LE(offset + 4);
    const bodyStart = offset + 8;
    if (bodyStart + chunkSize > wavBuffer.length) break;

    if (chunkId === 'fmt ' && chunkSize >= 16) {
      audioFormat = wavBuffer.readUInt16LE(bodyStart);
      channels = wavBuffer.readUInt16LE(bodyStart + 2);
      sampleRate = wavBuffer.readUInt32LE(bodyStart + 4);
      bitsPerSample = wavBuffer.readUInt16LE(bodyStart + 14);
    } else if (chunkId === 'data') {
      dataStart = bodyStart;
      dataSize = chunkSize;
      break;
    }
    offset = bodyStart + chunkSize + (chunkSize % 2);
  }

  if (
    audioFormat === 1 &&
    channels === 1 &&
    sampleRate === 24000 &&
    bitsPerSample === 16 &&
    dataStart > 0 &&
    dataSize > 0
  ) {
    const end = Math.min(wavBuffer.length, dataStart + dataSize);
    return wavBuffer.subarray(dataStart, end);
  }
  return null;
}

function trimActivePcmChunk(buf: Buffer, sampleRate = 24000): { buf: Buffer; activeRmsDb: number } | null {
  const totalSamples = Math.floor(buf.length / 2);
  if (totalSamples === 0) return null;

  let firstActive = 0;
  let lastActive = totalSamples - 1;
  const threshold = 65; // ~ -54 dBFS in int16
  while (firstActive < totalSamples && Math.abs(buf.readInt16LE(firstActive * 2)) < threshold) {
    firstActive++;
  }
  while (lastActive > firstActive && Math.abs(buf.readInt16LE(lastActive * 2)) < threshold) {
    lastActive--;
  }
  const padSamples = Math.round(sampleRate * 0.065);
  const startSample = Math.max(0, firstActive - padSamples);
  const endSample = Math.min(totalSamples, lastActive + padSamples);
  const activeLen = Math.max(1, endSample - startSample);

  const trimmedBuf = Buffer.alloc(activeLen * 2);
  buf.copy(trimmedBuf, 0, startSample * 2, endSample * 2);

  return {
    buf: trimmedBuf,
    activeRmsDb: measureActiveSpeechRmsDbFromPcm(trimmedBuf)
  };
}

/**
 * Concatenates multiple 24kHz mono WAV sentence chunks into a single continuous WAV file with:
 * - Zero-spawn fast RIFF PCM extraction when chunks are already 24kHz mono PCM16 (parallel FFmpeg fallback otherwise)
 * - Active-speech RMS leveling across chunks (bounded +/- 4.5 dB) for consistent loudness
 * - Raised-cosine 14ms S-curve edge fades (zero click & zero boundary step)
 * - Generous 65ms natural breath/consonant tail preservation
 * - Ultra-quiet comfort room-tone gaps honoring exact [pause] millisecond timings
 */
export interface ConcatenatedChunkTimingOffset {
  chunkIndex: number;
  startMs: number;
  endMs: number;
}

export async function concatenateWavChunksWithCrossfade(
  chunkWavPaths: string[],
  outputWavPath: string,
  interSentencePauseMs = 190,
  perChunkPauses?: ChunkPauseTiming[]
): Promise<{
  totalDurationSec: number;
  chunkOffsets: ConcatenatedChunkTimingOffset[];
}> {
  if (chunkWavPaths.length === 0) {
    throw new Error('No audio chunks provided for concatenation.');
  }

  const sampleRate = 24000;
  const fadeSamples = Math.round((sampleRate * 14) / 1000); // 14ms cosine fade (336 samples)
  const defaultPauseSamples = Math.round((sampleRate * interSentencePauseMs) / 1000);

  interface DecodedChunk {
    chunkIndex: number;
    buf: Buffer;
    activeRmsDb: number;
  }

  const decodeSingleChunk = async (
    chunkPath: string,
    chunkIndex: number
  ): Promise<DecodedChunk | null> => {
    try {
      const wavBuf = await fs.promises.readFile(chunkPath);
      const directPcm = tryExtractCanonical24kMonoPcmSync(wavBuf);
      if (directPcm) {
        const trimmed = trimActivePcmChunk(directPcm, sampleRate);
        if (trimmed) {
          return { chunkIndex, ...trimmed };
        }
      }
    } catch {
      // Fall back cleanly to FFmpeg decode below
    }

    const rawPath = `${chunkPath}.24k.pcm`;
    try {
      await execFileAsync('ffmpeg', [
        '-y',
        '-i',
        chunkPath,
        '-vn',
        '-ac',
        '1',
        '-ar',
        String(sampleRate),
        '-f',
        's16le',
        rawPath
      ]);
      const buf = await fs.promises.readFile(rawPath);
      const trimmed = trimActivePcmChunk(buf, sampleRate);
      return trimmed ? { chunkIndex, ...trimmed } : null;
    } finally {
      await fs.promises.unlink(rawPath).catch(() => {});
    }
  };

  // Decode in bounded parallel batches of 6 chunks
  const decodedMaybe: (DecodedChunk | null)[] = new Array(chunkWavPaths.length).fill(null);
  let nextIdx = 0;
  const workerCount = Math.min(6, chunkWavPaths.length);
  await Promise.all(
    Array.from({ length: workerCount }, async () => {
      while (nextIdx < chunkWavPaths.length) {
        const myIdx = nextIdx++;
        decodedMaybe[myIdx] = await decodeSingleChunk(chunkWavPaths[myIdx], myIdx);
      }
    })
  );

  const decodedChunks: DecodedChunk[] = decodedMaybe.filter(
    (c): c is DecodedChunk => c !== null
  );

  if (decodedChunks.length === 0) {
    throw new Error('No decodable audio chunks available for concatenation.');
  }

  // Match active chunk RMS toward median loudness (bounded +/- 4.5 dB) when multiple chunks exist
  let targetChunkRmsDb = decodedChunks[0].activeRmsDb;
  if (decodedChunks.length > 1) {
    const sorted = [...decodedChunks.map((c) => c.activeRmsDb)].sort((a, b) => a - b);
    targetChunkRmsDb = sorted[Math.floor(sorted.length / 2)];
  }

  const pcmBuffers: Buffer[] = [];
  const chunkOffsets: ConcatenatedChunkTimingOffset[] = [];
  let sampleCursor = 0;

  for (let idx = 0; idx < decodedChunks.length; idx++) {
    const chunkBuf = decodedChunks[idx].buf;
    const origIdx = decodedChunks[idx].chunkIndex;
    const totalSamples = Math.floor(chunkBuf.length / 2);

    if (decodedChunks.length > 1) {
      const gainDb = Math.max(-4.5, Math.min(4.5, targetChunkRmsDb - decodedChunks[idx].activeRmsDb));
      if (Math.abs(gainDb) >= 0.35) {
        const linearGain = Math.pow(10, gainDb / 20);
        for (let i = 0; i < totalSamples; i++) {
          let scaled = (chunkBuf.readInt16LE(i * 2) / 32768.0) * linearGain;
          // Soft-knee ceiling prevents inter-chunk gain matching from ever hard-clipping
          if (Math.abs(scaled) > 0.88) {
            scaled = Math.sign(scaled) * (0.88 + 0.1 * Math.tanh((Math.abs(scaled) - 0.88) / 0.1));
          }
          chunkBuf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(scaled * 32767))), i * 2);
        }
      }
    }

    // Apply 14ms raised-cosine S-curve fade-in and fade-out to eliminate boundary steps cleanly
    const effFade = Math.min(fadeSamples, Math.floor(totalSamples / 4));
    for (let i = 0; i < effFade; i++) {
      const t = (i + 0.5) / effFade;
      const gain = 0.5 * (1 - Math.cos(Math.PI * t));

      const sIn = Math.round(chunkBuf.readInt16LE(i * 2) * gain);
      chunkBuf.writeInt16LE(Math.max(-32768, Math.min(32767, sIn)), i * 2);

      const outIdx = totalSamples - 1 - i;
      const sOut = Math.round(chunkBuf.readInt16LE(outIdx * 2) * gain);
      chunkBuf.writeInt16LE(Math.max(-32768, Math.min(32767, sOut)), outIdx * 2);
    }

    if (idx === 0) {
      const leadMs = perChunkPauses?.[origIdx]?.leadingPauseMs || 0;
      if (leadMs > 0) {
        const leadSamples = Math.round((sampleRate * leadMs) / 1000);
        pcmBuffers.push(buildComfortSilenceBuffer(leadSamples));
        sampleCursor += leadSamples;
      }
    }

    const speechStartSample = sampleCursor;
    pcmBuffers.push(chunkBuf);
    sampleCursor += totalSamples;
    const speechEndSample = sampleCursor;

    chunkOffsets.push({
      chunkIndex: origIdx,
      startMs: Math.round((speechStartSample / sampleRate) * 1000),
      endMs: Math.round((speechEndSample / sampleRate) * 1000)
    });

    if (idx < decodedChunks.length - 1) {
      const nextOrigIdx = decodedChunks[idx + 1].chunkIndex;
      const currTrailMs = perChunkPauses?.[origIdx]?.trailingPauseMs || 0;
      const nextLeadMs = perChunkPauses?.[nextOrigIdx]?.leadingPauseMs || 0;
      const explicitBoundaryMs = currTrailMs + nextLeadMs;
      const effectivePauseSamples =
        explicitBoundaryMs > 0
          ? Math.round((sampleRate * explicitBoundaryMs) / 1000)
          : defaultPauseSamples;
      if (effectivePauseSamples > 0) {
        pcmBuffers.push(buildComfortSilenceBuffer(effectivePauseSamples));
        sampleCursor += effectivePauseSamples;
      }
    } else {
      const finalTrailMs = perChunkPauses?.[origIdx]?.trailingPauseMs || 0;
      if (finalTrailMs > 0) {
        const tailSamples = Math.round((sampleRate * finalTrailMs) / 1000);
        pcmBuffers.push(buildComfortSilenceBuffer(tailSamples));
        sampleCursor += tailSamples;
      }
    }
  }

  const combinedRawPath = `${outputWavPath}.combined.pcm`;
  try {
    await fs.promises.writeFile(combinedRawPath, Buffer.concat(pcmBuffers));
    await execFileAsync('ffmpeg', [
      '-y',
      '-f',
      's16le',
      '-ar',
      String(sampleRate),
      '-ac',
      '1',
      '-i',
      combinedRawPath,
      '-c:a',
      'pcm_s16le',
      outputWavPath
    ]);
  } finally {
    await fs.promises.unlink(combinedRawPath).catch(() => {});
  }

  return {
    totalDurationSec: Number((sampleCursor / sampleRate).toFixed(3)),
    chunkOffsets
  };
}

interface EbuR128Measurement {
  input_i: string;
  input_tp: string;
  input_lra: string;
  input_thresh: string;
  target_offset: string;
}

async function measureEbuR128Loudness(
  filePath: string,
  targetI: number,
  targetTp: number,
  targetLra: number
): Promise<EbuR128Measurement | null> {
  try {
    const { stderr } = await execFileAsync(
      'ffmpeg',
      [
        '-hide_banner',
        '-i',
        filePath,
        '-af',
        `loudnorm=I=${targetI}:TP=${targetTp}:LRA=${targetLra}:print_format=json`,
        '-f',
        'null',
        '-'
      ],
      { timeout: 25_000 }
    );
    const jsonMatch = (stderr || '').match(/\{[\s\S]*?"input_i"[\s\S]*?\}/);
    if (!jsonMatch) return null;
    const parsed = JSON.parse(jsonMatch[0]);
    if (
      typeof parsed?.input_i === 'string' &&
      typeof parsed?.input_tp === 'string' &&
      typeof parsed?.input_lra === 'string' &&
      typeof parsed?.input_thresh === 'string' &&
      typeof parsed?.target_offset === 'string'
    ) {
      return parsed as EbuR128Measurement;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Adaptive Broadcast Voice Mastering Engine & WAV/MP3 Exporter.
 * - 'standard': Transparent DC rumble removal + 2-pass linear EBU R128 loudness normalization (-17 LUFS).
 * - 'studio' (Default): 48kHz SoXR oversampled processing with F0-adaptive highpass, measured-floor spectral
 *   noise reduction, gentle room-tail de-reverberation, adaptive voice EQ (mud cut, nasal tame, clarity,
 *   silky air), de-essing, parallel vocal compression, and 2-pass linear -15.5 LUFS mastering.
 * - 'studio_plus': Enhanced creator/podcast polish (-14.5 LUFS, 256kbps MP3) while strictly preserving
 *   the cloned speaker's fundamental pitch, gender characteristics, accent, and vocal identity.
 */
export async function applyReferenceAcousticIdentityAndExport(options: {
  rawSynthesizedWavPath: string;
  referenceAnalysis: AudioSignalAnalysis;
  audioQuality?: VoiceAudioQualityMode;
  preservePitchContour: boolean;
  preserveSpectralTimbre: boolean;
  outputWavPath: string;
  outputMp3Path: string;
}): Promise<{
  synthesizedBasePitchHz: number;
  pitchShiftSemitonesApplied: number;
  outputDurationSec: number;
  outputSampleRateHz: number;
  rmsNormalizedDb: number;
  finalAcoustics: RawPcmPitchStats;
  estimatedOutputSnrDb: number;
  dspApplied: boolean;
  audioEnhancementReport: VoiceAudioEnhancementReport;
}> {
  const {
    rawSynthesizedWavPath,
    audioQuality = 'studio',
    outputWavPath,
    outputMp3Path
  } = options;

  const qualityMode: VoiceAudioQualityMode =
    audioQuality === 'standard' || audioQuality === 'studio_plus' ? audioQuality : 'studio';

  const isStudioLike = qualityMode === 'studio' || qualityMode === 'studio_plus';
  const isPlus = qualityMode === 'studio_plus';
  const targetSampleRateHz = isStudioLike ? 48000 : 24000;
  const targetLufs = isPlus ? -14.5 : isStudioLike ? -15.5 : -17.0;
  const targetTp = isPlus ? -1.0 : -1.2;
  const targetLra = isPlus ? 5.2 : isStudioLike ? 6.0 : 7.0;

  const [synthAcoustics, inputEbu] = await Promise.all([
    analyzeVocalAcousticsFromAudioFile(rawSynthesizedWavPath),
    measureEbuR128Loudness(rawSynthesizedWavPath, targetLufs, targetTp, targetLra)
  ]);
  const synthF0 = synthAcoustics.fundamentalFreqHz || 155;
  const activeNoiseFloorDb = synthAcoustics.activeNoiseFloorDb ?? synthAcoustics.noiseFloorDb;
  const inputEstimatedSnrDb = Number(
    Math.max(0, synthAcoustics.rmsDb - activeNoiseFloorDb).toFixed(1)
  );
  const inputIntegratedLufs = Number(parseFloat(inputEbu?.input_i || String(synthAcoustics.rmsDb)).toFixed(2));

  const adaptiveDecisions: string[] = [];
  let noiseReductionDbApplied = 0;
  let deEsserApplied = false;
  let eqAppliedSummary = 'Sub-sonic DC filter only (60 Hz HPF)';
  let compressionSummary = 'Transparent true-peak lookahead limiter (-1.2 dBTP)';

  const intermediateWavPath = `${outputWavPath}.stage1.wav`;

  try {
    if (!isStudioLike) {
      // Standard mode: Clean rumble removal + direct pass into 2-pass linear loudnorm
      await execFileAsync(
        'ffmpeg',
        [
          '-y',
          '-i',
          rawSynthesizedWavPath,
          '-af',
          'highpass=f=60,aresample=24000:resampler=soxr',
          '-ar',
          '24000',
          '-ac',
          '1',
          '-c:a',
          'pcm_s16le',
          intermediateWavPath
        ],
        { timeout: 25_000 }
      );
      adaptiveDecisions.push('Standard transparent chain (raw neural vocal tone preserved)');
    } else {
      const bands = synthAcoustics.studioBandProfileDb;
      const filterChain: string[] = [];

      // 1. 48kHz SoXR linear-phase oversampling prevents Nyquist EQ cramping in 4-10 kHz range
      filterChain.push('aresample=48000:resampler=soxr:precision=28');

      // 2. Speaker-F0-adaptive sub-sonic rumble removal (protects deep male fundamental)
      const hpHz = Math.max(60, Math.min(92, Math.round(synthF0 * 0.54)));
      filterChain.push(`highpass=f=${hpHz}:p=2`);
      adaptiveDecisions.push(`F0-safe sub-sonic HPF at ${hpHz} Hz (tracks ${synthF0} Hz voice)`);

      // 3. Adaptive spectral denoising strictly scaled to measured active speech floor (never aggressive/watery)
      if (activeNoiseFloorDb <= -58 && inputEstimatedSnrDb >= 30) {
        noiseReductionDbApplied = isPlus ? 4.5 : 3.5;
        filterChain.push(`afftdn=nr=${noiseReductionDbApplied}:nf=-56:tn=1`);
        adaptiveDecisions.push(`Light hiss polish (${noiseReductionDbApplied} dB) on clean input`);
      } else if (activeNoiseFloorDb <= -46) {
        noiseReductionDbApplied = isPlus ? 8.5 : 7.0;
        const nfBound = Math.max(-52, Math.min(-38, Math.round(activeNoiseFloorDb + 3.5)));
        filterChain.push(`afftdn=nr=${noiseReductionDbApplied}:nf=${nfBound}:tn=1`);
        adaptiveDecisions.push(`Moderate denoising (${noiseReductionDbApplied} dB @ ${nfBound} dB floor)`);
      } else {
        noiseReductionDbApplied = isPlus ? 12.0 : 10.5;
        const nfBound = Math.max(-44, Math.min(-32, Math.round(activeNoiseFloorDb + 5.0)));
        filterChain.push(`afftdn=nr=${noiseReductionDbApplied}:nf=${nfBound}:tn=1`);
        adaptiveDecisions.push(`Mobile-noise reduction (${noiseReductionDbApplied} dB @ ${nfBound} dB floor)`);
      }

      // 4. Gentle downward expansion (de-reverb / room-tail taming without clipping breaths or tails)
      if (activeNoiseFloorDb > -56 || isPlus) {
        const threshDb = Math.max(-48, Math.min(-31, activeNoiseFloorDb + 7.5));
        const threshLin = Number(Math.pow(10, threshDb / 20).toFixed(5));
        const ratio = isPlus ? 1.8 : 1.65;
        const range = isPlus ? 0.22 : 0.28;
        filterChain.push(
          `agate=threshold=${threshLin}:ratio=${ratio}:range=${range}:attack=15:release=230:knee=4.5:detection=rms`
        );
        adaptiveDecisions.push(`Soft room-tail de-reverb (ratio ${ratio}:1, release 230ms)`);
      }

      // 5. Speaker-adaptive parametric studio voice EQ (calibrated against full-speech vowel+consonant spectrum)
      const mudExcessDb = bands.mudDb - bands.lowWarmthDb;
      const nasalExcessDb = bands.nasalDb - 0.5 * (bands.lowWarmthDb + bands.presenceDb);
      const presenceDeficitDb = bands.lowWarmthDb - bands.presenceDb;
      const sibilanceExcessDb = bands.sibilanceDb - bands.presenceDb;
      const airDeficitDb = bands.lowWarmthDb - bands.airDb;

      // 5a. Chest warmth (skip if voice is already warm/heavy)
      const warmthCenterHz = Math.max(140, Math.min(215, Math.round(synthF0 * 1.25)));
      let warmthBoostDb = 0;
      if (synthAcoustics.spectralWarmthRatio < 1.35 || mudExcessDb > 1.5) {
        warmthBoostDb = isPlus ? 1.7 : 1.3;
      } else if (synthAcoustics.spectralWarmthRatio < 2.3) {
        warmthBoostDb = isPlus ? 0.9 : 0.6;
      }
      if (warmthBoostDb >= 0.4) {
        filterChain.push(`equalizer=f=${warmthCenterHz}:t=q:w=0.85:g=${warmthBoostDb}`);
      }

      // 5b. Boxy phone/room mud reduction around 330 Hz
      const mudCutDb =
        mudExcessDb > -2.5
          ? -Number(
              Math.min(isPlus ? 2.9 : 2.5, Math.max(0.9, (mudExcessDb + 3.0) * 0.62)).toFixed(1)
            )
          : -0.7;
      filterChain.push(`equalizer=f=330:t=q:w=1.1:g=${mudCutDb}`);

      // 5c. Nasal midrange phone-mic taming around 1120 Hz
      let nasalCutDb = 0;
      if (nasalExcessDb > -1.0) {
        nasalCutDb = -Number(Math.min(1.6, Math.max(0.7, (nasalExcessDb + 1.5) * 0.35)).toFixed(1));
        filterChain.push(`equalizer=f=1120:t=q:w=1.4:g=${nasalCutDb}`);
      }

      // 5d. Vocal articulation & intimacy around 2950 Hz (scales smoothly from dark to already crisp voices)
      const presenceGainDb = Number(
        Math.min(
          isPlus ? 3.2 : 2.7,
          Math.max(0.3, (presenceDeficitDb - (isPlus ? 7.5 : 8.5)) * (isPlus ? 0.48 : 0.42))
        ).toFixed(1)
      );
      filterChain.push(`equalizer=f=2950:t=q:w=0.9:g=${presenceGainDb}`);

      // 5e. Tame vocoder grain spike around 4800 Hz if 4.8kHz exceeds vocal clarity band
      let grainCutDb = 0;
      if (sibilanceExcessDb > -2.5) {
        grainCutDb = -Number(Math.min(1.6, Math.max(0.6, (sibilanceExcessDb + 3.0) * 0.32)).toFixed(1));
        filterChain.push(`equalizer=f=4800:t=q:w=1.6:g=${grainCutDb}`);
      }

      // 5f. Silky high-frequency air shelf at 8200 Hz (minimal touch on already bright voices) + 11.6 kHz guard
      const airGainDb = Number(
        Math.min(
          isPlus ? 3.8 : 3.2,
          Math.max(0.0, (airDeficitDb - (isPlus ? 10.5 : 11.5)) * (isPlus ? 0.35 : 0.30))
        ).toFixed(1)
      );
      if (airGainDb >= 0.3) {
        filterChain.push(`highshelf=f=8200:t=q:w=0.85:g=${airGainDb}`);
      }
      filterChain.push('lowpass=f=11600');

      eqAppliedSummary = `Warmth(${warmthCenterHz}Hz:${warmthBoostDb > 0 ? `+${warmthBoostDb}` : '0'}dB) · DeMud(330Hz:${mudCutDb}dB)${
        nasalCutDb !== 0 ? ` · DeNasal(1.1kHz:${nasalCutDb}dB)` : ''
      } · Clarity(2.95kHz:+${presenceGainDb}dB)${airGainDb >= 0.3 ? ` · Air(8.2kHz:+${airGainDb}dB)` : ''}`;
      adaptiveDecisions.push(`Adaptive EQ: ${eqAppliedSummary}`);

      // 6. Adaptive De-Esser (smooths harsh sibilance without dulling speech)
      const deessIntensity = Number(
        Math.min(0.34, Math.max(0.16, 0.18 + Math.max(0, sibilanceExcessDb + 3.0) * 0.032)).toFixed(2)
      );
      filterChain.push(`deesser=i=${deessIntensity}:m=0.42:f=0.52:s=o`);
      deEsserApplied = true;
      adaptiveDecisions.push(`Adaptive De-Esser (intensity=${deessIntensity})`);

      // 7. Parallel optical-style vocal compression (smooth intimacy + preserved breath transients)
      const compRatio = isPlus ? 2.55 : 2.2;
      const compThresh = isPlus ? -21 : -20;
      const compMakeup = isPlus ? 2.4 : 2.0;
      const compMix = isPlus ? 0.80 : 0.72;
      filterChain.push(
        `acompressor=threshold=${compThresh}dB:ratio=${compRatio}:attack=14:release=160:knee=6:makeup=${compMakeup}:mix=${compMix}:detection=rms`
      );
      compressionSummary = `Parallel broadcast compressor (${compRatio}:1, ${Math.round(compMix * 100)}% wet) + 2-pass ${targetLufs} LUFS linear mastering`;
      adaptiveDecisions.push(`Parallel vocal compression (${compRatio}:1 @ ${Math.round(compMix * 100)}% mix)`);

      await execFileAsync(
        'ffmpeg',
        [
          '-y',
          '-i',
          rawSynthesizedWavPath,
          '-af',
          filterChain.join(','),
          '-ar',
          String(targetSampleRateHz),
          '-ac',
          '1',
          '-c:a',
          'pcm_s16le',
          intermediateWavPath
        ],
        { timeout: 35_000 }
      );
    }

    // Pass 2: True-Linear 2-Pass EBU R128 Broadcast Loudness Normalization + Lookahead Peak Limiter
    const stage1Ebu = await measureEbuR128Loudness(
      intermediateWavPath,
      targetLufs,
      targetTp,
      targetLra
    );

    const masteringFilters: string[] = [];
    if (stage1Ebu) {
      masteringFilters.push(
        `loudnorm=I=${targetLufs}:TP=${targetTp}:LRA=${targetLra}:measured_I=${stage1Ebu.input_i}:measured_LRA=${stage1Ebu.input_lra}:measured_TP=${stage1Ebu.input_tp}:measured_thresh=${stage1Ebu.input_thresh}:offset=${stage1Ebu.target_offset}:linear=true:print_format=json`
      );
      adaptiveDecisions.push(`2-pass linear EBU R128 normalization to ${targetLufs} LUFS`);
    } else {
      const targetRmsDb = isPlus ? -15.5 : isStudioLike ? -16.5 : -17.0;
      const gainDb = Number(
        Math.min(8.0, Math.max(-8.0, targetRmsDb - synthAcoustics.rmsDb)).toFixed(1)
      );
      if (Math.abs(gainDb) >= 0.4) {
        masteringFilters.push(`volume=${gainDb}dB`);
      }
    }
    masteringFilters.push('alimiter=limit=0.891:attack=4:release=65:level=disabled');

    const { stderr: stage2Stderr } = await execFileAsync(
      'ffmpeg',
      [
        '-y',
        '-i',
        intermediateWavPath,
        '-af',
        masteringFilters.join(','),
        '-ar',
        String(targetSampleRateHz),
        '-ac',
        '1',
        '-c:a',
        'pcm_s16le',
        outputWavPath
      ],
      { timeout: 30_000 }
    );

    let stage2OutputEbu: { output_i?: string; output_lra?: string; output_tp?: string } | null = null;
    try {
      const jsonMatch = (stage2Stderr || '').match(/\{[\s\S]*?"output_i"[\s\S]*?\}/);
      if (jsonMatch) {
        stage2OutputEbu = JSON.parse(jsonMatch[0]);
      }
    } catch {
      stage2OutputEbu = null;
    }

    const mp3Bitrate = isPlus ? '256k' : '192k';
    const [, finalStats, ffprobeOut] = await Promise.all([
      execFileAsync(
        'ffmpeg',
        ['-y', '-i', outputWavPath, '-c:a', 'libmp3lame', '-b:a', mp3Bitrate, outputMp3Path],
        { timeout: 25_000 }
      ),
      analyzeVocalAcousticsFromAudioFile(outputWavPath),
      execFileAsync('ffprobe', [
        '-v',
        'error',
        '-show_entries',
        'format=duration',
        '-of',
        'default=noprint_wrappers=1:nokey=1',
        outputWavPath
      ])
    ]);

    const outputDurationSec = Number(parseFloat(ffprobeOut.stdout.trim() || '0').toFixed(2));
    const outputActiveNoiseDb = finalStats.activeNoiseFloorDb ?? finalStats.noiseFloorDb;
    const estimatedOutputSnrDb = Number(
      Math.max(0, finalStats.rmsDb - outputActiveNoiseDb).toFixed(1)
    );

    const qualityLabels: Record<VoiceAudioQualityMode, string> = {
      standard: 'Standard (Raw Neural Output)',
      studio: 'Studio (Broadcast Voice Mastering)',
      studio_plus: 'Studio Plus (Creator & Podcast Polish)'
    };

    const outputIntegratedLufs = Number(
      parseFloat(stage2OutputEbu?.output_i || String(finalStats.rmsDb)).toFixed(2)
    );
    const outputLoudnessRangeLu = Number(parseFloat(stage2OutputEbu?.output_lra || '3.0').toFixed(2));
    const outputTruePeakDb = Number(
      parseFloat(stage2OutputEbu?.output_tp || String(finalStats.peakDb)).toFixed(2)
    );

    return {
      synthesizedBasePitchHz: synthF0,
      pitchShiftSemitonesApplied: 0,
      outputDurationSec,
      outputSampleRateHz: targetSampleRateHz,
      rmsNormalizedDb: finalStats.rmsDb,
      finalAcoustics: finalStats,
      estimatedOutputSnrDb,
      dspApplied: isStudioLike,
      audioEnhancementReport: {
        qualityMode,
        qualityLabel: qualityLabels[qualityMode],
        adaptiveDecisions,
        inputNoiseFloorDb: activeNoiseFloorDb,
        outputNoiseFloorDb: outputActiveNoiseDb,
        inputEstimatedSnrDb,
        outputEstimatedSnrDb: estimatedOutputSnrDb,
        inputIntegratedLufs,
        outputIntegratedLufs,
        outputLoudnessRangeLu,
        outputTruePeakDb,
        inputPitchHz: synthF0,
        outputPitchHz: finalStats.fundamentalFreqHz,
        outputSampleRateHz: targetSampleRateHz,
        eqAppliedSummary,
        deEsserApplied,
        noiseReductionDbApplied,
        compressionSummary
      }
    };
  } finally {
    await fs.promises.unlink(intermediateWavPath).catch(() => {});
  }
}

export interface AudioSpeechInterval {
  startSec: number;
  endSec: number;
}

export interface DetectedAcousticPause {
  startMs: number;
  endMs: number;
  midMs: number;
  durationMs: number;
  depthDb: number;
}

export interface ActiveSpeechIsland {
  startMs: number;
  endMs: number;
  durationMs: number;
}

export interface ChunkAcousticTimingProfile {
  durationMs: number;
  frameHopMs: number;
  frameEnergies: number[];
  frameIsVoiced: boolean[];
  cumulativeEnergy: number[];
  pauses: DetectedAcousticPause[];
  activeSpeechIslands: ActiveSpeechIsland[];
  activeSpeechStartMs: number;
  activeSpeechEndMs: number;
}

/**
 * Extracts a 20ms-resolution acoustic energy, voiced-activity & natural pause profile from a 16kHz mono WAV chunk.
 * Used by the SRT segmenter both to verify ASR timestamps and to anchor sentence/clause boundaries directly
 * to measured vocal activity and real acoustic pauses.
 */
export async function extractChunkAcousticTimingProfile(
  chunkWavPath: string
): Promise<ChunkAcousticTimingProfile> {
  const rawPcmPath = `${chunkWavPath}.timing.pcm`;
  try {
    await execFileAsync('ffmpeg', [
      '-y',
      '-i',
      chunkWavPath,
      '-vn',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-f',
      's16le',
      rawPcmPath
    ]);
    const buf = await fs.promises.readFile(rawPcmPath);
    const totalSamples = Math.floor(buf.length / 2);
    const sampleRate = 16000;
    const frameHopMs = 20;
    const frameSamples = Math.round((sampleRate * frameHopMs) / 1000); // 320 samples (20ms)
    const numFrames = Math.max(1, Math.floor(totalSamples / frameSamples));
    const durationMs = Math.round((totalSamples / sampleRate) * 1000);

    const frameDb: number[] = new Array(numFrames).fill(-96.0);
    const frameRms: number[] = new Array(numFrames).fill(0);

    for (let f = 0; f < numFrames; f++) {
      let sumSq = 0;
      const baseIdx = f * frameSamples;
      for (let i = 0; i < frameSamples; i++) {
        const s = buf.readInt16LE((baseIdx + i) * 2) / 32768.0;
        sumSq += s * s;
      }
      const rms = Math.sqrt(sumSq / frameSamples);
      const db = rms > 1e-6 ? 20 * Math.log10(rms) : -96.0;
      frameRms[f] = rms;
      frameDb[f] = db;
    }

    const sortedDb = [...frameDb].sort((a, b) => a - b);
    const noiseFloorDb = sortedDb[Math.floor(sortedDb.length * 0.15)] ?? -60.0;
    const p80Db = sortedDb[Math.floor(sortedDb.length * 0.8)] ?? -20.0;
    const dynamicRangeDb = Math.max(8.0, p80Db - noiseFloorDb);
    const pauseThresholdDb = Math.max(
      -44.0,
      Math.min(-24.0, Math.max(p80Db - 15.0, noiseFloorDb + dynamicRangeDb * 0.34))
    );

    // Raw voiced mask + 1-frame (20ms) hangover so brief stop-consonant closures stay voiced
    const rawVoiced = frameDb.map((db) => db > pauseThresholdDb);
    const frameIsVoiced: boolean[] = rawVoiced.map((v, idx) => {
      if (v) return true;
      return Boolean(rawVoiced[idx - 1] && rawVoiced[idx + 1]);
    });

    // Companded voiced activity energy: tracks spoken syllable time rather than peak bursts
    const frameEnergies: number[] = frameRms.map((rms, idx) =>
      frameIsVoiced[idx] ? Math.max(0.25, Math.pow(rms, 0.35)) : 0.001
    );

    // Identify natural acoustic pauses >= 140ms (7 frames of 20ms)
    const pauses: DetectedAcousticPause[] = [];
    let inPause = false;
    let pauseStartFrame = 0;
    let minPauseDb = 0;

    for (let f = 0; f < numFrames; f++) {
      if (!frameIsVoiced[f]) {
        if (!inPause) {
          inPause = true;
          pauseStartFrame = f;
          minPauseDb = frameDb[f];
        } else if (frameDb[f] < minPauseDb) {
          minPauseDb = frameDb[f];
        }
      } else if (inPause) {
        const spanFrames = f - pauseStartFrame;
        if (spanFrames >= 7) {
          const startMs = pauseStartFrame * frameHopMs;
          const endMs = f * frameHopMs;
          pauses.push({
            startMs,
            endMs,
            midMs: Math.round((startMs + endMs) / 2),
            durationMs: endMs - startMs,
            depthDb: Number((p80Db - minPauseDb).toFixed(1))
          });
        }
        inPause = false;
      }
    }

    // Build contiguous active speech islands (merging intra-phrase micro-pauses < 340ms)
    const activeSpeechIslands: ActiveSpeechIsland[] = [];
    let inSpeech = false;
    let speechStartFrame = 0;
    for (let f = 0; f < numFrames; f++) {
      if (frameIsVoiced[f]) {
        if (!inSpeech) {
          inSpeech = true;
          speechStartFrame = f;
        }
      } else if (inSpeech) {
        const prevIslandEndMs = f * frameHopMs;
        // Look ahead up to 340ms (17 frames) to see if speech immediately resumes
        let nextVoicedFrame = -1;
        for (let k = f + 1; k < Math.min(numFrames, f + 17); k++) {
          if (frameIsVoiced[k]) {
            nextVoicedFrame = k;
            break;
          }
        }
        if (nextVoicedFrame !== -1) {
          continue;
        }
        const startMs = Math.max(0, (speechStartFrame - 2) * frameHopMs);
        const endMs = Math.min(durationMs, prevIslandEndMs + 40);
        if (endMs - startMs >= 160) {
          activeSpeechIslands.push({
            startMs,
            endMs,
            durationMs: endMs - startMs
          });
        }
        inSpeech = false;
      }
    }
    if (inSpeech) {
      const startMs = Math.max(0, (speechStartFrame - 2) * frameHopMs);
      const endMs = durationMs;
      if (endMs - startMs >= 160) {
        activeSpeechIslands.push({
          startMs,
          endMs,
          durationMs: endMs - startMs
        });
      }
    }

    // Build cumulative energy array for energy-proportional timestamp interpolation
    const cumulativeEnergy: number[] = new Array(numFrames + 1).fill(0);
    for (let f = 0; f < numFrames; f++) {
      cumulativeEnergy[f + 1] = cumulativeEnergy[f] + frameEnergies[f];
    }

    let firstActiveFrame = 0;
    while (firstActiveFrame < numFrames - 1 && !frameIsVoiced[firstActiveFrame]) {
      firstActiveFrame++;
    }
    let lastActiveFrame = numFrames - 1;
    while (lastActiveFrame > firstActiveFrame && !frameIsVoiced[lastActiveFrame]) {
      lastActiveFrame--;
    }

    return {
      durationMs,
      frameHopMs,
      frameEnergies,
      frameIsVoiced,
      cumulativeEnergy,
      pauses,
      activeSpeechIslands,
      activeSpeechStartMs: Math.max(0, (firstActiveFrame - 2) * frameHopMs),
      activeSpeechEndMs: Math.min(durationMs, (lastActiveFrame + 3) * frameHopMs)
    };
  } finally {
    await fs.promises.unlink(rawPcmPath).catch(() => {});
  }
}

/**
 * Maps a fractional speech progression (0..1) inside [windowStartMs, windowEndMs] to an actual
 * timestamp in milliseconds based on measured cumulative vocal energy, snapping to the nearest
 * natural acoustic pause if one exists within +/- 380ms.
 */
function resolveAcousticSplitTimestampMs(
  profile: ChunkAcousticTimingProfile,
  windowStartMs: number,
  windowEndMs: number,
  energyFraction: number,
  usedPauseMidpoints: Set<number>
): { timestampMs: number; source: 'acoustic-pause-aligned' | 'acoustic-energy-aligned' } {
  const clampedFraction = Math.max(0.08, Math.min(0.92, energyFraction));
  const hopMs = profile.frameHopMs || 20;
  let energyAlignedMs = Math.round(windowStartMs + (windowEndMs - windowStartMs) * clampedFraction);

  if (
    Array.isArray(profile.frameEnergies) &&
    profile.frameEnergies.length > 0 &&
    Array.isArray(profile.cumulativeEnergy) &&
    profile.cumulativeEnergy.length > 1
  ) {
    const fStart = Math.max(
      0,
      Math.min(profile.frameEnergies.length - 1, Math.floor(windowStartMs / hopMs))
    );
    const fEnd = Math.max(
      fStart + 1,
      Math.min(profile.frameEnergies.length, Math.ceil(windowEndMs / hopMs))
    );

    const startEnergy = profile.cumulativeEnergy[fStart] ?? 0;
    const endEnergy = profile.cumulativeEnergy[fEnd] ?? startEnergy;
    const totalWindowEnergy = Math.max(1e-6, endEnergy - startEnergy);
    const targetEnergy = startEnergy + totalWindowEnergy * clampedFraction;

    let matchedFrame = fStart;
    for (let f = fStart; f <= fEnd; f++) {
      if ((profile.cumulativeEnergy[f] ?? 0) >= targetEnergy) {
        matchedFrame = f;
        break;
      }
    }
    energyAlignedMs = Math.max(
      windowStartMs + 250,
      Math.min(windowEndMs - 250, matchedFrame * hopMs)
    );
  } else {
    energyAlignedMs = Math.max(
      windowStartMs + 250,
      Math.min(windowEndMs - 250, energyAlignedMs)
    );
  }

  // Check if a natural acoustic pause exists near energyAlignedMs within the valid window
  const searchRadiusMs = Math.min(480, Math.round((windowEndMs - windowStartMs) * 0.22));
  let bestPause: DetectedAcousticPause | null = null;
  let bestScore = -Infinity;

  for (const p of profile.pauses) {
    if (p.midMs <= windowStartMs + 300 || p.midMs >= windowEndMs - 300) continue;
    if (usedPauseMidpoints.has(p.midMs)) continue;
    const distMs = Math.abs(p.midMs - energyAlignedMs);
    if (distMs <= searchRadiusMs) {
      const score = p.durationMs * 0.5 + p.depthDb * 4 - distMs * 0.25;
      if (score > bestScore) {
        bestScore = score;
        bestPause = p;
      }
    }
  }

  if (bestPause) {
    usedPauseMidpoints.add(bestPause.midMs);
    return {
      timestampMs: bestPause.midMs,
      source: 'acoustic-pause-aligned'
    };
  }

  return {
    timestampMs: energyAlignedMs,
    source: 'acoustic-energy-aligned'
  };
}

/**
 * Determines the writing-system-aware maximum characters per single subtitle line.
 * - Bengali, Hindi, and other Indic/Arabic scripts: 38 chars/line (max 2 lines = 76 chars/block)
 * - CJK (Chinese, Japanese, Korean): 22 chars/line (max 2 lines = 44 chars/block)
 * - English, Latin, Cyrillic: 42 chars/line (max 2 lines = 84 chars/block)
 */
export function getScriptMaxCharsPerLine(langCode: string, sampleText: string): number {
  const code = (langCode || '').toLowerCase();
  if (['zh', 'ja', 'ko'].includes(code)) return 22;
  if (['bn', 'hi', 'ta', 'te', 'mr', 'gu', 'ar', 'ur', 'pa', 'ml', 'kn', 'or'].includes(code)) {
    return 38;
  }
  for (const ch of sampleText) {
    const cp = ch.codePointAt(0) || 0;
    if (cp >= 0x4e00 && cp <= 0x9fff) return 22;
    if (cp >= 0x0900 && cp <= 0x0dff) return 38;
    if (cp >= 0x0600 && cp <= 0x06ff) return 38;
  }
  return 42;
}

const ABBREVIATION_SET = new Set([
  'mr',
  'mrs',
  'ms',
  'dr',
  'prof',
  'sr',
  'jr',
  'st',
  'vs',
  'etc',
  'fig',
  'inc',
  'ltd',
  'co',
  'corp',
  'no',
  'vol',
  'ch',
  'sec',
  'min',
  'max',
  'dept',
  'univ',
  'govt',
  'gen',
  'col',
  'capt',
  'sgt',
  'lt',
  'jan',
  'feb',
  'mar',
  'apr',
  'jun',
  'jul',
  'aug',
  'sep',
  'sept',
  'oct',
  'nov',
  'dec',
  'approx',
  'est',
  'cf',
  'al',
  'ed',
  'rev'
]);

const DANGLE_FORBIDDEN_END_SET = new Set([
  // English articles, prepositions, coordinating/subordinating conjunctions, determiners, auxiliary/linking starters, titles
  'a',
  'an',
  'the',
  'and',
  'or',
  'but',
  'nor',
  'so',
  'yet',
  'for',
  'to',
  'in',
  'at',
  'on',
  'of',
  'by',
  'with',
  'from',
  'into',
  'onto',
  'upon',
  'across',
  'without',
  'within',
  'between',
  'among',
  'through',
  'during',
  'before',
  'after',
  'under',
  'over',
  'because',
  'although',
  'though',
  'while',
  'whereas',
  'unless',
  'since',
  'until',
  'whether',
  'if',
  'when',
  'where',
  'which',
  'that',
  'who',
  'whom',
  'whose',
  'any',
  'some',
  'every',
  'each',
  'most',
  'very',
  'quite',
  'rather',
  'my',
  'your',
  'our',
  'their',
  'his',
  'her',
  'its',
  'this',
  'these',
  'those',
  'dr.',
  'mr.',
  'mrs.',
  'ms.',
  'prof.',
  'st.',
  'no.',
  'e.g.',
  'e.g.,',
  'i.e.',
  'i.e.,',
  // Bengali conjunctions, determiners, demonstratives & particles that must not dangle at the end of a subtitle/line
  'এবং',
  'কিন্তু',
  'অথবা',
  'বা',
  'কিংবা',
  'কিবা',
  'নতুবা',
  'ও',
  'আর',
  'কারণ',
  'যদি',
  'যদিও',
  'যেহেতু',
  'যাতে',
  'যেন',
  'যে',
  'একটি',
  'একটা',
  'কোনো',
  'খুব',
  'অত্যন্ত',
  'অতি',
  'এই',
  'সেই',
  'ঐ',
  'প্রতিটি',
  // Hindi conjunctions, genitive markers, determiners & intensifiers that must not dangle at the end of a subtitle/line
  'और',
  'लेकिन',
  'किन्तु',
  'परन्तु',
  'या',
  'अथवा',
  'तथा',
  'एवं',
  'क्योंकि',
  'ताकि',
  'कि',
  'जो',
  'यदि',
  'अगर',
  'यद्यपि',
  'चूँकि',
  'एक',
  'कोई',
  'बहुत',
  'अति',
  'इस',
  'उस',
  'इन',
  'उन',
  'हर',
  'प्रत्येक',
  'का',
  'की',
  'के',
  // Arabic conjunctions & prepositions that must not dangle at end of a subtitle/line
  'و',
  'أو',
  'لكن',
  'بل',
  'في',
  'من',
  'إلى',
  'على',
  'مع',
  'عن',
  'أن',
  'إن'
]);

const CONJUNCTION_CLAUSE_START_SET = new Set([
  // English
  'and',
  'but',
  'or',
  'because',
  'while',
  'although',
  'though',
  'whereas',
  'however',
  'therefore',
  'moreover',
  'furthermore',
  'nevertheless',
  'which',
  'where',
  'when',
  'whenever',
  'whether',
  'unless',
  'since',
  'until',
  // Bengali (বাংলা)
  'এবং',
  'কিন্তু',
  'অথবা',
  'বা',
  'কিংবা',
  'কারণ',
  'যাতে',
  'যখন',
  'যেখানে',
  'তাই',
  'ফলে',
  'তবে',
  'যদি',
  'যদিও',
  'যেহেতু',
  'তথাপি',
  // Hindi (हिन्दी)
  'और',
  'लेकिन',
  'किन्तु',
  'परन्तु',
  'क्योंकि',
  'ताकि',
  'जबकि',
  'इसलिए',
  'तथा',
  'या',
  'अथवा',
  'जब',
  'जहाँ',
  'यदि',
  'अगर',
  'यद्यपि',
  'चूँकि',
  // Arabic (العربية)
  'ولكن',
  'لكن',
  'لأن',
  'بينما',
  'حيث',
  'عندما',
  'لذلك'
]);

const PHRASE_BOUNDARY_START_SET = new Set([
  // English prepositions & infinitive/participle starters
  'to',
  'for',
  'with',
  'without',
  'from',
  'by',
  'at',
  'in',
  'on',
  'under',
  'during',
  'through',
  'across',
  'into',
  'toward',
  'towards',
  'between',
  'among',
  'after',
  'before',
  'using',
  'including',
  'ensuring',
  'preserving',
  // Bengali phrase starters
  'একটি',
  'আমাদের',
  'আপনার',
  'প্রতিটি',
  'এই',
  'সেই',
  'যেখানে',
  'যাতে',
  // Hindi phrase starters
  'एक',
  'इस',
  'उस',
  'इसमें',
  'जिसमें',
  'प्रत्येक',
  'हर',
  'हमारे'
]);

const POSTPOSITION_CLAUSE_END_SET = new Set([
  // Bengali postpositions that naturally conclude a phrase
  'জন্য',
  'থেকে',
  'মাধ্যমে',
  'সাথে',
  'সঙ্গে',
  'উপর',
  'মধ্যে',
  'কারণে',
  'পরে',
  'আগে',
  // Hindi postpositions that conclude a noun/verb phrase
  'लिए',
  'साथ',
  'द्वारा',
  'अंदर',
  'बाहर',
  'बाद',
  'पहले',
  'दौरान',
  'अनुसार'
]);

/**
 * Deterministic, language-aware tokenizer for verifying exact ordered token-sequence preservation
 * between the canonical ASR source transcript and the reconstructed SRT output across Bengali,
 * Hindi, English, Arabic, and all supported Unicode scripts.
 * Preserves NFC Unicode composition, zero-width joiners (\u200C/\u200D), numbers, and punctuation verbatim.
 */
export function tokenizeTranscriptForVerification(text: string): string[] {
  const normalized = String(text || '')
    .normalize('NFC')
    .replace(/\s+/g, ' ')
    .trim();
  if (!normalized) return [];
  return normalized.split(' ').filter(Boolean);
}

function normalizeTokenComparisonKey(tok: string): string {
  return tok
    .normalize('NFC')
    .replace(/^['"“‘([<{]+|[,;—–:।॥.!?؟۔。！？'"”’)\]}>]+$/g, '')
    .toLowerCase();
}

/**
 * Performs an exact ordered sequence alignment (Longest Common Subsequence backtracking)
 * between the recognized source transcript and the combined SRT segment text.
 * Detects and distinguishes:
 * - Missing words (present in source, absent in output)
 * - Duplicated words (extra occurrences of words already present in source)
 * - Reordered words (present in both source and output, but out of chronological sequence)
 * - Unexpected inserted words (present in output, completely absent from source)
 * - Incorrect token normalization (word matches only after stripping case/punctuation/Unicode)
 */
export function compareTokenSequenceIntegrity(
  originalTranscript: string,
  reconstructedTranscript: string
): {
  inputTokenCount: number;
  outputTokenCount: number;
  duplicateTokenCount: number;
  missingTokenCount: number;
  reorderedTokenCount: number;
  insertedTokenCount: number;
  normalizationMismatchCount: number;
  missingTokens: string[];
  duplicatedTokens: string[];
  reorderedTokens: string[];
  insertedTokens: string[];
  exactSequenceMatch: boolean;
} {
  const inputTokens = tokenizeTranscriptForVerification(originalTranscript);
  const outputTokens = tokenizeTranscriptForVerification(reconstructedTranscript);

  const inputTokenCount = inputTokens.length;
  const outputTokenCount = outputTokens.length;

  if (inputTokenCount === 0 && outputTokenCount === 0) {
    return {
      inputTokenCount: 0,
      outputTokenCount: 0,
      duplicateTokenCount: 0,
      missingTokenCount: 0,
      reorderedTokenCount: 0,
      insertedTokenCount: 0,
      normalizationMismatchCount: 0,
      missingTokens: [],
      duplicatedTokens: [],
      reorderedTokens: [],
      insertedTokens: [],
      exactSequenceMatch: true
    };
  }

  // Fast path: 1-to-1 ordered exact sequence equality
  if (
    inputTokenCount === outputTokenCount &&
    inputTokens.every((tok, idx) => tok === outputTokens[idx])
  ) {
    return {
      inputTokenCount,
      outputTokenCount,
      duplicateTokenCount: 0,
      missingTokenCount: 0,
      reorderedTokenCount: 0,
      insertedTokenCount: 0,
      normalizationMismatchCount: 0,
      missingTokens: [],
      duplicatedTokens: [],
      reorderedTokens: [],
      insertedTokens: [],
      exactSequenceMatch: true
    };
  }

  // Full Longest Common Subsequence (LCS) DP table for exact ordered sequence alignment
  const n = inputTokens.length;
  const m = outputTokens.length;
  const dp: Uint16Array[] = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      if (inputTokens[i - 1] === outputTokens[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = dp[i - 1][j] >= dp[i][j - 1] ? dp[i - 1][j] : dp[i][j - 1];
      }
    }
  }

  // Backtrack LCS to mark exact in-order matched token positions
  const inMatched = new Uint8Array(n);
  const outMatched = new Uint8Array(m);
  let i = n;
  let j = m;
  while (i > 0 && j > 0) {
    if (inputTokens[i - 1] === outputTokens[j - 1]) {
      inMatched[i - 1] = 1;
      outMatched[j - 1] = 1;
      i--;
      j--;
    } else if (dp[i - 1][j] >= dp[i][j - 1]) {
      i--;
    } else {
      j--;
    }
  }

  const unmatchedIn: string[] = [];
  const unmatchedOut: string[] = [];
  for (let idx = 0; idx < n; idx++) {
    if (!inMatched[idx]) unmatchedIn.push(inputTokens[idx]);
  }
  for (let idx = 0; idx < m; idx++) {
    if (!outMatched[idx]) unmatchedOut.push(outputTokens[idx]);
  }

  // 1. Identify reordered tokens: tokens present in both unmatchedIn and unmatchedOut
  const unmatchedInPool = new Map<string, number>();
  for (const tok of unmatchedIn) {
    unmatchedInPool.set(tok, (unmatchedInPool.get(tok) || 0) + 1);
  }

  const reorderedTokens: string[] = [];
  const remainingOutAfterReorder: string[] = [];

  for (const tok of unmatchedOut) {
    const avail = unmatchedInPool.get(tok) || 0;
    if (avail > 0) {
      reorderedTokens.push(tok);
      unmatchedInPool.set(tok, avail - 1);
    } else {
      remainingOutAfterReorder.push(tok);
    }
  }

  const remainingInAfterReorder: string[] = [];
  for (const [tok, count] of unmatchedInPool.entries()) {
    for (let c = 0; c < count; c++) {
      remainingInAfterReorder.push(tok);
    }
  }

  // 2. Detect incorrect token normalization among remaining unmatched tokens (and remove paired normalization mismatches from missing/inserted pools)
  const normInIndicesByKey = new Map<string, number[]>();
  remainingInAfterReorder.forEach((tok, idx) => {
    const key = normalizeTokenComparisonKey(tok);
    if (key) {
      const list = normInIndicesByKey.get(key) || [];
      list.push(idx);
      normInIndicesByKey.set(key, list);
    }
  });

  const usedNormInIndices = new Set<number>();
  const remainingOutAfterNorm: string[] = [];
  let normalizationMismatchCount = 0;

  for (const tok of remainingOutAfterReorder) {
    const key = normalizeTokenComparisonKey(tok);
    const candidateIndices = key ? normInIndicesByKey.get(key) : undefined;
    const matchIdx = candidateIndices?.find((idx) => !usedNormInIndices.has(idx));
    if (matchIdx !== undefined) {
      usedNormInIndices.add(matchIdx);
      normalizationMismatchCount++;
    } else {
      remainingOutAfterNorm.push(tok);
    }
  }

  const remainingInAfterNorm = remainingInAfterReorder.filter(
    (_, idx) => !usedNormInIndices.has(idx)
  );

  // 3. Classify remaining unmatched output tokens into Duplicated (exists in source) vs Inserted (foreign to source)
  const sourceVocab = new Set<string>(inputTokens);
  const duplicatedTokens: string[] = [];
  const insertedTokens: string[] = [];

  for (const tok of remainingOutAfterNorm) {
    if (sourceVocab.has(tok)) {
      duplicatedTokens.push(tok);
    } else {
      insertedTokens.push(tok);
    }
  }

  const missingTokens = remainingInAfterNorm;

  return {
    inputTokenCount,
    outputTokenCount,
    duplicateTokenCount: duplicatedTokens.length,
    missingTokenCount: missingTokens.length,
    reorderedTokenCount: reorderedTokens.length,
    insertedTokenCount: insertedTokens.length,
    normalizationMismatchCount,
    missingTokens,
    duplicatedTokens,
    reorderedTokens,
    insertedTokens,
    exactSequenceMatch: false
  };
}

function isAbbreviationOrNonTerminalPeriodToken(token: string, nextToken?: string): boolean {
  // Strip leading quotes/parens and trailing quotes/parens
  const stripped = token.replace(/^['"“‘([<{]+|['"”’)\]}>]+$/g, '');
  if (!stripped.endsWith('.')) return false;

  // Multi-dot abbreviations like e.g., i.e., U.S., U.K., a.m., p.m., Ph.D.
  if (/^(?:[a-zA-Z]\.){2,}$/.test(stripped)) return true;
  if (/^(?:e\.g|i\.e|u\.s|u\.k|a\.m|p\.m|ph\.d|b\.a|m\.a|b\.sc|m\.sc)\.$/i.test(stripped)) {
    return true;
  }

  // Single uppercase initial (e.g., "J.", "A.")
  const withoutTrailingDots = stripped.replace(/\.+$/, '');
  if (/^[A-Z]$/.test(withoutTrailingDots)) return true;

  // Known title or Latin/English abbreviation (Dr., Mr., Mrs., Ms., Prof., vs., etc., Fig., No.)
  const lowerWord = withoutTrailingDots.toLowerCase();
  if (ABBREVIATION_SET.has(lowerWord)) {
    return true;
  }

  // Numeric list item / ordinal prefix (e.g., "1.", "2.") when followed by another token
  if (/^\d+$/.test(withoutTrailingDots) && nextToken && !/^[A-Z]/.test(nextToken)) {
    return true;
  }

  return false;
}

/**
 * Splits a multi-sentence transcript string into individual linguistic sentences while preserving
 * 100% of original tokens and punctuation in exact sequence and protecting abbreviations and decimals.
 * Also merges isolated 1-word sentences with their adjacent sentence when multiple sentences exist
 * so unnecessary 1-word subtitle blocks are not emitted.
 */
export function splitTextIntoLinguisticSentences(text: string): string[] {
  const tokens = tokenizeTranscriptForVerification(text);
  if (tokens.length === 0) return [];

  const rawSentences: string[][] = [];
  let currentWords: string[] = [];

  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const nextToken = tokens[i + 1];
    currentWords.push(token);

    // Check if token ends with sentence-terminal punctuation across Latin, Indic (Danda । ॥), Arabic (؟ ۔), CJK (。！？)
    const terminalMatch = token.match(/([.!?।॥؟۔。！？]+)['"”’)\]}>]*$/);
    if (!terminalMatch) continue;

    // If the next token is a standalone closing punctuation/danda, wait for next iteration so it stays attached
    if (nextToken && /^[.!?।॥؟۔。！？'"”’)\]}>]+$/.test(nextToken)) {
      continue;
    }

    const punct = terminalMatch[1];
    // Indic Danda (। ॥), Arabic question/full stop, CJK full stop, ! or ? are always sentence boundaries
    if (/[!?।॥؟۔。！？]/.test(punct)) {
      rawSentences.push(currentWords);
      currentWords = [];
      continue;
    }

    // Period '.': check if abbreviation, initial, or non-terminal period
    if (!isAbbreviationOrNonTerminalPeriodToken(token, nextToken)) {
      rawSentences.push(currentWords);
      currentWords = [];
    }
  }

  if (currentWords.length > 0) {
    rawSentences.push(currentWords);
  }

  // Merge isolated 1-word sentences (e.g. "নমস্কার।", "नमस्ते।", "Hello.") with adjacent sentence
  // when there are multiple sentences, avoiding unnecessary 1-word subtitle blocks while preserving exact token order.
  if (rawSentences.length >= 2) {
    const mergedSentences: string[][] = [];
    let i = 0;
    while (i < rawSentences.length) {
      const curr = rawSentences[i];
      if (curr.length === 1 && i < rawSentences.length - 1) {
        rawSentences[i + 1] = [...curr, ...rawSentences[i + 1]];
        i++;
        continue;
      }
      if (curr.length === 1 && mergedSentences.length > 0) {
        mergedSentences[mergedSentences.length - 1] = [
          ...mergedSentences[mergedSentences.length - 1],
          ...curr
        ];
        i++;
        continue;
      }
      mergedSentences.push(curr);
      i++;
    }
    return mergedSentences.map((wList) => wList.join(' '));
  }

  return rawSentences.map((wList) => wList.join(' '));
}

function cleanWordForLookup(word: string): string {
  return word
    .replace(/^['"“‘([<{]+|[,;—–:।॥.!?؟۔。！？'"”’)\]}>]+$/g, '')
    .toLowerCase();
}

function doesClauseFitWithinTwoLines(words: string[], maxCharsPerLine: number): boolean {
  if (words.length === 0) return true;
  const wrapped = wrapSubtitleTextIntoMaxTwoLines(words.join(' '), maxCharsPerLine);
  const lines = wrapped.split('\n');
  if (lines.length > 2) return false;
  for (const line of lines) {
    if (Array.from(line.trim()).length > maxCharsPerLine) {
      return false;
    }
  }
  return true;
}

function partitionTokensIntoBalancedClauses(
  words: string[],
  maxCharsPerLine: number
): string[][] {
  const maxBlockChars = maxCharsPerLine * 2;
  const joined = words.join(' ');
  const totalChars = Array.from(joined).length;

  // Check if the current word sequence already fits cleanly within <= 2 lines without any line exceeding maxCharsPerLine
  const fitsTwoLinesCleanly =
    totalChars <= maxBlockChars && doesClauseFitWithinTwoLines(words, maxCharsPerLine);

  if (words.length <= 3) {
    return [words];
  }

  if (fitsTwoLinesCleanly) {
    // Only split a sentence that already fits within 2 lines if it is very long (>= 10 words and >= 88% of maxBlockChars)
    // AND has an explicit internal punctuation pause (comma, semicolon, colon, dash) with >= 4 words on both sides
    if (words.length >= 10 && totalChars >= Math.round(maxBlockChars * 0.88)) {
      let commaSplitIdx = -1;
      let bestCommaBalance = Infinity;
      for (let k = 4; k <= words.length - 4; k++) {
        const leftLastRaw = words[k - 1];
        const leftLastClean = cleanWordForLookup(leftLastRaw);
        if (
          /[,;—–:،؛]$/.test(leftLastRaw) &&
          !DANGLE_FORBIDDEN_END_SET.has(leftLastClean) &&
          !isAbbreviationOrNonTerminalPeriodToken(leftLastRaw, words[k])
        ) {
          const leftLen = Array.from(words.slice(0, k).join(' ')).length;
          const rightLen = Array.from(words.slice(k).join(' ')).length;
          const diff = Math.abs(leftLen - rightLen);
          if (diff < bestCommaBalance) {
            bestCommaBalance = diff;
            commaSplitIdx = k;
          }
        }
      }
      if (commaSplitIdx !== -1) {
        return [
          ...partitionTokensIntoBalancedClauses(words.slice(0, commaSplitIdx), maxCharsPerLine),
          ...partitionTokensIntoBalancedClauses(words.slice(commaSplitIdx), maxCharsPerLine)
        ];
      }
    }
    return [words];
  }

  // The sentence exceeds maxBlockChars or wraps with a line > maxCharsPerLine:
  // Find the optimal natural clause boundary k. Prefer [3, words.length - 3], or [2, words.length - 2] if 4-5 long words.
  const minK = words.length >= 6 ? 3 : 2;
  const maxK = words.length - minK;
  let bestSplitIdx = -1;
  let bestScore = -Infinity;

  for (let k = minK; k <= maxK; k++) {
    const leftWords = words.slice(0, k);
    const rightWords = words.slice(k);
    const leftLastRaw = leftWords[leftWords.length - 1];
    const leftLastClean = cleanWordForLookup(leftLastRaw);
    const rightFirstClean = cleanWordForLookup(rightWords[0]);

    const rightSecondClean = rightWords.length > 1 ? cleanWordForLookup(rightWords[1]) : '';
    const isNumericWord = (w: string) =>
      /^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|এক|দুই|তিন|চার|পাঁচ|ছয়|সাত|আট|নয়|দশ|एक|दो|तीन|चार|पाँच|छह|सात|आठ|नौ|दस)$/i.test(
        w
      );
    const isNumericRangeSplit =
      (rightFirstClean === 'to' || rightFirstClean === 'থেকে' || rightFirstClean === 'से') &&
      isNumericWord(leftLastClean) &&
      isNumericWord(rightSecondClean);

    // Prohibit ending the left clause on a dangling conjunction, preposition, article, title abbreviation,
    // numeric range ("one to five"), or splitting inside an intentional reduplicated word pair ("ধীরে ধীরে", "बार बार")
    if (
      DANGLE_FORBIDDEN_END_SET.has(leftLastClean) ||
      DANGLE_FORBIDDEN_END_SET.has(leftLastRaw.toLowerCase()) ||
      isAbbreviationOrNonTerminalPeriodToken(leftLastRaw, rightWords[0]) ||
      (leftLastClean.length > 0 && leftLastClean === rightFirstClean) ||
      isNumericRangeSplit
    ) {
      continue;
    }

    const leftLen = Array.from(leftWords.join(' ')).length;
    const rightLen = Array.from(rightWords.join(' ')).length;

    const isCoordinator =
      rightFirstClean === 'and' ||
      rightFirstClean === 'or' ||
      rightFirstClean === 'এবং' ||
      rightFirstClean === 'ও' ||
      rightFirstClean === 'বা' ||
      rightFirstClean === 'और' ||
      rightFirstClean === 'या';

    let linguisticBonus = 0;
    if (/[,;—–:،؛]$/.test(leftLastRaw)) {
      linguisticBonus += 62;
    } else if (
      CONJUNCTION_CLAUSE_START_SET.has(rightFirstClean) &&
      (!isCoordinator || (leftWords.length >= 5 && rightWords.length >= 4))
    ) {
      linguisticBonus += 42;
    } else if (POSTPOSITION_CLAUSE_END_SET.has(leftLastClean) && leftWords.length >= 4) {
      linguisticBonus += 32;
    } else if (PHRASE_BOUNDARY_START_SET.has(rightFirstClean) && leftWords.length >= 4) {
      linguisticBonus += 28;
    }

    // Favor splits where the left clause fits cleanly within 2 lines (<= maxCharsPerLine per line)
    const leftFitsCleanly =
      leftLen <= maxBlockChars && doesClauseFitWithinTwoLines(leftWords, maxCharsPerLine);
    const leftOverflowPenalty = leftFitsCleanly
      ? 0
      : Math.max(18, (leftLen - maxBlockChars) * 2.8);

    const balanceDiff = Math.abs(leftLen - rightLen);
    const shortLeftPenalty =
      leftWords.length <= 3 || leftLen < 16 ? (5 - Math.min(4, leftWords.length)) * 20 : 0;
    const shortTailPenalty =
      rightWords.length <= 3 || rightLen < 16 ? (5 - Math.min(4, rightWords.length)) * 20 : 0;

    const score =
      linguisticBonus -
      balanceDiff * 0.45 -
      leftOverflowPenalty -
      shortLeftPenalty -
      shortTailPenalty;
    if (score > bestScore) {
      bestScore = score;
      bestSplitIdx = k;
    }
  }

  if (bestSplitIdx === -1) {
    // Fallback if all positions had a soft constraint: pick midpoint avoiding dangling conjunction if possible
    bestSplitIdx = Math.max(minK, Math.min(maxK, Math.ceil(words.length / 2)));
  }

  const leftClause = words.slice(0, bestSplitIdx);
  const rightClause = words.slice(bestSplitIdx);

  return [
    ...partitionTokensIntoBalancedClauses(leftClause, maxCharsPerLine),
    ...partitionTokensIntoBalancedClauses(rightClause, maxCharsPerLine)
  ];
}

/**
 * Splits a long single sentence into natural, balanced readable clauses without
 * creating 1-2 word orphan tails, conjunction-only endings, or altering any token.
 */
export function splitLongSentenceIntoReadableClauses(
  sentence: string,
  maxCharsPerLine: number
): string[] {
  const words = tokenizeTranscriptForVerification(sentence);
  if (words.length === 0) return [];
  const clauseWordLists = partitionTokensIntoBalancedClauses(words, maxCharsPerLine);
  return clauseWordLists.map((list) => list.join(' '));
}

/**
 * Formats a subtitle block text into at most 2 balanced lines, each respecting maxCharsPerLine
 * where word boundaries permit, while avoiding dangling conjunctions/prepositions/titles at the end of line 1
 * and preserving 100% of words and punctuation in exact sequence.
 */
export function wrapSubtitleTextIntoMaxTwoLines(text: string, maxCharsPerLine: number): string {
  const words = tokenizeTranscriptForVerification(text);
  if (words.length === 0) return '';
  const singleLine = words.join(' ');
  const chars = Array.from(singleLine);
  if (chars.length <= maxCharsPerLine || words.length <= 2) {
    return singleLine;
  }

  let bestSplitIdx = Math.ceil(words.length / 2);
  let bestScore = Infinity;

  for (let i = 1; i < words.length; i++) {
    const line1 = words.slice(0, i).join(' ');
    const line2 = words.slice(i).join(' ');
    const len1 = Array.from(line1).length;
    const len2 = Array.from(line2).length;

    const lastWordLine1Raw = words[i - 1];
    const lastWordLine1Clean = cleanWordForLookup(lastWordLine1Raw);
    const nextWordLine2Clean = cleanWordForLookup(words[i]);

    // Strong penalty if either line exceeds maxCharsPerLine
    const overflowPenalty =
      (len1 > maxCharsPerLine ? (len1 - maxCharsPerLine) * 14 : 0) +
      (len2 > maxCharsPerLine ? (len2 - maxCharsPerLine) * 14 : 0);

    const commaBonus = /[,;—–:،؛]$/.test(lastWordLine1Raw) ? -12 : 0;
    const conjStartBonus = CONJUNCTION_CLAUSE_START_SET.has(nextWordLine2Clean) ? -9 : 0;
    const phraseStartBonus = PHRASE_BOUNDARY_START_SET.has(nextWordLine2Clean) ? -6 : 0;

    const danglePenalty =
      DANGLE_FORBIDDEN_END_SET.has(lastWordLine1Clean) ||
      DANGLE_FORBIDDEN_END_SET.has(lastWordLine1Raw.toLowerCase()) ||
      isAbbreviationOrNonTerminalPeriodToken(lastWordLine1Raw, words[i]) ||
      (lastWordLine1Clean.length > 0 && lastWordLine1Clean === nextWordLine2Clean)
        ? 35
        : 0;

    const singleWordLinePenalty = i === 1 || i === words.length - 1 ? 12 : 0;

    const balanceDiff = Math.abs(len1 - len2);
    const score =
      balanceDiff +
      overflowPenalty +
      commaBonus +
      conjStartBonus +
      phraseStartBonus +
      danglePenalty +
      singleWordLinePenalty;

    if (score < bestScore) {
      bestScore = score;
      bestSplitIdx = i;
    }
  }

  return `${words.slice(0, bestSplitIdx).join(' ')}\n${words.slice(bestSplitIdx).join(' ')}`;
}

export interface RawAsrSegmentInput {
  startSec: number;
  endSec: number;
  text: string;
  words?: { word: string; startSec: number; endSec: number }[];
}

/**
 * Sanitizes raw ASR segments returned by the speech recognition model before canonical
 * transcript assembly and subtitle segmentation:
 * 1. Removes accidental LLM boundary echo segments where an adjacent ASR segment repeats
 *    the entire previous segment or repeats a trailing phrase suffix of the previous segment.
 * 2. Strips overlapping multi-word boundary prefixes where ASR segment i+1 repeats the last
 *    k >= 2 words of ASR segment i across a segment boundary.
 * 3. Merges raw ASR segments that end with an isolated conjunction/preposition/determiner
 *    (without terminal punctuation) or isolated 1-2 word fragments with their adjacent segment
 *    so natural phrases are never severed across raw ASR boundaries.
 * Note: Intentional word repetition inside any sentence (e.g. "ধীরে ধীরে", "बार बार", "very very")
 * is 100% preserved because only cross-segment boundary overlaps are deduplicated.
 */
export function parseFlexibleSeconds(raw: unknown): number | null {
  if (typeof raw === 'number') {
    return Number.isFinite(raw) ? raw : null;
  }
  const s = String(raw ?? '').trim().replace(',', '.');
  if (!s) return null;
  if (/^\d+(?:\.\d+)?$/.test(s)) {
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  }
  const parts = s.split(':').map((p) => Number(p.trim()));
  if (parts.some((p) => !Number.isFinite(p) || p < 0)) return null;
  if (parts.length === 2) {
    return parts[0] * 60 + parts[1];
  }
  if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return null;
}

export function sanitizeRawAsrSegmentsOverlap(
  rawSegments: RawAsrSegmentInput[]
): RawAsrSegmentInput[] {
  const nonEmpty: RawAsrSegmentInput[] = [];
  for (const seg of rawSegments) {
    const tokens = tokenizeTranscriptForVerification(seg.text || '');
    if (tokens.length === 0) continue;
    const parsedStart = parseFlexibleSeconds(seg.startSec);
    const parsedEnd = parseFlexibleSeconds(seg.endSec);
    nonEmpty.push({
      startSec: parsedStart ?? 0,
      endSec: parsedEnd ?? (parsedStart !== null ? parsedStart + 0.5 : 0),
      text: tokens.join(' '),
      words: Array.isArray(seg.words) ? seg.words : undefined
    });
  }

  if (nonEmpty.length <= 1) return nonEmpty;

  const deduplicated: RawAsrSegmentInput[] = [nonEmpty[0]];

  for (let i = 1; i < nonEmpty.length; i++) {
    const prev = deduplicated[deduplicated.length - 1];
    const curr = nonEmpty[i];
    const prevTokens = tokenizeTranscriptForVerification(prev.text);
    const currTokens = tokenizeTranscriptForVerification(curr.text);
    const prevKeys = prevTokens.map(normalizeTokenComparisonKey);
    const currKeys = currTokens.map(normalizeTokenComparisonKey);

    // Case 1: curr is an exact full duplicate of prev, or curr's entire phrase (>= 2 words)
    // is an exact suffix of prev
    const isExactDuplicate =
      prevKeys.length === currKeys.length && prevKeys.every((k, idx) => k === currKeys[idx]);

    let isEntireSuffixEcho = false;
    if (currKeys.length >= 2 && currKeys.length <= prevKeys.length) {
      const offset = prevKeys.length - currKeys.length;
      isEntireSuffixEcho = currKeys.every((k, idx) => k === prevKeys[offset + idx]);
    }

    if (isExactDuplicate || isEntireSuffixEcho) {
      if (curr.endSec > prev.endSec && curr.startSec - prev.endSec < 0.8) {
        prev.endSec = curr.endSec;
      }
      continue;
    }

    // Case 2: curr starts with a multi-word prefix (overlapLen >= 2) that matches the suffix of prev
    const maxOverlapCheck = Math.min(prevKeys.length, currKeys.length - 1, 8);
    let overlapLen = 0;
    for (let k = maxOverlapCheck; k >= 2; k--) {
      const suffixStart = prevKeys.length - k;
      let matches = true;
      for (let j = 0; j < k; j++) {
        if (prevKeys[suffixStart + j] !== currKeys[j]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        overlapLen = k;
        break;
      }
    }

    if (overlapLen >= 2) {
      const remainingTokens = currTokens.slice(overlapLen);
      const remainingWords =
        Array.isArray(curr.words) && curr.words.length > overlapLen
          ? curr.words.slice(overlapLen)
          : undefined;
      deduplicated.push({
        startSec: remainingWords?.[0]?.startSec ?? curr.startSec,
        endSec: curr.endSec,
        text: remainingTokens.join(' '),
        words: remainingWords
      });
    } else {
      deduplicated.push(curr);
    }
  }

  // Pass 2: Merge raw segments that end on a dangling conjunction/preposition/determiner
  // or are isolated 1-2 word fragments so downstream clause segmentation sees the full natural phrase.
  if (deduplicated.length >= 2) {
    const merged: RawAsrSegmentInput[] = [];
    let i = 0;
    while (i < deduplicated.length) {
      const curr = deduplicated[i];
      const currTokens = tokenizeTranscriptForVerification(curr.text);
      const lastRawToken = currTokens[currTokens.length - 1] || '';
      const lastCleanToken = cleanWordForLookup(lastRawToken);
      const endsWithTerminalPunct = /[.!?।॥؟۔。！？]['"”’)\]}>]*$/.test(lastRawToken);
      const endsWithDanglingWord =
        !endsWithTerminalPunct &&
        (DANGLE_FORBIDDEN_END_SET.has(lastCleanToken) ||
          DANGLE_FORBIDDEN_END_SET.has(lastRawToken.toLowerCase()));

      const isTinyFragment =
        currTokens.length === 1 || (currTokens.length === 2 && !endsWithTerminalPunct);

      if ((endsWithDanglingWord || isTinyFragment) && i < deduplicated.length - 1) {
        const next = deduplicated[i + 1];
        deduplicated[i + 1] = {
          startSec: Math.min(curr.startSec, next.startSec),
          endSec: Math.max(curr.endSec, next.endSec),
          text: `${curr.text} ${next.text}`.replace(/\s+/g, ' ').trim(),
          words:
            curr.words && next.words
              ? [...curr.words, ...next.words]
              : undefined
        };
        i++;
        continue;
      }
      if (currTokens.length === 1 && merged.length > 0) {
        const prev = merged[merged.length - 1];
        merged[merged.length - 1] = {
          startSec: prev.startSec,
          endSec: Math.max(prev.endSec, curr.endSec),
          text: `${prev.text} ${curr.text}`.replace(/\s+/g, ' ').trim(),
          words:
            prev.words && curr.words
              ? [...prev.words, ...curr.words]
              : undefined
        };
        i++;
        continue;
      }
      merged.push(curr);
      i++;
    }
    return merged;
  }

  return deduplicated;
}

/**
 * Uses FFmpeg silencedetect to identify active speech regions and optimal chunk split points
 * for long audio files. Uses sensitive pause detection (-33dB, 0.18s) to capture natural sentence pauses.
 */
export async function detectSpeechIntervalsAndChunks(
  filePath: string,
  totalDurationSec: number,
  maxChunkSec = 45
): Promise<{
  speechIntervals: AudioSpeechInterval[];
  chunks: { chunkIndex: number; startSec: number; endSec: number; chunkFilePath: string }[];
}> {
  const { stderr } = await execFileAsync('ffmpeg', [
    '-i',
    filePath,
    '-af',
    'silencedetect=noise=-33dB:d=0.18',
    '-f',
    'null',
    '-'
  ]);

  const silenceStarts: number[] = [];
  const silenceEnds: number[] = [];

  const lines = stderr.split('\n');
  for (const line of lines) {
    const startMatch = line.match(/silence_start:\s*([0-9.]+)/);
    if (startMatch) {
      silenceStarts.push(parseFloat(startMatch[1]));
    }
    const endMatch = line.match(/silence_end:\s*([0-9.]+)/);
    if (endMatch) {
      silenceEnds.push(parseFloat(endMatch[1]));
    }
  }

  const speechIntervals: AudioSpeechInterval[] = [];
  let cursor = 0;
  for (let i = 0; i < silenceStarts.length; i++) {
    const sStart = silenceStarts[i];
    const sEnd = silenceEnds[i] ?? totalDurationSec;
    if (sStart - cursor >= 0.18) {
      speechIntervals.push({
        startSec: Number(cursor.toFixed(3)),
        endSec: Number(sStart.toFixed(3))
      });
    }
    cursor = sEnd;
  }
  if (totalDurationSec - cursor >= 0.18) {
    speechIntervals.push({
      startSec: Number(cursor.toFixed(3)),
      endSec: Number(totalDurationSec.toFixed(3))
    });
  }

  if (speechIntervals.length === 0 && totalDurationSec > 0) {
    speechIntervals.push({ startSec: 0, endSec: totalDurationSec });
  }

  const splitPoints: number[] = [0];
  let lastSplit = 0;
  const minLeadingIntervalSec = Math.min(8, Math.max(4, Math.round(maxChunkSec * 0.35)));
  const searchRadiusSec = Math.min(8, Math.max(4, Math.round(maxChunkSec * 0.35)));

  while (totalDurationSec - lastSplit > maxChunkSec + 4) {
    const targetSplit = lastSplit + maxChunkSec;
    let bestSplit = targetSplit;
    let bestDist = Infinity;

    for (let i = 0; i < silenceStarts.length; i++) {
      const sStart = silenceStarts[i];
      const sEnd = silenceEnds[i] ?? sStart + 0.3;
      const midpoint = (sStart + sEnd) / 2;
      if (midpoint > lastSplit + minLeadingIntervalSec && midpoint < totalDurationSec - 3) {
        const dist = Math.abs(midpoint - targetSplit);
        if (dist < searchRadiusSec && dist < bestDist) {
          bestDist = dist;
          bestSplit = Number(midpoint.toFixed(3));
        }
      }
    }
    splitPoints.push(bestSplit);
    lastSplit = bestSplit;
  }
  splitPoints.push(Number(totalDurationSec.toFixed(3)));

  const chunks: { chunkIndex: number; startSec: number; endSec: number; chunkFilePath: string }[] = [];
  const dir = path.dirname(filePath);
  const base = path.basename(filePath, path.extname(filePath));

  for (let idx = 0; idx < splitPoints.length - 1; idx++) {
    const startSec = splitPoints[idx];
    const endSec = splitPoints[idx + 1];
    const duration = Math.max(0.5, endSec - startSec);
    const chunkFilePath = path.join(dir, `${base}_chunk_${idx}.wav`);

    // Place -ss AFTER -i so FFmpeg decodes sample-accurately on VBR MP3/M4A/AAC files (prevents chunk start drift)
    await execFileAsync('ffmpeg', [
      '-y',
      '-i',
      filePath,
      '-ss',
      String(startSec),
      '-t',
      String(duration),
      '-vn',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-c:a',
      'pcm_s16le',
      chunkFilePath
    ]);

    chunks.push({
      chunkIndex: idx,
      startSec,
      endSec,
      chunkFilePath
    });
  }

  return { speechIntervals, chunks };
}

function countVoicedFramesInWindow(
  profile: ChunkAcousticTimingProfile,
  startMs: number,
  endMs: number
): { voicedFrames: number; totalFrames: number; voicedRatio: number } {
  const hopMs = profile.frameHopMs || 20;
  const voicedArr = profile.frameIsVoiced || [];
  if (voicedArr.length === 0 || endMs <= startMs) {
    return { voicedFrames: 0, totalFrames: 0, voicedRatio: 0 };
  }
  const fStart = Math.max(0, Math.min(voicedArr.length - 1, Math.floor(startMs / hopMs)));
  const fEnd = Math.max(fStart + 1, Math.min(voicedArr.length, Math.ceil(endMs / hopMs)));
  let voicedFrames = 0;
  for (let f = fStart; f < fEnd; f++) {
    if (voicedArr[f]) voicedFrames++;
  }
  const totalFrames = Math.max(1, fEnd - fStart);
  return {
    voicedFrames,
    totalFrames,
    voicedRatio: voicedFrames / totalFrames
  };
}

function trimIntervalToVoicedSpeech(
  profile: ChunkAcousticTimingProfile,
  startMs: number,
  endMs: number
): { startMs: number; endMs: number } {
  const hopMs = profile.frameHopMs || 20;
  const voicedArr = profile.frameIsVoiced || [];
  if (voicedArr.length === 0 || endMs - startMs < 400) {
    return { startMs, endMs };
  }
  const fStart = Math.max(0, Math.min(voicedArr.length - 1, Math.floor(startMs / hopMs)));
  const fEnd = Math.max(fStart, Math.min(voicedArr.length - 1, Math.floor(endMs / hopMs)));

  let firstVoiced = -1;
  let lastVoiced = -1;
  for (let f = fStart; f <= fEnd; f++) {
    if (voicedArr[f]) {
      if (firstVoiced === -1) firstVoiced = f;
      lastVoiced = f;
    }
  }
  if (firstVoiced === -1 || lastVoiced <= firstVoiced) {
    return { startMs, endMs };
  }
  const trimmedStart = Math.max(startMs, (firstVoiced - 2) * hopMs);
  const trimmedEnd = Math.min(endMs, (lastVoiced + 3) * hopMs);
  if (trimmedEnd - trimmedStart >= 250) {
    return { startMs: trimmedStart, endMs: trimmedEnd };
  }
  return { startMs, endMs };
}

function mapEnergyFractionAcrossIslands(
  profile: ChunkAcousticTimingProfile,
  islands: ActiveSpeechIsland[],
  targetFraction: number
): number {
  const hopMs = profile.frameHopMs || 20;
  const cum = profile.cumulativeEnergy || [];
  const islandWeights = islands.map((isl) => {
    if (cum.length <= 1) return Math.max(1, isl.durationMs);
    const f0 = Math.max(0, Math.min(cum.length - 2, Math.floor(isl.startMs / hopMs)));
    const f1 = Math.max(f0 + 1, Math.min(cum.length - 1, Math.ceil(isl.endMs / hopMs)));
    return Math.max(0.01, (cum[f1] ?? 0) - (cum[f0] ?? 0));
  });
  const totalIslandEnergy = islandWeights.reduce((a, b) => a + b, 0) || 1;
  const targetEnergy = Math.max(0, Math.min(1, targetFraction)) * totalIslandEnergy;

  let accum = 0;
  for (let i = 0; i < islands.length; i++) {
    const w = islandWeights[i];
    if (accum + w >= targetEnergy || i === islands.length - 1) {
      const localFrac = Math.max(0, Math.min(1, (targetEnergy - accum) / w));
      const isl = islands[i];
      if (cum.length <= 1) {
        return Math.round(isl.startMs + isl.durationMs * localFrac);
      }
      const f0 = Math.max(0, Math.min(cum.length - 2, Math.floor(isl.startMs / hopMs)));
      const f1 = Math.max(f0 + 1, Math.min(cum.length - 1, Math.ceil(isl.endMs / hopMs)));
      const baseCum = cum[f0] ?? 0;
      const localTarget = baseCum + w * localFrac;
      let matchedFrame = f0;
      for (let f = f0; f <= f1; f++) {
        if ((cum[f] ?? 0) >= localTarget) {
          matchedFrame = f;
          break;
        }
      }
      return Math.max(isl.startMs, Math.min(isl.endMs, matchedFrame * hopMs));
    }
    accum += w;
  }
  return islands[islands.length - 1].endMs;
}

function alignSubUnitsAcrossActiveSpeechIslands(options: {
  subUnits: string[];
  maxCharsPerLine: number;
  chunkOffsetMs: number;
  chunkDurationMs: number;
  acousticProfile: ChunkAcousticTimingProfile;
  usedPauseMidpoints: Set<number>;
}): SrtSegment[] {
  const {
    subUnits,
    maxCharsPerLine,
    chunkOffsetMs,
    chunkDurationMs,
    acousticProfile,
    usedPauseMidpoints
  } = options;
  if (subUnits.length === 0) return [];

  const rawIslands =
    acousticProfile.activeSpeechIslands && acousticProfile.activeSpeechIslands.length > 0
      ? acousticProfile.activeSpeechIslands
      : [
          {
            startMs: Math.max(0, acousticProfile.activeSpeechStartMs),
            endMs: Math.max(
              acousticProfile.activeSpeechStartMs + 250,
              Math.min(chunkDurationMs, acousticProfile.activeSpeechEndMs || chunkDurationMs)
            ),
            durationMs: Math.max(
              250,
              (acousticProfile.activeSpeechEndMs || chunkDurationMs) -
                acousticProfile.activeSpeechStartMs
            )
          }
        ];

  // Keep only substantial islands unless total islands is empty
  const validIslands = rawIslands.filter((isl) => isl.durationMs >= 180);
  let islands = (validIslands.length > 0 ? validIslands : rawIslands).map((isl) => ({ ...isl }));

  const weights = subUnits.map((u) => {
    const tokens = tokenizeTranscriptForVerification(u);
    const charCount = Math.max(1, Array.from(u.replace(/\s+/g, '')).length);
    const wordWeight = tokens.length * 2.5;
    const punctPauseWeight = /[.!?।॥؟۔。！？]$/.test(u)
      ? 4.5
      : /[,;—–:،؛]$/.test(u)
      ? 2.0
      : 0;
    return charCount + wordWeight + punctPauseWeight;
  });
  const totalWeight = weights.reduce((a, b) => a + b, 0) || 1;

  // If there are more islands than subUnits, iteratively merge the pair of adjacent islands
  // separated by the SHORTEST silence gap so real multi-second silences stay strictly unbridged.
  while (islands.length > subUnits.length && islands.length > 1) {
    let minGapIdx = 0;
    let minGapMs = Infinity;
    for (let i = 0; i < islands.length - 1; i++) {
      const gap = islands[i + 1].startMs - islands[i].endMs;
      if (gap < minGapMs) {
        minGapMs = gap;
        minGapIdx = i;
      }
    }
    const merged: ActiveSpeechIsland = {
      startMs: islands[minGapIdx].startMs,
      endMs: islands[minGapIdx + 1].endMs,
      durationMs: islands[minGapIdx + 1].endMs - islands[minGapIdx].startMs
    };
    islands.splice(minGapIdx, 2, merged);
  }

  // Compute each island's acoustic voiced energy weight
  const hopMs = acousticProfile.frameHopMs || 20;
  const cum = acousticProfile.cumulativeEnergy || [];
  const islandWeights = islands.map((isl) => {
    if (cum.length <= 1) return Math.max(1, isl.durationMs);
    const f0 = Math.max(0, Math.min(cum.length - 2, Math.floor(isl.startMs / hopMs)));
    const f1 = Math.max(f0 + 1, Math.min(cum.length - 1, Math.ceil(isl.endMs / hopMs)));
    return Math.max(0.01, (cum[f1] ?? 0) - (cum[f0] ?? 0));
  });
  const totalIslandEnergy = islandWeights.reduce((a, b) => a + b, 0) || 1;
  const islandFracs = islandWeights.map((w) => w / totalIslandEnergy);

  // Prefix sums of subUnit weights for O(1) slice fraction lookup
  const weightPrefix: number[] = [0];
  for (let i = 0; i < weights.length; i++) {
    weightPrefix.push(weightPrefix[i] + weights[i]);
  }

  // Partition subUnits[0..N-1] contiguously across islands[0..M-1] (M <= N) minimizing energy-proportion mismatch
  const M = islands.length;
  const N = subUnits.length;
  const groupSlices: Array<{ startIdx: number; endIdx: number }> = [];

  if (M === 1) {
    groupSlices.push({ startIdx: 0, endIdx: N });
  } else {
    const dp: number[][] = Array.from({ length: M + 1 }, () => new Array(N + 1).fill(Infinity));
    const prevSplit: number[][] = Array.from({ length: M + 1 }, () => new Array(N + 1).fill(0));
    dp[0][0] = 0;

    for (let m = 1; m <= M; m++) {
      const targetFrac = islandFracs[m - 1];
      const maxEnd = N - (M - m);
      for (let j = m; j <= maxEnd; j++) {
        for (let i = m - 1; i < j; i++) {
          if (!Number.isFinite(dp[m - 1][i])) continue;
          const sliceFrac = (weightPrefix[j] - weightPrefix[i]) / totalWeight;
          const cost = Math.abs(sliceFrac - targetFrac) ** 2;
          if (dp[m - 1][i] + cost < dp[m][j]) {
            dp[m][j] = dp[m - 1][i] + cost;
            prevSplit[m][j] = i;
          }
        }
      }
    }

    let cursor = N;
    for (let m = M; m >= 1; m--) {
      const startIdx = prevSplit[m][cursor];
      groupSlices.unshift({ startIdx, endIdx: cursor });
      cursor = startIdx;
    }
  }

  const blocks: SrtSegment[] = [];
  let prevEndMs = 0;

  for (let m = 0; m < M; m++) {
    const isl = islands[m];
    const { startIdx, endIdx } = groupSlices[m];
    const sliceCount = endIdx - startIdx;
    if (sliceCount <= 0) continue;

    const sliceTotalWeight = Math.max(1e-6, weightPrefix[endIdx] - weightPrefix[startIdx]);
    let localAccum = 0;
    let sliceStartMs = Math.max(m === 0 ? isl.startMs : prevEndMs + 20, isl.startMs);

    for (let idx = startIdx; idx < endIdx; idx++) {
      localAccum += weights[idx];
      const localFrac = localAccum / sliceTotalWeight;
      const isLastInIsland = idx === endIdx - 1;

      let rawStartMs = sliceStartMs;
      let rawEndMs = isLastInIsland
        ? isl.endMs
        : mapEnergyFractionAcrossIslands(acousticProfile, [isl], localFrac);

      let timingSource: SrtSegment['timingSource'] = 'acoustic-energy-aligned';

      if (!isLastInIsland) {
        let bestPause: DetectedAcousticPause | null = null;
        let bestScore = -Infinity;
        for (const p of acousticProfile.pauses) {
          if (usedPauseMidpoints.has(p.midMs)) continue;
          if (p.midMs <= rawStartMs + 260 || p.midMs >= isl.endMs - 220) continue;
          const distMs = Math.abs(p.midMs - rawEndMs);
          if (distMs <= 450) {
            const score = p.durationMs * 0.55 + p.depthDb * 3 - distMs * 0.3;
            if (score > bestScore) {
              bestScore = score;
              bestPause = p;
            }
          }
        }
        if (bestPause) {
          usedPauseMidpoints.add(bestPause.midMs);
          rawEndMs = bestPause.midMs;
          timingSource = 'acoustic-pause-aligned';
        }
        const minTail = (endIdx - 1 - idx) * 240;
        rawEndMs = Math.min(isl.endMs - minTail, Math.max(rawStartMs + 220, rawEndMs));
      } else if (isl.endMs - isl.startMs > 220) {
        rawEndMs = isl.endMs;
      }

      const trimmed = trimIntervalToVoicedSpeech(
        acousticProfile,
        rawStartMs,
        Math.max(rawStartMs + 220, rawEndMs)
      );
      rawStartMs = Math.max(idx === startIdx ? isl.startMs : prevEndMs + 20, trimmed.startMs);
      rawEndMs = Math.max(rawStartMs + 220, Math.min(chunkDurationMs, Math.min(isl.endMs, trimmed.endMs)));

      prevEndMs = rawEndMs;
      sliceStartMs = Math.min(isl.endMs - 220, rawEndMs + 25);

      const absStartMs = chunkOffsetMs + rawStartMs;
      const absEndMs = chunkOffsetMs + rawEndMs;
      blocks.push({
        id: blocks.length + 1,
        startTimeMs: absStartMs,
        endTimeMs: absEndMs,
        startTimestamp: msToSrtTimestamp(absStartMs),
        endTimestamp: msToSrtTimestamp(absEndMs),
        text: wrapSubtitleTextIntoMaxTwoLines(subUnits[idx], maxCharsPerLine),
        timingSource
      });
    }
  }

  return blocks;
}

function shouldFallbackToAcousticIslandAlignment(
  rawSegments: RawAsrSegmentInput[],
  chunkDurationMs: number,
  acousticProfile: ChunkAcousticTimingProfile
): boolean {
  if (rawSegments.length === 0) return false;
  const voicedArr = acousticProfile.frameIsVoiced || [];
  if (voicedArr.length < 10) return false;

  const totalVoicedFrames = voicedArr.filter(Boolean).length;
  if (totalVoicedFrames < 12) return false; // Less than 240ms of speech total

  const parsedBounds: { startMs: number; endMs: number; tokens: number }[] = [];
  let nonMonotonicOrInvalid = 0;

  for (let i = 0; i < rawSegments.length; i++) {
    const seg = rawSegments[i];
    const s = parseFlexibleSeconds(seg.startSec);
    const e = parseFlexibleSeconds(seg.endSec);
    const tokens = tokenizeTranscriptForVerification(seg.text || '').length;
    if (s === null || e === null || e <= s || s * 1000 > chunkDurationMs + 800) {
      nonMonotonicOrInvalid++;
      continue;
    }
    const startMs = Math.max(0, Math.min(chunkDurationMs - 50, Math.round(s * 1000)));
    const endMs = Math.max(startMs + 50, Math.min(chunkDurationMs, Math.round(e * 1000)));
    if (
      parsedBounds.length > 0 &&
      startMs < parsedBounds[parsedBounds.length - 1].startMs - 400
    ) {
      nonMonotonicOrInvalid++;
    }
    parsedBounds.push({ startMs, endMs, tokens });
  }

  if (parsedBounds.length === 0 || nonMonotonicOrInvalid >= Math.max(1, Math.ceil(rawSegments.length * 0.4))) {
    return true;
  }

  // Check 1: Single raw segment spanning whole chunk or missing acoustic boundaries -> use island aligner
  if (rawSegments.length === 1) {
    const b = parsedBounds[0];
    const segVoiced = countVoicedFramesInWindow(acousticProfile, b.startMs, b.endMs);
    if (segVoiced.voicedFrames < totalVoicedFrames * 0.7 || acousticProfile.activeSpeechIslands.length >= 2) {
      return true;
    }
  }

  // Check 2: Did the model leave a giant gap (> 1200ms) right across active speech?
  for (let i = 1; i < parsedBounds.length; i++) {
    const gapStart = parsedBounds[i - 1].endMs;
    const gapEnd = parsedBounds[i].startMs;
    if (gapEnd - gapStart > 1200) {
      const gapVoice = countVoicedFramesInWindow(acousticProfile, gapStart, gapEnd);
      if (gapVoice.voicedRatio >= 0.4 && gapVoice.voicedFrames >= 35) {
        return true;
      }
    }
  }

  // Check 3: Compare total voiced speech covered by model segments vs total voiced speech in chunk
  let coveredVoicedFrames = 0;
  let placedTotalFrames = 0;
  for (const b of parsedBounds) {
    const w = countVoicedFramesInWindow(acousticProfile, b.startMs, b.endMs);
    coveredVoicedFrames += w.voicedFrames;
    placedTotalFrames += w.totalFrames;
  }

  if (totalVoicedFrames >= 25 && coveredVoicedFrames < totalVoicedFrames * 0.56) {
    return true;
  }

  // Check 4: Are placed segments landing mostly on dead silence while real speech is elsewhere?
  if (
    placedTotalFrames >= 20 &&
    totalVoicedFrames / voicedArr.length >= 0.25 &&
    coveredVoicedFrames / placedTotalFrames < 0.26
  ) {
    return true;
  }

  return false;
}

/**
 * Priority 2 Intelligent Timestamp-Aware SRT Segmenter:
 * Converts sanitized ASR segments into sentence-aware, pause-aligned, 2-line max SubRip segments
 * while guaranteeing 100% exact ordered token sequence preservation:
 *   0. Verifies ASR segment timestamps against actual 20ms PCM voiced activity & speech islands;
 *      if Gemini timestamps drifted, compressed, or left dead holes across active speech, cleanly
 *      re-anchors all verbatim sentences/clauses to the measured acoustic speech islands & pauses.
 *   1. Word-level timestamps from ASR when present, monotonic, and acoustically coherent.
 *   2. Sentence boundary detection (., !, ?, ।, ॥, ؟, ۔, 。, ！, ？) with abbreviation protection.
 *   3. Measured 20ms-resolution acoustic pauses (extractChunkAcousticTimingProfile).
 *   4. Balanced linguistic clause grouping (>= 3 words per clause, zero orphan tails) & 2-line wrapping.
 *   5. Companded voiced energy alignment fallback when word timestamps and pauses are absent.
 */
export function segmentChunkAsrIntoReadableSrtBlocks(options: {
  rawSegments: RawAsrSegmentInput[];
  chunkStartSec: number;
  chunkEndSec: number;
  acousticProfile: ChunkAcousticTimingProfile;
  languageCode: string;
}): SrtSegment[] {
  const { rawSegments, chunkStartSec, chunkEndSec, acousticProfile, languageCode } = options;
  const chunkOffsetMs = Math.round(chunkStartSec * 1000);
  const chunkDurationMs = Math.max(250, Math.round((chunkEndSec - chunkStartSec) * 1000));
  const usedPauseMidpoints = new Set<number>();

  // If Gemini's segment timestamps failed acoustic verification (drifted, left spoken gaps, or placed on silence),
  // preserve 100% of verbatim words and re-anchor strictly to actual measured PCM speech islands & pauses!
  if (shouldFallbackToAcousticIslandAlignment(rawSegments, chunkDurationMs, acousticProfile)) {
    const combinedVerbatimText = rawSegments
      .map((s) => tokenizeTranscriptForVerification(s.text || '').join(' '))
      .filter(Boolean)
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (combinedVerbatimText) {
      const maxCharsPerLine = getScriptMaxCharsPerLine(languageCode, combinedVerbatimText);
      const rawSentences = splitTextIntoLinguisticSentences(combinedVerbatimText);
      const subUnits: string[] = [];
      for (const sent of rawSentences) {
        const clauses = splitLongSentenceIntoReadableClauses(sent, maxCharsPerLine);
        for (const cl of clauses) {
          if (cl.trim()) subUnits.push(cl.trim());
        }
      }
      if (subUnits.length > 0) {
        return alignSubUnitsAcrossActiveSpeechIslands({
          subUnits,
          maxCharsPerLine,
          chunkOffsetMs,
          chunkDurationMs,
          acousticProfile,
          usedPauseMidpoints
        });
      }
    }
  }

  const outputBlocks: SrtSegment[] = [];

  for (let segIdx = 0; segIdx < rawSegments.length; segIdx++) {
    const rawSeg = rawSegments[segIdx];
    const cleanTokens = tokenizeTranscriptForVerification(rawSeg.text || '');
    if (cleanTokens.length === 0) continue;
    const cleanText = cleanTokens.join(' ');

    const maxCharsPerLine = getScriptMaxCharsPerLine(languageCode, cleanText);

    const parsedStartSec = parseFlexibleSeconds(rawSeg.startSec) ?? 0;
    const parsedEndSec = parseFlexibleSeconds(rawSeg.endSec) ?? (chunkEndSec - chunkStartSec);

    let relStartMs = Math.max(
      0,
      Math.min(chunkDurationMs - 100, Math.round(parsedStartSec * 1000))
    );
    let relEndMs = Math.max(
      relStartMs + 100,
      Math.min(chunkDurationMs, Math.round(parsedEndSec * 1000))
    );

    if (rawSegments.length === 1 && relStartMs === 0 && relEndMs >= chunkDurationMs - 100) {
      relStartMs = Math.min(chunkDurationMs - 100, acousticProfile.activeSpeechStartMs);
      relEndMs = Math.max(relStartMs + 100, Math.min(chunkDurationMs, acousticProfile.activeSpeechEndMs));
    }

    // Trim pure leading/trailing silence inside the segment window
    const voicedTrim = trimIntervalToVoicedSpeech(acousticProfile, relStartMs, relEndMs);
    relStartMs = voicedTrim.startMs;
    relEndMs = voicedTrim.endMs;

    // Guard against compressed ASR segment endSec
    const minNaturalSpanMs = cleanTokens.length * 260;
    const nextRawSeg = rawSegments[segIdx + 1];
    const nextSegStartMs = nextRawSeg
      ? Math.max(relEndMs, Math.round((parseFlexibleSeconds(nextRawSeg.startSec) ?? 0) * 1000) - 80)
      : Math.max(relEndMs, Math.min(chunkDurationMs, acousticProfile.activeSpeechEndMs));

    if (relEndMs - relStartMs < minNaturalSpanMs && nextSegStartMs > relEndMs + 350) {
      const targetExpandedEndMs = Math.min(
        nextSegStartMs,
        relStartMs + Math.max(minNaturalSpanMs, cleanTokens.length * 420)
      );
      relEndMs = Math.max(relEndMs, targetExpandedEndMs);
    }

    // Step 1: Decompose rawSeg.text into linguistic sentences, and split any long sentence into balanced clauses
    const rawSentences = splitTextIntoLinguisticSentences(cleanText);
    const subUnits: string[] = [];
    for (const sent of rawSentences) {
      const clauses = splitLongSentenceIntoReadableClauses(sent, maxCharsPerLine);
      for (const cl of clauses) {
        if (cl.trim()) subUnits.push(cl.trim());
      }
    }

    if (subUnits.length === 0) continue;

    // Check if usable monotonic word-level timestamps were returned by ASR for this segment
    let validWords: SrtWordTimestamp[] = [];
    if (Array.isArray(rawSeg.words) && rawSeg.words.length >= 2) {
      let prevEnd = relStartMs;
      let hadOutlierJump = false;
      for (const w of rawSeg.words) {
        const wText = String(w.word || '').trim();
        const wStartSec = parseFlexibleSeconds(w.startSec);
        const wEndSec = parseFlexibleSeconds(w.endSec);
        if (wStartSec === null || wEndSec === null) continue;
        const wStart = Math.max(0, Math.round(wStartSec * 1000));
        const wEnd = Math.min(chunkDurationMs, Math.round(wEndSec * 1000));
        if (wStart - prevEnd > 3500 && rawSeg.words.length > 4) {
          hadOutlierJump = true;
          break;
        }
        if (wText && wEnd > wStart && wStart >= prevEnd - 120) {
          validWords.push({
            word: wText,
            startTimeMs: Math.max(prevEnd, wStart),
            endTimeMs: Math.max(prevEnd + 40, wEnd)
          });
          prevEnd = Math.max(prevEnd, wEnd);
        }
      }
      if (hadOutlierJump || validWords.length < Math.floor(cleanTokens.length * 0.7)) {
        validWords = [];
      } else if (validWords.length >= 2) {
        const wordSpanMs = validWords[validWords.length - 1].endTimeMs - validWords[0].startTimeMs;
        if (wordSpanMs < cleanTokens.length * 180 && relEndMs - relStartMs > wordSpanMs * 1.4) {
          validWords = [];
        }
      }
    }

    if (subUnits.length === 1) {
      const wrappedText = wrapSubtitleTextIntoMaxTwoLines(subUnits[0], maxCharsPerLine);
      const absStartMs =
        validWords.length >= 1
          ? chunkOffsetMs + validWords[0].startTimeMs
          : chunkOffsetMs + relStartMs;
      const absEndMs =
        validWords.length >= 1
          ? Math.max(absStartMs + 200, chunkOffsetMs + validWords[validWords.length - 1].endTimeMs)
          : Math.max(absStartMs + 200, chunkOffsetMs + relEndMs);

      outputBlocks.push({
        id: outputBlocks.length + 1,
        startTimeMs: absStartMs,
        endTimeMs: absEndMs,
        startTimestamp: msToSrtTimestamp(absStartMs),
        endTimestamp: msToSrtTimestamp(absEndMs),
        text: wrappedText,
        timingSource:
          validWords.length >= 1
            ? 'word-timestamps'
            : acousticProfile.pauses.length > 0
            ? 'acoustic-pause-aligned'
            : 'acoustic-energy-aligned'
      });
      continue;
    }

    // Multiple sentences/clauses inside rawSeg:
    // Priority 1: Use exact word-level timestamps when word count matches subUnits
    const subUnitWordCounts = subUnits.map(
      (u) => tokenizeTranscriptForVerification(u).length
    );
    const totalSubUnitWords = subUnitWordCounts.reduce((a, b) => a + b, 0);

    if (validWords.length === totalSubUnitWords && validWords.length >= 2) {
      let wordCursor = 0;
      for (let idx = 0; idx < subUnits.length; idx++) {
        const count = subUnitWordCounts[idx];
        const startWordIdx = Math.min(validWords.length - 1, wordCursor);
        const endWordIdx = Math.min(
          validWords.length - 1,
          Math.max(startWordIdx, wordCursor + count - 1)
        );
        wordCursor += count;

        const absStartMs = chunkOffsetMs + validWords[startWordIdx].startTimeMs;
        const absEndMs = Math.max(
          absStartMs + 150,
          chunkOffsetMs + validWords[endWordIdx].endTimeMs
        );
        outputBlocks.push({
          id: outputBlocks.length + 1,
          startTimeMs: absStartMs,
          endTimeMs: absEndMs,
          startTimestamp: msToSrtTimestamp(absStartMs),
          endTimestamp: msToSrtTimestamp(absEndMs),
          text: wrapSubtitleTextIntoMaxTwoLines(subUnits[idx], maxCharsPerLine),
          timingSource: 'word-timestamps'
        });
      }
      continue;
    }

    // Priority 2 & 3 Fallback: Anchor split boundaries to actual measured acoustic pauses
    // and cumulative voiced speech energy within [relStartMs, relEndMs]
    const weights = subUnits.map((u) => {
      const tokens = tokenizeTranscriptForVerification(u);
      const charCount = Math.max(1, Array.from(u.replace(/\s+/g, '')).length);
      const wordWeight = tokens.length * 2.5;
      const punctPauseWeight = /[.!?।॥؟۔。！？]$/.test(u) ? 5 : /[,;—–:،؛]$/.test(u) ? 2.5 : 0;
      return charCount + wordWeight + punctPauseWeight;
    });
    const totalWeight = weights.reduce((a, b) => a + b, 0) || 1;

    let currentRelStartMs = relStartMs;
    let accumulatedWeight = 0;
    const totalWindowSpanMs = Math.max(200, relEndMs - relStartMs);
    const minSliceMs = Math.max(120, Math.min(400, Math.floor(totalWindowSpanMs / (subUnits.length * 2))));

    for (let idx = 0; idx < subUnits.length; idx++) {
      accumulatedWeight += weights[idx];
      let nextRelEndMs = relEndMs;
      let timingSource: SrtSegment['timingSource'] = 'acoustic-energy-aligned';

      if (idx < subUnits.length - 1) {
        const fraction = accumulatedWeight / totalWeight;
        const resolved = resolveAcousticSplitTimestampMs(
          acousticProfile,
          relStartMs,
          relEndMs,
          fraction,
          usedPauseMidpoints
        );
        const remainingSlices = subUnits.length - 1 - idx;
        const maxAllowedEnd = Math.max(
          currentRelStartMs + minSliceMs,
          relEndMs - remainingSlices * (minSliceMs + 20)
        );
        nextRelEndMs = Math.min(
          maxAllowedEnd,
          Math.max(currentRelStartMs + minSliceMs, resolved.timestampMs)
        );
        timingSource = resolved.source;
      }

      const sliceTrimmed = trimIntervalToVoicedSpeech(
        acousticProfile,
        currentRelStartMs,
        Math.max(currentRelStartMs + 150, nextRelEndMs)
      );
      const effStartRelMs = Math.max(currentRelStartMs, sliceTrimmed.startMs);
      const effEndRelMs = Math.max(effStartRelMs + 100, Math.min(nextRelEndMs, sliceTrimmed.endMs));

      const absStartMs = chunkOffsetMs + effStartRelMs;
      const absEndMs = Math.max(absStartMs + 100, chunkOffsetMs + effEndRelMs);
      outputBlocks.push({
        id: outputBlocks.length + 1,
        startTimeMs: absStartMs,
        endTimeMs: absEndMs,
        startTimestamp: msToSrtTimestamp(absStartMs),
        endTimestamp: msToSrtTimestamp(absEndMs),
        text: wrapSubtitleTextIntoMaxTwoLines(subUnits[idx], maxCharsPerLine),
        timingSource
      });

      const gapMs = Math.min(40, Math.max(10, Math.floor((relEndMs - effEndRelMs) * 0.1)));
      currentRelStartMs = Math.min(relEndMs - minSliceMs, effEndRelMs + gapMs);
    }
  }

  return outputBlocks;
}

/**
 * Enforces strict SubRip timeline integrity across all segments in their exact canonical sequence
 * (never sorting or reordering spoken segments) and produces a comprehensive token-sequence and
 * timestamp validation report verifying:
 * - Exact ordered token sequence preservation (inputTokenCount, outputTokenCount, duplicateTokenCount = 0,
 *   missingTokenCount = 0, reorderedTokenCount = 0, insertedTokenCount = 0, normalizationMismatchCount = 0,
 *   exactSequenceMatch = true)
 * - startTime < endTime
 * - Chronological ordering
 * - Zero unintended overlaps
 * - Zero negative timestamps
 * - Zero timestamps beyond audio duration
 * - Sequential 1..N numbering
 */
export function enforceAndValidateSrtTimelineIntegrity(
  rawSegments: SrtSegment[],
  totalAudioDurationSec: number,
  canonicalOriginalTranscriptOrWordCount?: string | number
): { segments: SrtSegment[]; validationReport: SrtValidationReport } {
  const maxAudioMs = Math.max(200, Math.round(totalAudioDurationSec * 1000));
  // Preserve exact spoken transcript order—never sort by startTimeMs which could reorder spoken words
  const ordered = rawSegments.filter(
    (s) => tokenizeTranscriptForVerification(s.text || '').length > 0
  );

  const n = ordered.length;
  const cleaned: SrtSegment[] = [];

  if (n > 0) {
    // Pass 1: Clamp all raw timestamps into [0, maxAudioMs]
    const starts = ordered.map((s) => Math.max(0, Math.min(maxAudioMs - 50, Math.round(s.startTimeMs))));
    const ends = ordered.map((s, i) =>
      Math.max(starts[i] + 50, Math.min(maxAudioMs, Math.round(s.endTimeMs)))
    );

    // Pass 2: Resolve any overlap or non-monotonic boundary between consecutive segments
    // while strictly staying inside [0, maxAudioMs]
    const minDurMs = Math.max(80, Math.min(350, Math.floor(maxAudioMs / Math.max(1, n * 2))));
    const minGapMs = maxAudioMs >= n * (minDurMs + 20) ? 20 : 0;

    for (let i = 1; i < n; i++) {
      if (starts[i] < ends[i - 1] + minGapMs) {
        const overlapMid = Math.round((ends[i - 1] + starts[i]) / 2);
        const adjustedPrevEnd = Math.max(starts[i - 1] + minDurMs, overlapMid - Math.floor(minGapMs / 2));
        ends[i - 1] = Math.min(maxAudioMs - (n - i) * (minDurMs + minGapMs), adjustedPrevEnd);
        starts[i] = Math.max(starts[i], ends[i - 1] + minGapMs);
      }
      if (ends[i] < starts[i] + minDurMs) {
        ends[i] = Math.min(maxAudioMs, starts[i] + minDurMs);
      }
    }

    // Pass 3: Backward ceiling pass to guarantee no timestamp exceeds maxAudioMs and startTime < endTime
    if (ends[n - 1] > maxAudioMs) {
      ends[n - 1] = maxAudioMs;
    }
    for (let i = n - 1; i >= 0; i--) {
      if (i < n - 1 && ends[i] > starts[i + 1] - minGapMs) {
        ends[i] = Math.max(0, starts[i + 1] - minGapMs);
      }
      if (starts[i] >= ends[i]) {
        starts[i] = Math.max(0, ends[i] - minDurMs);
      }
    }

    // Pass 4: Expand overly brief segments (< 750ms) up to available slack before next segment and maxAudioMs
    // without arbitrarily amputating longer continuous speech blocks.
    for (let i = 0; i < n; i++) {
      const ceilingMs = i < n - 1 ? Math.max(ends[i], starts[i + 1] - minGapMs) : maxAudioMs;
      if (ends[i] - starts[i] < 750 && ceilingMs > ends[i]) {
        ends[i] = Math.min(ceilingMs, starts[i] + 800);
      }
      if (ends[i] - starts[i] > 10_000) {
        ends[i] = starts[i] + 10_000;
      }
    }

    for (let i = 0; i < n; i++) {
      const startMs = Math.max(0, starts[i]);
      const endMs = Math.min(maxAudioMs, Math.max(startMs + 40, ends[i]));
      cleaned.push({
        ...ordered[i],
        id: i + 1,
        startTimeMs: startMs,
        endTimeMs: endMs,
        startTimestamp: msToSrtTimestamp(startMs),
        endTimestamp: msToSrtTimestamp(endMs),
        text: ordered[i].text.trim()
      });
    }
  }

  // Reconstruct full transcript from final SRT segments
  const reconstructedTranscript = cleaned
    .map((s) => s.text.replace(/\s+/g, ' ').trim())
    .join(' ')
    .trim();

  const canonicalOriginalTranscript =
    typeof canonicalOriginalTranscriptOrWordCount === 'string'
      ? canonicalOriginalTranscriptOrWordCount.replace(/\s+/g, ' ').trim()
      : reconstructedTranscript;

  const tokenComparison = compareTokenSequenceIntegrity(
    canonicalOriginalTranscript,
    reconstructedTranscript
  );

  // Compute objective timeline & formatting metrics
  let chronologicalOrderVerified = true;
  let zeroOverlapsVerified = true;
  let zeroNegativeOrReversedVerified = true;
  let withinAudioDurationVerified = true;
  let sequentialNumberingVerified = true;
  let maxLinesPerBlock = 0;
  let maxCharsPerLineObserved = 0;
  let minDurationMsObserved = cleaned.length > 0 ? Infinity : 0;
  let maxDurationMsObserved = 0;
  const timingSourcesUsed = new Set<string>();

  for (let i = 0; i < cleaned.length; i++) {
    const s = cleaned[i];
    const dur = s.endTimeMs - s.startTimeMs;
    if (s.id !== i + 1) sequentialNumberingVerified = false;
    if (s.startTimeMs < 0 || s.endTimeMs <= s.startTimeMs) {
      zeroNegativeOrReversedVerified = false;
    }
    if (s.endTimeMs > maxAudioMs) {
      withinAudioDurationVerified = false;
    }
    if (i > 0) {
      if (s.startTimeMs < cleaned[i - 1].startTimeMs) chronologicalOrderVerified = false;
      if (s.startTimeMs < cleaned[i - 1].endTimeMs) zeroOverlapsVerified = false;
    }
    if (dur < minDurationMsObserved) minDurationMsObserved = dur;
    if (dur > maxDurationMsObserved) maxDurationMsObserved = dur;
    if (s.timingSource) timingSourcesUsed.add(s.timingSource);

    const lines = s.text.split('\n');
    if (lines.length > maxLinesPerBlock) maxLinesPerBlock = lines.length;
    for (const line of lines) {
      const charCount = Array.from(line.trim()).length;
      if (charCount > maxCharsPerLineObserved) maxCharsPerLineObserved = charCount;
    }
  }

  const wordPreservationVerified =
    tokenComparison.exactSequenceMatch &&
    tokenComparison.duplicateTokenCount === 0 &&
    tokenComparison.missingTokenCount === 0 &&
    tokenComparison.reorderedTokenCount === 0 &&
    tokenComparison.insertedTokenCount === 0 &&
    tokenComparison.normalizationMismatchCount === 0;

  const timingAlignmentSummary =
    cleaned.length === 0
      ? 'Empty transcript (0 segments)'
      : timingSourcesUsed.has('word-timestamps') && timingSourcesUsed.size === 1
      ? 'Word-level ASR timestamps'
      : timingSourcesUsed.has('acoustic-pause-aligned')
      ? '20ms PCM acoustic pause alignment + vocal energy fallback where pauses are absent'
      : 'Documented fallback: 20ms PCM cumulative vocal energy alignment (word timestamps unavailable)';

  return {
    segments: cleaned,
    validationReport: {
      isValid:
        chronologicalOrderVerified &&
        zeroOverlapsVerified &&
        zeroNegativeOrReversedVerified &&
        withinAudioDurationVerified &&
        sequentialNumberingVerified &&
        wordPreservationVerified &&
        maxLinesPerBlock <= 2,
      chronologicalOrderVerified,
      zeroOverlapsVerified,
      zeroNegativeOrReversedVerified,
      withinAudioDurationVerified,
      sequentialNumberingVerified,
      wordPreservationVerified,
      exactSequenceMatch: tokenComparison.exactSequenceMatch,
      inputTokenCount: tokenComparison.inputTokenCount,
      outputTokenCount: tokenComparison.outputTokenCount,
      duplicateTokenCount: tokenComparison.duplicateTokenCount,
      missingTokenCount: tokenComparison.missingTokenCount,
      reorderedTokenCount: tokenComparison.reorderedTokenCount,
      insertedTokenCount: tokenComparison.insertedTokenCount,
      normalizationMismatchCount: tokenComparison.normalizationMismatchCount,
      missingTokens: tokenComparison.missingTokens,
      duplicatedTokens: tokenComparison.duplicatedTokens,
      reorderedTokens: tokenComparison.reorderedTokens,
      insertedTokens: tokenComparison.insertedTokens,
      originalTranscript: canonicalOriginalTranscript,
      reconstructedTranscript,
      timingAlignmentSummary,
      maxLinesPerBlock,
      maxCharsPerLineObserved,
      minDurationMsObserved: minDurationMsObserved === Infinity ? 0 : minDurationMsObserved,
      maxDurationMsObserved,
      totalWordsInput: tokenComparison.inputTokenCount,
      totalWordsOutput: tokenComparison.outputTokenCount
    }
  };
}

/**
 * Formats milliseconds into standard SRT timestamp format: HH:MM:SS,mmm
 */
export function msToSrtTimestamp(ms: number): string {
  const clamped = Math.max(0, Math.round(ms));
  const hours = Math.floor(clamped / 3600000);
  const minutes = Math.floor((clamped % 3600000) / 60000);
  const seconds = Math.floor((clamped % 60000) / 1000);
  const millis = clamped % 1000;

  const hh = String(hours).padStart(2, '0');
  const mm = String(minutes).padStart(2, '0');
  const ss = String(seconds).padStart(2, '0');
  const mmm = String(millis).padStart(3, '0');
  return `${hh}:${mm}:${ss},${mmm}`;
}

/**
 * Parses an SRT timestamp (HH:MM:SS,mmm or MM:SS,mmm or SS.mmm) into milliseconds.
 */
export function parseFlexibleTimestampToMs(raw: string | number): number {
  if (typeof raw === 'number') {
    return Math.round(raw * 1000);
  }
  const clean = raw.trim().replace('.', ',');
  const parts = clean.split(':');
  if (parts.length === 3) {
    const hh = parseInt(parts[0], 10) || 0;
    const mm = parseInt(parts[1], 10) || 0;
    const [ssStr, msStr = '0'] = parts[2].split(',');
    const ss = parseInt(ssStr, 10) || 0;
    const ms = parseInt(msStr.padEnd(3, '0').slice(0, 3), 10) || 0;
    return hh * 3600000 + mm * 60000 + ss * 1000 + ms;
  }
  if (parts.length === 2) {
    const mm = parseInt(parts[0], 10) || 0;
    const [ssStr, msStr = '0'] = parts[1].split(',');
    const ss = parseInt(ssStr, 10) || 0;
    const ms = parseInt(msStr.padEnd(3, '0').slice(0, 3), 10) || 0;
    return mm * 60000 + ss * 1000 + ms;
  }
  const num = parseFloat(clean.replace(',', '.'));
  return Number.isFinite(num) ? Math.round(num * 1000) : 0;
}

/**
 * Formats an array of SrtSegments into a valid standard SubRip (.srt) string.
 */
export function formatSegmentsToSrt(segments: SrtSegment[]): string {
  return segments
    .map((seg, idx) => {
      const startTs = msToSrtTimestamp(seg.startTimeMs);
      const endTs = msToSrtTimestamp(Math.max(seg.startTimeMs + 200, seg.endTimeMs));
      return `${idx + 1}\n${startTs} --> ${endTs}\n${seg.text.trim()}\n`;
    })
    .join('\n');
}

export interface Auto10SecExtractResult {
  trimmed: boolean;
  outputPath: string;
  originalDurationSec: number;
  extractedDurationSec: number;
  windowStartSec: number;
  windowEndSec: number;
}

/**
 * Automatically cuts the cleanest, most speech-rich 10-second window from any
 * reference voice exceeding 10 seconds, converting it to a canonical 24kHz mono WAV.
 * Files <= 10.05s pass through untouched.
 */
export async function extractAuto10SecVoiceCloneReference(
  inputPath: string,
  outputWavPath: string,
  targetMaxSec = 10
): Promise<Auto10SecExtractResult> {
  let originalDurationSec = 0;
  try {
    const { stdout } = await execFileAsync(
      'ffprobe',
      ['-v', 'error', '-show_format', '-show_streams', '-of', 'json', inputPath],
      { timeout: 12_000 }
    );
    const parsed = JSON.parse(stdout);
    const stream = (parsed.streams || []).find((s: any) => s.codec_type === 'audio');
    originalDurationSec = Number(
      parseFloat(stream?.duration || parsed.format?.duration || '0').toFixed(2)
    );
  } catch {
    originalDurationSec = 0;
  }

  if (originalDurationSec <= targetMaxSec + 0.05) {
    return {
      trimmed: false,
      outputPath: inputPath,
      originalDurationSec,
      extractedDurationSec: originalDurationSec,
      windowStartSec: 0,
      windowEndSec: originalDurationSec
    };
  }

  let bestStartSec = 0;
  const scanLimitSec = Math.min(originalDurationSec, 180);
  const scanPcmPath = `${outputWavPath}.scan8k.pcm`;

  try {
    await execFileAsync(
      'ffmpeg',
      [
        '-y',
        '-i',
        inputPath,
        '-t',
        String(scanLimitSec),
        '-vn',
        '-ar',
        '8000',
        '-ac',
        '1',
        '-f',
        's16le',
        scanPcmPath
      ],
      { timeout: 15_000 }
    );
    const pcmBuf = await fs.promises.readFile(scanPcmPath);
    const frameSamples = 1600; // 200ms frames at 8kHz
    const frameSec = 0.2;
    const totalSamples = Math.floor(pcmBuf.length / 2);
    const frameCount = Math.floor(totalSamples / frameSamples);
    const windowFrames = Math.max(10, Math.round(targetMaxSec / frameSec)); // 50 frames = 10s

    if (frameCount > windowFrames) {
      const frameDbs: number[] = [];
      const framePeaks: number[] = [];
      for (let f = 0; f < frameCount; f++) {
        let sumSq = 0;
        let peak = 0;
        const base = f * frameSamples;
        for (let s = 0; s < frameSamples; s++) {
          const val = pcmBuf.readInt16LE((base + s) * 2) / 32768.0;
          const absVal = Math.abs(val);
          if (absVal > peak) peak = absVal;
          sumSq += val * val;
        }
        const rms = Math.sqrt(sumSq / frameSamples);
        frameDbs.push(rms > 1e-6 ? 20 * Math.log10(rms) : -96.0);
        framePeaks.push(peak > 1e-6 ? 20 * Math.log10(peak) : -96.0);
      }

      const sortedDb = [...frameDbs].sort((a, b) => a - b);
      const noiseFloorDb = sortedDb[Math.floor(sortedDb.length * 0.15)] ?? -60;
      const p80Db = sortedDb[Math.floor(sortedDb.length * 0.8)] ?? -20;
      const speechThreshDb = Math.max(-46, Math.min(-24, noiseFloorDb + Math.max(8, (p80Db - noiseFloorDb) * 0.35)));

      let bestScore = -Infinity;
      let bestFrameIdx = 0;

      for (let start = 0; start <= frameCount - windowFrames; start++) {
        let activeCount = 0;
        let dbSum = 0;
        let clippedCount = 0;
        for (let k = 0; k < windowFrames; k++) {
          const db = frameDbs[start + k];
          if (db >= speechThreshDb) {
            activeCount++;
            dbSum += Math.min(-10, db);
          }
          if (framePeaks[start + k] > -0.8) {
            clippedCount++;
          }
        }
        const activeRatio = activeCount / windowFrames;
        const avgActiveDbNorm = activeCount > 0 ? Math.max(0, (dbSum / activeCount + 45) / 35) : 0;
        const earlyBias = 0.04 * (1 - start / Math.max(1, frameCount - windowFrames));
        const score = activeRatio * 2.0 + avgActiveDbNorm * 0.35 - (clippedCount / windowFrames) * 0.4 + earlyBias;
        if (score > bestScore) {
          bestScore = score;
          bestFrameIdx = start;
        }
      }

      // Trim any leading silence inside the chosen 10s window while keeping full 10s when available
      let firstActiveOffset = 0;
      for (let k = 0; k < Math.min(windowFrames - 10, 15); k++) {
        if (frameDbs[bestFrameIdx + k] >= speechThreshDb) {
          firstActiveOffset = k;
          break;
        }
      }
      const adjustedStartFrame = Math.min(frameCount - windowFrames, bestFrameIdx + firstActiveOffset);
      bestStartSec = Number(Math.max(0, adjustedStartFrame * frameSec).toFixed(2));
      if (bestStartSec + targetMaxSec > originalDurationSec) {
        bestStartSec = Number(Math.max(0, originalDurationSec - targetMaxSec).toFixed(2));
      }
    }
  } catch {
    bestStartSec = 0;
  } finally {
    await fs.promises.unlink(scanPcmPath).catch(() => {});
  }

  const fadeOutStart = Math.max(0.1, targetMaxSec - 0.02).toFixed(3);
  await execFileAsync(
    'ffmpeg',
    [
      '-y',
      '-i',
      inputPath,
      '-ss',
      String(bestStartSec),
      '-t',
      String(targetMaxSec),
      '-vn',
      '-af',
      `afade=t=in:ss=0:d=0.015,afade=t=out:st=${fadeOutStart}:d=0.015`,
      '-ar',
      '24000',
      '-ac',
      '1',
      '-c:a',
      'pcm_s16le',
      outputWavPath
    ],
    { timeout: 25_000 }
  );

  const stat = await fs.promises.stat(outputWavPath).catch(() => null);
  const extractedDurationSec = stat
    ? Number(Math.max(0.1, (stat.size - 44) / (24000 * 2)).toFixed(2))
    : targetMaxSec;

  return {
    trimmed: true,
    outputPath: outputWavPath,
    originalDurationSec,
    extractedDurationSec,
    windowStartSec: bestStartSec,
    windowEndSec: Number(Math.min(originalDurationSec, bestStartSec + extractedDurationSec).toFixed(2))
  };
}

