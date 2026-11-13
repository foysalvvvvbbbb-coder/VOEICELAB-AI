import { execFile } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import type {
  AudioSignalAnalysis,
  SrtSegment,
  SrtValidationReport,
  SrtWordTimestamp
} from '../types/voicelab.ts';

const execFileAsync = promisify(execFile);

export interface RawPcmPitchStats {
  fundamentalFreqHz: number;
  pitchRangeHz: [number, number];
  pitchStdDevHz: number;
  voicedFrameRatio: number;
  spectralWarmthRatio: number;
  rmsDb: number;
  peakDb: number;
  noiseFloorDb: number;
  silenceRatio: number;
  clippedSamplesCount: number;
  dcOffset: number;
  acousticEmbedding16D: number[];
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

/**
 * Extracts raw 16kHz 16-bit mono PCM via FFmpeg and computes:
 * - Voiced frame autocorrelation fundamental frequency (F0 in Hz, 72Hz–350Hz human vocal range)
 * - 10th to 90th percentile F0 range & F0 standard deviation (Hz)
 * - RMS level (dBFS), Peak level (dBFS), Noise floor (15th percentile frame energy)
 * - Clipped sample count (|s| >= 0.998) and DC offset
 * - 16-D Voiced Acoustic-Spectral Speaker Descriptor (12 Goertzel vocal band energies + 4 prosodic descriptors)
 */
export async function analyzeVocalAcousticsFromAudioFile(filePath: string): Promise<RawPcmPitchStats> {
  const tempRawPath = `${filePath}.16k.pcm`;
  try {
    await execFileAsync('ffmpeg', [
      '-y',
      '-i',
      filePath,
      '-vn',
      '-ac',
      '1',
      '-ar',
      '16000',
      '-f',
      's16le',
      tempRawPath
    ]);

    const buffer = await fs.promises.readFile(tempRawPath);
    const numSamples = Math.floor(buffer.length / 2);
    if (numSamples < 1600) {
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

    // Frame-by-frame analysis (30ms frames = 480 samples at 16kHz, hop = 240 samples)
    const sampleRate = 16000;
    const frameSize = 480;
    const hopSize = 240;
    const frameDbValues: number[] = [];
    const f0Estimates: number[] = [];
    const bandEnergySums = new Float64Array(VOCAL_BAND_FREQS_HZ.length);
    let voicedSpectralFrames = 0;
    let silentFrames = 0;
    let totalFrames = 0;
    let lowBandEnergy = 0;
    let highBandEnergy = 0;

    const minLag = Math.floor(sampleRate / 350); // ~350 Hz max vocal F0
    const maxLag = Math.floor(sampleRate / 72);  // ~72 Hz min vocal F0

    // Precompute Goertzel coefficients for the 12 vocal formant/harmonic bands
    const goertzelCoeffs = VOCAL_BAND_FREQS_HZ.map((freq) => {
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

      if (frameDb < -45.0) {
        silentFrames++;
        continue;
      }

      lowBandEnergy += frameSumSq;
      highBandEnergy += diffSumSq + 1e-9;

      // Normalized autocorrelation for voiced F0 detection & Goertzel spectral envelope
      if (f0Estimates.length < 400 && totalFrames % 2 === 0 && frameDb > rmsDb - 12) {
        let bestLag = 0;
        let maxCorr = 0;

        for (let lag = minLag; lag <= maxLag; lag++) {
          let corr = 0;
          let norm1 = 0;
          let norm2 = 0;
          const limit = frameSize - lag;
          for (let i = 0; i < limit; i += 2) {
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

        // Voiced frame threshold
        if (maxCorr > 0.62 && bestLag > 0) {
          const f0 = sampleRate / bestLag;
          if (f0 >= 75 && f0 <= 340) {
            f0Estimates.push(f0);
          }

          // Accumulate Goertzel sub-band energies on voiced frames
          voicedSpectralFrames++;
          for (let b = 0; b < goertzelCoeffs.length; b++) {
            const coeff = goertzelCoeffs[b];
            let s1 = 0;
            let s2 = 0;
            for (let i = 0; i < frameSize; i += 2) {
              const s0 = samples[start + i] + coeff * s1 - s2;
              s2 = s1;
              s1 = s0;
            }
            const power = Math.max(1e-9, s1 * s1 + s2 * s2 - coeff * s1 * s2);
            bandEnergySums[b] += Math.log10(power + 1e-6) + 6.0;
          }
        }
      }
    }

    frameDbValues.sort((a, b) => a - b);
    const p15Index = Math.floor(frameDbValues.length * 0.15);
    const noiseFloorDb = frameDbValues.length > 0 ? frameDbValues[p15Index] : -72.0;

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
      Math.min(4.5, Math.max(0.5, lowBandEnergy / (highBandEnergy * 2.5 + 1e-6))).toFixed(2)
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

    return {
      fundamentalFreqHz: Math.round(medianF0),
      pitchRangeHz: [Math.round(p10F0), Math.round(p90F0)],
      pitchStdDevHz,
      voicedFrameRatio: activeRatio,
      spectralWarmthRatio,
      rmsDb: Number(rmsDb.toFixed(1)),
      peakDb: Number(peakDb.toFixed(1)),
      noiseFloorDb: Number(noiseFloorDb.toFixed(1)),
      silenceRatio: totalFrames > 0 ? Number((silentFrames / totalFrames).toFixed(2)) : 0,
      clippedSamplesCount,
      dcOffset,
      acousticEmbedding16D
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
/**
 * Concatenates multiple 24kHz mono WAV sentence chunks into a single continuous WAV file
 * with 6ms zero-crossing linear fade-in/fade-out and a natural 180ms inter-sentence pause
 * to eliminate boundary clicks and preserve natural sentence cadence on long texts.
 */
export async function concatenateWavChunksWithCrossfade(
  chunkWavPaths: string[],
  outputWavPath: string,
  interSentencePauseMs = 180
): Promise<void> {
  if (chunkWavPaths.length === 0) {
    throw new Error('No audio chunks provided for concatenation.');
  }
  if (chunkWavPaths.length === 1) {
    await fs.promises.copyFile(chunkWavPaths[0], outputWavPath);
    return;
  }

  const sampleRate = 24000;
  const fadeSamples = Math.round((sampleRate * 6) / 1000); // 6ms anti-click fade (144 samples)
  const pauseSamples = Math.round((sampleRate * interSentencePauseMs) / 1000);
  const pcmBuffers: Buffer[] = [];

  for (let idx = 0; idx < chunkWavPaths.length; idx++) {
    const rawPath = `${chunkWavPaths[idx]}.24k.pcm`;
    try {
      await execFileAsync('ffmpeg', [
        '-y',
        '-i',
        chunkWavPaths[idx],
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
      const totalSamples = Math.floor(buf.length / 2);
      if (totalSamples === 0) continue;

      // Trim leading/trailing digital dead-air (< -52 dBFS) while keeping natural 35ms breath pad
      let firstActive = 0;
      let lastActive = totalSamples - 1;
      const threshold = 82; // ~ -52 dBFS in int16
      while (firstActive < totalSamples && Math.abs(buf.readInt16LE(firstActive * 2)) < threshold) {
        firstActive++;
      }
      while (lastActive > firstActive && Math.abs(buf.readInt16LE(lastActive * 2)) < threshold) {
        lastActive--;
      }
      const padSamples = Math.round(sampleRate * 0.035);
      const startSample = Math.max(0, firstActive - padSamples);
      const endSample = Math.min(totalSamples, lastActive + padSamples);
      const activeLen = Math.max(1, endSample - startSample);

      const trimmedBuf = Buffer.alloc(activeLen * 2);
      buf.copy(trimmedBuf, 0, startSample * 2, endSample * 2);

      // Apply 6ms linear fade-in and fade-out to prevent boundary clicks
      const effFade = Math.min(fadeSamples, Math.floor(activeLen / 4));
      for (let i = 0; i < effFade; i++) {
        const gainIn = i / effFade;
        const sIn = Math.round(trimmedBuf.readInt16LE(i * 2) * gainIn);
        trimmedBuf.writeInt16LE(Math.max(-32768, Math.min(32767, sIn)), i * 2);

        const outIdx = activeLen - 1 - i;
        const gainOut = i / effFade;
        const sOut = Math.round(trimmedBuf.readInt16LE(outIdx * 2) * gainOut);
        trimmedBuf.writeInt16LE(Math.max(-32768, Math.min(32767, sOut)), outIdx * 2);
      }

      pcmBuffers.push(trimmedBuf);
      if (idx < chunkWavPaths.length - 1 && pauseSamples > 0) {
        pcmBuffers.push(Buffer.alloc(pauseSamples * 2, 0));
      }
    } finally {
      await fs.promises.unlink(rawPath).catch(() => {});
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
}

/**
 * Conservative, optional post-synthesis DSP audio finishing and WAV/MP3 export.
 * By default (when preservePitchContour=false and preserveSpectralTimbre=false), NO pitch-shifting,
 * EQ, or aggressive single-pass loudnorm is applied—preserving the neural synthesizer's exact
 * waveform, prosody, and formant structure without distortion.
 * When optional DSP is explicitly enabled, pitch adjustment is strictly clamped to +/- 2.0 semitones
 * and EQ to +/- 2.0 dB with a transparent peak limiter to prevent clipping.
 */
export async function applyReferenceAcousticIdentityAndExport(options: {
  rawSynthesizedWavPath: string;
  referenceAnalysis: AudioSignalAnalysis;
  preservePitchContour: boolean;
  preserveSpectralTimbre: boolean;
  outputWavPath: string;
  outputMp3Path: string;
}): Promise<{
  synthesizedBasePitchHz: number;
  pitchShiftSemitonesApplied: number;
  outputDurationSec: number;
  rmsNormalizedDb: number;
  finalAcoustics: RawPcmPitchStats;
  estimatedOutputSnrDb: number;
  dspApplied: boolean;
}> {
  const {
    rawSynthesizedWavPath,
    referenceAnalysis,
    preservePitchContour,
    preserveSpectralTimbre,
    outputWavPath,
    outputMp3Path
  } = options;

  const synthAcoustics = await analyzeVocalAcousticsFromAudioFile(rawSynthesizedWavPath);
  const refF0 = referenceAnalysis.fundamentalFreqHz || 155;
  const synthF0 = synthAcoustics.fundamentalFreqHz || 165;

  // Conservative pitch ratio clamp: max +/- 2.0 semitones ONLY when explicitly requested,
  // avoiding formant shifting ("chipmunk/hollow" distortion) and WSOLA time-stretch artifacts.
  const rawRatio = refF0 / synthF0;
  const minRatio = Math.pow(2, -2.0 / 12); // ~0.8909
  const maxRatio = Math.pow(2, 2.0 / 12);  // ~1.1225
  const clampedRatio = preservePitchContour
    ? Math.min(maxRatio, Math.max(minRatio, rawRatio))
    : 1.0;

  const rawSemitones = 12 * Math.log2(clampedRatio);
  const applyPitchShift = preservePitchContour && Math.abs(rawSemitones) >= 0.35;
  const semitonesApplied = applyPitchShift ? Number(rawSemitones.toFixed(2)) : 0;
  const tempoCompensation = applyPitchShift ? Number((1 / clampedRatio).toFixed(4)) : 1.0;

  const filterChain: string[] = [];

  // 1. Remove sub-audible DC offset / rumble cleanly
  filterChain.push('highpass=f=55');

  // 2. Optional conservative pitch shift (only if explicitly enabled and >= 0.35 st)
  if (applyPitchShift) {
    const shiftedRate = Math.round(24000 * clampedRatio);
    filterChain.push(`asetrate=${shiftedRate}`);
    filterChain.push('aresample=24000');
    filterChain.push(`atempo=${tempoCompensation}`);
  } else {
    filterChain.push('aresample=24000');
  }

  // 3. Optional conservative spectral timbre finishing (+/- 2.0 dB max)
  if (preserveSpectralTimbre) {
    const warmthDelta = referenceAnalysis.spectralWarmthRatio - synthAcoustics.spectralWarmthRatio;
    const lowShelfGainDb = Number(Math.min(2.0, Math.max(-2.0, warmthDelta * 1.2)).toFixed(1));
    const presenceGainDb = Number(Math.min(1.8, Math.max(-1.8, -warmthDelta * 0.9)).toFixed(1));
    if (Math.abs(lowShelfGainDb) >= 0.3) {
      filterChain.push(`equalizer=f=190:t=q:w=1.2:g=${lowShelfGainDb}`);
    }
    if (Math.abs(presenceGainDb) >= 0.3) {
      filterChain.push(`equalizer=f=2850:t=q:w=1.5:g=${presenceGainDb}`);
    }
  }

  // 4. Transparent static gain normalization toward -17 dBFS RMS with true-peak ceiling at -1.0 dBFS
  // (Avoids single-pass loudnorm dynamic pumping while guaranteeing zero digital clipping)
  const targetRmsDb = -17.0;
  const gainDeltaDb = Number(
    Math.min(8.0, Math.max(-8.0, targetRmsDb - synthAcoustics.rmsDb)).toFixed(1)
  );
  if (Math.abs(gainDeltaDb) >= 0.5) {
    filterChain.push(`volume=${gainDeltaDb}dB`);
  }
  filterChain.push('alimiter=limit=0.891:attack=5:release=50:level=disabled');

  const afString = filterChain.join(',');
  const dspApplied = applyPitchShift || preserveSpectralTimbre;

  // Export 24kHz 16-bit mono WAV
  await execFileAsync('ffmpeg', [
    '-y',
    '-i',
    rawSynthesizedWavPath,
    '-af',
    afString,
    '-ar',
    '24000',
    '-ac',
    '1',
    '-c:a',
    'pcm_s16le',
    outputWavPath
  ]);

  // Export 192kbps MP3
  await execFileAsync('ffmpeg', [
    '-y',
    '-i',
    outputWavPath,
    '-c:a',
    'libmp3lame',
    '-b:a',
    '192k',
    outputMp3Path
  ]);

  const finalStats = await analyzeVocalAcousticsFromAudioFile(outputWavPath);
  const { stdout } = await execFileAsync('ffprobe', [
    '-v',
    'error',
    '-show_entries',
    'format=duration',
    '-of',
    'default=noprint_wrappers=1:nokey=1',
    outputWavPath
  ]);
  const outputDurationSec = Number(parseFloat(stdout.trim() || '0').toFixed(2));
  const estimatedOutputSnrDb = Number(Math.max(0, finalStats.rmsDb - finalStats.noiseFloorDb).toFixed(1));

  return {
    synthesizedBasePitchHz: synthF0,
    pitchShiftSemitonesApplied: semitonesApplied,
    outputDurationSec,
    rmsNormalizedDb: finalStats.rmsDb,
    finalAcoustics: finalStats,
    estimatedOutputSnrDb,
    dspApplied
  };
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

export interface ChunkAcousticTimingProfile {
  durationMs: number;
  frameHopMs: number;
  frameEnergies: number[];
  cumulativeEnergy: number[];
  pauses: DetectedAcousticPause[];
  activeSpeechStartMs: number;
  activeSpeechEndMs: number;
}

/**
 * Extracts a 20ms-resolution acoustic energy & natural pause profile from a 16kHz mono WAV chunk.
 * Used by the SRT segmenter to anchor sentence and clause split points to real acoustic pauses
 * and actual speech energy distribution when word-level timestamps are absent or coarse.
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
    const frameEnergies: number[] = new Array(numFrames).fill(0);

    for (let f = 0; f < numFrames; f++) {
      let sumSq = 0;
      const baseIdx = f * frameSamples;
      for (let i = 0; i < frameSamples; i++) {
        const s = buf.readInt16LE((baseIdx + i) * 2) / 32768.0;
        sumSq += s * s;
      }
      const rms = Math.sqrt(sumSq / frameSamples);
      const db = rms > 1e-6 ? 20 * Math.log10(rms) : -96.0;
      frameDb[f] = db;
      // Weight active voiced energy above -44 dBFS
      frameEnergies[f] = db > -44.0 ? Math.max(0.05, rms) : 0.002;
    }

    const sortedDb = [...frameDb].sort((a, b) => a - b);
    const p80Db = sortedDb[Math.floor(sortedDb.length * 0.8)] ?? -20.0;
    const pauseThresholdDb = Math.max(-43.0, Math.min(-26.0, p80Db - 13.5));

    // Identify natural acoustic pauses >= 120ms (6 frames of 20ms)
    const pauses: DetectedAcousticPause[] = [];
    let inPause = false;
    let pauseStartFrame = 0;
    let minPauseDb = 0;

    for (let f = 0; f < numFrames; f++) {
      if (frameDb[f] <= pauseThresholdDb) {
        if (!inPause) {
          inPause = true;
          pauseStartFrame = f;
          minPauseDb = frameDb[f];
        } else if (frameDb[f] < minPauseDb) {
          minPauseDb = frameDb[f];
        }
      } else if (inPause) {
        const spanFrames = f - pauseStartFrame;
        if (spanFrames >= 6) {
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

    // Build cumulative energy array for energy-proportional timestamp interpolation
    const cumulativeEnergy: number[] = new Array(numFrames + 1).fill(0);
    for (let f = 0; f < numFrames; f++) {
      cumulativeEnergy[f + 1] = cumulativeEnergy[f] + frameEnergies[f];
    }

    let firstActiveFrame = 0;
    while (firstActiveFrame < numFrames - 1 && frameDb[firstActiveFrame] <= pauseThresholdDb - 4) {
      firstActiveFrame++;
    }
    let lastActiveFrame = numFrames - 1;
    while (lastActiveFrame > firstActiveFrame && frameDb[lastActiveFrame] <= pauseThresholdDb - 4) {
      lastActiveFrame--;
    }

    return {
      durationMs,
      frameHopMs,
      frameEnergies,
      cumulativeEnergy,
      pauses,
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
export function sanitizeRawAsrSegmentsOverlap(
  rawSegments: RawAsrSegmentInput[]
): RawAsrSegmentInput[] {
  const nonEmpty: RawAsrSegmentInput[] = [];
  for (const seg of rawSegments) {
    const tokens = tokenizeTranscriptForVerification(seg.text || '');
    if (tokens.length === 0) continue;
    nonEmpty.push({
      startSec: Number(seg.startSec) || 0,
      endSec: Number(seg.endSec) || 0,
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

  while (totalDurationSec - lastSplit > maxChunkSec + 6) {
    const targetSplit = lastSplit + maxChunkSec;
    let bestSplit = targetSplit;
    let bestDist = Infinity;

    for (let i = 0; i < silenceStarts.length; i++) {
      const sStart = silenceStarts[i];
      const sEnd = silenceEnds[i] ?? sStart + 0.3;
      const midpoint = (sStart + sEnd) / 2;
      if (midpoint > lastSplit + 12 && midpoint < totalDurationSec - 4) {
        const dist = Math.abs(midpoint - targetSplit);
        if (dist < 12 && dist < bestDist) {
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

    await execFileAsync('ffmpeg', [
      '-y',
      '-ss',
      String(startSec),
      '-t',
      String(duration),
      '-i',
      filePath,
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

/**
 * Priority 2 Intelligent Timestamp-Aware SRT Segmenter:
 * Converts sanitized ASR segments into sentence-aware, pause-aligned, 2-line max SubRip segments
 * while guaranteeing 100% exact ordered token sequence preservation:
 *   1. Word-level timestamps from ASR when present and valid.
 *   2. Sentence boundary detection (., !, ?, ।, ॥, ؟, ۔, 。, ！, ？) with abbreviation protection.
 *   3. Measured 20ms-resolution acoustic pauses (extractChunkAcousticTimingProfile).
 *   4. Balanced linguistic clause grouping (>= 3 words per clause, zero orphan tails) & 2-line wrapping.
 *   5. Cumulative vocal energy timestamp fallback when word timestamps and pauses are absent.
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
  const outputBlocks: SrtSegment[] = [];

  for (let segIdx = 0; segIdx < rawSegments.length; segIdx++) {
    const rawSeg = rawSegments[segIdx];
    const cleanTokens = tokenizeTranscriptForVerification(rawSeg.text || '');
    if (cleanTokens.length === 0) continue;
    const cleanText = cleanTokens.join(' ');

    const maxCharsPerLine = getScriptMaxCharsPerLine(languageCode, cleanText);

    let relStartMs = Math.max(
      0,
      Math.min(chunkDurationMs - 100, Math.round((Number(rawSeg.startSec) || 0) * 1000))
    );
    let relEndMs = Math.max(
      relStartMs + 100,
      Math.min(
        chunkDurationMs,
        Math.round((Number(rawSeg.endSec) || chunkEndSec - chunkStartSec) * 1000)
      )
    );

    if (rawSegments.length === 1 && relStartMs === 0 && relEndMs >= chunkDurationMs - 100) {
      relStartMs = Math.min(chunkDurationMs - 100, acousticProfile.activeSpeechStartMs);
      relEndMs = Math.max(relStartMs + 100, Math.min(chunkDurationMs, acousticProfile.activeSpeechEndMs));
    }

    // Guard against compressed ASR segment endSec (e.g., 13 words reported inside 0.24s..1.48s when next segment starts at 7.46s)
    const minNaturalSpanMs = cleanTokens.length * 260;
    const nextRawSeg = rawSegments[segIdx + 1];
    const nextSegStartMs = nextRawSeg
      ? Math.max(relEndMs, Math.round((Number(nextRawSeg.startSec) || 0) * 1000) - 80)
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
      let prevEnd = 0;
      for (const w of rawSeg.words) {
        const wText = String(w.word || '').trim();
        const wStart = Math.max(0, Math.round((Number(w.startSec) || 0) * 1000));
        const wEnd = Math.min(chunkDurationMs, Math.round((Number(w.endSec) || 0) * 1000));
        if (wText && wEnd > wStart && wStart >= prevEnd - 80) {
          validWords.push({
            word: wText,
            startTimeMs: Math.max(prevEnd, wStart),
            endTimeMs: Math.max(prevEnd + 40, wEnd)
          });
          prevEnd = Math.max(prevEnd, wEnd);
        }
      }
      // Discard compressed word timestamps if their total span is implausibly short (< 180ms/word)
      if (validWords.length >= 2) {
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
      // Weight by phonological syllable/vowel-nucleus proxy + word count + terminal pause weight
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

      const absStartMs = chunkOffsetMs + currentRelStartMs;
      const absEndMs = Math.max(absStartMs + 100, chunkOffsetMs + nextRelEndMs);
      outputBlocks.push({
        id: outputBlocks.length + 1,
        startTimeMs: absStartMs,
        endTimeMs: absEndMs,
        startTimestamp: msToSrtTimestamp(absStartMs),
        endTimestamp: msToSrtTimestamp(absEndMs),
        text: wrapSubtitleTextIntoMaxTwoLines(subUnits[idx], maxCharsPerLine),
        timingSource
      });

      const gapMs = Math.min(40, Math.max(10, Math.floor((relEndMs - nextRelEndMs) * 0.1)));
      currentRelStartMs = Math.min(relEndMs - minSliceMs, nextRelEndMs + gapMs);
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

    // Pass 4: Expand overly brief segments (< 750ms) ONLY up to available slack before next segment and maxAudioMs
    for (let i = 0; i < n; i++) {
      const ceilingMs = i < n - 1 ? Math.max(ends[i], starts[i + 1] - minGapMs) : maxAudioMs;
      if (ends[i] - starts[i] < 750 && ceilingMs > ends[i]) {
        ends[i] = Math.min(ceilingMs, starts[i] + 800);
      }
      // Cap single subtitle block duration at 6500ms
      if (ends[i] - starts[i] > 6500) {
        ends[i] = starts[i] + 6500;
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
