import type { SrtSegment, VoiceCloneResult } from '../types/voicelab';

export function msToSrtTimestampUtil(ms: number): string {
  const clamped = Math.max(0, Math.round(ms));
  const hours = Math.floor(clamped / 3600000);
  const minutes = Math.floor((clamped % 3600000) / 60000);
  const seconds = Math.floor((clamped % 60000) / 1000);
  const millis = clamped % 1000;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(
    seconds
  ).padStart(2, '0')},${String(millis).padStart(3, '0')}`;
}

export function msToVttTimestampUtil(ms: number): string {
  return msToSrtTimestampUtil(ms).replace(',', '.');
}

function wrapIntoTwoBalancedLines(text: string, maxCharsPerLine: number): string {
  const clean = text.replace(/\s+/g, ' ').trim();
  const chars = Array.from(clean);
  if (chars.length <= maxCharsPerLine) return clean;

  const words = clean.split(' ').filter(Boolean);
  if (words.length <= 1) return clean;

  let bestSplit = Math.ceil(words.length / 2);
  let bestDiff = Infinity;
  for (let i = 1; i < words.length; i++) {
    const line1 = words.slice(0, i).join(' ');
    const line2 = words.slice(i).join(' ');
    const len1 = Array.from(line1).length;
    const len2 = Array.from(line2).length;
    if (len1 <= maxCharsPerLine && len2 <= maxCharsPerLine) {
      const diff = Math.abs(len1 - len2);
      if (diff < bestDiff) {
        bestDiff = diff;
        bestSplit = i;
      }
    }
  }

  return `${words.slice(0, bestSplit).join(' ')}\n${words.slice(bestSplit).join(' ')}`;
}

function splitTextIntoReadableSubtitleCues(text: string, maxCharsPerBlock: number): string[] {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (!clean) return [];
  const sentences = clean
    .split(/(?<=[.!?।॥])\s+/u)
    .map((s) => s.trim())
    .filter(Boolean);

  const cues: string[] = [];
  for (const sent of sentences.length > 0 ? sentences : [clean]) {
    if (Array.from(sent).length <= maxCharsPerBlock) {
      cues.push(sent);
      continue;
    }
    const words = sent.split(/\s+/).filter(Boolean);
    let buf = '';
    for (const w of words) {
      const candidate = buf ? `${buf} ${w}` : w;
      if (Array.from(candidate).length <= maxCharsPerBlock) {
        buf = candidate;
      } else {
        if (buf) cues.push(buf);
        buf = w;
      }
    }
    if (buf) cues.push(buf);
  }
  return cues;
}

/**
 * Generates chronologically aligned, non-overlapping SubRip (.srt) segments directly
 * from a VoiceLab AI synthesized VoiceCloneResult (using real per-chunk acoustic durations
 * and explicit pause timings).
 */
export function buildSrtSegmentsFromVoiceClone(result: VoiceCloneResult): SrtSegment[] {
  const chunks = result.chunks || [];
  if (chunks.length === 0) return [];

  const totalAudioMs = Math.max(1000, Math.round((result.outputDurationSec || 1) * 1000));
  const maxCharsPerLine = result.language === 'en' ? 42 : 38;
  const maxCharsPerBlock = maxCharsPerLine * 2;

  const hasExactOffsets = chunks.every(
    (c) =>
      typeof c.actualStartMs === 'number' &&
      typeof c.actualEndMs === 'number' &&
      c.actualEndMs > c.actualStartMs
  );

  const segments: SrtSegment[] = [];

  if (hasExactOffsets) {
    const rawLastEndMs = Math.max(
      1000,
      chunks[chunks.length - 1].actualEndMs! + (chunks[chunks.length - 1].trailingPauseMs || 0)
    );
    const scale = totalAudioMs / rawLastEndMs;

    for (const chk of chunks) {
      const chunkStartMs = Math.round((chk.actualStartMs || 0) * scale);
      const chunkEndMs = Math.min(totalAudioMs, Math.round((chk.actualEndMs || 0) * scale));
      const cues = splitTextIntoReadableSubtitleCues(chk.spokenText, maxCharsPerBlock);
      const totalChars = cues.reduce((s, c) => s + Math.max(1, Array.from(c).length), 0) || 1;
      const chunkSpanMs = Math.max(400, chunkEndMs - chunkStartMs);

      let innerCursor = chunkStartMs;
      for (let i = 0; i < cues.length; i++) {
        const cueText = cues[i];
        const weight = Math.max(1, Array.from(cueText).length) / totalChars;
        const isLastCue = i === cues.length - 1;
        const proposedEnd = isLastCue
          ? chunkEndMs
          : Math.min(chunkEndMs, innerCursor + Math.max(380, Math.round(chunkSpanMs * weight)));
        const safeStart = Math.max(
          segments.length > 0 ? segments[segments.length - 1].endTimeMs + 20 : 0,
          innerCursor
        );
        const safeEnd = Math.max(safeStart + 350, proposedEnd);

        segments.push({
          id: segments.length + 1,
          startTimeMs: safeStart,
          endTimeMs: safeEnd,
          startTimestamp: msToSrtTimestampUtil(safeStart),
          endTimestamp: msToSrtTimestampUtil(safeEnd),
          text: wrapIntoTwoBalancedLines(cueText, maxCharsPerLine),
          timingSource: 'acoustic-pause-aligned'
        });

        innerCursor = safeEnd + 20;
      }
    }
  } else {
    const rawWindows = chunks.map((chk, idx) => {
      const leadMs = Math.max(0, chk.leadingPauseMs || 0);
      const trailMs = Math.max(0, chk.trailingPauseMs || 0);
      const interChunkDefaultPauseMs =
        idx < chunks.length - 1 && trailMs === 0 && (chunks[idx + 1]?.leadingPauseMs || 0) === 0
          ? 180
          : 0;
      const speechMs = Math.max(500, Math.round((chk.durationSec || 2.0) * 1000));
      return { chk, leadMs, speechMs, trailMs: trailMs + interChunkDefaultPauseMs };
    });

    const rawTotalMs =
      rawWindows.reduce((acc, w) => acc + w.leadMs + w.speechMs + w.trailMs, 0) || totalAudioMs;
    const scale = totalAudioMs / rawTotalMs;

    let cursorMs = 0;

    for (const win of rawWindows) {
      cursorMs += Math.round(win.leadMs * scale);
      const scaledSpeechMs = Math.max(450, Math.round(win.speechMs * scale));
      const cues = splitTextIntoReadableSubtitleCues(win.chk.spokenText, maxCharsPerBlock);
      const totalChars = cues.reduce((s, c) => s + Math.max(1, Array.from(c).length), 0) || 1;

      let innerCursor = cursorMs;
      const chunkEnd = Math.min(totalAudioMs, cursorMs + scaledSpeechMs);

      for (let i = 0; i < cues.length; i++) {
        const cueText = cues[i];
        const weight = Math.max(1, Array.from(cueText).length) / totalChars;
        const isLastCue = i === cues.length - 1;
        const proposedEnd = isLastCue
          ? chunkEnd
          : Math.min(chunkEnd, innerCursor + Math.max(400, Math.round(scaledSpeechMs * weight)));
        const safeStart = Math.max(
          segments.length > 0 ? segments[segments.length - 1].endTimeMs + 20 : 0,
          innerCursor
        );
        const safeEnd = Math.max(safeStart + 350, proposedEnd);

        segments.push({
          id: segments.length + 1,
          startTimeMs: safeStart,
          endTimeMs: safeEnd,
          startTimestamp: msToSrtTimestampUtil(safeStart),
          endTimestamp: msToSrtTimestampUtil(safeEnd),
          text: wrapIntoTwoBalancedLines(cueText, maxCharsPerLine),
          timingSource: 'acoustic-pause-aligned'
        });

        innerCursor = safeEnd + 20;
      }

      cursorMs = chunkEnd + Math.round(win.trailMs * scale);
    }
  }

  // Clamp final segment smoothly to total output audio length if needed
  if (segments.length > 0 && segments[segments.length - 1].endTimeMs > totalAudioMs) {
    const last = segments[segments.length - 1];
    last.endTimeMs = Math.max(last.startTimeMs + 350, totalAudioMs);
    last.endTimestamp = msToSrtTimestampUtil(last.endTimeMs);
  }

  return segments;
}

export function formatSegmentsToSrtString(segments: SrtSegment[]): string {
  return segments
    .map((seg, idx) => `${idx + 1}\n${seg.startTimestamp} --> ${seg.endTimestamp}\n${seg.text.trim()}\n`)
    .join('\n');
}

export function formatSegmentsToVttString(segments: SrtSegment[]): string {
  const body = segments
    .map(
      (seg, idx) =>
        `${idx + 1}\n${msToVttTimestampUtil(seg.startTimeMs)} --> ${msToVttTimestampUtil(
          seg.endTimeMs
        )}\n${seg.text.trim()}\n`
    )
    .join('\n');
  return `WEBVTT\n\n${body}`;
}

export function formatSegmentsToPlainText(segments: SrtSegment[]): string {
  return segments
    .map((seg) => seg.text.replace(/\n+/g, ' ').trim())
    .filter(Boolean)
    .join('\n');
}
