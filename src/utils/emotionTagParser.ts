export type RecognizedEmotionTagName =
  | 'happy'
  | 'sad'
  | 'angry'
  | 'excited'
  | 'calm'
  | 'serious'
  | 'whispering'
  | 'softly'
  | 'confidently'
  | 'shouting'
  | 'pause';

export type EmotionTagCategory = 'emotion' | 'delivery' | 'timing';

export const MAX_SUPPORTED_CLONING_SCRIPT_CHARS = 12000;

export interface EmotionTagDefinition {
  name: RecognizedEmotionTagName;
  rawTag: `[${RecognizedEmotionTagName}]`;
  token: `[${RecognizedEmotionTagName}]`;
  label: string;
  bengaliLabel: string;
  category: EmotionTagCategory;
  styleDirective: string;
  pauseDurationMs?: number;
  badgeClass: string;
  badgeColorClass: string;
  inlineHighlightClass: string;
  tempoMultiplier: number;
  volumeDeltaDb: number;
  pitchDeltaSemitones: number;
  prosodyHint: {
    tempoMultiplier: number;
    volumeDeltaDb: number;
    pitchDeltaSemitones: number;
  };
}

export const RECOGNIZED_EMOTION_TAGS: Record<RecognizedEmotionTagName, EmotionTagDefinition> = {
  happy: {
    name: 'happy',
    rawTag: '[happy]',
    token: '[happy]',
    label: 'Happy',
    bengaliLabel: 'আনন্দিত',
    category: 'emotion',
    styleDirective: 'Cheerful, warm, joyful, and upbeat vocal expression with a natural smile in the voice',
    badgeClass: 'bg-amber-500/20 text-amber-200 border-amber-400/50',
    badgeColorClass: 'bg-amber-500/20 text-amber-200 border-amber-400/50',
    inlineHighlightClass: 'bg-amber-500/25 text-amber-200 rounded px-0.5',
    tempoMultiplier: 1.04,
    volumeDeltaDb: 0.8,
    pitchDeltaSemitones: 0.4,
    prosodyHint: { tempoMultiplier: 1.04, volumeDeltaDb: 0.8, pitchDeltaSemitones: 0.4 }
  },
  sad: {
    name: 'sad',
    rawTag: '[sad]',
    token: '[sad]',
    label: 'Sad',
    bengaliLabel: 'দুঃখিত',
    category: 'emotion',
    styleDirective: 'Melancholic, sorrowful, heavy-hearted, and gentle slow-paced emotional delivery',
    badgeClass: 'bg-sky-500/20 text-sky-200 border-sky-400/50',
    badgeColorClass: 'bg-sky-500/20 text-sky-200 border-sky-400/50',
    inlineHighlightClass: 'bg-sky-500/25 text-sky-200 rounded px-0.5',
    tempoMultiplier: 0.92,
    volumeDeltaDb: -1.5,
    pitchDeltaSemitones: -0.5,
    prosodyHint: { tempoMultiplier: 0.92, volumeDeltaDb: -1.5, pitchDeltaSemitones: -0.5 }
  },
  angry: {
    name: 'angry',
    rawTag: '[angry]',
    token: '[angry]',
    label: 'Angry',
    bengaliLabel: 'রাগান্বিত',
    category: 'emotion',
    styleDirective: 'Intense, indignant, firm, and sharply articulated delivery with strong vocal tension',
    badgeClass: 'bg-rose-500/20 text-rose-200 border-rose-400/50',
    badgeColorClass: 'bg-rose-500/20 text-rose-200 border-rose-400/50',
    inlineHighlightClass: 'bg-rose-500/25 text-rose-200 rounded px-0.5',
    tempoMultiplier: 1.06,
    volumeDeltaDb: 2.0,
    pitchDeltaSemitones: 0.3,
    prosodyHint: { tempoMultiplier: 1.06, volumeDeltaDb: 2.0, pitchDeltaSemitones: 0.3 }
  },
  excited: {
    name: 'excited',
    rawTag: '[excited]',
    token: '[excited]',
    label: 'Excited',
    bengaliLabel: 'উচ্ছ্বসিত',
    category: 'emotion',
    styleDirective: 'Enthusiastic, high-energy, animated, and vibrant delivery with lively inflection',
    badgeClass: 'bg-orange-500/20 text-orange-200 border-orange-400/50',
    badgeColorClass: 'bg-orange-500/20 text-orange-200 border-orange-400/50',
    inlineHighlightClass: 'bg-orange-500/25 text-orange-200 rounded px-0.5',
    tempoMultiplier: 1.08,
    volumeDeltaDb: 1.5,
    pitchDeltaSemitones: 0.6,
    prosodyHint: { tempoMultiplier: 1.08, volumeDeltaDb: 1.5, pitchDeltaSemitones: 0.6 }
  },
  calm: {
    name: 'calm',
    rawTag: '[calm]',
    token: '[calm]',
    label: 'Calm',
    bengaliLabel: 'শান্ত',
    category: 'emotion',
    styleDirective: 'Serene, peaceful, composed, and unhurried soothing vocal delivery',
    badgeClass: 'bg-teal-500/20 text-teal-200 border-teal-400/50',
    badgeColorClass: 'bg-teal-500/20 text-teal-200 border-teal-400/50',
    inlineHighlightClass: 'bg-teal-500/25 text-teal-200 rounded px-0.5',
    tempoMultiplier: 0.94,
    volumeDeltaDb: -0.8,
    pitchDeltaSemitones: -0.2,
    prosodyHint: { tempoMultiplier: 0.94, volumeDeltaDb: -0.8, pitchDeltaSemitones: -0.2 }
  },
  serious: {
    name: 'serious',
    rawTag: '[serious]',
    token: '[serious]',
    label: 'Serious',
    bengaliLabel: 'গম্ভীর',
    category: 'emotion',
    styleDirective: 'Grave, authoritative, solemn, and measured broadcast delivery with steady cadence',
    badgeClass: 'bg-indigo-500/20 text-indigo-200 border-indigo-400/50',
    badgeColorClass: 'bg-indigo-500/20 text-indigo-200 border-indigo-400/50',
    inlineHighlightClass: 'bg-indigo-500/25 text-indigo-200 rounded px-0.5',
    tempoMultiplier: 0.96,
    volumeDeltaDb: 0.5,
    pitchDeltaSemitones: -0.4,
    prosodyHint: { tempoMultiplier: 0.96, volumeDeltaDb: 0.5, pitchDeltaSemitones: -0.4 }
  },
  whispering: {
    name: 'whispering',
    rawTag: '[whispering]',
    token: '[whispering]',
    label: 'Whispering',
    bengaliLabel: 'ফিসফিস করে',
    category: 'delivery',
    styleDirective: 'Hushed, breathy, intimate whispered delivery at a very low, close-mic vocal projection',
    badgeClass: 'bg-purple-500/20 text-purple-200 border-purple-400/50',
    badgeColorClass: 'bg-purple-500/20 text-purple-200 border-purple-400/50',
    inlineHighlightClass: 'bg-purple-500/25 text-purple-200 rounded px-0.5',
    tempoMultiplier: 0.93,
    volumeDeltaDb: -4.5,
    pitchDeltaSemitones: 0,
    prosodyHint: { tempoMultiplier: 0.93, volumeDeltaDb: -4.5, pitchDeltaSemitones: 0 }
  },
  softly: {
    name: 'softly',
    rawTag: '[softly]',
    token: '[softly]',
    label: 'Softly',
    bengaliLabel: 'কোমলভাবে',
    category: 'delivery',
    styleDirective: 'Gentle, tender, quiet, and delicately articulated soft vocal projection',
    badgeClass: 'bg-pink-500/20 text-pink-200 border-pink-400/50',
    badgeColorClass: 'bg-pink-500/20 text-pink-200 border-pink-400/50',
    inlineHighlightClass: 'bg-pink-500/25 text-pink-200 rounded px-0.5',
    tempoMultiplier: 0.95,
    volumeDeltaDb: -2.5,
    pitchDeltaSemitones: 0,
    prosodyHint: { tempoMultiplier: 0.95, volumeDeltaDb: -2.5, pitchDeltaSemitones: 0 }
  },
  confidently: {
    name: 'confidently',
    rawTag: '[confidently]',
    token: '[confidently]',
    label: 'Confidently',
    bengaliLabel: 'আত্মবিশ্বাসের সাথে',
    category: 'delivery',
    styleDirective: 'Assured, resonant, clear, and decisive vocal delivery with strong projection',
    badgeClass: 'bg-emerald-500/20 text-emerald-200 border-emerald-400/50',
    badgeColorClass: 'bg-emerald-500/20 text-emerald-200 border-emerald-400/50',
    inlineHighlightClass: 'bg-emerald-500/25 text-emerald-200 rounded px-0.5',
    tempoMultiplier: 1.02,
    volumeDeltaDb: 1.2,
    pitchDeltaSemitones: 0.1,
    prosodyHint: { tempoMultiplier: 1.02, volumeDeltaDb: 1.2, pitchDeltaSemitones: 0.1 }
  },
  shouting: {
    name: 'shouting',
    rawTag: '[shouting]',
    token: '[shouting]',
    label: 'Shouting',
    bengaliLabel: 'উচ্চস্বরে',
    category: 'delivery',
    styleDirective: 'Loud, emphatic, high-projection exclamatory vocal delivery with urgent energy',
    badgeClass: 'bg-red-500/20 text-red-200 border-red-400/50',
    badgeColorClass: 'bg-red-500/20 text-red-200 border-red-400/50',
    inlineHighlightClass: 'bg-red-500/25 text-red-200 rounded px-0.5',
    tempoMultiplier: 1.05,
    volumeDeltaDb: 3.2,
    pitchDeltaSemitones: 0.7,
    prosodyHint: { tempoMultiplier: 1.05, volumeDeltaDb: 3.2, pitchDeltaSemitones: 0.7 }
  },
  pause: {
    name: 'pause',
    rawTag: '[pause]',
    token: '[pause]',
    label: 'Pause (450ms)',
    bengaliLabel: 'বিরতি',
    category: 'timing',
    styleDirective: 'Explicit 450ms acoustic silence pause',
    pauseDurationMs: 450,
    badgeClass: 'bg-cyan-500/20 text-cyan-200 border-cyan-400/50',
    badgeColorClass: 'bg-cyan-500/20 text-cyan-200 border-cyan-400/50',
    inlineHighlightClass: 'bg-cyan-500/25 text-cyan-200 rounded px-0.5',
    tempoMultiplier: 1.0,
    volumeDeltaDb: 0,
    pitchDeltaSemitones: 0,
    prosodyHint: { tempoMultiplier: 1.0, volumeDeltaDb: 0, pitchDeltaSemitones: 0 }
  }
};

export const ORDERED_EMOTION_TAG_LIST: EmotionTagDefinition[] = [
  RECOGNIZED_EMOTION_TAGS.happy,
  RECOGNIZED_EMOTION_TAGS.sad,
  RECOGNIZED_EMOTION_TAGS.angry,
  RECOGNIZED_EMOTION_TAGS.excited,
  RECOGNIZED_EMOTION_TAGS.calm,
  RECOGNIZED_EMOTION_TAGS.serious,
  RECOGNIZED_EMOTION_TAGS.whispering,
  RECOGNIZED_EMOTION_TAGS.softly,
  RECOGNIZED_EMOTION_TAGS.confidently,
  RECOGNIZED_EMOTION_TAGS.shouting,
  RECOGNIZED_EMOTION_TAGS.pause
];

export const ALL_RECOGNIZED_TAG_LIST = ORDERED_EMOTION_TAG_LIST;

export interface ScriptEditorToken {
  id: string;
  type: 'text' | 'recognized-tag' | 'unknown-tag' | 'supported-tag' | 'pause-tag' | 'unsupported-tag';
  raw: string;
  rawText: string;
  startIndex: number;
  endIndex: number;
  recognizedName?: RecognizedEmotionTagName;
  tagName?: string;
  tagDef?: EmotionTagDefinition;
}

export interface ParsedEmotionSpan {
  spanIndex: number;
  spokenText: string;
  activeEmotion: RecognizedEmotionTagName | null;
  activeDelivery: RecognizedEmotionTagName | null;
  activeTags: RecognizedEmotionTagName[];
  styleDirective: string;
  leadingPauseMs: number;
  trailingPauseMs: number;
}

export interface ParsedScriptAnalysis {
  rawScript: string;
  /**
   * Pure spoken text with ALL bracket tags removed and spaces normalized.
   * Guaranteed to contain zero "[...]" tag tokens.
   */
  spokenText: string;
  editorTokens: ScriptEditorToken[];
  spans: ParsedEmotionSpan[];
  emotionInstructions: {
    tag: RecognizedEmotionTagName;
    category: 'emotion' | 'delivery';
    styleDirective: string;
    appliedToExcerpt: string;
  }[];
  pauseInstructions: {
    index: number;
    durationMs: number;
    afterTextExcerpt: string;
  }[];
  recognizedTagsUsed: RecognizedEmotionTagName[];
  unsupportedTags: string[];
  charCountTotal: number;
  rawCharCount: number;
  charCountSpoken: number;
  spokenCharCount: number;
  wordCountSpoken: number;
  spokenWordCount: number;
  estimatedDurationSec: number;
}

export interface SynthesizableScriptChunk {
  chunkIndex: number;
  chunkId: string;
  totalChunks: number;
  spokenText: string;
  activeEmotion: RecognizedEmotionTagName | null;
  activeDelivery: RecognizedEmotionTagName | null;
  activeTags: RecognizedEmotionTagName[];
  styleDirective: string;
  combinedStyleDirective: string;
  leadingPauseMs: number;
  trailingPauseMs: number;
  charCount: number;
  wordCount: number;
}

export type EmotionScriptChunk = SynthesizableScriptChunk;

const BRACKET_TAG_REGEX = /\[([^\][\r\n]{1,40})\]/g;

/**
 * Tokenizes a raw script into lossless editor tokens (preserving every character, newline, and space)
 * so the live highlighted editor can render recognized emotion tags, pause tags, and unknown tags
 * in real time while typing or pasting.
 */
export function tokenizeScriptForLiveEditor(rawScript: string): ScriptEditorToken[] {
  const source = String(rawScript ?? '');
  if (!source) return [];

  const tokens: ScriptEditorToken[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  const regex = new RegExp(BRACKET_TAG_REGEX.source, 'g');

  while ((match = regex.exec(source)) !== null) {
    const matchStart = match.index;
    const matchEnd = matchStart + match[0].length;

    if (matchStart > lastIndex) {
      const slice = source.slice(lastIndex, matchStart);
      tokens.push({
        id: `txt_${lastIndex}`,
        type: 'text',
        raw: slice,
        rawText: slice,
        startIndex: lastIndex,
        endIndex: matchStart
      });
    }

    const innerNormalized = match[1].trim().toLowerCase();
    const recognized = Object.prototype.hasOwnProperty.call(RECOGNIZED_EMOTION_TAGS, innerNormalized)
      ? RECOGNIZED_EMOTION_TAGS[innerNormalized as RecognizedEmotionTagName]
      : undefined;

    if (recognized) {
      tokens.push({
        id: `tag_${matchStart}`,
        type: 'recognized-tag',
        raw: match[0],
        rawText: match[0],
        startIndex: matchStart,
        endIndex: matchEnd,
        recognizedName: recognized.name,
        tagName: recognized.name,
        tagDef: recognized
      });
    } else {
      tokens.push({
        id: `unk_${matchStart}`,
        type: 'unknown-tag',
        raw: match[0],
        rawText: match[0],
        startIndex: matchStart,
        endIndex: matchEnd,
        tagName: innerNormalized
      });
    }

    lastIndex = matchEnd;
  }

  if (lastIndex < source.length) {
    const tail = source.slice(lastIndex);
    tokens.push({
      id: `txt_${lastIndex}`,
      type: 'text',
      raw: tail,
      rawText: tail,
      startIndex: lastIndex,
      endIndex: source.length
    });
  }

  return tokens;
}

function buildCombinedStyleDirective(
  activeEmotion: RecognizedEmotionTagName | null,
  activeDelivery: RecognizedEmotionTagName | null
): string {
  const parts: string[] = [];
  if (activeEmotion && RECOGNIZED_EMOTION_TAGS[activeEmotion]) {
    parts.push(RECOGNIZED_EMOTION_TAGS[activeEmotion].styleDirective);
  }
  if (activeDelivery && RECOGNIZED_EMOTION_TAGS[activeDelivery]) {
    parts.push(RECOGNIZED_EMOTION_TAGS[activeDelivery].styleDirective);
  }
  return parts.length > 0
    ? parts.join('. ')
    : 'Natural, expressive, authentic studio vocal delivery matching the reference speaker.';
}

function normalizeSpokenSegmentWhitespace(raw: string): string {
  return raw
    .replace(/[ \t]+/g, ' ')
    .replace(/\s+([,.;:!?।॥])/g, '$1')
    .trim();
}

/**
 * Parses a script containing bracketed emotion/delivery/pause tags into:
 * - spokenText (100% free of any bracket tags so TTS never speaks "[sad]" or "[happy]")
 * - spans (preserving emotion and delivery tag scope across sentences until changed)
 * - emotionInstructions
 * - pauseInstructions
 * - unsupportedTags (never silently dropped; surfaced for user warning while stripped from spokenText)
 */
export function parseEmotionTaggedScript(rawScript: string): ParsedScriptAnalysis {
  const editorTokens = tokenizeScriptForLiveEditor(rawScript);

  // State always starts clean for each independent script
  let currentEmotion: RecognizedEmotionTagName | null = null;
  let currentDelivery: RecognizedEmotionTagName | null = null;
  let pendingLeadingPauseMs = 0;

  const spans: ParsedEmotionSpan[] = [];
  const emotionInstructions: ParsedScriptAnalysis['emotionInstructions'] = [];
  const pauseInstructions: ParsedScriptAnalysis['pauseInstructions'] = [];
  const recognizedSet = new Set<RecognizedEmotionTagName>();
  const unsupportedSet = new Set<string>();

  let currentTextBuffer = '';

  const flushCurrentTextBuffer = () => {
    const cleaned = normalizeSpokenSegmentWhitespace(currentTextBuffer);
    currentTextBuffer = '';
    if (!cleaned) return;

    const activeTags: RecognizedEmotionTagName[] = [];
    if (currentEmotion) activeTags.push(currentEmotion);
    if (currentDelivery) activeTags.push(currentDelivery);

    const styleDirective = buildCombinedStyleDirective(currentEmotion, currentDelivery);
    const span: ParsedEmotionSpan = {
      spanIndex: spans.length,
      spokenText: cleaned,
      activeEmotion: currentEmotion,
      activeDelivery: currentDelivery,
      activeTags,
      styleDirective,
      leadingPauseMs: pendingLeadingPauseMs,
      trailingPauseMs: 0
    };
    pendingLeadingPauseMs = 0;
    spans.push(span);

    if (currentEmotion) {
      emotionInstructions.push({
        tag: currentEmotion,
        category: 'emotion',
        styleDirective: RECOGNIZED_EMOTION_TAGS[currentEmotion].styleDirective,
        appliedToExcerpt: cleaned.slice(0, 80)
      });
    }
    if (currentDelivery) {
      emotionInstructions.push({
        tag: currentDelivery,
        category: 'delivery',
        styleDirective: RECOGNIZED_EMOTION_TAGS[currentDelivery].styleDirective,
        appliedToExcerpt: cleaned.slice(0, 80)
      });
    }
  };

  for (const token of editorTokens) {
    if (token.type === 'text') {
      currentTextBuffer += token.rawText;
      continue;
    }

    if (token.type === 'unknown-tag' || token.type === 'unsupported-tag') {
      unsupportedSet.add(token.rawText);
      if (currentTextBuffer.length > 0 && !/\s$/.test(currentTextBuffer)) {
        currentTextBuffer += ' ';
      }
      continue;
    }

    if (token.tagDef && token.tagDef.category === 'timing') {
      recognizedSet.add('pause');
      const pauseMs = token.tagDef.pauseDurationMs || 450;
      flushCurrentTextBuffer();

      const lastSpan = spans[spans.length - 1];
      pauseInstructions.push({
        index: pauseInstructions.length,
        durationMs: pauseMs,
        afterTextExcerpt: lastSpan ? lastSpan.spokenText.slice(-50) : '(start of script)'
      });

      if (lastSpan) {
        lastSpan.trailingPauseMs += pauseMs;
      } else {
        pendingLeadingPauseMs += pauseMs;
      }
      continue;
    }

    if (token.tagDef) {
      const def = token.tagDef;
      recognizedSet.add(def.name);
      flushCurrentTextBuffer();

      if (def.category === 'emotion') {
        currentEmotion = def.name;
      } else if (def.category === 'delivery') {
        currentDelivery = def.name;
      }
    }
  }

  flushCurrentTextBuffer();

  if (pendingLeadingPauseMs > 0 && spans.length > 0) {
    spans[spans.length - 1].trailingPauseMs += pendingLeadingPauseMs;
    pendingLeadingPauseMs = 0;
  }

  for (const s of spans) {
    s.spokenText = s.spokenText.replace(/\[[^\]]*\]/g, ' ').replace(/\s+/g, ' ').trim();
  }
  const nonEmptySpans = spans.filter((s) => s.spokenText.length > 0);

  const fullSpokenText = nonEmptySpans.map((s) => s.spokenText).join(' ').replace(/\s+/g, ' ').trim();
  const charCountTotal = Array.from(String(rawScript || '')).length;
  const charCountSpoken = Array.from(fullSpokenText).length;
  const words = fullSpokenText ? fullSpokenText.split(/\s+/).filter(Boolean) : [];
  const wordCountSpoken = words.length;
  const totalPauseSec = pauseInstructions.reduce((acc, p) => acc + p.durationMs / 1000, 0);
  const estimatedDurationSec =
    wordCountSpoken > 0
      ? Math.max(1, Math.round(wordCountSpoken / 2.2 + totalPauseSec))
      : 0;

  return {
    rawScript,
    spokenText: fullSpokenText,
    editorTokens,
    spans: nonEmptySpans,
    emotionInstructions,
    pauseInstructions,
    recognizedTagsUsed: Array.from(recognizedSet),
    unsupportedTags: Array.from(unsupportedSet),
    charCountTotal,
    rawCharCount: charCountTotal,
    charCountSpoken,
    spokenCharCount: charCountSpoken,
    wordCountSpoken,
    spokenWordCount: wordCountSpoken,
    estimatedDurationSec
  };
}

function splitLongSentenceOnSafeWordBoundaries(sentence: string, maxChunkChars: number): string[] {
  const chars = Array.from(sentence.trim());
  if (chars.length <= maxChunkChars) {
    return [sentence.trim()];
  }

  const words = sentence.trim().split(/\s+/).filter(Boolean);
  const parts: string[] = [];
  let current = '';

  for (const word of words) {
    if (!current) {
      current = word;
      continue;
    }
    const candidate = `${current} ${word}`;
    if (Array.from(candidate).length <= maxChunkChars) {
      current = candidate;
    } else {
      parts.push(current);
      current = word;
    }
  }
  if (current) {
    parts.push(current);
  }
  return parts;
}

export function buildEmotionPreservingChunks(
  parsed: ParsedScriptAnalysis,
  sentenceSplitter: (text: string) => string[],
  maxChunkChars = 320
): SynthesizableScriptChunk[] {
  const rawChunks: Omit<SynthesizableScriptChunk, 'chunkIndex' | 'chunkId' | 'totalChunks'>[] = [];

  for (const span of parsed.spans) {
    const sentences = sentenceSplitter(span.spokenText);
    const atomicUnits: string[] = [];

    for (const sent of sentences) {
      const subUnits = splitLongSentenceOnSafeWordBoundaries(sent, maxChunkChars);
      for (const u of subUnits) {
        if (u.trim()) atomicUnits.push(u.trim());
      }
    }

    if (atomicUnits.length === 0 && span.spokenText.trim()) {
      atomicUnits.push(span.spokenText.trim());
    }

    const spanGroupedTexts: string[] = [];
    let currentGroup = '';

    for (const unit of atomicUnits) {
      if (!currentGroup) {
        currentGroup = unit;
      } else if (Array.from(`${currentGroup} ${unit}`).length <= maxChunkChars) {
        currentGroup = `${currentGroup} ${unit}`;
      } else {
        spanGroupedTexts.push(currentGroup);
        currentGroup = unit;
      }
    }
    if (currentGroup) {
      spanGroupedTexts.push(currentGroup);
    }

    for (let i = 0; i < spanGroupedTexts.length; i++) {
      const textPiece = spanGroupedTexts[i];
      rawChunks.push({
        spokenText: textPiece,
        activeEmotion: span.activeEmotion,
        activeDelivery: span.activeDelivery,
        activeTags: [...span.activeTags],
        styleDirective: span.styleDirective,
        combinedStyleDirective: span.styleDirective,
        leadingPauseMs: i === 0 ? span.leadingPauseMs : 0,
        trailingPauseMs: i === spanGroupedTexts.length - 1 ? span.trailingPauseMs : 0,
        charCount: Array.from(textPiece).length,
        wordCount: textPiece.split(/\s+/).filter(Boolean).length
      });
    }
  }

  return rawChunks.map((c, idx) => ({
    ...c,
    chunkIndex: idx,
    chunkId: `chunk_${idx + 1}_${c.activeTags.join('-') || 'neutral'}`,
    totalChunks: rawChunks.length
  }));
}

export function verifyZeroLiteralTagsInTranscript(transcript: string): {
  passed: boolean;
  verified: boolean;
  leakedTags: string[];
} {
  const text = String(transcript || '');
  const leaked: string[] = [];

  const bracketMatches = text.match(/\[[^\]]+\]/g);
  if (bracketMatches) {
    leaked.push(...bracketMatches);
  }

  const tagNames = Object.keys(RECOGNIZED_EMOTION_TAGS);
  const lower = text.toLowerCase();
  for (const t of tagNames) {
    if (lower.includes(`[${t}]`) || lower.includes(`bracket ${t}`)) {
      leaked.push(`[${t}]`);
    }
  }

  const isClean = leaked.length === 0;
  return {
    passed: isClean,
    verified: isClean,
    leakedTags: Array.from(new Set(leaked))
  };
}
