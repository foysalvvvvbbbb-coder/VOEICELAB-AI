export type CloningLanguageCode = 'bn' | 'hi' | 'en';

export interface CloningLanguageConfig {
  code: CloningLanguageCode;
  nativeName: string;
  englishName: string;
  isoTag: string;
  scriptName: string;
  unicodeRangeDescription: string;
  routedModelId: string;
  routedModelName: string;
  openSourceEquivalent: string;
  sampleTexts: {
    label: string;
    text: string;
  }[];
}

export interface AudioSignalAnalysis {
  fileId: string;
  fileName: string;
  mimeType: string;
  formatName: string;
  codecName: string;
  fileSizeBytes: number;
  durationSeconds: number;
  sampleRateHz: number;
  channels: number;
  bitRateKbps: number;
  rmsLevelDb: number;
  peakLevelDb: number;
  noiseFloorDb: number;
  estimatedSnrDb: number;
  silenceRatio: number;
  fundamentalFreqHz: number;
  pitchRangeHz: [number, number];
  spectralWarmthRatio: number;
  acousticEmbedding16D?: number[];
  qualityGrade: 'Studio Grade' | 'Good' | 'Acceptable (Noisy)' | 'Poor / Corrupted';
  qualityWarnings: string[];
  isValid: boolean;
  detectedSpokenLanguage?: {
    code: string;
    name: string;
    confidence: string;
    transcriptExcerpt: string;
    vocalIdentitySummary: {
      register: string;
      timbre: string;
      pace: string;
      accentNotes: string;
      recommendedBasePersona: 'Puck' | 'Charon' | 'Kore' | 'Fenrir' | 'Zephyr';
    };
  };
  previewUrl: string;
}

export interface VoiceCloneRequest {
  referenceFileId: string;
  language: CloningLanguageCode;
  text: string;
  preservePitchContour: boolean;
  preserveSpectralTimbre: boolean;
  consentConfirmed: boolean;
  localEndpointUrl?: string;
}

export interface VoiceCloneResult {
  jobId: string;
  language: CloningLanguageCode;
  cloningMode: 'local-neural-zero-shot' | 'cloud-persona-conditioned-tts';
  routedModelUsed: string;
  inputCharCount: number;
  referenceDurationSec: number;
  outputDurationSec: number;
  wavUrl: string;
  mp3Url: string;
  speakerConditioningInfo: {
    embeddingExtractorUsed: string;
    neuralEmbeddingDimension: number;
    acousticVectorDimension: number;
    conditionedVoicePersona: string;
    zeroShotNeuralActive: boolean;
  };
  acousticTransferMetrics: {
    referencePitchHz: number;
    synthesizedPitchHz: number;
    pitchShiftSemitonesApplied: number;
    spectralEqMatched: boolean;
    rmsNormalizedDb: number;
  };
  qualityControlReport: {
    passed: boolean;
    speakerSimilarityScore: number | null;
    pronunciationAccuracy: string;
    naturalnessGrade: string;
    languageConsistencyVerified: boolean;
    verifiedTranscriptBackcheck: string;
    artifactsDetected: string;
    separatedMetrics: {
      speakerIdentitySimilarity: {
        verifiedScorePct: number | null;
        ecapaCosineSimilarity: number | null;
        verificationModelUsed: string;
        statusLabel: string;
      };
      generalAudioEmbeddingSimilarity: {
        cosineSimilarity: number | null;
        modelUsed: string;
        dimension: number;
        note: string;
      };
      spectralFeatureSimilarity: {
        cosineSimilarity16D: number;
        extractorUsed: string;
        dimension: number;
      };
      asrTranscriptAccuracy: {
        characterMatchRatePct: number;
        cerPct: number;
        grade: string;
        scriptVerified: string;
      };
      audioSignalQuality: {
        sampleRateHz: number;
        rmsLevelDb: number;
        peakLevelDb: number;
        estimatedSnrDb: number;
        clippedSamplesCount: number;
        dcOffset: number;
        voicedFrameRatio: number;
        pitchStdDevHz: number;
        dspApplied: boolean;
      };
    };
  };
  createdAt: string;
}

export interface SrtWordTimestamp {
  word: string;
  startTimeMs: number;
  endTimeMs: number;
}

export interface SrtSegment {
  id: number;
  startTimeMs: number;
  endTimeMs: number;
  startTimestamp: string; // HH:MM:SS,mmm
  endTimestamp: string;   // HH:MM:SS,mmm
  text: string;
  confidence?: number;
  speaker?: string;
  timingSource?: 'word-timestamps' | 'acoustic-pause-aligned' | 'acoustic-energy-aligned';
}

export interface SrtValidationReport {
  isValid: boolean;
  chronologicalOrderVerified: boolean;
  zeroOverlapsVerified: boolean;
  zeroNegativeOrReversedVerified: boolean;
  withinAudioDurationVerified: boolean;
  sequentialNumberingVerified: boolean;
  wordPreservationVerified: boolean;
  exactSequenceMatch: boolean;
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
  originalTranscript: string;
  reconstructedTranscript: string;
  timingAlignmentSummary: string;
  maxLinesPerBlock: number;
  maxCharsPerLineObserved: number;
  minDurationMsObserved: number;
  maxDurationMsObserved: number;
  totalWordsInput: number;
  totalWordsOutput: number;
}

export interface SrtTranscriptionResult {
  jobId: string;
  fileName: string;
  durationSeconds: number;
  requestedLanguage: string; // 'auto' or ISO code
  detectedLanguageCode: string;
  detectedLanguageName: string;
  nativeScriptName: string;
  modelUsed: string;
  chunkCount: number;
  chunksProcessed: {
    chunkIndex: number;
    startSec: number;
    endSec: number;
    segmentCount: number;
    status: 'completed' | 'recovered';
  }[];
  segments: SrtSegment[];
  rawSrtContent: string;
  audioPreviewUrl: string;
  processingTimeMs: number;
  validationReport?: SrtValidationReport;
}

export interface ModelResearchEntry {
  id: string;
  name: string;
  category: 'Voice Cloning' | 'Multilingual ASR';
  officialLanguages: string;
  bengaliSupport: {
    supported: boolean;
    statusLabel: 'Verified Native' | 'Unsupported / Absent' | 'Experimental / Accent-Only';
    pronunciationQuality: string;
    notes: string;
  };
  hindiSupport: {
    supported: boolean;
    statusLabel: 'Verified Native' | 'High Accuracy' | 'Experimental';
    pronunciationQuality: string;
  };
  englishSupport: {
    supported: boolean;
    statusLabel: 'Verified Studio' | 'High Accuracy';
    pronunciationQuality: string;
  };
  referenceAudioReq: string;
  speakerSimilarity: string;
  naturalness: string;
  modelSize: string;
  cpuInference: string;
  gpuInference: string;
  ramRequirement: string;
  license: string;
  commercialUse: string;
  deploymentFeasibility: string;
  roleInVoiceLab: string;
}

export interface RealWorldTestItem {
  id: string;
  workspace: 'Voice Cloning' | 'Universal SRT';
  language: string;
  testName: string;
  inputDescription: string;
  status: 'Passed' | 'Partially Passed' | 'Failed' | 'Not Tested';
  latencyMs?: number;
  metrics: {
    label: string;
    value: string;
  }[];
  verdictNotes: string;
  sampleAudioUrl?: string;
  outputAudioUrl?: string;
  outputSrtPreview?: string;
}
