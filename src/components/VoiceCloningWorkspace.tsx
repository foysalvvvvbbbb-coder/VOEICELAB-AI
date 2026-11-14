import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  Upload,
  Mic,
  Square,
  Download,
  AlertTriangle,
  CheckCircle2,
  ShieldCheck,
  Volume2,
  Sliders,
  RefreshCw,
  FileAudio,
  Lock,
  ChevronDown,
  ChevronUp,
  Cpu,
  Sparkles,
  Maximize2,
  Minimize2,
  X,
  Layers,
  Copy,
  Check,
  Terminal
} from 'lucide-react';
import {
  AudioSignalAnalysis,
  CloningLanguageCode,
  VoiceCloneResult
} from '../types/voicelab';
import {
  CLONING_LANGUAGES,
  validateSameLanguageText
} from '../data/researchAndConfigs';
import { buildColabOneCellSetupScript } from '../data/colabSetupScript';
import {
  ALL_RECOGNIZED_TAG_LIST,
  buildEmotionPreservingChunks,
  MAX_SUPPORTED_CLONING_SCRIPT_CHARS,
  parseEmotionTaggedScript,
  RECOGNIZED_EMOTION_TAGS,
  tokenizeScriptForLiveEditor
} from '../utils/emotionTagParser';
import { fetchApiJson } from '../utils/apiClient';
import {
  completeOperation,
  connectBackendOperationStream,
  failOperation,
  recordOperationRetry,
  startOperation,
  updateOperationStage,
  updateUploadBytes
} from '../utils/activityManager';

interface VoiceCloningWorkspaceProps {
  onNavigateBackToDashboard?: () => void;
  onActivityComplete?: (info: {
    languageLabel: string;
    downloadUrl: string;
    downloadFilename: string;
  }) => void;
}

const PIPELINE_A_STEPS = [
  '01. Reference Audio Validation & Signal Check',
  '02. Emotion Tag Parsing & Same-Language Script Guard',
  '03. Reference Speaker Identity & Emotion-Scoped Chunk Synthesis',
  '04. Reference F0 & 16-Band Spectral Acoustic Transfer',
  '05. ASR Back-Transcription & Zero-Literal-Tag Verification'
];

function splitPreviewSentences(text: string): string[] {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (!clean) return [];
  const rawParts = clean.split(/(?<=[.!?।॥\n])\s+/u);
  return rawParts.map((s) => s.trim()).filter(Boolean);
}

export const VoiceCloningWorkspace: React.FC<VoiceCloningWorkspaceProps> = ({
  onActivityComplete
}) => {
  const [selectedLang, setSelectedLang] = useState<CloningLanguageCode>('bn');
  const [referenceAnalysis, setReferenceAnalysis] = useState<AudioSignalAnalysis | null>(null);
  const [referenceTranscript, setReferenceTranscript] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [inputText, setInputText] = useState<string>(
    CLONING_LANGUAGES.bn.sampleTexts[0].text
  );
  const preservePitchContour = false;
  const preserveSpectralTimbre = false;
  const [localEndpointUrl, setLocalEndpointUrl] = useState<string>('');
  const [localNodeToken, setLocalNodeToken] = useState<string>('voicelab-local-token');
  const [enableCloudVerification, setEnableCloudVerification] = useState<boolean>(false);
  const [showColabGuide, setShowColabGuide] = useState<boolean>(false);
  const [copiedColabScript, setCopiedColabScript] = useState<boolean>(false);
  const [consentConfirmed, setConsentConfirmed] = useState(true);

  // Script Editor UI states
  const [isFullscreenEditor, setIsFullscreenEditor] = useState(false);

  // Collapsible progressive disclosure states
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [showSignalDetails, setShowSignalDetails] = useState(false);
  const [showDetailedMetrics, setShowDetailedMetrics] = useState(false);
  const [showChunkBreakdown, setShowChunkBreakdown] = useState(false);

  const [isGenerating, setIsGenerating] = useState(false);
  const [activePipelineStep, setActivePipelineStep] = useState(0);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [cloneResult, setCloneResult] = useState<VoiceCloneResult | null>(null);
  const [nodeHealthStatus, setNodeHealthStatus] = useState<{
    checking: boolean;
    message?: string;
    ok?: boolean;
  }>({ checking: false });

  // Microphone recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scriptTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const highlightMirrorRef = useRef<HTMLDivElement | null>(null);

  const currentLangConfig = CLONING_LANGUAGES[selectedLang];

  // Live emotion tag parsing, tokenizing, and chunk preview
  const liveTokens = useMemo(() => tokenizeScriptForLiveEditor(inputText), [inputText]);
  const parsedScript = useMemo(() => parseEmotionTaggedScript(inputText), [inputText]);
  const previewChunks = useMemo(
    () => buildEmotionPreservingChunks(parsedScript, splitPreviewSentences, 320),
    [parsedScript]
  );
  const scriptValidation = useMemo(
    () => validateSameLanguageText(inputText, selectedLang),
    [inputText, selectedLang]
  );

  // Estimate spoken duration (~12.5 chars/sec for Indic/English speech + explicit [pause] tags)
  const estimatedDurationSec = useMemo(() => {
    const speechSec = parsedScript.spokenCharCount / 12.5;
    const pauseSec =
      parsedScript.pauseInstructions.reduce((sum, p) => sum + p.durationMs, 0) / 1000;
    return Math.max(0, Math.round(speechSec + pauseSec));
  }, [parsedScript]);

  const activeScriptTagTokens = useMemo(
    () =>
      liveTokens.filter(
        (tok) => tok.type === 'recognized-tag' || tok.type === 'unknown-tag'
      ),
    [liveTokens]
  );

  const detectedRefCode = referenceAnalysis?.detectedSpokenLanguage?.code;
  const isReferenceLanguageMismatch = Boolean(
    detectedRefCode &&
      ['bn', 'hi', 'en'].includes(detectedRefCode) &&
      detectedRefCode !== selectedLang
  );

  const syncEditorScroll = () => {
    if (scriptTextareaRef.current && highlightMirrorRef.current) {
      highlightMirrorRef.current.scrollTop = scriptTextareaRef.current.scrollTop;
      highlightMirrorRef.current.scrollLeft = scriptTextareaRef.current.scrollLeft;
    }
  };

  const handleInsertEmotionTagAtCursor = (tagToken: string) => {
    const textarea = scriptTextareaRef.current;
    if (!textarea) {
      setInputText((prev) => `${prev}${prev.endsWith(' ') || !prev ? '' : ' '}${tagToken} `);
      return;
    }

    const start = textarea.selectionStart ?? inputText.length;
    const end = textarea.selectionEnd ?? inputText.length;
    const before = inputText.slice(0, start);
    const after = inputText.slice(end);
    const needsLeadingSpace = before.length > 0 && !/\s$/.test(before);
    const needsTrailingSpace = after.length === 0 || !/^\s/.test(after);
    const insertion = `${needsLeadingSpace ? ' ' : ''}${tagToken}${needsTrailingSpace ? ' ' : ''}`;
    const nextValue = `${before}${insertion}${after}`;
    setInputText(nextValue);

    const nextCursorPos = start + insertion.length;
    requestAnimationFrame(() => {
      textarea.focus();
      textarea.setSelectionRange(nextCursorPos, nextCursorPos);
    });
  };

  const handleRemoveTagTokenAtRange = (startIndex: number, endIndex: number) => {
    const before = inputText.slice(0, startIndex);
    const after = inputText.slice(endIndex);
    // Collapse double space left behind when removing an inline tag without touching words
    const cleaned =
      before.endsWith(' ') && after.startsWith(' ')
        ? `${before}${after.slice(1)}`
        : `${before}${after}`;
    setInputText(cleaned);
  };

  const handleStripAllBracketTags = () => {
    setInputText(parsedScript.spokenText);
  };

  const handleSelectLanguage = (lang: CloningLanguageCode) => {
    setSelectedLang(lang);
    setGenerationError(null);
    const checkExisting = validateSameLanguageText(inputText, lang);
    if (!checkExisting.isValid) {
      setInputText(CLONING_LANGUAGES[lang].sampleTexts[0].text);
    }
    const op = startOperation({
      feature: 'voice-loading',
      title: `Switch to ${CLONING_LANGUAGES[lang].englishName} (${CLONING_LANGUAGES[lang].nativeName})`,
      stages: [
        `Locking same-language script guard to ${CLONING_LANGUAGES[lang].scriptName}`,
        `Loading ${CLONING_LANGUAGES[lang].nativeName} synthesis presets`
      ],
      initialProgressPct: 100,
      progressSource: 'task-steps',
      canCancel: false
    });
    completeOperation(
      op.operationId,
      `Switched to ${CLONING_LANGUAGES[lang].englishName} (${CLONING_LANGUAGES[lang].nativeName})`
    );
  };

  const handleLoadAuthorizedSample = async (lang: CloningLanguageCode) => {
    setIsUploading(true);
    setUploadError(null);
    setGenerationError(null);
    const langCfg = CLONING_LANGUAGES[lang];
    const op = startOperation({
      feature: 'voice-loading',
      title: `Load ${langCfg.englishName} Sample Voice`,
      stages: [
        `Requesting authorized ${langCfg.nativeName} studio sample`,
        'Preparing WAV audio & running FFprobe/FFmpeg signal check',
        'Applying reference vocal profile & alignment transcript'
      ],
      initialProgressPct: null,
      progressSource: 'indeterminate-inference',
      canCancel: true
    });
    const stopStream = connectBackendOperationStream(op.operationId);

    try {
      const data = await fetchApiJson<{ status: string; analysis: AudioSignalAnalysis }>(
        `/api/sample-reference/${lang}`,
        {
          operationId: op.operationId,
          signal: op.signal,
          onRetry: (_, reason) => recordOperationRetry(op.operationId, reason)
        }
      );
      if (op.isStale()) return;

      updateOperationStage(op.operationId, {
        stageIndex: 2,
        stageLabel: 'Applying reference vocal profile & alignment transcript...'
      });
      setReferenceAnalysis(data.analysis);
      setReferenceTranscript(
        parseEmotionTaggedScript(CLONING_LANGUAGES[lang].sampleTexts[0].text).spokenText
      );
      if (selectedLang !== lang) {
        setSelectedLang(lang);
        const checkExisting = validateSameLanguageText(inputText, lang);
        if (!checkExisting.isValid) {
          setInputText(CLONING_LANGUAGES[lang].sampleTexts[0].text);
        }
      }
      completeOperation(
        op.operationId,
        `Loaded ${langCfg.englishName} sample (${data.analysis.durationSeconds}s · ${data.analysis.fundamentalFreqHz} Hz)`
      );
    } catch (err: any) {
      if (err?.name === 'AbortError' || op.isStale()) return;
      const msg = err?.message || 'Could not load studio sample reference.';
      setUploadError(msg);
      failOperation(op.operationId, msg);
    } finally {
      stopStream();
      setIsUploading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setUploadError(null);
    setGenerationError(null);

    const op = startOperation({
      feature: 'audio-upload',
      title: `Upload & Analyze ${file.name}`,
      stages: [
        'Uploading reference audio bytes',
        'Verifying audio container signature & running FFprobe/FFmpeg DSP',
        'Analyzing spoken language & reference vocal identity'
      ],
      initialProgressPct: 0,
      progressSource: 'upload-bytes',
      canCancel: true,
      cancelPreviousSameFeature: true
    });
    const stopStream = connectBackendOperationStream(op.operationId);

    try {
      const formData = new FormData();
      formData.append('audio', file);
      formData.append('workspace', 'voice-cloning');

      const data = await fetchApiJson<{ status: string; analysis: AudioSignalAnalysis }>(
        '/api/audio/upload',
        {
          method: 'POST',
          body: formData,
          operationId: op.operationId,
          signal: op.signal,
          onUploadProgress: (loaded, total) => updateUploadBytes(op.operationId, loaded, total),
          onRetry: (_, reason) => recordOperationRetry(op.operationId, reason)
        }
      );
      if (op.isStale()) return;
      setReferenceAnalysis(data.analysis);
      completeOperation(
        op.operationId,
        `Validated ${data.analysis.fileName} (${data.analysis.durationSeconds}s · ${data.analysis.fundamentalFreqHz} Hz)`
      );
    } catch (err: any) {
      if (err?.name === 'AbortError' || op.isStale()) return;
      const msg = err?.message || 'Failed to upload and validate reference audio.';
      setUploadError(msg);
      failOperation(op.operationId, msg);
    } finally {
      stopStream();
      setIsUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const startMicrophoneRecording = async () => {
    setUploadError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recordedChunksRef.current = [];

      recorder.ondataavailable = (ev) => {
        if (ev.data.size > 0) {
          recordedChunksRef.current.push(ev.data);
        }
      };

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(recordedChunksRef.current, { type: 'audio/webm' });
        const recordedFile = new File(
          [blob],
          `mic_reference_${selectedLang}_${Date.now()}.webm`,
          { type: 'audio/webm' }
        );
        await handleFileUpload(recordedFile);
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
      setRecordingSeconds(0);
      recordingTimerRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch {
      setUploadError(
        'Microphone access unavailable. Please upload an audio file or load a studio sample.'
      );
    }
  };

  const stopMicrophoneRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
    }
  };

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
    };
  }, []);

  const handleGenerateClonedVoice = async (reuseCompletedChunks = true) => {
    if (!referenceAnalysis) {
      setGenerationError('Please upload or load an authorized reference voice first.');
      return;
    }
    if (!scriptValidation.isValid) {
      setGenerationError(scriptValidation.errorMessage || 'Text language mismatch.');
      return;
    }
    if (isReferenceLanguageMismatch) {
      setGenerationError(
        `Cross-Language Blocked: Reference audio language (${detectedRefCode?.toUpperCase()}) does not match selected target language (${selectedLang.toUpperCase()}).`
      );
      return;
    }
    if (!consentConfirmed) {
      setGenerationError('Please confirm voice authorization before generating.');
      return;
    }

    setIsGenerating(true);
    setGenerationError(null);
    setActivePipelineStep(0);

    const op = startOperation({
      feature: 'voice-cloning',
      title: `${currentLangConfig.englishName} (${currentLangConfig.nativeName}) Synthesis (${previewChunks.length} chunk${previewChunks.length === 1 ? '' : 's'})`,
      stages: [
        'Parsing emotion tags & verifying genuine cloning capability',
        `Synthesizing ${previewChunks.length} emotion-scoped chunk(s) conditioned on reference audio`,
        'Finalizing lossless 24kHz WAV/MP3 loudness & anti-clipping mastering',
        'Computing 3072-D & 16-D audio embeddings',
        'Running single-pass ASR back-transcription & zero-literal-tag check'
      ],
      initialProgressPct: null,
      progressSource: 'indeterminate-inference',
      canCancel: true
    });
    const stopStream = connectBackendOperationStream(op.operationId);

    const stepInterval = window.setInterval(() => {
      setActivePipelineStep((prev) => (prev < PIPELINE_A_STEPS.length - 1 ? prev + 1 : prev));
    }, 1200);

    try {
      const data = await fetchApiJson<{ status: string; result: VoiceCloneResult }>(
        '/api/voice-clone/generate',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          operationId: op.operationId,
          signal: op.signal,
          onRetry: (_, reason) => recordOperationRetry(op.operationId, reason),
          body: JSON.stringify({
            referenceFileId: referenceAnalysis.fileId,
            language: selectedLang,
            text: inputText,
            referenceTranscript: referenceTranscript.trim() || undefined,
            preservePitchContour,
            preserveSpectralTimbre,
            reuseCompletedChunks,
            localEndpointUrl: localEndpointUrl.trim() || undefined,
            localNodeToken: localNodeToken.trim() || undefined,
            enableCloudVerification,
            consentConfirmed
          })
        }
      );

      if (op.isStale()) return;

      const resObj: VoiceCloneResult = data.result;
      setCloneResult(resObj);
      setActivePipelineStep(PIPELINE_A_STEPS.length - 1);
      const matchPct =
        resObj.qualityControlReport.separatedMetrics.asrTranscriptAccuracy.characterMatchRatePct;
      completeOperation(
        op.operationId,
        `${currentLangConfig.englishName} speech ready (${resObj.outputDurationSec}s · ${resObj.chunkCount} chunk${resObj.chunkCount === 1 ? '' : 's'} · ${matchPct !== null ? `${matchPct}% ASR match` : '0-Quota Mode 2'})`
      );

      if (onActivityComplete) {
        onActivityComplete({
          languageLabel: `${currentLangConfig.englishName} (${currentLangConfig.nativeName})`,
          downloadUrl: `${resObj.wavUrl}?download=1`,
          downloadFilename: `voicelab_${resObj.language}_${resObj.jobId}.wav`
        });
      }
    } catch (err: any) {
      if (err?.name === 'AbortError' || op.isStale()) return;
      const msg = err?.message || 'Voice cloning pipeline encountered an error.';
      setGenerationError(msg);
      failOperation(op.operationId, msg);
    } finally {
      stopStream();
      clearInterval(stepInterval);
      setIsGenerating(false);
    }
  };

  const handleTrackAudioDownload = (format: 'WAV' | 'MP3', filename: string) => {
    const op = startOperation({
      feature: 'export-download',
      title: `Export ${format} Audio`,
      stages: [
        `Preparing ${format} audio stream (${filename})`,
        'Starting browser file download'
      ],
      initialProgressPct: 100,
      progressSource: 'task-steps',
      canCancel: false
    });
    completeOperation(op.operationId, `Downloading ${filename}`);
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const handleVerifyLocalNode = async () => {
    const raw = localEndpointUrl.trim();
    if (!raw) {
      setNodeHealthStatus({
        checking: false,
        ok: false,
        message:
          'Enter your Mode 2 Neural Node URL first (e.g. https://xxxx.trycloudflare.com from Colab T4 or http://127.0.0.1:8001).'
      });
      return;
    }
    setNodeHealthStatus({ checking: true });
    try {
      const resp = await fetchApiJson<{
        reachable: boolean;
        origin?: string;
        health?: { readyForLanguage?: Record<string, boolean> };
        error?: string;
      }>('/api/voice-clone/node-health', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          localEndpointUrl: raw,
          localNodeToken: localNodeToken.trim() || undefined
        })
      });
      if (!resp.reachable) {
        setNodeHealthStatus({
          checking: false,
          ok: false,
          message: resp.error || `Node unreachable at ${resp.origin || raw}.`
        });
        return;
      }
      const langReady = resp.health?.readyForLanguage?.[selectedLang];
      setNodeHealthStatus({
        checking: false,
        ok: langReady !== false,
        message:
          langReady === false
            ? `Node reachable at ${resp.origin}, but ${currentLangConfig.englishName} (${selectedLang}) checkpoint is not loaded.`
            : `Verified Mode 2 Neural Node online at ${resp.origin} (${currentLangConfig.englishName} ready · 0 Cloud Quota).`
      });
    } catch (err: any) {
      setNodeHealthStatus({
        checking: false,
        ok: false,
        message: err?.message || 'Failed to verify Mode 2 Neural Node.'
      });
    }
  };

  const handleCopyColabScript = async () => {
    const scriptText = buildColabOneCellSetupScript(localNodeToken);
    try {
      await navigator.clipboard.writeText(scriptText);
      setCopiedColabScript(true);
      window.setTimeout(() => setCopiedColabScript(false), 2500);
    } catch {
      setShowColabGuide(true);
    }
  };

  const scriptFontClass =
    selectedLang === 'bn' ? 'font-bengali' : selectedLang === 'hi' ? 'font-hindi' : 'font-sans';

  return (
    <div className="space-y-5">
      {/* Compact Workspace Header + Language Lock + Mode Status Panel */}
      <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 sm:p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-xl sm:text-2xl font-bold text-[#F4F8F5] tracking-tight">
            Voice Cloning Studio
          </h1>
          <p className="text-xs text-[#A2B5AA]">
            Reference-conditioned same-language voice synthesis with inline emotion &amp; delivery
            tags and automatic multi-chunk long-script generation.
          </p>
        </div>

        {/* Compact Status & Restriction Panel */}
        <div className="flex flex-wrap items-center gap-2.5 text-xs">
          <div className="bg-[#0B1512] border border-[#19352A] rounded-xl px-3 py-2 flex items-center gap-2">
            <Lock className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
            <span className="text-[#F4F8F5] font-medium whitespace-nowrap">
              {currentLangConfig.nativeName} → {currentLangConfig.nativeName} Only
            </span>
          </div>

          <div className="bg-[#0B1512] border border-[#19352A] rounded-xl px-3 py-2 flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
            <span className="text-[#A2B5AA] font-mono whitespace-nowrap">
              {localEndpointUrl.trim()
                ? `Mode 2: Zero-Shot Neural (${currentLangConfig.openSourceEquivalent.split(' ')[0]})`
                : 'No Mode 2 Node (Cloud Models Do Not Clone)'}
            </span>
          </div>
        </div>
      </div>

      {/* Compact Language Selector Tabs */}
      <div className="grid grid-cols-3 gap-2.5">
        {(['bn', 'hi', 'en'] as CloningLanguageCode[]).map((langKey) => {
          const cfg = CLONING_LANGUAGES[langKey];
          const isSelected = selectedLang === langKey;
          return (
            <button
              key={langKey}
              type="button"
              onClick={() => handleSelectLanguage(langKey)}
              className={`p-3 sm:p-3.5 rounded-xl border text-left transition-colors cursor-pointer ${
                isSelected
                  ? 'bg-[#19352A] border-[#16A36A] text-[#F4F8F5]'
                  : 'bg-[#12231D] border-[#19352A] text-[#A2B5AA] hover:border-[#16A36A]/50 hover:text-[#F4F8F5]'
              }`}
            >
              <div className="flex items-center justify-between gap-1">
                <span
                  className={`text-sm sm:text-base font-bold truncate ${
                    langKey === 'bn'
                      ? 'font-bengali'
                      : langKey === 'hi'
                      ? 'font-hindi'
                      : ''
                  }`}
                >
                  {cfg.nativeName}
                </span>
                <span className="hidden sm:inline text-[11px] font-mono text-[#42D9A0] shrink-0">
                  {cfg.isoTag}
                </span>
              </div>
              <div className="text-[11px] text-[#A2B5AA] truncate mt-0.5">
                {cfg.englishName}
              </div>
            </button>
          );
        })}
      </div>

      {/* Main Two-Column Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column (5 cols): Reference Audio Upload, Recording & Reference Vocal Identity */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 sm:p-5 space-y-4">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-sm font-semibold text-[#F4F8F5]">
                01. Reference Voice ({currentLangConfig.nativeName})
              </h2>
              <span className="text-[11px] font-mono text-[#A2B5AA]">5–30s audio</span>
            </div>

            {/* Compact Drag and Drop Zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="border border-dashed border-[#19352A] hover:border-[#16A36A] bg-[#0B1512]/70 rounded-xl p-4 text-center cursor-pointer transition-colors"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.mp3,.m4a,.flac,.ogg,audio/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) handleFileUpload(file);
                }}
              />
              <Upload className="w-5 h-5 text-[#42D9A0] mx-auto mb-1.5" />
              <div className="text-xs sm:text-sm font-medium text-[#F4F8F5]">
                {isUploading
                  ? 'Inspecting audio signal...'
                  : 'Upload Reference Audio (Drop or Browse)'}
              </div>
              <div className="text-[11px] text-[#A2B5AA] mt-0.5 font-mono">
                WAV · MP3 · M4A · FLAC · OGG
              </div>
            </div>

            {/* Record Mic OR Load Studio Sample */}
            <div className="grid grid-cols-2 gap-2">
              {!isRecording ? (
                <button
                  type="button"
                  onClick={startMicrophoneRecording}
                  disabled={isUploading}
                  className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl bg-[#0B1512] hover:bg-[#19352A] border border-[#19352A] text-xs font-medium text-[#F4F8F5] transition-colors cursor-pointer whitespace-nowrap"
                >
                  <Mic className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
                  <span>Record Mic</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={stopMicrophoneRecording}
                  className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl bg-red-950/80 hover:bg-red-900 border border-red-500/50 text-xs font-medium text-red-100 transition-colors cursor-pointer whitespace-nowrap"
                >
                  <Square className="w-3.5 h-3.5 text-red-400 shrink-0" />
                  <span className="font-mono tabular-nums">Stop ({recordingSeconds}s)</span>
                </button>
              )}

              <button
                type="button"
                onClick={() => handleLoadAuthorizedSample(selectedLang)}
                disabled={isUploading}
                className="flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl bg-[#19352A] hover:bg-[#16A36A] text-xs font-medium text-[#F4F8F5] transition-colors cursor-pointer whitespace-nowrap"
              >
                <Volume2 className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
                <span>Load {currentLangConfig.nativeName} Sample</span>
              </button>
            </div>

            {uploadError && (
              <div className="bg-red-950/50 border border-red-500/40 rounded-xl p-3 text-xs text-red-200 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <span>{uploadError}</span>
              </div>
            )}

            {/* Compact Reference Audio Card + Vocal Identity Lock Summary */}
            {referenceAnalysis && (
              <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3.5 space-y-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <FileAudio className="w-4 h-4 text-[#42D9A0] shrink-0" />
                    <span className="text-xs font-semibold text-[#F4F8F5] truncate">
                      {referenceAnalysis.fileName}
                    </span>
                  </div>
                  <span className="text-[11px] font-mono text-[#42D9A0] shrink-0">
                    {referenceAnalysis.durationSeconds}s · {referenceAnalysis.fundamentalFreqHz} Hz
                  </span>
                </div>

                <audio
                  controls
                  src={referenceAnalysis.previewUrl}
                  className="w-full h-9 rounded-lg"
                />

                {/* Reference Speaker Conditioning Status Badge */}
                <div className="bg-[#12231D] border border-[#19352A] rounded-lg p-2.5 text-[11px] space-y-1">
                  <div className="flex items-center justify-between text-[#42D9A0] font-semibold">
                    <span>Reference Speaker Conditioning</span>
                    <span className="font-mono">
                      {localEndpointUrl.trim() ? 'Mode 2 Zero-Shot' : 'Mode 2 Required'}
                    </span>
                  </div>
                  <p className="text-[#A2B5AA] leading-relaxed">
                    {localEndpointUrl.trim() ? (
                      <>
                        Mode 2 Local Zero-Shot Neural Node configured (
                        <span className="text-[#F4F8F5] font-mono">
                          {currentLangConfig.openSourceEquivalent}
                        </span>
                        ). Every chunk receives your exact uploaded reference audio bytes for
                        consistent speaker identity.
                      </>
                    ) : (
                      <>
                        Cloud Gemini speech models only provide 5 fixed prebuilt voices and cannot
                        clone uploaded reference audio. VoiceLab AI will not substitute a generic
                        Gemini voice—connect Mode 2 Local Neural Node (
                        <span className="text-[#F4F8F5] font-mono">http://127.0.0.1:8001</span>)
                        under settings to synthesize true reference-conditioned speech.
                      </>
                    )}
                  </p>
                </div>

                {/* Cross-Language Mismatch Alert */}
                {isReferenceLanguageMismatch && (
                  <div className="bg-amber-950/60 border border-amber-500/50 rounded-lg p-2.5 text-xs text-amber-200 flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                    <div>
                      Detected{' '}
                      <span className="font-semibold">
                        {referenceAnalysis.detectedSpokenLanguage?.name}
                      </span>{' '}
                      speech, which mismatches {currentLangConfig.nativeName}.{' '}
                      <button
                        type="button"
                        onClick={() =>
                          handleSelectLanguage(detectedRefCode as CloningLanguageCode)
                        }
                        className="underline text-[#42D9A0] font-semibold cursor-pointer"
                      >
                        Switch to{' '}
                        {CLONING_LANGUAGES[detectedRefCode as CloningLanguageCode]?.nativeName}
                      </button>
                    </div>
                  </div>
                )}

                {/* Collapsible Signal Details */}
                <button
                  type="button"
                  onClick={() => setShowSignalDetails((prev) => !prev)}
                  className="w-full pt-2 border-t border-[#19352A] flex items-center justify-between text-[11px] text-[#A2B5AA] hover:text-[#F4F8F5] cursor-pointer"
                >
                  <span>
                    Signal Analysis ({formatBytes(referenceAnalysis.fileSizeBytes)} ·{' '}
                    {referenceAnalysis.estimatedSnrDb} dB SNR)
                  </span>
                  {showSignalDetails ? (
                    <ChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5" />
                  )}
                </button>

                {showSignalDetails && (
                  <div className="grid grid-cols-2 gap-2 pt-1 text-[11px] font-mono tabular-nums text-[#A2B5AA]">
                    <div>
                      Pitch Range:{' '}
                      <span className="text-[#F4F8F5]">
                        {referenceAnalysis.pitchRangeHz[0]}–{referenceAnalysis.pitchRangeHz[1]} Hz
                      </span>
                    </div>
                    <div>
                      Sample Rate:{' '}
                      <span className="text-[#F4F8F5]">{referenceAnalysis.sampleRateHz} Hz</span>
                    </div>
                    <div>
                      RMS / Peak:{' '}
                      <span className="text-[#F4F8F5]">
                        {referenceAnalysis.rmsLevelDb} / {referenceAnalysis.peakLevelDb} dB
                      </span>
                    </div>
                    <div>
                      Detected:{' '}
                      <span className="text-[#42D9A0]">
                        {referenceAnalysis.detectedSpokenLanguage?.name || 'Verified'}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Reference Transcript Box */}
            <div className="pt-2 border-t border-[#19352A] space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <label className="font-medium text-[#F4F8F5]">
                  Reference Transcript (Optional Alignment)
                </label>
                <span className="text-[11px] text-[#A2B5AA]">For IndicF5 Conditioning</span>
              </div>
              <textarea
                rows={2}
                value={referenceTranscript}
                onChange={(e) => setReferenceTranscript(e.target.value)}
                placeholder={`Optional spoken transcript of the ${currentLangConfig.nativeName} reference audio...`}
                className={`w-full rounded-xl bg-[#0B1512] border border-[#19352A] focus:border-[#16A36A] p-2.5 text-xs text-[#F4F8F5] placeholder-[#A2B5AA]/50 focus:outline-none ${scriptFontClass}`}
              />
            </div>
          </div>
        </div>

        {/* Right Column (7 cols): Emotion-Tagged Script Editor, Long-Script Telemetry, Settings & Output */}
        <div className="lg:col-span-7 space-y-4">
          <div
            className={`${
              isFullscreenEditor
                ? 'fixed inset-3 z-50 overflow-y-auto bg-[#0B1512] border-2 border-[#16A36A] shadow-2xl'
                : 'bg-[#12231D] border border-[#19352A]'
            } rounded-2xl p-4 sm:p-5 space-y-4`}
          >
            {/* Header + Expand Toggle */}
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold text-[#F4F8F5]">
                  02. Emotion-Tagged Synthesis Script ({currentLangConfig.nativeName})
                </h2>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#0B1512] border border-[#19352A] text-[#42D9A0]">
                  Long-Script Auto-Chunking
                </span>
              </div>

              <div className="flex items-center gap-2">
                <span className="text-xs font-mono tabular-nums text-[#A2B5AA]">
                  {parsedScript.rawCharCount.toLocaleString()} /{' '}
                  {MAX_SUPPORTED_CLONING_SCRIPT_CHARS.toLocaleString()} chars
                </span>
                <button
                  type="button"
                  onClick={() => setIsFullscreenEditor((prev) => !prev)}
                  title={isFullscreenEditor ? 'Exit Expanded Editor' : 'Expand Script Editor'}
                  className="p-1.5 rounded-lg bg-[#0B1512] hover:bg-[#19352A] border border-[#19352A] text-[#A2B5AA] hover:text-[#F4F8F5] transition-colors cursor-pointer"
                >
                  {isFullscreenEditor ? (
                    <Minimize2 className="w-3.5 h-3.5" />
                  ) : (
                    <Maximize2 className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>
            </div>

            {/* Sample Presets */}
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] text-[#A2B5AA] mr-1">Presets:</span>
              {currentLangConfig.sampleTexts.map((preset, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setInputText(preset.text)}
                  className="px-2.5 py-1 rounded-lg bg-[#0B1512] hover:bg-[#19352A] border border-[#19352A] text-xs text-[#F4F8F5] transition-colors cursor-pointer whitespace-nowrap"
                >
                  {preset.label}
                </button>
              ))}
            </div>

            {/* Interactive Emotion, Delivery & Timing Tag Palette (Click to Insert at Cursor) */}
            <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3 space-y-2">
              <div className="flex items-center justify-between gap-2 text-[11px]">
                <span className="text-[#A2B5AA] flex items-center gap-1.5 font-medium">
                  <Sparkles className="w-3.5 h-3.5 text-[#42D9A0]" />
                  <span>
                    Insert Emotion / Delivery Tag at Cursor (Never spoken aloud — controls prosody):
                  </span>
                </span>
                {activeScriptTagTokens.length > 0 && (
                  <button
                    type="button"
                    onClick={handleStripAllBracketTags}
                    className="text-[11px] text-[#A2B5AA] hover:text-red-300 underline cursor-pointer whitespace-nowrap"
                  >
                    Strip All Tags
                  </button>
                )}
              </div>

              <div className="flex flex-wrap gap-1.5">
                {ALL_RECOGNIZED_TAG_LIST.map((tagSpec) => (
                  <button
                    key={tagSpec.name}
                    type="button"
                    onClick={() => handleInsertEmotionTagAtCursor(tagSpec.token)}
                    title={`Insert ${tagSpec.token} (${tagSpec.label})`}
                    className={`px-2 py-0.5 rounded-md border text-[11px] font-mono transition-transform active:scale-95 cursor-pointer ${tagSpec.badgeColorClass}`}
                  >
                    {tagSpec.token}
                  </button>
                ))}
              </div>
            </div>

            {/* Live Syntax-Highlighted Script Editor (Overlay Mirror + Native Textarea) */}
            <div className="space-y-2">
              <div
                className={`relative w-full rounded-xl bg-[#0B1512] border transition-colors ${
                  !scriptValidation.isValid && inputText.trim().length > 0
                    ? 'border-red-500/70 focus-within:border-red-400'
                    : 'border-[#19352A] focus-within:border-[#16A36A]'
                }`}
              >
                {/* Highlighted Backdrop Mirror Layer */}
                <div
                  ref={highlightMirrorRef}
                  aria-hidden="true"
                  className={`pointer-events-none w-full p-3.5 text-sm leading-relaxed whitespace-pre-wrap break-words overflow-y-auto ${
                    isFullscreenEditor ? 'min-h-[340px] max-h-[520px]' : 'min-h-[150px] max-h-[300px]'
                  } ${scriptFontClass}`}
                >
                  {liveTokens.map((tok, i) => {
                    if (tok.type === 'recognized-tag' && tok.recognizedName) {
                      const spec = RECOGNIZED_EMOTION_TAGS[tok.recognizedName];
                      return (
                        <mark
                          key={`${tok.startIndex}_${i}`}
                          className={`bg-transparent rounded px-0.5 font-mono ${spec.inlineHighlightClass}`}
                        >
                          {tok.raw}
                        </mark>
                      );
                    }
                    if (tok.type === 'unknown-tag') {
                      return (
                        <mark
                          key={`${tok.startIndex}_${i}`}
                          className="bg-red-500/30 text-red-200 underline decoration-wavy decoration-red-400 rounded px-0.5 font-mono"
                        >
                          {tok.raw}
                        </mark>
                      );
                    }
                    return (
                      <span key={`${tok.startIndex}_${i}`} className="text-[#F4F8F5]">
                        {tok.raw}
                      </span>
                    );
                  })}
                  {inputText.endsWith('\n') ? '\n ' : ''}
                </div>

                {/* Transparent-Text Native Textarea Layer (Preserves Cursor, Undo/Redo, Copy/Paste & Mobile Input) */}
                <textarea
                  ref={scriptTextareaRef}
                  rows={isFullscreenEditor ? 14 : 6}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onScroll={syncEditorScroll}
                  spellCheck={false}
                  placeholder={`Type or paste ${currentLangConfig.nativeName} (${currentLangConfig.englishName}) script with tags like [sad], [happy], [whispering], [confidently], [pause]...`}
                  className={`absolute inset-0 w-full h-full resize-none bg-transparent p-3.5 text-sm leading-relaxed text-transparent caret-[#42D9A0] selection:bg-[#16A36A]/35 placeholder-[#A2B5AA]/50 focus:outline-none overflow-y-auto ${scriptFontClass}`}
                />
              </div>

              {/* Active Tag Chips Strip (Interactive Removable Chips) */}
              {activeScriptTagTokens.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                  <span className="text-[11px] text-[#A2B5AA] mr-1">
                    Active tags in script ({activeScriptTagTokens.length}):
                  </span>
                  {activeScriptTagTokens.map((tok, idx) => {
                    const spec =
                      tok.type === 'recognized-tag' && tok.recognizedName
                        ? RECOGNIZED_EMOTION_TAGS[tok.recognizedName]
                        : null;
                    return (
                      <span
                        key={`${tok.startIndex}_${idx}`}
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md border text-[11px] font-mono ${
                          spec
                            ? spec.badgeColorClass
                            : 'bg-red-500/20 border-red-400/50 text-red-200'
                        }`}
                      >
                        <span>{tok.raw}</span>
                        <button
                          type="button"
                          onClick={() => handleRemoveTagTokenAtRange(tok.startIndex, tok.endIndex)}
                          title={`Remove ${tok.raw} from script`}
                          className="hover:opacity-100 opacity-75 cursor-pointer"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    );
                  })}
                </div>
              )}

              {/* Unsupported Tag Warning Banner */}
              {parsedScript.unsupportedTags.length > 0 && (
                <div className="bg-amber-950/60 border border-amber-500/50 rounded-xl p-2.5 text-xs text-amber-200 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    Unrecognized bracket tag(s) detected:{' '}
                    <span className="font-mono font-semibold">
                      {parsedScript.unsupportedTags.join(', ')}
                    </span>
                    . Unrecognized tags are automatically stripped before synthesis so they are{' '}
                    <span className="font-semibold underline">never spoken aloud</span>. Supported
                    tags: [happy], [sad], [angry], [excited], [calm], [serious], [whispering],
                    [softly], [confidently], [shouting], [pause].
                  </div>
                </div>
              )}

              {/* Live Long-Script Telemetry Bar */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-[11px] font-mono">
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg px-2.5 py-1.5 flex items-center justify-between">
                  <span className="text-[#A2B5AA]">Spoken Chars</span>
                  <span className="text-[#F4F8F5] font-semibold tabular-nums">
                    {parsedScript.spokenCharCount.toLocaleString()}
                  </span>
                </div>
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg px-2.5 py-1.5 flex items-center justify-between">
                  <span className="text-[#A2B5AA]">Words</span>
                  <span className="text-[#F4F8F5] font-semibold tabular-nums">
                    {parsedScript.spokenWordCount.toLocaleString()}
                  </span>
                </div>
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg px-2.5 py-1.5 flex items-center justify-between">
                  <span className="text-[#A2B5AA]">Est. Duration</span>
                  <span className="text-[#42D9A0] font-semibold tabular-nums">
                    ~{estimatedDurationSec}s
                  </span>
                </div>
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg px-2.5 py-1.5 flex items-center justify-between">
                  <span className="text-[#A2B5AA]">Auto-Chunks</span>
                  <span className="text-[#42D9A0] font-semibold tabular-nums">
                    {previewChunks.length} chunk{previewChunks.length === 1 ? '' : 's'}
                  </span>
                </div>
              </div>

              {/* Script Guard Status */}
              {!scriptValidation.isValid && inputText.trim().length > 0 ? (
                <div className="mt-2 bg-red-950/60 border border-red-500/50 rounded-xl p-3 text-xs text-red-200 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                  <span>{scriptValidation.errorMessage}</span>
                </div>
              ) : (
                <div className="mt-1 flex items-center justify-between text-[11px] text-[#A2B5AA]">
                  <span>
                    Script Guard:{' '}
                    <span className="text-[#42D9A0] font-medium">
                      Valid {currentLangConfig.nativeName} ({currentLangConfig.scriptName}) ·
                      Bracket tags excluded from speech
                    </span>
                  </span>
                  <span className="font-mono tabular-nums">
                    bn:{scriptValidation.bengaliCharCount} · hi:{scriptValidation.hindiCharCount} ·
                    en:{scriptValidation.latinCharCount}
                  </span>
                </div>
              )}
            </div>

            {/* Collapsible Advanced Node & DSP Settings */}
            <div className="rounded-xl bg-[#0B1512] border border-[#19352A]">
              <button
                type="button"
                onClick={() => setShowAdvancedSettings((prev) => !prev)}
                className="w-full px-3.5 py-2.5 flex items-center justify-between text-xs font-medium text-[#F4F8F5] hover:text-[#42D9A0] transition-colors cursor-pointer"
              >
                <span className="flex items-center gap-2">
                  <Sliders className="w-3.5 h-3.5 text-[#42D9A0]" />
                  <span>Reference Conditioning &amp; DSP Settings</span>
                </span>
                <span className="flex items-center gap-1.5 text-[11px] font-mono text-[#A2B5AA]">
                  <span>
                    {localEndpointUrl.trim() ? 'Mode 2 Local Connected' : 'Configure Mode 2 Node'}
                  </span>
                  {showAdvancedSettings ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </span>
              </button>

              {showAdvancedSettings && (
                <div className="p-3.5 pt-2 border-t border-[#19352A] space-y-3 text-xs">
                  <div className="space-y-1.5">
                    <label className="block font-medium text-[#F4F8F5]">
                      Mode 2 Neural Node URL (Free Colab T4 HTTPS Tunnel or Local 127.0.0.1:8001)
                    </label>
                    <div className="flex gap-2">
                      <input
                        type="url"
                        value={localEndpointUrl}
                        onChange={(e) => {
                          setLocalEndpointUrl(e.target.value);
                          setNodeHealthStatus({ checking: false });
                        }}
                        placeholder="https://xxxx.trycloudflare.com or http://127.0.0.1:8001"
                        className="flex-1 rounded-lg bg-[#12231D] border border-[#19352A] focus:border-[#16A36A] px-3 py-1.5 text-xs font-mono text-[#F4F8F5] placeholder-[#A2B5AA]/50 focus:outline-none"
                      />
                      <button
                        type="button"
                        onClick={handleVerifyLocalNode}
                        disabled={nodeHealthStatus.checking}
                        className="px-3 py-1.5 rounded-lg bg-[#19352A] hover:bg-[#16A36A] text-xs font-medium text-[#F4F8F5] transition-colors cursor-pointer whitespace-nowrap"
                      >
                        {nodeHealthStatus.checking ? 'Checking...' : 'Verify Node'}
                      </button>
                    </div>
                    {nodeHealthStatus.message && (
                      <div
                        className={`text-[11px] px-2.5 py-1.5 rounded-lg border ${
                          nodeHealthStatus.ok
                            ? 'bg-emerald-950/50 border-emerald-500/40 text-emerald-200'
                            : 'bg-amber-950/50 border-amber-500/40 text-amber-200'
                        }`}
                      >
                        {nodeHealthStatus.message}
                      </div>
                    )}
                  </div>

                  <div className="space-y-1.5">
                    <label className="block font-medium text-[#F4F8F5]">
                      Node Auth Token (<span className="font-mono text-[#42D9A0]">X-VoiceLab-Node-Token</span>)
                    </label>
                    <input
                      type="text"
                      value={localNodeToken}
                      onChange={(e) => {
                        setLocalNodeToken(e.target.value);
                        setNodeHealthStatus({ checking: false });
                      }}
                      placeholder="voicelab-local-token"
                      className="w-full rounded-lg bg-[#12231D] border border-[#19352A] focus:border-[#16A36A] px-3 py-1.5 text-xs font-mono text-[#F4F8F5] placeholder-[#A2B5AA]/50 focus:outline-none"
                    />
                  </div>

                  <label className="flex items-start gap-2 text-[11px] text-[#A2B5AA] cursor-pointer">
                    <input
                      type="checkbox"
                      checked={enableCloudVerification}
                      onChange={(e) => setEnableCloudVerification(e.target.checked)}
                      className="accent-[#16A36A] mt-0.5 shrink-0"
                    />
                    <span>
                      Run optional Gemini Cloud ASR &amp; 3072-D Embedding check (uses 3 Gemini API calls per generation; keep unchecked for <strong className="text-[#42D9A0]">0 Cloud Quota unlimited cloning</strong>).
                    </span>
                  </label>

                  {/* 1-Click Free Google Colab T4 GPU Setup */}
                  <div className="rounded-lg bg-[#12231D] border border-[#19352A] p-2.5 space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <button
                        type="button"
                        onClick={() => setShowColabGuide((prev) => !prev)}
                        className="flex items-center gap-1.5 text-xs font-semibold text-[#42D9A0] hover:underline cursor-pointer"
                      >
                        <Terminal className="w-3.5 h-3.5" />
                        <span>Free Google Colab T4 GPU Setup (৳0 Unlimited Cloning)</span>
                      </button>
                      <button
                        type="button"
                        onClick={handleCopyColabScript}
                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md bg-[#19352A] hover:bg-[#16A36A] text-[11px] font-medium text-[#F4F8F5] transition-colors cursor-pointer"
                      >
                        {copiedColabScript ? (
                          <>
                            <Check className="w-3 h-3 text-[#42D9A0]" />
                            <span>Copied Colab Cell!</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            <span>Copy 1-Cell Colab Script</span>
                          </>
                        )}
                      </button>
                    </div>

                    {showColabGuide && (
                      <div className="space-y-2 pt-1 border-t border-[#19352A] text-[11px] text-[#A2B5AA]">
                        <ol className="list-decimal list-inside space-y-1 text-[#F4F8F5]/90">
                          <li>
                            Open <span className="font-mono text-[#42D9A0]">colab.research.google.com</span> → Runtime → Change runtime type → <strong className="text-[#F4F8F5]">T4 GPU</strong> (Free).
                          </li>
                          <li>
                            Copy the 1-Cell script above, paste it into a Colab cell, and run it.
                          </li>
                          <li>
                            Copy the printed <span className="font-mono text-[#42D9A0]">https://*.trycloudflare.com</span> URL into the box above and click <strong className="text-[#F4F8F5]">Verify Node</strong>.
                          </li>
                        </ol>
                        <pre className="max-h-40 overflow-y-auto rounded-lg bg-[#0B1512] border border-[#19352A] p-2 text-[10px] font-mono text-[#A2B5AA] whitespace-pre">
                          {buildColabOneCellSetupScript(localNodeToken)}
                        </pre>
                      </div>
                    )}
                  </div>

                  <div className="text-[11px] text-[#A2B5AA] bg-[#12231D] border border-[#19352A] rounded-lg p-2.5 leading-relaxed">
                    Speaker identity is preserved directly by Mode 2 neural weights (without DSP pitch
                    shifting over synthesized output). Output audio uses lossless 24kHz loudness
                    normalization and anti-clipping limiting only.
                  </div>
                </div>
              )}
            </div>

            {/* Compact Consent Checkbox */}
            <label className="flex items-center gap-2.5 px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] cursor-pointer text-xs text-[#A2B5AA]">
              <input
                type="checkbox"
                checked={consentConfirmed}
                onChange={(e) => setConsentConfirmed(e.target.checked)}
                className="accent-[#16A36A] shrink-0"
              />
              <ShieldCheck className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
              <span className="truncate">
                I confirm ownership or explicit authorization to clone this reference voice.
              </span>
            </label>

            {generationError && (
              <div className="bg-red-950/60 border border-red-500/50 rounded-xl p-3 text-xs text-red-200 space-y-2">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                  <span>{generationError}</span>
                </div>
                {previewChunks.length > 1 && (
                  <button
                    type="button"
                    onClick={() => handleGenerateClonedVoice(true)}
                    disabled={isGenerating}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-900/60 hover:bg-red-800 border border-red-400/40 text-xs font-semibold text-red-100 cursor-pointer"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Retry Failed Chunks (Reuse Completed Cached Chunks)</span>
                  </button>
                )}
              </div>
            )}

            {/* Primary Generate Action */}
            <button
              type="button"
              onClick={() => handleGenerateClonedVoice(true)}
              disabled={
                isGenerating ||
                !referenceAnalysis ||
                !scriptValidation.isValid ||
                isReferenceLanguageMismatch ||
                !consentConfirmed
              }
              className="w-full py-3 px-5 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 disabled:bg-[#19352A] disabled:text-[#A2B5AA]/50 text-[#F4F8F5] font-semibold text-sm transition-colors flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed shadow-sm whitespace-nowrap"
            >
              {isGenerating ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin shrink-0" />
                  <span>
                    Synthesizing {currentLangConfig.nativeName} Speech ({previewChunks.length}{' '}
                    chunk{previewChunks.length === 1 ? '' : 's'})...
                  </span>
                </>
              ) : (
                <>
                  <Volume2 className="w-4 h-4 shrink-0" />
                  <span>
                    Generate {currentLangConfig.nativeName} Speech ({previewChunks.length} chunk
                    {previewChunks.length === 1 ? '' : 's'})
                  </span>
                </>
              )}
            </button>

            {/* Compact Pipeline Progress */}
            {isGenerating && (
              <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3 text-xs font-mono text-[#42D9A0] flex items-center justify-between">
                <span className="truncate">{PIPELINE_A_STEPS[activePipelineStep]}</span>
                <span className="tabular-nums shrink-0 ml-2">
                  {activePipelineStep + 1}/{PIPELINE_A_STEPS.length}
                </span>
              </div>
            )}
          </div>

          {/* Output Audio Player, Emotion & Chunk Verification, & Quality Metrics */}
          {cloneResult && (
            <div className="bg-[#12231D] border border-[#16A36A]/50 rounded-2xl p-4 sm:p-5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-2 min-w-0">
                  <CheckCircle2 className="w-4 h-4 text-[#42D9A0] shrink-0" />
                  <h3 className="text-sm sm:text-base font-bold text-[#F4F8F5] truncate">
                    {CLONING_LANGUAGES[cloneResult.language].nativeName} Output Ready (
                    {cloneResult.outputDurationSec}s · {cloneResult.chunkCount || 1} chunk
                    {(cloneResult.chunkCount || 1) === 1 ? '' : 's'})
                  </h3>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <a
                    href={`${cloneResult.wavUrl}?download=1`}
                    download={`voicelab_${cloneResult.language}_${cloneResult.jobId}.wav`}
                    onClick={() =>
                      handleTrackAudioDownload(
                        'WAV',
                        `voicelab_${cloneResult.language}_${cloneResult.jobId}.wav`
                      )
                    }
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] transition-colors whitespace-nowrap"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>WAV</span>
                  </a>
                  <a
                    href={`${cloneResult.mp3Url}?download=1`}
                    download={`voicelab_${cloneResult.language}_${cloneResult.jobId}.mp3`}
                    onClick={() =>
                      handleTrackAudioDownload(
                        'MP3',
                        `voicelab_${cloneResult.language}_${cloneResult.jobId}.mp3`
                      )
                    }
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#0B1512] hover:bg-[#19352A] border border-[#19352A] text-xs font-semibold text-[#42D9A0] transition-colors whitespace-nowrap"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>MP3</span>
                  </a>
                </div>
              </div>

              {/* Audio Player */}
              <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3">
                <audio controls src={cloneResult.wavUrl} className="w-full h-9" />
              </div>

              {/* Emotion & Speaker Identity Verification Banner */}
              {cloneResult.emotionTagReport && (
                <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3 text-xs space-y-1.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-semibold text-[#F4F8F5] flex items-center gap-1.5">
                      <Sparkles className="w-3.5 h-3.5 text-[#42D9A0]" />
                      <span>Emotion &amp; Speaker Consistency Verification</span>
                    </span>
                    <span className="font-mono text-[11px] text-[#42D9A0]">
                      {cloneResult.emotionTagReport.zeroLiteralTagsInAudioVerified
                        ? '0 Literal Tags Spoken (Verified)'
                        : 'Check Transcript'}
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-[11px] text-[#A2B5AA]">
                    <span>
                      Persona:{' '}
                      <strong className="text-[#F4F8F5] font-mono">
                        {cloneResult.speakerConditioningInfo.conditionedVoicePersona}
                      </strong>
                    </span>
                    <span>·</span>
                    <span>
                      Tags Applied:{' '}
                      <strong className="text-[#42D9A0] font-mono">
                        {cloneResult.emotionTagReport.recognizedTagsUsed.length > 0
                          ? cloneResult.emotionTagReport.recognizedTagsUsed.join(' ')
                          : 'Neutral studio delivery'}
                      </strong>
                    </span>
                    {cloneResult.emotionTagReport.pauseCount > 0 && (
                      <>
                        <span>·</span>
                        <span>
                          Explicit Pauses:{' '}
                          <strong className="text-[#F4F8F5] font-mono">
                            {cloneResult.emotionTagReport.pauseCount} (
                            {cloneResult.emotionTagReport.totalPauseDurationMs}ms)
                          </strong>
                        </span>
                      </>
                    )}
                  </div>
                </div>
              )}

              {/* Compact Summary Metrics Strip */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5">
                  <div className="text-[11px] text-[#A2B5AA]">ASR Match</div>
                  <div className="font-mono tabular-nums font-semibold text-[#42D9A0]">
                    {cloneResult.qualityControlReport.separatedMetrics.asrTranscriptAccuracy
                      .characterMatchRatePct !== null
                      ? `${cloneResult.qualityControlReport.separatedMetrics.asrTranscriptAccuracy.characterMatchRatePct}%`
                      : '0 Quota (Mode 2)'}
                  </div>
                </div>
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5">
                  <div className="text-[11px] text-[#A2B5AA]">Spectral Match</div>
                  <div className="font-mono tabular-nums font-semibold text-[#F4F8F5]">
                    {(
                      cloneResult.qualityControlReport.separatedMetrics.spectralFeatureSimilarity
                        .cosineSimilarity16D * 100
                    ).toFixed(1)}
                    %
                  </div>
                </div>
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5">
                  <div className="text-[11px] text-[#A2B5AA]">SNR / Pitch Shift</div>
                  <div className="font-mono tabular-nums font-semibold text-[#F4F8F5]">
                    {
                      cloneResult.qualityControlReport.separatedMetrics.audioSignalQuality
                        .estimatedSnrDb
                    }{' '}
                    dB · {cloneResult.acousticTransferMetrics.pitchShiftSemitonesApplied} st
                  </div>
                </div>
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5">
                  <div className="text-[11px] text-[#A2B5AA]">Clipped</div>
                  <div className="font-mono tabular-nums font-semibold text-[#42D9A0]">
                    {
                      cloneResult.qualityControlReport.separatedMetrics.audioSignalQuality
                        .clippedSamplesCount
                    }{' '}
                    samples
                  </div>
                </div>
              </div>

              {/* Collapsible Per-Chunk Synthesis Breakdown */}
              {cloneResult.chunks && cloneResult.chunks.length > 0 && (
                <div className="border-t border-[#19352A] pt-2">
                  <button
                    type="button"
                    onClick={() => setShowChunkBreakdown((prev) => !prev)}
                    className="w-full flex items-center justify-between text-xs text-[#A2B5AA] hover:text-[#F4F8F5] cursor-pointer py-1"
                  >
                    <span className="flex items-center gap-1.5">
                      <Layers className="w-3.5 h-3.5 text-[#42D9A0]" />
                      <span>
                        Synthesized Chunk Breakdown ({cloneResult.chunks.length} chunk
                        {cloneResult.chunks.length === 1 ? '' : 's'} · Deterministic Cache Enabled)
                      </span>
                    </span>
                    {showChunkBreakdown ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </button>

                  {showChunkBreakdown && (
                    <div className="mt-2 space-y-2 max-h-60 overflow-y-auto pr-1">
                      {cloneResult.chunks.map((chk) => (
                        <div
                          key={chk.chunkId}
                          className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5 text-xs space-y-1"
                        >
                          <div className="flex items-center justify-between gap-2 font-mono text-[11px]">
                            <span className="text-[#42D9A0] font-semibold">
                              Chunk #{chk.chunkIndex + 1} ({chk.durationSec ?? 0}s · {chk.charCount}{' '}
                              chars)
                            </span>
                            <span className="text-[#A2B5AA]">
                              {chk.activeEmotionTags.length > 0
                                ? chk.activeEmotionTags.join(' ')
                                : '[neutral]'}{' '}
                              · {chk.status.toUpperCase()}
                            </span>
                          </div>
                          <div className={`text-[#F4F8F5] ${scriptFontClass}`}>
                            {chk.spokenText}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Collapsible Detailed 5-Category Quality Report */}
              <div className="border-t border-[#19352A] pt-2">
                <button
                  type="button"
                  onClick={() => setShowDetailedMetrics((prev) => !prev)}
                  className="w-full flex items-center justify-between text-xs text-[#A2B5AA] hover:text-[#F4F8F5] cursor-pointer py-1"
                >
                  <span>
                    Detailed Quality &amp; Speaker Verification Metrics (5 Categories)
                  </span>
                  {showDetailedMetrics ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </button>

                {showDetailedMetrics && (
                  <div className="mt-3 space-y-2.5 text-xs">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                      <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5 space-y-0.5">
                        <div className="text-[11px] font-mono text-[#42D9A0]">
                          1. Speaker Conditioning &amp; Identity
                        </div>
                        <div className="text-[#F4F8F5] font-mono">
                          {
                            cloneResult.qualityControlReport.separatedMetrics
                              .speakerIdentitySimilarity.statusLabel
                          }
                        </div>
                      </div>

                      <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5 space-y-0.5">
                        <div className="text-[11px] font-mono text-[#42D9A0]">
                          2. General Audio Embedding (3072-D)
                        </div>
                        <div className="text-[#F4F8F5] font-mono">
                          {cloneResult.qualityControlReport.separatedMetrics
                            .generalAudioEmbeddingSimilarity.cosineSimilarity !== null
                            ? `${(
                                cloneResult.qualityControlReport.separatedMetrics
                                  .generalAudioEmbeddingSimilarity.cosineSimilarity * 100
                              ).toFixed(2)}% (gemini-embedding-2)`
                            : 'Unavailable'}
                        </div>
                      </div>

                      <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5 space-y-0.5">
                        <div className="text-[11px] font-mono text-[#42D9A0]">
                          3. Spectral Feature Cosine (16-D)
                        </div>
                        <div className="text-[#F4F8F5] font-mono">
                          {(
                            cloneResult.qualityControlReport.separatedMetrics
                              .spectralFeatureSimilarity.cosineSimilarity16D * 100
                          ).toFixed(2)}
                          %
                        </div>
                      </div>

                      <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5 space-y-0.5">
                        <div className="text-[11px] font-mono text-[#42D9A0]">
                          4. Signal &amp; DSP Telemetry
                        </div>
                        <div className="text-[#F4F8F5] font-mono">
                          RMS{' '}
                          {
                            cloneResult.qualityControlReport.separatedMetrics.audioSignalQuality
                              .rmsLevelDb
                          }{' '}
                          dB · Peak{' '}
                          {
                            cloneResult.qualityControlReport.separatedMetrics.audioSignalQuality
                              .peakLevelDb
                          }{' '}
                          dB · Shift {cloneResult.acousticTransferMetrics.pitchShiftSemitonesApplied}{' '}
                          st
                        </div>
                      </div>
                    </div>

                    <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5">
                      <div className="text-[11px] text-[#A2B5AA] mb-1">
                        5. ASR Back-Transcription Verification (Clean Spoken Output):
                      </div>
                      <div
                        className={`text-[#F4F8F5] ${
                          cloneResult.language === 'bn'
                            ? 'font-bengali'
                            : cloneResult.language === 'hi'
                            ? 'font-hindi'
                            : ''
                        }`}
                      >
                        {cloneResult.qualityControlReport.verifiedTranscriptBackcheck}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

