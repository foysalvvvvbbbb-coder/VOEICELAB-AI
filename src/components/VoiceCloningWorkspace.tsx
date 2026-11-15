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
  Terminal,
  BookmarkPlus,
  Pencil,
  Trash2,
  Play,
  LogIn,
  UserCheck,
  Captions,
  ArrowRight
} from 'lucide-react';
import {
  AudioSignalAnalysis,
  CloningLanguageCode,
  SavedVoiceProfile,
  SavedVoiceSourceType,
  SrtSegment,
  VoiceAudioQualityMode,
  VoiceCloneResult
} from '../types/voicelab';
import { useFirebaseAuth } from '../contexts/FirebaseAuthContext';
import {
  CLONING_LANGUAGES,
  validateSameLanguageText
} from '../data/researchAndConfigs';
import { buildColabOneCellSetupScript } from '../data/colabSetupScript';
import {
  ALL_RECOGNIZED_TAG_LIST,
  buildEmotionPreservingChunks,
  DEFAULT_CLONING_CHUNK_MAX_CHARS,
  MAX_SUPPORTED_CLONING_SCRIPT_CHARS,
  parseEmotionTaggedScript,
  RECOGNIZED_EMOTION_TAGS,
  tokenizeScriptForLiveEditor
} from '../utils/emotionTagParser';
import {
  buildSrtSegmentsFromVoiceClone,
  formatSegmentsToSrtString
} from '../utils/voiceToSrt';
import { fetchApiJson } from '../utils/apiClient';
import {
  completeOperation,
  connectBackendOperationStream,
  failOperation,
  recordOperationRetry,
  startOperation,
  updateOperationStage,
  updateUploadBytes,
  useLiveActivityState
} from '../utils/activityManager';

export interface VoiceToSrtHandoff {
  fileName: string;
  audioUrl: string;
  languageCode: string;
  languageName: string;
  scriptName: string;
  durationSeconds: number;
  segments: SrtSegment[];
}

interface VoiceCloningWorkspaceProps {
  onNavigateBackToDashboard?: () => void;
  onSendToSrtEditor?: (handoff: VoiceToSrtHandoff) => void;
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
  onSendToSrtEditor,
  onActivityComplete
}) => {
  const {
    user,
    savedVoices,
    savedVoicesLoading,
    saveVoiceProfile,
    renameSavedVoiceProfile,
    deleteSavedVoiceProfile,
    recordSavedVoiceUsage,
    fetchSavedVoicePreviewBlobUrl,
    signInWithGoogle
  } = useFirebaseAuth();

  const [selectedLang, setSelectedLang] = useState<CloningLanguageCode>('bn');
  const [referenceAnalysis, setReferenceAnalysis] = useState<AudioSignalAnalysis | null>(null);
  const [referenceTranscript, setReferenceTranscript] = useState<string>('');
  const [lastReferenceSourceType, setLastReferenceSourceType] =
    useState<SavedVoiceSourceType>('reference_upload');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Persistent Saved Voice Profiles UI State
  const [selectedSavedVoiceId, setSelectedSavedVoiceId] = useState<string | null>(null);
  const [newVoiceNameInput, setNewVoiceNameInput] = useState<string>('');
  const [isSavingVoice, setIsSavingVoice] = useState<boolean>(false);
  const [saveVoiceFeedback, setSaveVoiceFeedback] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);
  const [renamingVoiceId, setRenamingVoiceId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState<string>('');
  const [renameLoadingId, setRenameLoadingId] = useState<string | null>(null);
  const [deletingVoiceId, setDeletingVoiceId] = useState<string | null>(null);
  const [deleteLoadingId, setDeleteLoadingId] = useState<string | null>(null);
  const [previewVoiceId, setPreviewVoiceId] = useState<string | null>(null);
  const [previewBlobUrl, setPreviewBlobUrl] = useState<string | null>(null);
  const [previewLoadingId, setPreviewLoadingId] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const [inputText, setInputText] = useState<string>(
    CLONING_LANGUAGES.bn.sampleTexts[0].text
  );
  const [audioQuality, setAudioQuality] = useState<VoiceAudioQualityMode>(() => {
    try {
      const saved = window.localStorage.getItem('voicelab_audio_quality');
      if (saved === 'standard' || saved === 'studio' || saved === 'studio_plus') return saved;
    } catch {
      // Ignore storage errors
    }
    return 'studio';
  });
  const preservePitchContour = false;
  const preserveSpectralTimbre = false;
  const [localEndpointUrl, setLocalEndpointUrl] = useState<string>(() => {
    try {
      return window.localStorage.getItem('voicelab_local_node_url') || '';
    } catch {
      return '';
    }
  });
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
  const [showAllActiveTags, setShowAllActiveTags] = useState(false);
  const [showAllUnsupportedTags, setShowAllUnsupportedTags] = useState(false);
  const [highlightConsentWarning, setHighlightConsentWarning] = useState(false);

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
  const advancedSettingsRef = useRef<HTMLDivElement | null>(null);
  const nodeUrlInputRef = useRef<HTMLInputElement | null>(null);
  const uploadSectionRef = useRef<HTMLDivElement | null>(null);
  const scriptEditorRef = useRef<HTMLDivElement | null>(null);
  const consentCardRef = useRef<HTMLDivElement | null>(null);

  const liveActivity = useLiveActivityState();
  const currentLangConfig = CLONING_LANGUAGES[selectedLang];

  const selectedSavedVoice = useMemo(
    () => savedVoices.find((v) => v.voiceId === selectedSavedVoiceId) || null,
    [savedVoices, selectedSavedVoiceId]
  );

  // Live emotion tag parsing, tokenizing, and chunk preview (matches 300-char server chunking)
  const liveTokens = useMemo(() => tokenizeScriptForLiveEditor(inputText), [inputText]);
  const parsedScript = useMemo(() => parseEmotionTaggedScript(inputText), [inputText]);
  const previewChunks = useMemo(
    () =>
      buildEmotionPreservingChunks(
        parsedScript,
        splitPreviewSentences,
        DEFAULT_CLONING_CHUNK_MAX_CHARS
      ),
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
    !selectedSavedVoice &&
      detectedRefCode &&
      ['bn', 'hi', 'en'].includes(detectedRefCode) &&
      detectedRefCode !== selectedLang
  );
  const isSavedVoiceLanguageMismatch = Boolean(
    selectedSavedVoice && selectedSavedVoice.language !== selectedLang
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
      setSelectedSavedVoiceId(null);
      setLastReferenceSourceType('studio_sample');
      setSaveVoiceFeedback(null);
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
      setSelectedSavedVoiceId(null);
      setLastReferenceSourceType(
        file.name.startsWith('mic_reference_') ? 'microphone' : 'reference_upload'
      );
      setSaveVoiceFeedback(null);
      const excerpt = data.analysis.detectedSpokenLanguage?.transcriptExcerpt?.trim();
      if (excerpt) {
        setReferenceTranscript((prev) => (prev.trim() ? prev : excerpt));
      }
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

  useEffect(() => {
    return () => {
      if (previewBlobUrl) {
        URL.revokeObjectURL(previewBlobUrl);
      }
    };
  }, [previewBlobUrl]);

  useEffect(() => {
    try {
      window.localStorage.setItem('voicelab_audio_quality', audioQuality);
    } catch {
      // Ignore storage errors
    }
  }, [audioQuality]);

  useEffect(() => {
    try {
      window.localStorage.setItem('voicelab_local_node_url', localEndpointUrl);
    } catch {
      // Ignore storage errors
    }
  }, [localEndpointUrl]);

  const lastCloneStorageKey = `voicelab_last_clone_result_v1_${user?.uid || 'guest'}`;

  useEffect(() => {
    let cancelled = false;
    try {
      const raw = window.localStorage.getItem(lastCloneStorageKey);
      if (!raw) {
        setCloneResult(null);
        return;
      }
      const parsed = JSON.parse(raw) as VoiceCloneResult;
      if (parsed && parsed.jobId && parsed.wavUrl) {
        fetch(parsed.wavUrl, { method: 'HEAD' })
          .then((res) => {
            if (!cancelled && res.ok) {
              setCloneResult(parsed);
            } else if (!cancelled) {
              window.localStorage.removeItem(lastCloneStorageKey);
            }
          })
          .catch(() => {});
      }
    } catch {
      // Ignore storage parse errors
    }
    return () => {
      cancelled = true;
    };
  }, [lastCloneStorageKey]);

  useEffect(() => {
    if (!cloneResult) return;
    try {
      window.localStorage.setItem(lastCloneStorageKey, JSON.stringify(cloneResult));
    } catch {
      // Ignore storage quota errors
    }
  }, [cloneResult, lastCloneStorageKey]);

  useEffect(() => {
    if (!user?.uid) return;
    const key = `voicelab_selected_voice_${user.uid}`;
    if (selectedSavedVoiceId) {
      try {
        window.localStorage.setItem(key, selectedSavedVoiceId);
      } catch {
        // Ignore storage errors
      }
      return;
    }
    if (!savedVoicesLoading && savedVoices.length > 0 && !referenceAnalysis) {
      try {
        const storedId = window.localStorage.getItem(key);
        if (storedId) {
          const found = savedVoices.find((v) => v.voiceId === storedId && v.hasConditioningAudio);
          if (found) {
            setSelectedSavedVoiceId(found.voiceId);
            setReferenceTranscript(found.referenceTranscript || '');
            setSelectedLang(found.language);
          }
        }
      } catch {
        // Ignore storage errors
      }
    }
  }, [user?.uid, selectedSavedVoiceId, savedVoices, savedVoicesLoading, referenceAnalysis]);

  useEffect(() => {
    if (!user && selectedSavedVoiceId) {
      setSelectedSavedVoiceId(null);
      setPreviewVoiceId(null);
      setPreviewBlobUrl(null);
    } else if (
      selectedSavedVoiceId &&
      !savedVoicesLoading &&
      savedVoices.length > 0 &&
      !savedVoices.some((v) => v.voiceId === selectedSavedVoiceId)
    ) {
      setSelectedSavedVoiceId(null);
      if (user?.uid) {
        try {
          window.localStorage.removeItem(`voicelab_selected_voice_${user.uid}`);
        } catch {
          // Ignore storage errors
        }
      }
    }
  }, [user, savedVoices, savedVoicesLoading, selectedSavedVoiceId]);

  const handleSelectSavedVoice = (voice: SavedVoiceProfile) => {
    setSelectedSavedVoiceId(voice.voiceId);
    setGenerationError(null);
    setSaveVoiceFeedback(null);
    setReferenceTranscript(voice.referenceTranscript || '');
    if (voice.language !== selectedLang) {
      setSelectedLang(voice.language);
      const checkExisting = validateSameLanguageText(inputText, voice.language);
      if (!checkExisting.isValid) {
        setInputText(CLONING_LANGUAGES[voice.language].sampleTexts[0].text);
      }
    }
  };

  const handlePreviewSavedVoice = async (voice: SavedVoiceProfile) => {
    if (previewVoiceId === voice.voiceId && previewBlobUrl) {
      URL.revokeObjectURL(previewBlobUrl);
      setPreviewBlobUrl(null);
      setPreviewVoiceId(null);
      return;
    }
    setPreviewError(null);
    setPreviewLoadingId(voice.voiceId);
    try {
      const nextUrl = await fetchSavedVoicePreviewBlobUrl(voice.voiceId);
      setPreviewBlobUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return nextUrl;
      });
      setPreviewVoiceId(voice.voiceId);
    } catch (err: any) {
      setPreviewError(err?.message || 'Unable to load private voice preview.');
    } finally {
      setPreviewLoadingId(null);
    }
  };

  const handleSaveCurrentReferenceVoice = async () => {
    if (!referenceAnalysis) return;
    if (!user || !user.emailVerified) {
      try {
        await signInWithGoogle();
      } catch {
        // Error surfaced by auth context
      }
      return;
    }
    if (!consentConfirmed) {
      setHighlightConsentWarning(true);
      setSaveVoiceFeedback({
        type: 'error',
        text: 'Please check the Voice Authorization box first before saving a voice profile.'
      });
      consentCardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }

    setIsSavingVoice(true);
    setSaveVoiceFeedback(null);
    try {
      const resp = await saveVoiceProfile({
        referenceFileId: referenceAnalysis.fileId,
        name: newVoiceNameInput.trim() || undefined,
        language: selectedLang,
        referenceTranscript: referenceTranscript.trim() || undefined,
        sourceType: lastReferenceSourceType,
        consentConfirmed: true,
        deduplicateByPromptHash: false
      });
      setSelectedSavedVoiceId(resp.voice.voiceId);
      setNewVoiceNameInput('');
      setSaveVoiceFeedback({
        type: 'success',
        text: `Saved "${resp.voice.name}" (${resp.voice.voiceId}) to My Saved Voices.`
      });
    } catch (err: any) {
      setSaveVoiceFeedback({
        type: 'error',
        text: err?.message || 'Could not save voice profile.'
      });
    } finally {
      setIsSavingVoice(false);
    }
  };

  const handleConfirmRenameSavedVoice = async (voiceId: string) => {
    const clean = renameDraft.trim();
    if (!clean) return;
    setRenameLoadingId(voiceId);
    try {
      await renameSavedVoiceProfile(voiceId, clean);
      setRenamingVoiceId(null);
      setRenameDraft('');
    } catch (err: any) {
      setSaveVoiceFeedback({
        type: 'error',
        text: err?.message || 'Failed to rename saved voice.'
      });
    } finally {
      setRenameLoadingId(null);
    }
  };

  const handleConfirmDeleteSavedVoice = async (voiceId: string) => {
    setDeleteLoadingId(voiceId);
    try {
      await deleteSavedVoiceProfile(voiceId);
      if (selectedSavedVoiceId === voiceId) {
        setSelectedSavedVoiceId(null);
      }
      if (previewVoiceId === voiceId) {
        if (previewBlobUrl) URL.revokeObjectURL(previewBlobUrl);
        setPreviewBlobUrl(null);
        setPreviewVoiceId(null);
      }
      setDeletingVoiceId(null);
    } catch (err: any) {
      setSaveVoiceFeedback({
        type: 'error',
        text: err?.message || 'Failed to delete saved voice.'
      });
    } finally {
      setDeleteLoadingId(null);
    }
  };

  const handleGenerateClonedVoice = async (reuseCompletedChunks = true) => {
    const activeSavedVoice = selectedSavedVoice;
    if (!activeSavedVoice && !referenceAnalysis) {
      setGenerationError(
        'Please select a saved voice profile or upload/record/load a reference voice first.'
      );
      uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }
    if (activeSavedVoice && !activeSavedVoice.hasConditioningAudio) {
      setGenerationError(
        `Saved voice "${activeSavedVoice.name}" is missing speaker-conditioning audio on server storage. Please re-upload or record a reference once to restore it.`
      );
      return;
    }
    if (!scriptValidation.isValid) {
      setGenerationError(scriptValidation.errorMessage || 'Text language mismatch.');
      scriptEditorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }
    if (isSavedVoiceLanguageMismatch && activeSavedVoice) {
      setGenerationError(
        `Cross-Language Blocked: Saved voice "${activeSavedVoice.name}" is locked to ${activeSavedVoice.languageName}, which mismatches ${selectedLang.toUpperCase()}.`
      );
      return;
    }
    if (isReferenceLanguageMismatch) {
      setGenerationError(
        `Cross-Language Blocked: Reference audio language (${detectedRefCode?.toUpperCase()}) does not match selected target language (${selectedLang.toUpperCase()}).`
      );
      uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }
    if (!consentConfirmed) {
      setHighlightConsentWarning(true);
      setGenerationError(
        'Please check the Voice Authorization confirmation box directly above the Generate button to start cloning.'
      );
      consentCardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return;
    }

    setHighlightConsentWarning(false);
    setIsGenerating(true);
    setGenerationError(null);
    setActivePipelineStep(0);

    // Allow generous time for multi-chunk zero-shot CPU synthesis (up to 30 min)
    const dynamicTimeoutMs = Math.max(
      300_000,
      Math.min(30 * 60 * 1000, Math.max(1, previewChunks.length) * 120_000)
    );

    const op = startOperation({
      feature: 'voice-cloning',
      title: `${currentLangConfig.englishName} (${currentLangConfig.nativeName}) Synthesis (${previewChunks.length} chunk${previewChunks.length === 1 ? '' : 's'})`,
      stages: [
        'Parsing emotion tags & verifying genuine cloning capability',
        `Synthesizing ${previewChunks.length} emotion-scoped chunk(s) conditioned on reference audio`,
        audioQuality === 'standard'
          ? 'Finalizing 24kHz WAV/MP3 loudness & anti-clipping mastering'
          : audioQuality === 'studio_plus'
          ? 'Applying Studio Plus 48kHz broadcast voice mastering'
          : 'Applying adaptive 48kHz Studio voice mastering',
        'Computing 3072-D & 16-D audio embeddings',
        'Running single-pass ASR back-transcription & zero-literal-tag check'
      ],
      initialProgressPct: null,
      progressSource: 'indeterminate-inference',
      timeoutMs: dynamicTimeoutMs,
      canCancel: true
    });
    const stopStream = connectBackendOperationStream(op.operationId);

    const stepInterval = window.setInterval(() => {
      setActivePipelineStep((prev) => (prev < PIPELINE_A_STEPS.length - 1 ? prev + 1 : prev));
    }, 1200);

    let activeBackendJobId: string | null = null;
    const onAbortSignal = () => {
      if (activeBackendJobId) {
        fetchApiJson(`/api/voice-clone/jobs/${encodeURIComponent(activeBackendJobId)}/cancel`, {
          method: 'POST'
        }).catch(() => {});
      }
    };
    op.signal.addEventListener('abort', onAbortSignal, { once: true });

    try {
      const startData = await fetchApiJson<{
        status?: string;
        jobId?: string;
        result?: VoiceCloneResult;
      }>('/api/voice-clone/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        operationId: op.operationId,
        signal: op.signal,
        onRetry: (_, reason) => recordOperationRetry(op.operationId, reason),
        body: JSON.stringify({
          referenceFileId: activeSavedVoice ? undefined : referenceAnalysis?.fileId,
          savedVoiceId: activeSavedVoice ? activeSavedVoice.voiceId : undefined,
          language: selectedLang,
          text: inputText,
          referenceTranscript: referenceTranscript.trim() || undefined,
          audioQuality,
          preservePitchContour,
          preserveSpectralTimbre,
          reuseCompletedChunks,
          localEndpointUrl: localEndpointUrl.trim() || undefined,
          localNodeToken: localNodeToken.trim() || undefined,
          enableCloudVerification,
          consentConfirmed,
          asyncJob: true
        })
      });

      if (op.isStale()) return;

      let resObj: VoiceCloneResult | undefined = startData.result;

      if (!resObj && startData.jobId) {
        activeBackendJobId = startData.jobId;
        let transientPollErrors = 0;

        while (!resObj) {
          if (op.signal.aborted || op.isStale()) {
            const abortErr = new Error('Voice cloning operation ended before completion.');
            abortErr.name = 'AbortError';
            throw abortErr;
          }

          await new Promise<void>((resolve) => {
            const delayMs = transientPollErrors > 0 ? Math.min(5000, 1600 + transientPollErrors * 900) : 1600;
            const t = window.setTimeout(resolve, delayMs);
            const onWake = () => {
              window.clearTimeout(t);
              resolve();
            };
            op.signal.addEventListener('abort', onWake, { once: true });
          });

          if (op.signal.aborted || op.isStale()) {
            const abortErr = new Error('Voice cloning operation ended before completion.');
            abortErr.name = 'AbortError';
            throw abortErr;
          }

          try {
            const pollResp = await fetchApiJson<{
              jobId: string;
              status: 'running' | 'succeeded' | 'failed' | 'cancelled';
              operation?: {
                stageIndex?: number;
                stageLabel?: string;
                stageDetail?: string;
                progressPct?: number | null;
                progressSource?: 'backend-chunks' | 'indeterminate-inference' | 'completed';
                completedUnits?: number;
                totalUnits?: number;
              } | null;
              result?: VoiceCloneResult;
              error?: string;
            }>(`/api/voice-clone/jobs/${encodeURIComponent(startData.jobId)}`, {
              method: 'GET',
              signal: op.signal
            });
            transientPollErrors = 0;

            if (pollResp.operation && pollResp.status === 'running') {
              updateOperationStage(op.operationId, {
                stageIndex: pollResp.operation.stageIndex,
                stageLabel: pollResp.operation.stageLabel,
                stageDetail: pollResp.operation.stageDetail,
                progressPct: pollResp.operation.progressPct,
                progressSource: pollResp.operation.progressSource,
                completedUnits: pollResp.operation.completedUnits,
                totalUnits: pollResp.operation.totalUnits
              });
              if (typeof pollResp.operation.stageIndex === 'number') {
                setActivePipelineStep(pollResp.operation.stageIndex);
              }
            }

            if (pollResp.status === 'succeeded' && pollResp.result) {
              resObj = pollResp.result;
              break;
            }
            if (pollResp.status === 'failed') {
              const terminalErr: any = new Error(pollResp.error || 'Voice cloning pipeline failed.');
              terminalErr.isJobTerminalFailure = true;
              throw terminalErr;
            }
            if (pollResp.status === 'cancelled') {
              const cancelErr = new Error('Voice cloning job was cancelled.');
              cancelErr.name = 'AbortError';
              throw cancelErr;
            }
          } catch (pollErr: any) {
            if (pollErr?.name === 'AbortError' || op.signal.aborted || op.isStale()) {
              throw pollErr;
            }
            if (pollErr?.isJobTerminalFailure) {
              throw pollErr;
            }
            transientPollErrors += 1;
            const msg = String(pollErr?.message || '');
            const isFatal =
              transientPollErrors >= 15 ||
              msg.includes('JOB_NOT_FOUND') ||
              msg.includes('not found') ||
              msg.includes('ZeroGPU') ||
              msg.includes('Mode 2');
            if (isFatal) {
              throw pollErr;
            }
          }
        }
      }

      if (!resObj) {
        throw new Error('Voice cloning did not return synthesized audio.');
      }

      if (op.isStale()) return;

      setCloneResult(resObj);
      setActivePipelineStep(PIPELINE_A_STEPS.length - 1);
      const matchPct =
        resObj.qualityControlReport.separatedMetrics.asrTranscriptAccuracy.characterMatchRatePct;
      completeOperation(
        op.operationId,
        `${currentLangConfig.englishName} speech ready (${resObj.outputDurationSec}s · ${resObj.chunkCount} chunk${resObj.chunkCount === 1 ? '' : 's'} · ${matchPct !== null ? `${matchPct}% ASR match` : '0-Quota Mode 2'})`
      );

      if (activeSavedVoice) {
        void recordSavedVoiceUsage(activeSavedVoice.voiceId).catch(() => {});
      } else if (user && user.emailVerified && referenceAnalysis) {
        try {
          const savedResp = await saveVoiceProfile({
            referenceFileId: referenceAnalysis.fileId,
            name: newVoiceNameInput.trim() || undefined,
            language: selectedLang,
            referenceTranscript: referenceTranscript.trim() || undefined,
            modelCheckpoint: resObj.routedModelUsed,
            sourceType: lastReferenceSourceType,
            consentConfirmed: true,
            deduplicateByPromptHash: true
          });
          setSelectedSavedVoiceId(savedResp.voice.voiceId);
          setNewVoiceNameInput('');
          setSaveVoiceFeedback({
            type: 'success',
            text: savedResp.deduplicated
              ? `Using saved speaker profile "${savedResp.voice.name}" (${savedResp.voice.voiceId}).`
              : `Auto-saved speaker profile "${savedResp.voice.name}" (${savedResp.voice.voiceId}) to My Saved Voices for future sessions.`
          });
        } catch {
          // Non-fatal if auto-save fails after successful synthesis
        }
      }

      if (onActivityComplete) {
        onActivityComplete({
          languageLabel: `${currentLangConfig.englishName} (${currentLangConfig.nativeName})`,
          downloadUrl: `${resObj.wavUrl}?download=1`,
          downloadFilename: `voicelab_${resObj.language}_${resObj.jobId}.wav`
        });
      }
    } catch (err: any) {
      if (err?.name === 'AbortError' || op.isStale()) {
        // Check if abort came from timeout vs explicit user cancellation
        const timedOutMsg =
          liveActivity.primaryOperation?.operationId === op.operationId &&
          liveActivity.primaryOperation.status === 'failed'
            ? liveActivity.primaryOperation.error
            : null;
        if (timedOutMsg) {
          setGenerationError(
            `${timedOutMsg} Completed chunks are saved on the server—click "Retry Failed Chunks" to resume cleanly.`
          );
        }
        return;
      }
      const msg = err?.message || 'Voice cloning pipeline encountered an error.';
      setGenerationError(msg);
      failOperation(op.operationId, msg);
    } finally {
      op.signal.removeEventListener('abort', onAbortSignal);
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

  const handleDownloadGeneratedSrt = () => {
    if (!cloneResult) return;
    const srtSegments = buildSrtSegmentsFromVoiceClone(cloneResult);
    if (srtSegments.length === 0) return;
    const srtText = `\uFEFF${formatSegmentsToSrtString(srtSegments)}`;
    const filename = `voicelab_${cloneResult.language}_${cloneResult.jobId}.srt`;
    const blob = new Blob([srtText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    const op = startOperation({
      feature: 'export-download',
      title: `Export ${filename}`,
      stages: [`Generated ${srtSegments.length} aligned subtitle blocks`, 'Triggered UTF-8 BOM .SRT download'],
      initialProgressPct: 100,
      progressSource: 'task-steps',
      canCancel: false
    });
    completeOperation(op.operationId, `Exported ${filename} (${srtSegments.length} subtitle blocks)`);
  };

  const handleOpenGeneratedInSrtStudio = () => {
    if (!cloneResult || !onSendToSrtEditor) return;
    const srtSegments = buildSrtSegmentsFromVoiceClone(cloneResult);
    const langCfg = CLONING_LANGUAGES[cloneResult.language];
    onSendToSrtEditor({
      fileName: `voicelab_${cloneResult.language}_${cloneResult.jobId}.wav`,
      audioUrl: cloneResult.wavUrl,
      languageCode: cloneResult.language,
      languageName: `${langCfg.englishName} (${langCfg.nativeName})`,
      scriptName: langCfg.scriptName,
      durationSeconds: cloneResult.outputDurationSec,
      segments: srtSegments
    });
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

  const handleOpenMode2Setup = () => {
    setShowAdvancedSettings(true);
    setShowColabGuide(true);
    window.setTimeout(() => {
      advancedSettingsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      nodeUrlInputRef.current?.focus();
    }, 80);
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
            <span className="text-[#42D9A0] font-mono whitespace-nowrap">
              {localEndpointUrl.trim()
                ? `Mode 2 Custom Node (${selectedLang === 'en' ? 'OpenVoice V2' : 'IndicF5'})`
                : `Mode 2 Built-in Zero-Shot (${selectedLang === 'en' ? 'F5-TTS' : 'IndicF5'} · 0 Quota)`}
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

      {/* My Saved Voices — Persistent Account Speaker Profiles */}
      <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 sm:p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <UserCheck className="w-4 h-4 text-[#42D9A0]" />
              <h2 className="text-sm font-semibold text-[#F4F8F5]">My Saved Voices</h2>
              {user && user.emailVerified && (
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-[#0B1512] border border-[#19352A] text-[#42D9A0]">
                  {savedVoices.length} saved
                </span>
              )}
            </div>
            <p className="text-[11px] text-[#A2B5AA]">
              Saved speaker conditioning profiles persist across sessions for your account. Select
              any saved voice to generate new scripts without re-uploading audio.
            </p>
          </div>

          {selectedSavedVoice && (
            <button
              type="button"
              onClick={() => setSelectedSavedVoiceId(null)}
              className="px-2.5 py-1 rounded-lg bg-[#0B1512] hover:bg-[#19352A] border border-[#19352A] text-[11px] font-medium text-[#A2B5AA] hover:text-[#F4F8F5] transition-colors cursor-pointer"
            >
              Use New Reference Instead
            </button>
          )}
        </div>

        {!user || !user.emailVerified ? (
          <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <p className="text-xs text-[#A2B5AA]">
              Sign in with your verified Google account to save cloned voices and reuse them across
              sessions and devices.
            </p>
            <button
              type="button"
              onClick={() => void signInWithGoogle()}
              className="inline-flex items-center justify-center gap-1.5 px-3.5 py-2 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] transition-colors cursor-pointer whitespace-nowrap shrink-0"
            >
              <LogIn className="w-3.5 h-3.5" />
              <span>Sign in with Google</span>
            </button>
          </div>
        ) : savedVoicesLoading ? (
          <div className="text-xs text-[#A2B5AA] flex items-center gap-2 py-2">
            <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#42D9A0]" />
            <span>Loading your persistent saved voice profiles...</span>
          </div>
        ) : savedVoices.length === 0 ? (
          <div className="bg-[#0B1512] border border-dashed border-[#19352A] rounded-xl p-3 text-xs text-[#A2B5AA]">
            No saved voice profiles yet. Upload, record, or generate a voice clone in Step 01 — it
            will be saved here automatically for instant reuse.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 max-h-80 overflow-y-auto pr-1">
            {savedVoices.map((voice) => {
              const isSelected = selectedSavedVoiceId === voice.voiceId;
              const isRenaming = renamingVoiceId === voice.voiceId;
              const isConfirmDeleting = deletingVoiceId === voice.voiceId;
              const isPreviewing = previewVoiceId === voice.voiceId && Boolean(previewBlobUrl);
              const createdLabel = (() => {
                const d = new Date(voice.createdAt);
                return Number.isNaN(d.getTime()) ? 'Recent' : d.toLocaleDateString();
              })();

              return (
                <div
                  key={voice.voiceId}
                  className={`rounded-xl border p-3 space-y-2.5 transition-colors ${
                    isSelected
                      ? 'bg-[#19352A]/60 border-[#16A36A]'
                      : 'bg-[#0B1512] border-[#19352A] hover:border-[#16A36A]/50'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1 space-y-0.5">
                      {isRenaming ? (
                        <div className="flex items-center gap-1.5">
                          <input
                            type="text"
                            value={renameDraft}
                            onChange={(e) => setRenameDraft(e.target.value)}
                            maxLength={80}
                            className="flex-1 min-w-0 rounded-lg bg-[#12231D] border border-[#16A36A] px-2 py-1 text-xs text-[#F4F8F5] focus:outline-none"
                            placeholder="Voice profile name"
                          />
                          <button
                            type="button"
                            onClick={() => void handleConfirmRenameSavedVoice(voice.voiceId)}
                            disabled={renameLoadingId === voice.voiceId || !renameDraft.trim()}
                            className="px-2 py-1 rounded-lg bg-[#16A36A] text-[11px] font-semibold text-[#F4F8F5] cursor-pointer"
                          >
                            Save
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setRenamingVoiceId(null);
                              setRenameDraft('');
                            }}
                            className="px-2 py-1 rounded-lg bg-[#12231D] border border-[#19352A] text-[11px] text-[#A2B5AA] cursor-pointer"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5 min-w-0">
                          <span className="text-xs font-semibold text-[#F4F8F5] truncate">
                            {voice.name}
                          </span>
                          {isSelected && (
                            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#16A36A]/25 border border-[#16A36A]/60 text-[#42D9A0] shrink-0">
                              Active
                            </span>
                          )}
                        </div>
                      )}
                      <div className="text-[10px] font-mono text-[#A2B5AA] truncate">
                        ID: {voice.voiceId}
                      </div>
                    </div>

                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#12231D] border border-[#19352A] text-[#42D9A0] shrink-0">
                      {voice.language.toUpperCase()}
                    </span>
                  </div>

                  <div className="text-[11px] text-[#A2B5AA] space-y-0.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate">{voice.languageName}</span>
                      <span className="font-mono text-[#F4F8F5] shrink-0">
                        {voice.promptDurationSec}s prompt · {voice.fundamentalFreqHz} Hz
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-2 text-[10px] font-mono">
                      <span className="truncate">
                        Cond: {voice.promptConditioningSha256.slice(0, 10) || 'canonical-wav'}
                      </span>
                      <span className="shrink-0">
                        {createdLabel} · {voice.usageCount} uses
                      </span>
                    </div>
                  </div>

                  {voice.hasConditioningAudio ? (
                    <div className="text-[10px] text-[#42D9A0] bg-[#19352A]/40 border border-[#16A36A]/35 rounded-lg px-2 py-1 flex items-center gap-1.5">
                      <ShieldCheck className="w-3 h-3 shrink-0" />
                      <span>Permanent Cloud + Vault Speaker Conditioning Active</span>
                    </div>
                  ) : (
                    <div className="text-[10px] text-amber-300 bg-amber-950/50 border border-amber-500/40 rounded-lg px-2 py-1">
                      Speaker conditioning audio missing. Re-upload reference once to restore
                      cloud vault.
                    </div>
                  )}

                  {isPreviewing && previewBlobUrl && (
                    <audio controls autoPlay src={previewBlobUrl} className="w-full h-8 rounded-lg" />
                  )}

                  <div className="flex flex-wrap items-center justify-between gap-1.5 pt-1 border-t border-[#19352A]">
                    <button
                      type="button"
                      onClick={() => handleSelectSavedVoice(voice)}
                      disabled={!voice.hasConditioningAudio}
                      className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-colors cursor-pointer ${
                        isSelected
                          ? 'bg-[#16A36A] text-[#F4F8F5]'
                          : voice.hasConditioningAudio
                          ? 'bg-[#19352A] hover:bg-[#16A36A] text-[#F4F8F5]'
                          : 'bg-[#12231D] text-[#A2B5AA]/50 cursor-not-allowed'
                      }`}
                    >
                      {isSelected ? 'Selected' : 'Use Voice'}
                    </button>

                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => void handlePreviewSavedVoice(voice)}
                        disabled={!voice.hasConditioningAudio || previewLoadingId === voice.voiceId}
                        title="Preview saved reference voice"
                        className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-[#12231D] hover:bg-[#19352A] border border-[#19352A] text-[11px] text-[#A2B5AA] hover:text-[#F4F8F5] cursor-pointer disabled:opacity-40"
                      >
                        {previewLoadingId === voice.voiceId ? (
                          <RefreshCw className="w-3 h-3 animate-spin" />
                        ) : (
                          <Play className="w-3 h-3 text-[#42D9A0]" />
                        )}
                        <span>{isPreviewing ? 'Stop' : 'Preview'}</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => {
                          setRenamingVoiceId(voice.voiceId);
                          setRenameDraft(voice.name);
                          setDeletingVoiceId(null);
                        }}
                        title="Rename saved voice"
                        className="p-1.5 rounded-lg bg-[#12231D] hover:bg-[#19352A] border border-[#19352A] text-[#A2B5AA] hover:text-[#F4F8F5] cursor-pointer"
                      >
                        <Pencil className="w-3 h-3" />
                      </button>

                      {isConfirmDeleting ? (
                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => void handleConfirmDeleteSavedVoice(voice.voiceId)}
                            disabled={deleteLoadingId === voice.voiceId}
                            className="px-2 py-1 rounded-lg bg-red-600 hover:bg-red-500 text-[10px] font-semibold text-white cursor-pointer"
                          >
                            {deleteLoadingId === voice.voiceId ? '...' : 'Confirm'}
                          </button>
                          <button
                            type="button"
                            onClick={() => setDeletingVoiceId(null)}
                            className="px-1.5 py-1 rounded-lg bg-[#12231D] border border-[#19352A] text-[10px] text-[#A2B5AA] cursor-pointer"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button
                          type="button"
                          onClick={() => {
                            setDeletingVoiceId(voice.voiceId);
                            setRenamingVoiceId(null);
                          }}
                          title="Delete saved voice & conditioning audio"
                          className="p-1.5 rounded-lg bg-[#12231D] hover:bg-red-950/70 border border-[#19352A] hover:border-red-500/40 text-[#A2B5AA] hover:text-red-300 cursor-pointer"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {previewError && (
          <div className="text-xs text-red-200 bg-red-950/50 border border-red-500/40 rounded-lg px-3 py-2">
            {previewError}
          </div>
        )}
        {saveVoiceFeedback && (
          <div
            className={`text-xs rounded-lg px-3 py-2 border ${
              saveVoiceFeedback.type === 'success'
                ? 'bg-emerald-950/50 border-emerald-500/40 text-emerald-200'
                : 'bg-red-950/50 border-red-500/40 text-red-200'
            }`}
          >
            {saveVoiceFeedback.text}
          </div>
        )}
      </div>

      {/* Main Two-Column Studio Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column (5 cols): Reference Audio Upload, Recording & Reference Vocal Identity */}
        <div ref={uploadSectionRef} className="lg:col-span-5 space-y-4">
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
                      {localEndpointUrl.trim() ? 'Mode 2 Custom Node' : 'Mode 2 Zero-Shot Active'}
                    </span>
                  </div>
                  <p className="text-[#A2B5AA] leading-relaxed">
                    {localEndpointUrl.trim() ? (
                      <>
                        Custom Mode 2 Zero-Shot Neural Node configured (
                        <span className="text-[#F4F8F5] font-mono">
                          {currentLangConfig.openSourceEquivalent}
                        </span>
                        ). Every chunk receives your exact uploaded reference audio bytes for
                        consistent speaker identity.
                      </>
                    ) : (
                      <>
                        Using built-in zero-shot{' '}
                        <span className="text-[#F4F8F5] font-mono">
                          {selectedLang === 'en' ? 'SWivid/F5-TTS' : 'ai4bharat/IndicF5'}
                        </span>{' '}
                        speaker conditioning (<strong className="text-[#42D9A0]">0 Gemini Quota</strong>
                        ). Your exact uploaded reference voice is conditioned across every chunk.{' '}
                        <button
                          type="button"
                          onClick={handleOpenMode2Setup}
                          className="underline text-[#42D9A0] font-semibold cursor-pointer"
                        >
                          Optional: Private Colab T4 Node
                        </button>
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

                {/* Save Reference Speaker Profile to Account */}
                <div className="pt-2 border-t border-[#19352A] space-y-2">
                  {user && user.emailVerified ? (
                    <div className="flex flex-col sm:flex-row gap-2">
                      <input
                        type="text"
                        value={newVoiceNameInput}
                        onChange={(e) => setNewVoiceNameInput(e.target.value)}
                        maxLength={80}
                        placeholder="Custom voice name (optional)..."
                        className="flex-1 rounded-lg bg-[#12231D] border border-[#19352A] focus:border-[#16A36A] px-2.5 py-1.5 text-xs text-[#F4F8F5] placeholder-[#A2B5AA]/50 focus:outline-none"
                      />
                      <button
                        type="button"
                        onClick={() => void handleSaveCurrentReferenceVoice()}
                        disabled={isSavingVoice}
                        className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] transition-colors cursor-pointer whitespace-nowrap disabled:opacity-60"
                      >
                        {isSavingVoice ? (
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <BookmarkPlus className="w-3.5 h-3.5" />
                        )}
                        <span>{isSavingVoice ? 'Saving...' : 'Save Voice Profile'}</span>
                      </button>
                    </div>
                  ) : (
                    <button
                      type="button"
                      onClick={() => void signInWithGoogle()}
                      className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#12231D] hover:bg-[#19352A] border border-[#19352A] text-xs font-medium text-[#42D9A0] cursor-pointer"
                    >
                      <LogIn className="w-3.5 h-3.5" />
                      <span>Sign in with Google to Save Voice Profile</span>
                    </button>
                  )}
                </div>
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
        <div ref={scriptEditorRef} className="lg:col-span-7 space-y-4">
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

            {/* Saved Voice Selector for Script-to-Audio */}
            <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
              <div className="flex items-center gap-2 text-xs min-w-0">
                <UserCheck className="w-4 h-4 text-[#42D9A0] shrink-0" />
                <span className="font-medium text-[#F4F8F5] whitespace-nowrap">
                  Speaker Identity:
                </span>
                {selectedSavedVoice ? (
                  <span className="text-[#42D9A0] font-semibold truncate">
                    {selectedSavedVoice.name} ({selectedSavedVoice.languageName})
                  </span>
                ) : referenceAnalysis ? (
                  <span className="text-[#A2B5AA] truncate">
                    Step 01 Reference ({referenceAnalysis.fileName})
                  </span>
                ) : (
                  <span className="text-amber-300 truncate">
                    Select a saved voice or upload reference in Step 01
                  </span>
                )}
              </div>

              {user && user.emailVerified && savedVoices.length > 0 && (
                <div className="flex items-center gap-2 shrink-0">
                  <select
                    aria-label="Saved Voice Selector"
                    value={selectedSavedVoiceId || ''}
                    onChange={(e) => {
                      const nextId = e.target.value;
                      if (!nextId) {
                        setSelectedSavedVoiceId(null);
                        return;
                      }
                      const found = savedVoices.find((v) => v.voiceId === nextId);
                      if (found) handleSelectSavedVoice(found);
                    }}
                    className="rounded-lg bg-[#12231D] border border-[#19352A] focus:border-[#16A36A] px-2.5 py-1.5 text-xs text-[#F4F8F5] focus:outline-none max-w-[230px]"
                  >
                    <option value="">-- Use Step 01 Audio --</option>
                    {savedVoices.map((v) => (
                      <option key={v.voiceId} value={v.voiceId} disabled={!v.hasConditioningAudio}>
                        {v.name} ({v.language.toUpperCase()} · {v.promptDurationSec}s
                        {!v.hasConditioningAudio ? ' · Unavailable' : ''})
                      </option>
                    ))}
                  </select>
                </div>
              )}
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
              <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
                <span className="text-[#A2B5AA] flex items-center gap-1.5 font-medium">
                  <Sparkles className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
                  <span>
                    Insert Emotion / Delivery Tag (বাংলা যেকোনো স্টেজ-ট্যাগ যেমন{' '}
                    <code className="text-[#42D9A0]">[নিচু স্বরে]</code>,{' '}
                    <code className="text-[#42D9A0]">[দীর্ঘ বিরতি]</code>-ও অটোমেটিক কাজ করে):
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
                    title={`Insert ${tagSpec.token} (${tagSpec.label} / ${tagSpec.bengaliLabel})`}
                    className={`px-2 py-0.5 rounded-md border text-[11px] font-mono transition-transform active:scale-95 cursor-pointer inline-flex items-center gap-1 ${tagSpec.badgeColorClass}`}
                  >
                    <span>{tagSpec.token}</span>
                    {selectedLang === 'bn' && (
                      <span className="font-bengali opacity-80 text-[10px]">
                        {tagSpec.bengaliLabel}
                      </span>
                    )}
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

              {/* Active Tag Chips Strip (Interactive Removable Chips — collapsed on long scripts for mobile clarity) */}
              {activeScriptTagTokens.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                  <span className="text-[11px] text-[#A2B5AA] mr-1">
                    Active tags in script ({activeScriptTagTokens.length}):
                  </span>
                  {(showAllActiveTags
                    ? activeScriptTagTokens
                    : activeScriptTagTokens.slice(0, 6)
                  ).map((tok, idx) => {
                    const spec =
                      tok.type === 'recognized-tag' && tok.recognizedName
                        ? RECOGNIZED_EMOTION_TAGS[tok.recognizedName]
                        : null;
                    return (
                      <span
                        key={`${tok.startIndex}_${idx}`}
                        title={
                          tok.isNaturalDirection
                            ? `Auto-mapped to: ${[
                                ...(tok.mappedTags || []),
                                tok.mappedPauseMs ? `pause ${tok.mappedPauseMs}ms` : ''
                              ]
                                .filter(Boolean)
                                .join(' + ')}`
                            : undefined
                        }
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md border text-[11px] font-mono ${
                          spec
                            ? spec.badgeColorClass
                            : 'bg-red-500/20 border-red-400/50 text-red-200'
                        }`}
                      >
                        <span className="max-w-[180px] truncate">{tok.raw}</span>
                        {tok.isNaturalDirection && (
                          <span className="text-[10px] opacity-80">
                            →{[
                              ...(tok.mappedTags || []),
                              tok.mappedPauseMs ? `${tok.mappedPauseMs}ms` : ''
                            ]
                              .filter(Boolean)
                              .join('+')}
                          </span>
                        )}
                        <button
                          type="button"
                          onClick={() => handleRemoveTagTokenAtRange(tok.startIndex, tok.endIndex)}
                          title={`Remove ${tok.raw} from script`}
                          className="hover:opacity-100 opacity-75 cursor-pointer shrink-0"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    );
                  })}
                  {activeScriptTagTokens.length > 6 && (
                    <button
                      type="button"
                      onClick={() => setShowAllActiveTags((prev) => !prev)}
                      className="px-2 py-0.5 rounded-md border border-[#19352A] bg-[#0B1512] hover:bg-[#19352A] text-[11px] font-mono text-[#42D9A0] cursor-pointer"
                    >
                      {showAllActiveTags
                        ? 'Show fewer'
                        : `+${activeScriptTagTokens.length - 6} more`}
                    </button>
                  )}
                </div>
              )}

              {/* Unsupported Tag Warning Banner */}
              {parsedScript.unsupportedTags.length > 0 && (
                <div className="bg-amber-950/60 border border-amber-500/50 rounded-xl p-2.5 text-xs text-amber-200 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div className="space-y-1 min-w-0">
                    <div>
                      <span className="font-semibold">
                        {parsedScript.unsupportedTags.length} bracket stage-direction tag
                        {parsedScript.unsupportedTags.length === 1 ? '' : 's'}
                      </span>{' '}
                      detected:{' '}
                      <span className="font-mono font-semibold">
                        {(showAllUnsupportedTags
                          ? parsedScript.unsupportedTags
                          : parsedScript.unsupportedTags.slice(0, 3)
                        ).join(', ')}
                      </span>
                      {parsedScript.unsupportedTags.length > 3 && !showAllUnsupportedTags && (
                        <span>
                          {' '}
                          and {parsedScript.unsupportedTags.length - 3} more
                        </span>
                      )}
                      . Automatically stripped before synthesis so they are{' '}
                      <span className="font-semibold underline">never spoken aloud</span>.
                      {parsedScript.unsupportedTags.length > 3 && (
                        <button
                          type="button"
                          onClick={() => setShowAllUnsupportedTags((prev) => !prev)}
                          className="ml-1.5 underline text-amber-300 font-semibold cursor-pointer"
                        >
                          {showAllUnsupportedTags ? 'Hide list' : 'Show all'}
                        </button>
                      )}
                    </div>
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
            {/* Studio Output Audio Quality Selector */}
            <div className="rounded-xl bg-[#0B1512] border border-[#19352A] p-3 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <span className="flex items-center gap-1.5 text-xs font-medium text-[#F4F8F5]">
                  <Sparkles className="w-3.5 h-3.5 text-[#42D9A0]" />
                  <span>Audio Quality</span>
                </span>
                <span className="text-[11px] font-mono text-[#42D9A0]">
                  {audioQuality === 'standard'
                    ? '24 kHz · Raw Neural Balance'
                    : audioQuality === 'studio_plus'
                    ? '48 kHz · Creator Broadcast Polish'
                    : '48 kHz · Adaptive Studio Mastering'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {(
                  [
                    {
                      id: 'standard',
                      title: 'Standard',
                      subtitle: '24 kHz · -17 LUFS'
                    },
                    {
                      id: 'studio',
                      title: 'Studio',
                      subtitle: '48 kHz · Recommended'
                    },
                    {
                      id: 'studio_plus',
                      title: 'Studio Plus',
                      subtitle: '48 kHz · Rich Presence'
                    }
                  ] as const
                ).map((mode) => {
                  const selected = audioQuality === mode.id;
                  return (
                    <button
                      key={mode.id}
                      type="button"
                      onClick={() => setAudioQuality(mode.id)}
                      className={`px-2.5 py-2 rounded-lg border text-left transition-colors cursor-pointer ${
                        selected
                          ? 'bg-[#19352A] border-[#16A36A] text-[#F4F8F5]'
                          : 'bg-[#12231D] border-[#19352A] text-[#A2B5AA] hover:text-[#F4F8F5] hover:border-[#42D9A0]/40'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs font-semibold">{mode.title}</span>
                        {selected && <Check className="w-3 h-3 text-[#42D9A0] shrink-0" />}
                      </div>
                      <div className="text-[10px] font-mono text-[#A2B5AA] mt-0.5 truncate">
                        {mode.subtitle}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <div
              ref={advancedSettingsRef}
              className="rounded-xl bg-[#0B1512] border border-[#19352A]"
            >
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
                  <span className="text-[#42D9A0]">
                    {localEndpointUrl.trim()
                      ? 'Custom Mode 2 Connected'
                      : 'Built-in GPU Active (Optional Override)'}
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
                        ref={nodeUrlInputRef}
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
                    Speaker identity, pitch contour, accent, and vocal personality are preserved directly
                    by Mode 2 neural weights (0 semitone shift). Studio and Studio Plus apply adaptive
                    48 kHz noise cleanup, boxy-room EQ refinement, gentle de-essing, comfort-tone chunk
                    crossfades, and 2-pass linear EBU R128 loudness mastering.
                  </div>
                </div>
              )}
            </div>

            {/* Tap-Safe Voice Authorization Checkbox */}
            <div
              ref={consentCardRef}
              className={`rounded-xl border px-3 py-2.5 transition-colors ${
                !consentConfirmed
                  ? highlightConsentWarning
                    ? 'bg-amber-950/60 border-amber-400 ring-2 ring-amber-400/40'
                    : 'bg-amber-950/35 border-amber-500/50'
                  : 'bg-[#0B1512] border-[#19352A]'
              }`}
            >
              <label className="flex items-start gap-2.5 cursor-pointer text-xs">
                <input
                  type="checkbox"
                  checked={consentConfirmed}
                  onChange={(e) => {
                    setConsentConfirmed(e.target.checked);
                    if (e.target.checked) {
                      setHighlightConsentWarning(false);
                      setGenerationError(null);
                    }
                  }}
                  className="accent-[#16A36A] mt-0.5 h-4 w-4 shrink-0 cursor-pointer"
                />
                <ShieldCheck
                  className={`w-4 h-4 mt-0.5 shrink-0 ${
                    consentConfirmed ? 'text-[#42D9A0]' : 'text-amber-400'
                  }`}
                />
                <span className="text-[#A2B5AA] leading-snug">
                  <span className="text-[#F4F8F5] font-medium">
                    I confirm ownership or explicit authorization to clone this reference voice.
                  </span>
                  {!consentConfirmed && (
                    <span className="block text-[11px] text-amber-300 font-semibold mt-0.5">
                      Required: Tap checkbox above to enable voice generation.
                    </span>
                  )}
                </span>
              </label>
            </div>

            {generationError && (
              <div className="bg-red-950/60 border border-red-500/50 rounded-xl p-3 text-xs text-red-200 space-y-2">
                <div className="flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                  <span>{generationError}</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {previewChunks.length > 1 && (referenceAnalysis || selectedSavedVoice) && consentConfirmed && (
                    <button
                      type="button"
                      onClick={() => handleGenerateClonedVoice(true)}
                      disabled={isGenerating}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-900/60 hover:bg-red-800 border border-red-400/40 text-xs font-semibold text-red-100 cursor-pointer"
                    >
                      <RefreshCw className="w-3.5 h-3.5" />
                      <span>Retry Failed Chunks (Reuse Cached)</span>
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={handleOpenMode2Setup}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] cursor-pointer"
                  >
                    <Terminal className="w-3.5 h-3.5" />
                    <span>Connect Private Colab T4 GPU (Backup)</span>
                  </button>
                </div>
              </div>
            )}

            {/* Pre-flight readiness indicator & Primary Generate Action */}
            {(() => {
              const blockerReason =
                !selectedSavedVoice && !referenceAnalysis
                  ? 'Select a saved voice or upload/record/load a reference voice (Step 01) first'
                  : selectedSavedVoice && !selectedSavedVoice.hasConditioningAudio
                  ? 'Selected saved voice conditioning audio is missing on server storage'
                  : isSavedVoiceLanguageMismatch && selectedSavedVoice
                  ? `Saved voice "${selectedSavedVoice.name}" is locked to ${selectedSavedVoice.languageName}`
                  : !scriptValidation.isValid
                  ? 'Enter valid spoken text matching the selected language'
                  : isReferenceLanguageMismatch
                  ? 'Reference audio language mismatches selected target language'
                  : !consentConfirmed
                  ? 'Confirm voice authorization checkbox above'
                  : null;
              const isReadyToGenerate = blockerReason === null;
              const activeCloneOp =
                liveActivity.primaryOperation?.feature === 'voice-cloning' &&
                liveActivity.primaryOperation?.status === 'running'
                  ? liveActivity.primaryOperation
                  : null;

              return (
                <div className="space-y-2">
                  {blockerReason && !isGenerating && (
                    <div className="text-[11px] text-amber-300 flex items-center gap-1.5 px-1">
                      <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                      <span>{blockerReason}</span>
                    </div>
                  )}

                  <button
                    type="button"
                    onClick={() => handleGenerateClonedVoice(true)}
                    disabled={isGenerating}
                    aria-disabled={!isReadyToGenerate || isGenerating}
                    className={`w-full py-3 px-5 rounded-xl font-semibold text-sm transition-colors flex items-center justify-center gap-2 shadow-sm whitespace-nowrap ${
                      isGenerating
                        ? 'bg-[#19352A] text-[#F4F8F5]/80 cursor-wait'
                        : isReadyToGenerate
                        ? 'bg-[#16A36A] hover:bg-[#16A36A]/90 text-[#F4F8F5] cursor-pointer'
                        : 'bg-[#19352A] hover:bg-[#19352A]/80 text-[#F4F8F5]/80 border border-amber-500/40 cursor-pointer'
                    }`}
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

                  {isGenerating && (
                    <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3 text-xs font-mono text-[#42D9A0] space-y-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate">
                          {activeCloneOp?.currentStage || PIPELINE_A_STEPS[activePipelineStep]}
                        </span>
                        <span className="tabular-nums shrink-0 ml-2">
                          {activeCloneOp?.progressPct !== null &&
                          activeCloneOp?.progressPct !== undefined
                            ? `${activeCloneOp.progressPct}%`
                            : `${activePipelineStep + 1}/${PIPELINE_A_STEPS.length}`}
                        </span>
                      </div>
                      {activeCloneOp?.progressPct !== null &&
                        activeCloneOp?.progressPct !== undefined && (
                          <div className="w-full h-1.5 bg-[#12231D] rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[#16A36A] transition-all duration-300"
                              style={{ width: `${activeCloneOp.progressPct}%` }}
                            />
                          </div>
                        )}
                    </div>
                  )}
                </div>
              );
            })()}
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

                <div className="flex flex-wrap items-center gap-2 shrink-0">
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
                  {cloneResult.chunks && cloneResult.chunks.length > 0 && (
                    <>
                      <button
                        type="button"
                        onClick={handleDownloadGeneratedSrt}
                        title="Download aligned SubRip (.SRT) subtitles for this voiceover"
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#0B1512] hover:bg-[#19352A] border border-[#16A36A]/50 text-xs font-semibold text-[#42D9A0] transition-colors whitespace-nowrap cursor-pointer"
                      >
                        <Captions className="w-3.5 h-3.5" />
                        <span>.SRT</span>
                      </button>
                      {onSendToSrtEditor && (
                        <button
                          type="button"
                          onClick={handleOpenGeneratedInSrtStudio}
                          title="Open audio & subtitles in Audio-to-SRT Studio for fine-tuning"
                          className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-[#19352A] hover:bg-[#16A36A]/30 border border-[#16A36A]/40 text-xs font-medium text-[#F4F8F5] transition-colors whitespace-nowrap cursor-pointer"
                        >
                          <span>Edit SRT</span>
                          <ArrowRight className="w-3.5 h-3.5 text-[#42D9A0]" />
                        </button>
                      )}
                    </>
                  )}
                </div>
              </div>

              {/* Audio Player */}
              <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3 space-y-2">
                <audio controls src={cloneResult.wavUrl} className="w-full h-9" />
                {cloneResult.audioEnhancementReport && (
                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-[#A2B5AA] pt-1 border-t border-[#19352A]/70">
                    <span className="text-[#42D9A0]">
                      {cloneResult.audioEnhancementReport.qualityLabel}
                    </span>
                    <span>
                      {Math.round(cloneResult.audioEnhancementReport.outputSampleRateHz / 1000)} kHz ·{' '}
                      {cloneResult.audioEnhancementReport.outputIntegratedLufs} LUFS · Peak{' '}
                      {cloneResult.audioEnhancementReport.outputTruePeakDb} dBTP · Pitch{' '}
                      {cloneResult.audioEnhancementReport.outputPitchHz} Hz
                    </span>
                  </div>
                )}
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
                          4. Signal &amp; Studio Mastering Telemetry
                        </div>
                        <div className="text-[#F4F8F5] font-mono">
                          {cloneResult.qualityControlReport.separatedMetrics.audioSignalQuality.sampleRateHz} Hz ·{' '}
                          {cloneResult.audioEnhancementReport
                            ? `${cloneResult.audioEnhancementReport.outputIntegratedLufs} LUFS · `
                            : ''}
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

