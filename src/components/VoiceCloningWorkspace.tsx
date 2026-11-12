import React, { useState, useRef, useEffect } from 'react';
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
  Cpu
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
import { fetchApiJson } from '../utils/apiClient';

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
  '02. Speaker Embedding & Acoustic Extraction',
  '03. Same-Language Script & Identity Verification',
  '04. Neural Speech Synthesis',
  '05. Quality Control & WAV/MP3 Export'
];

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
  const [preservePitchContour, setPreservePitchContour] = useState(false);
  const [preserveSpectralTimbre, setPreserveSpectralTimbre] = useState(false);
  const [localEndpointUrl, setLocalEndpointUrl] = useState<string>('');
  const [consentConfirmed, setConsentConfirmed] = useState(true);

  // Collapsible progressive disclosure states
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [showSignalDetails, setShowSignalDetails] = useState(false);
  const [showDetailedMetrics, setShowDetailedMetrics] = useState(false);

  const [isGenerating, setIsGenerating] = useState(false);
  const [activePipelineStep, setActivePipelineStep] = useState(0);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [cloneResult, setCloneResult] = useState<VoiceCloneResult | null>(null);

  // Microphone recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<number | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const currentLangConfig = CLONING_LANGUAGES[selectedLang];
  const scriptValidation = validateSameLanguageText(inputText, selectedLang);

  const detectedRefCode = referenceAnalysis?.detectedSpokenLanguage?.code;
  const isReferenceLanguageMismatch = Boolean(
    detectedRefCode &&
      ['bn', 'hi', 'en'].includes(detectedRefCode) &&
      detectedRefCode !== selectedLang
  );

  const handleSelectLanguage = (lang: CloningLanguageCode) => {
    setSelectedLang(lang);
    setGenerationError(null);
    const checkExisting = validateSameLanguageText(inputText, lang);
    if (!checkExisting.isValid) {
      setInputText(CLONING_LANGUAGES[lang].sampleTexts[0].text);
    }
  };

  const handleLoadAuthorizedSample = async (lang: CloningLanguageCode) => {
    setIsUploading(true);
    setUploadError(null);
    setGenerationError(null);
    try {
      const data = await fetchApiJson<{ status: string; analysis: AudioSignalAnalysis }>(
        `/api/sample-reference/${lang}`
      );
      setReferenceAnalysis(data.analysis);
      setReferenceTranscript(CLONING_LANGUAGES[lang].sampleTexts[0].text);
      if (selectedLang !== lang) {
        handleSelectLanguage(lang);
      }
    } catch (err: any) {
      setUploadError(err?.message || 'Could not load studio sample reference.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setUploadError(null);
    setGenerationError(null);
    try {
      const formData = new FormData();
      formData.append('audio', file);
      formData.append('workspace', 'voice-cloning');

      const data = await fetchApiJson<{ status: string; analysis: AudioSignalAnalysis }>(
        '/api/audio/upload',
        {
          method: 'POST',
          body: formData
        }
      );
      setReferenceAnalysis(data.analysis);
    } catch (err: any) {
      setUploadError(err?.message || 'Failed to upload and validate reference audio.');
    } finally {
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

  const handleGenerateClonedVoice = async () => {
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

    const stepInterval = window.setInterval(() => {
      setActivePipelineStep((prev) => (prev < PIPELINE_A_STEPS.length - 1 ? prev + 1 : prev));
    }, 1200);

    try {
      const data = await fetchApiJson<{ status: string; result: VoiceCloneResult }>(
        '/api/voice-clone/generate',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            referenceFileId: referenceAnalysis.fileId,
            language: selectedLang,
            text: inputText,
            referenceTranscript: referenceTranscript.trim() || undefined,
            preservePitchContour,
            preserveSpectralTimbre,
            localEndpointUrl: localEndpointUrl.trim() || undefined,
            consentConfirmed
          })
        }
      );

      const resObj: VoiceCloneResult = data.result;
      setCloneResult(resObj);
      setActivePipelineStep(PIPELINE_A_STEPS.length - 1);

      if (onActivityComplete) {
        onActivityComplete({
          languageLabel: `${currentLangConfig.englishName} (${currentLangConfig.nativeName})`,
          downloadUrl: `${resObj.wavUrl}?download=1`,
          downloadFilename: `voicelab_${resObj.language}_${resObj.jobId}.wav`
        });
      }
    } catch (err: any) {
      setGenerationError(err?.message || 'Voice cloning pipeline encountered an error.');
    } finally {
      clearInterval(stepInterval);
      setIsGenerating(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="space-y-5">
      {/* Compact Workspace Header + Language Lock + Mode Status Panel */}
      <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 sm:p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-xl sm:text-2xl font-bold text-[#F4F8F5] tracking-tight">
            Voice Cloning Studio
          </h1>
          <p className="text-xs text-[#A2B5AA]">
            Same-language speech synthesis for Bengali, Hindi, and English.
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
                ? `Mode 2: Local (${currentLangConfig.openSourceEquivalent.split(' ')[0]})`
                : `Mode 1: Cloud Persona`}
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
        {/* Left Column (5 cols): Reference Audio Upload, Recording & Reference Transcript */}
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

            {/* Compact Reference Audio Card */}
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
                className={`w-full rounded-xl bg-[#0B1512] border border-[#19352A] focus:border-[#16A36A] p-2.5 text-xs text-[#F4F8F5] placeholder-[#A2B5AA]/50 focus:outline-none ${
                  selectedLang === 'bn'
                    ? 'font-bengali'
                    : selectedLang === 'hi'
                    ? 'font-hindi'
                    : ''
                }`}
              />
            </div>
          </div>
        </div>

        {/* Right Column (7 cols): Target Text, Collapsible Settings, Action & Compact Result */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 sm:p-5 space-y-4">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-sm font-semibold text-[#F4F8F5]">
                02. Target Synthesis Text ({currentLangConfig.nativeName})
              </h2>
              <span className="text-xs font-mono tabular-nums text-[#A2B5AA]">
                {Array.from(inputText).length} / 1,000
              </span>
            </div>

            {/* Compact Sample Presets */}
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

            {/* Target Textarea */}
            <div>
              <textarea
                rows={4}
                maxLength={1000}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder={`Enter ${currentLangConfig.nativeName} (${currentLangConfig.englishName}) text to synthesize...`}
                className={`w-full rounded-xl bg-[#0B1512] border ${
                  !scriptValidation.isValid && inputText.trim().length > 0
                    ? 'border-red-500/70 focus:border-red-400'
                    : 'border-[#19352A] focus:border-[#16A36A]'
                } p-3.5 text-sm text-[#F4F8F5] placeholder-[#A2B5AA]/50 focus:outline-none transition-colors leading-relaxed ${
                  selectedLang === 'bn'
                    ? 'font-bengali'
                    : selectedLang === 'hi'
                    ? 'font-hindi'
                    : ''
                }`}
              />

              {!scriptValidation.isValid && inputText.trim().length > 0 ? (
                <div className="mt-2 bg-red-950/60 border border-red-500/50 rounded-xl p-3 text-xs text-red-200 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                  <span>{scriptValidation.errorMessage}</span>
                </div>
              ) : (
                <div className="mt-1.5 flex items-center justify-between text-[11px] text-[#A2B5AA]">
                  <span>
                    Script Guard:{' '}
                    <span className="text-[#42D9A0] font-medium">
                      Valid {currentLangConfig.nativeName} ({currentLangConfig.scriptName})
                    </span>
                  </span>
                  <span className="font-mono tabular-nums">
                    bn:{scriptValidation.bengaliCharCount} · hi:{scriptValidation.hindiCharCount} · en:{scriptValidation.latinCharCount}
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
                  <span>Advanced Execution Mode &amp; DSP Settings</span>
                </span>
                <span className="flex items-center gap-1.5 text-[11px] font-mono text-[#A2B5AA]">
                  <span>
                    {localEndpointUrl.trim() ? 'Mode 2 Local' : 'Mode 1 Cloud'} · DSP{' '}
                    {preservePitchContour || preserveSpectralTimbre ? 'On' : 'Off'}
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
                  <div className="space-y-1">
                    <label className="block font-medium text-[#F4F8F5]">
                      Local Neural Node URL (Mode 2 · IndicF5 / OpenVoice V2)
                    </label>
                    <input
                      type="url"
                      value={localEndpointUrl}
                      onChange={(e) => setLocalEndpointUrl(e.target.value)}
                      placeholder="Leave blank for Cloud Mode 1, or enter http://127.0.0.1:8001"
                      className="w-full rounded-lg bg-[#12231D] border border-[#19352A] focus:border-[#16A36A] px-3 py-1.5 text-xs font-mono text-[#F4F8F5] placeholder-[#A2B5AA]/50 focus:outline-none"
                    />
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    <label className="flex items-start gap-2.5 p-2.5 rounded-lg bg-[#12231D] border border-[#19352A] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={preservePitchContour}
                        onChange={(e) => setPreservePitchContour(e.target.checked)}
                        className="mt-0.5 accent-[#16A36A]"
                      />
                      <div>
                        <div className="font-medium text-[#F4F8F5]">
                          Conservative F0 Pitch Alignment
                        </div>
                        <div className="text-[11px] text-[#A2B5AA]">
                          Bounded ±2.0 semitones max toward reference F0.
                        </div>
                      </div>
                    </label>

                    <label className="flex items-start gap-2.5 p-2.5 rounded-lg bg-[#12231D] border border-[#19352A] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={preserveSpectralTimbre}
                        onChange={(e) => setPreserveSpectralTimbre(e.target.checked)}
                        className="mt-0.5 accent-[#16A36A]"
                      />
                      <div>
                        <div className="font-medium text-[#F4F8F5]">
                          Gentle Spectral EQ (±2 dB)
                        </div>
                        <div className="text-[11px] text-[#A2B5AA]">
                          220 Hz &amp; 3.2 kHz shelf + -1.0 dBFS peak limiter.
                        </div>
                      </div>
                    </label>
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
              <div className="bg-red-950/60 border border-red-500/50 rounded-xl p-3 text-xs text-red-200 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <span>{generationError}</span>
              </div>
            )}

            {/* Primary Generate Action */}
            <button
              type="button"
              onClick={handleGenerateClonedVoice}
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
                  <span>Synthesizing {currentLangConfig.nativeName} Speech...</span>
                </>
              ) : (
                <>
                  <Volume2 className="w-4 h-4 shrink-0" />
                  <span>Generate {currentLangConfig.nativeName} Speech</span>
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

          {/* Compact Output Audio Player & Quality Metrics */}
          {cloneResult && (
            <div className="bg-[#12231D] border border-[#16A36A]/50 rounded-2xl p-4 sm:p-5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="flex items-center gap-2 min-w-0">
                  <CheckCircle2 className="w-4 h-4 text-[#42D9A0] shrink-0" />
                  <h3 className="text-sm sm:text-base font-bold text-[#F4F8F5] truncate">
                    {CLONING_LANGUAGES[cloneResult.language].nativeName} Output Ready (
                    {cloneResult.outputDurationSec}s)
                  </h3>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <a
                    href={`${cloneResult.wavUrl}?download=1`}
                    download={`voicelab_${cloneResult.language}_${cloneResult.jobId}.wav`}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] transition-colors whitespace-nowrap"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>WAV</span>
                  </a>
                  <a
                    href={`${cloneResult.mp3Url}?download=1`}
                    download={`voicelab_${cloneResult.language}_${cloneResult.jobId}.mp3`}
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

              {/* Compact Summary Metrics Strip */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5">
                  <div className="text-[11px] text-[#A2B5AA]">ASR Match</div>
                  <div className="font-mono tabular-nums font-semibold text-[#42D9A0]">
                    {
                      cloneResult.qualityControlReport.separatedMetrics.asrTranscriptAccuracy
                        .characterMatchRatePct
                    }
                    %
                  </div>
                </div>
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5">
                  <div className="text-[11px] text-[#A2B5AA]">Audio Cosine</div>
                  <div className="font-mono tabular-nums font-semibold text-[#F4F8F5]">
                    {cloneResult.qualityControlReport.separatedMetrics
                      .generalAudioEmbeddingSimilarity.cosineSimilarity !== null
                      ? `${(
                          cloneResult.qualityControlReport.separatedMetrics
                            .generalAudioEmbeddingSimilarity.cosineSimilarity * 100
                        ).toFixed(1)}%`
                      : 'N/A'}
                  </div>
                </div>
                <div className="bg-[#0B1512] border border-[#19352A] rounded-lg p-2.5">
                  <div className="text-[11px] text-[#A2B5AA]">SNR / RMS</div>
                  <div className="font-mono tabular-nums font-semibold text-[#F4F8F5]">
                    {
                      cloneResult.qualityControlReport.separatedMetrics.audioSignalQuality
                        .estimatedSnrDb
                    }{' '}
                    dB
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
                          1. Speaker Identity (ECAPA-TDNN)
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
                        5. ASR Back-Transcription Verification:
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
