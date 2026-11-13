import React, { useState, useRef, useMemo, useEffect } from 'react';
import {
  Upload,
  FileAudio,
  Download,
  Copy,
  Check,
  Scissors,
  Merge,
  Trash2,
  Plus,
  Play,
  AlertTriangle,
  RefreshCw,
  Globe2,
  Clock,
  FileText,
  Mic,
  Square,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import {
  AudioSignalAnalysis,
  SrtSegment,
  SrtTranscriptionResult
} from '../types/voicelab';
import { SRT_SUPPORTED_LANGUAGES } from '../data/researchAndConfigs';
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

interface SrtGeneratorWorkspaceProps {
  onActivityComplete?: (info: {
    languageLabel: string;
    srtContent: string;
    downloadFilename: string;
  }) => void;
}

const SRT_REGEX = /^\d{2}:\d{2}:\d{2},\d{3}$/;

function msToSrtTimestampClient(ms: number): string {
  const clamped = Math.max(0, Math.round(ms));
  const hours = Math.floor(clamped / 3600000);
  const minutes = Math.floor((clamped % 3600000) / 60000);
  const seconds = Math.floor((clamped % 60000) / 1000);
  const millis = clamped % 1000;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(
    seconds
  ).padStart(2, '0')},${String(millis).padStart(3, '0')}`;
}

function srtTimestampToMsClient(ts: string): number | null {
  const trimmed = ts.trim();
  if (!SRT_REGEX.test(trimmed)) return null;
  const [hms, msStr] = trimmed.split(',');
  const [hh, mm, ss] = hms.split(':').map((n) => parseInt(n, 10));
  const ms = parseInt(msStr, 10);
  return hh * 3600000 + mm * 60000 + ss * 1000 + ms;
}

function buildSrtStringFromSegments(segments: SrtSegment[]): string {
  return segments
    .map((seg, idx) => `${idx + 1}\n${seg.startTimestamp} --> ${seg.endTimestamp}\n${seg.text.trim()}\n`)
    .join('\n');
}

export const SrtGeneratorWorkspace: React.FC<SrtGeneratorWorkspaceProps> = ({
  onActivityComplete
}) => {
  const [selectedAudioFile, setSelectedAudioFile] = useState<File | null>(null);
  const [localAudioUrl, setLocalAudioUrl] = useState<string | null>(null);
  const [audioAnalysis, setAudioAnalysis] = useState<AudioSignalAnalysis | null>(null);
  const [requestedLanguage, setRequestedLanguage] = useState<string>('auto');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [isTranscribing, setIsTranscribing] = useState(false);
  const [transcriptionError, setTranscriptionError] = useState<string | null>(null);
  const [srtResult, setSrtResult] = useState<SrtTranscriptionResult | null>(null);

  // Microphone recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<number | null>(null);

  // Editable segments state
  const [segments, setSegments] = useState<SrtSegment[]>([]);
  const [currentPlayheadMs, setCurrentPlayheadMs] = useState<number>(0);
  const [copiedSrt, setCopiedSrt] = useState(false);
  const [includeUtf8Bom, setIncludeUtf8Bom] = useState(true);
  const [searchFilter, setSearchFilter] = useState('');

  // Collapsible progressive disclosure states
  const [showExportSettings, setShowExportSettings] = useState(false);
  const [showRawSrtPreview, setShowRawSrtPreview] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const operationSeqRef = useRef<number>(0);

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  useEffect(() => {
    return () => {
      if (localAudioUrl) {
        URL.revokeObjectURL(localAudioUrl);
      }
    };
  }, [localAudioUrl]);

  const handleFileUpload = async (file: File) => {
    // 1. Increment operation sequence & abort any in-flight upload or transcription
    operationSeqRef.current += 1;
    const currentOpSeq = operationSeqRef.current;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    // 2. Reset file input value so re-uploading or replacing files always fires onChange
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }

    // 3. Immediately clear all previous audio analysis, transcripts, SRT blocks, and errors
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      audioPlayerRef.current.currentTime = 0;
    }
    setSelectedAudioFile(file);
    setLocalAudioUrl((prevUrl) => {
      if (prevUrl) URL.revokeObjectURL(prevUrl);
      return URL.createObjectURL(file);
    });
    setAudioAnalysis(null);
    setSrtResult(null);
    setSegments([]);
    setCurrentPlayheadMs(0);
    setSearchFilter('');
    setUploadError(null);
    setTranscriptionError(null);
    setIsTranscribing(false);
    setIsUploading(true);

    const op = startOperation({
      feature: 'audio-upload',
      title: `Upload & Validate ${file.name}`,
      stages: [
        'Uploading audio file bytes',
        'Verifying audio container signature & running FFprobe/FFmpeg DSP'
      ],
      initialProgressPct: 0,
      progressSource: 'upload-bytes',
      canCancel: true,
      cancelPreviousSameFeature: true
    });
    const stopStream = connectBackendOperationStream(op.operationId);

    op.signal.addEventListener(
      'abort',
      () => {
        controller.abort();
      },
      { once: true }
    );

    try {
      const formData = new FormData();
      formData.append('audio', file);
      formData.append('workspace', 'srt-generator');

      const data = await fetchApiJson<{ status: string; analysis: AudioSignalAnalysis }>(
        '/api/audio/upload',
        {
          method: 'POST',
          body: formData,
          operationId: op.operationId,
          signal: controller.signal,
          onUploadProgress: (loaded, total) => updateUploadBytes(op.operationId, loaded, total),
          onRetry: (_, reason) => recordOperationRetry(op.operationId, reason)
        }
      );

      if (currentOpSeq !== operationSeqRef.current || op.isStale()) {
        return;
      }

      setAudioAnalysis(data.analysis);
      setSrtResult(null);
      setSegments([]);
      completeOperation(
        op.operationId,
        `Validated ${data.analysis.fileName} (${data.analysis.durationSeconds}s · ${data.analysis.sampleRateHz} Hz)`
      );
    } catch (err: any) {
      if (err?.name === 'AbortError' || currentOpSeq !== operationSeqRef.current || op.isStale()) {
        return;
      }
      const msg = err?.message || 'Failed to upload and validate audio file.';
      setSelectedAudioFile(null);
      setAudioAnalysis(null);
      setSrtResult(null);
      setSegments([]);
      setUploadError(msg);
      failOperation(op.operationId, msg);
    } finally {
      stopStream();
      if (currentOpSeq === operationSeqRef.current) {
        setIsUploading(false);
      }
    }
  };

  const startMicrophoneRecording = async () => {
    setUploadError(null);
    setTranscriptionError(null);
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
          `srt_recording_${Date.now()}.webm`,
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
        'Microphone access unavailable. Please upload an audio file from your device.'
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

  const handleRunTranscription = async () => {
    if (!audioAnalysis && !selectedAudioFile) {
      setTranscriptionError('Please upload an audio file or record from your microphone first.');
      return;
    }

    operationSeqRef.current += 1;
    const currentOpSeq = operationSeqRef.current;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    // Immediately clear any prior SRT output before starting a new transcription
    setIsTranscribing(true);
    setTranscriptionError(null);
    setSrtResult(null);
    setSegments([]);

    const targetName =
      audioAnalysis?.fileName || selectedAudioFile?.name || 'Uploaded Audio';
    const op = startOperation({
      feature: 'srt-transcription',
      title: `Transcribe ${targetName}`,
      stages: [
        'Detecting speech intervals & splitting silence-aligned chunks (FFmpeg)',
        'Transcribing speech audio with Gemini Multilingual ASR',
        'Verifying exact token sequence & chronological timeline integrity'
      ],
      initialProgressPct: null,
      progressSource: 'indeterminate-inference',
      canCancel: true,
      cancelPreviousSameFeature: true
    });
    const stopStream = connectBackendOperationStream(op.operationId);

    op.signal.addEventListener(
      'abort',
      () => {
        controller.abort();
      },
      { once: true }
    );

    try {
      const formData = new FormData();
      if (selectedAudioFile) {
        formData.append('audio', selectedAudioFile);
      }
      if (audioAnalysis?.fileId) {
        formData.append('fileId', audioAnalysis.fileId);
      }
      formData.append('requestedLanguage', requestedLanguage);

      const data = await fetchApiJson<{
        status: string;
        analysis?: AudioSignalAnalysis;
        result: SrtTranscriptionResult;
      }>('/api/srt/transcribe', {
        method: 'POST',
        body: formData,
        operationId: op.operationId,
        signal: controller.signal,
        onRetry: (_, reason) => recordOperationRetry(op.operationId, reason)
      });

      if (currentOpSeq !== operationSeqRef.current || op.isStale()) {
        return;
      }

      const result: SrtTranscriptionResult = data.result;
      if (!result || !Array.isArray(result.segments) || result.segments.length === 0) {
        throw new Error(
          'No spoken words were recognized in the uploaded audio file. Please verify the audio contains audible speech.'
        );
      }

      if (data.analysis) {
        setAudioAnalysis(data.analysis);
      }
      setSrtResult(result);
      setSegments(result.segments);

      completeOperation(
        op.operationId,
        `Generated ${result.segments.length} ${result.detectedLanguageName} SRT subtitle blocks (${result.validationReport?.exactSequenceMatch ? '100% token match' : 'validated'})`
      );

      if (onActivityComplete) {
        const activeFileName =
          data.analysis?.fileName || audioAnalysis?.fileName || selectedAudioFile?.name;
        const baseName = activeFileName
          ? activeFileName.replace(/\.[^/.]+$/, '')
          : 'voicelab_subtitles';
        onActivityComplete({
          languageLabel: result.detectedLanguageName,
          srtContent: result.rawSrtContent,
          downloadFilename: `${baseName}_${result.detectedLanguageCode}.srt`
        });
      }
    } catch (err: any) {
      if (err?.name === 'AbortError' || currentOpSeq !== operationSeqRef.current || op.isStale()) {
        return;
      }
      const msg = err?.message || 'Transcription pipeline failed.';
      setSrtResult(null);
      setSegments([]);
      setTranscriptionError(msg);
      failOperation(op.operationId, msg);
    } finally {
      stopStream();
      if (currentOpSeq === operationSeqRef.current) {
        setIsTranscribing(false);
      }
    }
  };

  const activeSegment = useMemo(() => {
    return segments.find(
      (s) => currentPlayheadMs >= s.startTimeMs && currentPlayheadMs <= s.endTimeMs
    );
  }, [segments, currentPlayheadMs]);

  const liveSrtText = useMemo(() => {
    return buildSrtStringFromSegments(segments);
  }, [segments]);

  const reindexSegments = (list: SrtSegment[]): SrtSegment[] => {
    return list.map((seg, idx) => ({
      ...seg,
      id: idx + 1
    }));
  };

  const handleUpdateSegmentText = (index: number, newText: string) => {
    setSegments((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], text: newText };
      return next;
    });
  };

  const handleUpdateTimestamp = (
    index: number,
    field: 'startTimestamp' | 'endTimestamp',
    value: string
  ) => {
    setSegments((prev) => {
      const next = [...prev];
      const target = { ...next[index], [field]: value };
      const parsedMs = srtTimestampToMsClient(value);
      if (parsedMs !== null) {
        if (field === 'startTimestamp') {
          target.startTimeMs = parsedMs;
        } else {
          target.endTimeMs = parsedMs;
        }
      }
      next[index] = target;
      return next;
    });
  };

  const handleSplitSegment = (index: number) => {
    const op = startOperation({
      feature: 'subtitle-processing',
      title: `Split Subtitle #${index + 1}`,
      stages: [
        `Calculating midpoint timestamp & splitting words for #${index + 1}`,
        'Re-indexing chronological SRT timeline'
      ],
      initialProgressPct: 50,
      progressSource: 'task-steps',
      canCancel: false
    });
    setSegments((prev) => {
      const target = prev[index];
      const durationMs = target.endTimeMs - target.startTimeMs;
      if (durationMs < 600) return prev;

      const midTimeMs = Math.round(target.startTimeMs + durationMs / 2);
      const midTimestamp = msToSrtTimestampClient(midTimeMs);

      const words = target.text.trim().split(/\s+/);
      let firstHalfText = target.text;
      let secondHalfText = '...';

      if (words.length >= 2) {
        const halfIdx = Math.ceil(words.length / 2);
        firstHalfText = words.slice(0, halfIdx).join(' ');
        secondHalfText = words.slice(halfIdx).join(' ');
      }

      const segA: SrtSegment = {
        ...target,
        endTimeMs: midTimeMs,
        endTimestamp: midTimestamp,
        text: firstHalfText
      };

      const segB: SrtSegment = {
        id: target.id + 1,
        startTimeMs: midTimeMs + 20,
        endTimeMs: target.endTimeMs,
        startTimestamp: msToSrtTimestampClient(midTimeMs + 20),
        endTimestamp: target.endTimestamp,
        text: secondHalfText
      };

      const updated = [...prev.slice(0, index), segA, segB, ...prev.slice(index + 1)];
      return reindexSegments(updated);
    });
    completeOperation(op.operationId, `Split subtitle segment #${index + 1} into two blocks`);
  };

  const handleMergeWithNext = (index: number) => {
    const op = startOperation({
      feature: 'subtitle-processing',
      title: `Merge Subtitles #${index + 1} & #${index + 2}`,
      stages: [
        `Joining text and spanning timestamps for #${index + 1}–#${index + 2}`,
        'Re-indexing chronological SRT timeline'
      ],
      initialProgressPct: 50,
      progressSource: 'task-steps',
      canCancel: false
    });
    setSegments((prev) => {
      if (index >= prev.length - 1) return prev;
      const current = prev[index];
      const nextSeg = prev[index + 1];

      const merged: SrtSegment = {
        id: current.id,
        startTimeMs: current.startTimeMs,
        endTimeMs: nextSeg.endTimeMs,
        startTimestamp: current.startTimestamp,
        endTimestamp: nextSeg.endTimestamp,
        text: `${current.text.trim()} ${nextSeg.text.trim()}`.trim()
      };

      const updated = [...prev.slice(0, index), merged, ...prev.slice(index + 2)];
      return reindexSegments(updated);
    });
    completeOperation(op.operationId, `Merged subtitle segments #${index + 1} & #${index + 2}`);
  };

  const handleDeleteSegment = (index: number) => {
    const op = startOperation({
      feature: 'subtitle-processing',
      title: `Delete Subtitle #${index + 1}`,
      stages: [`Removing segment #${index + 1}`, 'Re-indexing chronological SRT timeline'],
      initialProgressPct: 100,
      progressSource: 'task-steps',
      canCancel: false
    });
    setSegments((prev) => {
      const updated = prev.filter((_, idx) => idx !== index);
      return reindexSegments(updated);
    });
    completeOperation(op.operationId, `Deleted subtitle segment #${index + 1}`);
  };

  const handleInsertSegmentAfter = (index: number) => {
    const op = startOperation({
      feature: 'subtitle-processing',
      title: `Insert Subtitle After #${index + 1}`,
      stages: ['Allocating non-overlapping 2.0s cue window', 'Re-indexing chronological SRT timeline'],
      initialProgressPct: 100,
      progressSource: 'task-steps',
      canCancel: false
    });
    setSegments((prev) => {
      const refSeg = prev[index];
      const startMs = refSeg ? refSeg.endTimeMs + 50 : 0;
      const endMs = startMs + 2000;
      const newSeg: SrtSegment = {
        id: index + 2,
        startTimeMs: startMs,
        endTimeMs: endMs,
        startTimestamp: msToSrtTimestampClient(startMs),
        endTimestamp: msToSrtTimestampClient(endMs),
        text: 'New subtitle segment'
      };
      const updated = [...prev.slice(0, index + 1), newSeg, ...prev.slice(index + 1)];
      return reindexSegments(updated);
    });
    completeOperation(op.operationId, `Inserted new subtitle segment #${index + 2}`);
  };

  const handleSeekAudioToMs = (ms: number) => {
    if (audioPlayerRef.current) {
      audioPlayerRef.current.currentTime = ms / 1000;
      audioPlayerRef.current.play().catch(() => {});
    }
  };

  const handleDownloadSrtFile = () => {
    const baseName = audioAnalysis?.fileName
      ? audioAnalysis.fileName.replace(/\.[^/.]+$/, '')
      : 'voicelab_subtitles';
    const langSuffix = srtResult?.detectedLanguageCode || requestedLanguage || 'multilingual';
    const filename = `${baseName}_${langSuffix}.srt`;

    const op = startOperation({
      feature: 'export-download',
      title: `Export ${filename}`,
      stages: [
        `Formatting ${segments.length} subtitle blocks (${includeUtf8Bom ? 'UTF-8 BOM' : 'UTF-8'})`,
        'Triggering browser file download'
      ],
      initialProgressPct: 50,
      progressSource: 'task-steps',
      canCancel: false
    });

    const content = includeUtf8Bom ? `\uFEFF${liveSrtText}` : liveSrtText;
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    updateOperationStage(op.operationId, {
      stageIndex: 1,
      stageLabel: `Downloading ${filename}...`,
      progressPct: 100,
      progressSource: 'task-steps'
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    completeOperation(
      op.operationId,
      `Exported ${filename} (${segments.length} subtitle blocks)`
    );
  };

  const handleCopySrt = async () => {
    const op = startOperation({
      feature: 'export-download',
      title: 'Copy SubRip (.SRT) to Clipboard',
      stages: [`Copying ${segments.length} formatted SRT blocks to clipboard`],
      initialProgressPct: 100,
      progressSource: 'task-steps',
      canCancel: false
    });
    try {
      await navigator.clipboard.writeText(liveSrtText);
      setCopiedSrt(true);
      completeOperation(
        op.operationId,
        `Copied ${segments.length} SRT blocks to clipboard`
      );
      setTimeout(() => setCopiedSrt(false), 2000);
    } catch (err: any) {
      failOperation(op.operationId, err?.message || 'Clipboard copy failed.');
    }
  };

  const filteredSegments = useMemo(() => {
    if (!searchFilter.trim()) return segments.map((seg, idx) => ({ seg, originalIndex: idx }));
    const q = searchFilter.toLowerCase();
    return segments
      .map((seg, idx) => ({ seg, originalIndex: idx }))
      .filter(
        ({ seg }) =>
          seg.text.toLowerCase().includes(q) ||
          seg.startTimestamp.includes(q) ||
          seg.endTimestamp.includes(q)
      );
  }, [segments, searchFilter]);

  return (
    <div className="space-y-5">
      {/* Compact Workspace Header */}
      <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 sm:p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
        <div className="space-y-1">
          <h1 className="text-xl sm:text-2xl font-bold text-[#F4F8F5] tracking-tight">
            Audio to SRT Studio
          </h1>
          <p className="text-xs text-[#A2B5AA]">
            Intelligent sentence-segmented SubRip (<span className="font-mono text-[#42D9A0]">.srt</span>) transcription in native Unicode script.
          </p>
        </div>

        <div className="bg-[#0B1512] border border-[#19352A] rounded-xl px-3.5 py-2 flex items-center gap-2 text-xs self-start lg:self-auto">
          <Globe2 className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
          <span className="text-[#F4F8F5] font-medium whitespace-nowrap">
            Original Script Preserved (99+ Languages)
          </span>
        </div>
      </div>

      {/* Main Two-Column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* Left Column (4 cols): Upload / Record + Language Mode + Generate Action */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 sm:p-5 space-y-4">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-sm font-semibold text-[#F4F8F5]">
                01. Audio Source
              </h2>
              <span className="text-[11px] font-mono text-[#A2B5AA]">WAV · MP3 · M4A · FLAC</span>
            </div>

            {/* Compact Drag & Drop */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const file = e.dataTransfer.files?.[0];
                if (file) handleFileUpload(file);
              }}
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
                {isUploading ? 'Validating audio file...' : 'Upload Audio File (Drop or Browse)'}
              </div>
              <div className="text-[11px] text-[#A2B5AA] mt-0.5">
                Supports short clips &amp; long multi-minute recordings
              </div>
            </div>

            {/* Record Mic */}
            <div className="space-y-2">
              {!isRecording ? (
                <button
                  type="button"
                  onClick={startMicrophoneRecording}
                  disabled={isUploading}
                  className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl bg-[#0B1512] hover:bg-[#19352A] border border-[#19352A] text-xs font-medium text-[#F4F8F5] transition-colors cursor-pointer whitespace-nowrap"
                >
                  <Mic className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
                  <span>Record Microphone Audio</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={stopMicrophoneRecording}
                  className="w-full flex items-center justify-center gap-1.5 px-3 py-2 rounded-xl bg-red-950/80 hover:bg-red-900 border border-red-500/50 text-xs font-medium text-red-100 transition-colors cursor-pointer whitespace-nowrap"
                >
                  <Square className="w-3.5 h-3.5 text-red-400 shrink-0" />
                  <span className="font-mono tabular-nums">Stop Recording ({recordingSeconds}s)</span>
                </button>
              )}
            </div>

            {uploadError && (
              <div className="bg-red-950/50 border border-red-500/40 rounded-xl p-3 text-xs text-red-200 flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <span>{uploadError}</span>
              </div>
            )}

            {/* Language Selector */}
            <div className="pt-3 border-t border-[#19352A] space-y-3">
              <div>
                <label className="block text-xs font-semibold text-[#F4F8F5] mb-1.5">
                  02. Spoken Language Detection
                </label>
                <select
                  value={requestedLanguage}
                  onChange={(e) => setRequestedLanguage(e.target.value)}
                  className="w-full rounded-xl bg-[#0B1512] border border-[#19352A] px-3 py-2 text-xs text-[#F4F8F5] focus:outline-none focus:border-[#16A36A]"
                >
                  {SRT_SUPPORTED_LANGUAGES.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.label} — {lang.nativeLabel}
                    </option>
                  ))}
                </select>
              </div>

              {transcriptionError && (
                <div className="bg-red-950/50 border border-red-500/40 rounded-xl p-3 text-xs text-red-200 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                  <span>{transcriptionError}</span>
                </div>
              )}

              <button
                type="button"
                onClick={handleRunTranscription}
                disabled={(!audioAnalysis && !selectedAudioFile) || isUploading || isTranscribing}
                className="w-full py-3 px-5 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 disabled:bg-[#19352A] disabled:text-[#A2B5AA]/50 text-[#F4F8F5] font-semibold text-sm transition-colors flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed shadow-sm whitespace-nowrap"
              >
                {isTranscribing ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin shrink-0" />
                    <span>Segmenting &amp; Aligning SRT...</span>
                  </>
                ) : (
                  <>
                    <Clock className="w-4 h-4 shrink-0" />
                    <span>Generate Multilingual SRT</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column (7 cols): Audio Player, Live Monitor & Subtitle Editor */}
        <div className="lg:col-span-7 space-y-4">
          {audioAnalysis ? (
            <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 sm:p-5 space-y-3.5">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <div className="flex items-center gap-2 min-w-0">
                  <FileAudio className="w-4 h-4 text-[#42D9A0] shrink-0" />
                  <span className="font-semibold text-[#F4F8F5] truncate">
                    {audioAnalysis.fileName}
                  </span>
                </div>
                <div className="font-mono tabular-nums text-[#A2B5AA]">
                  {audioAnalysis.durationSeconds}s · Playhead:{' '}
                  <span className="text-[#42D9A0]">
                    {msToSrtTimestampClient(currentPlayheadMs)}
                  </span>
                </div>
              </div>

              <audio
                key={audioAnalysis.fileId}
                ref={audioPlayerRef}
                controls
                src={localAudioUrl || audioAnalysis.previewUrl}
                onTimeUpdate={(e) =>
                  setCurrentPlayheadMs(Math.round(e.currentTarget.currentTime * 1000))
                }
                className="w-full h-9"
              />

              {/* Compact Synced Caption Monitor */}
              <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3 min-h-[64px] flex flex-col items-center justify-center text-center">
                {activeSegment ? (
                  <>
                    <div className="text-[11px] font-mono tabular-nums text-[#42D9A0] mb-0.5">
                      #{activeSegment.id} · {activeSegment.startTimestamp} →{' '}
                      {activeSegment.endTimestamp}
                    </div>
                    <div className="text-sm sm:text-base font-medium text-[#F4F8F5] font-bengali font-hindi whitespace-pre-line">
                      {activeSegment.text}
                    </div>
                  </>
                ) : (
                  <div className="text-xs text-[#A2B5AA]">
                    {segments.length > 0
                      ? 'Play audio or click a segment below to preview synced captions.'
                      : 'Click "Generate Multilingual SRT" to transcribe and segment sentences.'}
                  </div>
                )}
              </div>

              {srtResult && (
                <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-[#19352A] text-xs text-[#A2B5AA]">
                  <div>
                    Detected:{' '}
                    <span className="text-[#42D9A0] font-semibold">
                      {srtResult.detectedLanguageName}
                    </span>{' '}
                    · <span className="text-[#F4F8F5]">{srtResult.nativeScriptName}</span>
                  </div>
                  <div className="font-mono tabular-nums">
                    {segments.length} blocks · {(srtResult.processingTimeMs / 1000).toFixed(1)}s
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-6 text-center space-y-2">
              <FileText className="w-7 h-7 text-[#42D9A0] mx-auto" />
              <div className="text-sm font-semibold text-[#F4F8F5]">
                No Audio Loaded
              </div>
              <p className="text-xs text-[#A2B5AA] max-w-sm mx-auto">
                Upload an audio file or record from your microphone on the left to generate SubRip (.srt) subtitles.
              </p>
            </div>
          )}

          {/* Readable Subtitle Editor & Export Actions */}
          {segments.length > 0 && (
            <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 sm:p-5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#19352A]">
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold text-[#F4F8F5]">
                    Subtitle Editor ({segments.length} segments)
                  </h3>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={handleCopySrt}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#0B1512] hover:bg-[#19352A] border border-[#19352A] text-xs font-medium text-[#F4F8F5] transition-colors cursor-pointer whitespace-nowrap"
                  >
                    {copiedSrt ? (
                      <>
                        <Check className="w-3.5 h-3.5 text-[#42D9A0]" />
                        <span>Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5 text-[#42D9A0]" />
                        <span>Copy SRT</span>
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={handleDownloadSrtFile}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] transition-colors cursor-pointer whitespace-nowrap"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download .SRT</span>
                  </button>
                </div>
              </div>

              {/* Filter Input */}
              <input
                type="text"
                value={searchFilter}
                onChange={(e) => setSearchFilter(e.target.value)}
                placeholder="Search subtitle text or timestamp..."
                className="w-full rounded-xl bg-[#0B1512] border border-[#19352A] px-3 py-2 text-xs text-[#F4F8F5] placeholder-[#A2B5AA]/50 focus:outline-none focus:border-[#16A36A]"
              />

              {/* Compact Segment Cards */}
              <div className="space-y-2.5 max-h-[440px] overflow-y-auto pr-1">
                {filteredSegments.map(({ seg, originalIndex }) => {
                  const isStartValid = SRT_REGEX.test(seg.startTimestamp.trim());
                  const isEndValid = SRT_REGEX.test(seg.endTimestamp.trim());
                  const isPlayingNow = activeSegment?.id === seg.id;

                  return (
                    <div
                      key={seg.id}
                      className={`rounded-xl border p-3 transition-colors space-y-2 ${
                        isPlayingNow
                          ? 'bg-[#19352A] border-[#16A36A]'
                          : 'bg-[#0B1512] border-[#19352A]'
                      }`}
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => handleSeekAudioToMs(seg.startTimeMs)}
                            title="Play from segment start"
                            className="inline-flex items-center gap-1 px-2 py-1 rounded bg-[#12231D] hover:bg-[#16A36A] text-xs font-mono tabular-nums text-[#42D9A0] hover:text-[#F4F8F5] transition-colors cursor-pointer"
                          >
                            <Play className="w-3 h-3" />
                            <span>#{seg.id}</span>
                          </button>

                          <input
                            type="text"
                            value={seg.startTimestamp}
                            onChange={(e) =>
                              handleUpdateTimestamp(originalIndex, 'startTimestamp', e.target.value)
                            }
                            className={`w-24 sm:w-28 rounded bg-[#12231D] border px-2 py-1 text-xs font-mono tabular-nums text-[#F4F8F5] focus:outline-none ${
                              isStartValid ? 'border-[#19352A] focus:border-[#16A36A]' : 'border-red-500'
                            }`}
                          />
                          <span className="text-[11px] font-mono text-[#A2B5AA]">→</span>
                          <input
                            type="text"
                            value={seg.endTimestamp}
                            onChange={(e) =>
                              handleUpdateTimestamp(originalIndex, 'endTimestamp', e.target.value)
                            }
                            className={`w-24 sm:w-28 rounded bg-[#12231D] border px-2 py-1 text-xs font-mono tabular-nums text-[#F4F8F5] focus:outline-none ${
                              isEndValid ? 'border-[#19352A] focus:border-[#16A36A]' : 'border-red-500'
                            }`}
                          />
                        </div>

                        <div className="flex items-center gap-1">
                          <button
                            type="button"
                            onClick={() => handleSplitSegment(originalIndex)}
                            title="Split segment"
                            className="inline-flex items-center gap-1 px-2 py-1 rounded bg-[#12231D] hover:bg-[#19352A] text-[11px] text-[#A2B5AA] hover:text-[#F4F8F5] transition-colors cursor-pointer"
                          >
                            <Scissors className="w-3 h-3" />
                            <span className="hidden sm:inline">Split</span>
                          </button>

                          {originalIndex < segments.length - 1 && (
                            <button
                              type="button"
                              onClick={() => handleMergeWithNext(originalIndex)}
                              title="Merge with next segment"
                              className="inline-flex items-center gap-1 px-2 py-1 rounded bg-[#12231D] hover:bg-[#19352A] text-[11px] text-[#A2B5AA] hover:text-[#F4F8F5] transition-colors cursor-pointer"
                            >
                              <Merge className="w-3 h-3" />
                              <span className="hidden sm:inline">Merge</span>
                            </button>
                          )}

                          <button
                            type="button"
                            onClick={() => handleInsertSegmentAfter(originalIndex)}
                            title="Insert segment after"
                            className="p-1 rounded bg-[#12231D] hover:bg-[#19352A] text-[#A2B5AA] hover:text-[#F4F8F5] transition-colors cursor-pointer"
                          >
                            <Plus className="w-3.5 h-3.5" />
                          </button>

                          <button
                            type="button"
                            onClick={() => handleDeleteSegment(originalIndex)}
                            title="Delete segment"
                            className="p-1 rounded bg-[#12231D] hover:bg-red-950/70 text-[#A2B5AA] hover:text-red-300 transition-colors cursor-pointer"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>

                      <textarea
                        rows={2}
                        value={seg.text}
                        onChange={(e) => handleUpdateSegmentText(originalIndex, e.target.value)}
                        className="w-full rounded-lg bg-[#12231D] border border-[#19352A] focus:border-[#16A36A] px-2.5 py-1.5 text-xs sm:text-sm text-[#F4F8F5] focus:outline-none font-bengali font-hindi"
                      />
                    </div>
                  );
                })}
              </div>

              {/* Collapsible Advanced Export & Validation Info */}
              <div className="pt-2 border-t border-[#19352A] space-y-2">
                <button
                  type="button"
                  onClick={() => setShowExportSettings((prev) => !prev)}
                  className="w-full flex items-center justify-between text-xs text-[#A2B5AA] hover:text-[#F4F8F5] cursor-pointer py-1"
                >
                  <span>Segmentation Validation &amp; Export Options</span>
                  {showExportSettings ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </button>

                {showExportSettings && (
                  <div className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3 space-y-2.5 text-xs">
                    <label className="flex items-center gap-2 text-[#F4F8F5] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={includeUtf8Bom}
                        onChange={(e) => setIncludeUtf8Bom(e.target.checked)}
                        className="accent-[#16A36A]"
                      />
                      <span>Include UTF-8 BOM (Recommended for Premiere Pro / DaVinci Resolve)</span>
                    </label>

                    {srtResult?.validationReport && (
                      <div className="space-y-2 pt-2 border-t border-[#19352A] font-mono tabular-nums text-[11px] text-[#A2B5AA]">
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                          <div>
                            Chronological:{' '}
                            <span className="text-[#42D9A0]">
                              {srtResult.validationReport.chronologicalOrderVerified ? 'Pass' : 'Check'}
                            </span>
                          </div>
                          <div>
                            Overlaps:{' '}
                            <span className="text-[#42D9A0]">
                              {srtResult.validationReport.zeroOverlapsVerified ? '0' : 'Detected'}
                            </span>
                          </div>
                          <div>
                            Max Lines:{' '}
                            <span className="text-[#F4F8F5]">
                              {srtResult.validationReport.maxLinesPerBlock}/block (≤
                              {srtResult.validationReport.maxCharsPerLineObserved} ch)
                            </span>
                          </div>
                          <div>
                            Token Seq:{' '}
                            <span className="text-[#42D9A0]">
                              {srtResult.validationReport.outputTokenCount}/
                              {srtResult.validationReport.inputTokenCount} (
                              {srtResult.validationReport.exactSequenceMatch ? 'Exact' : 'Diff'})
                            </span>
                          </div>
                        </div>
                        <div className="flex flex-wrap items-center justify-between gap-2 text-[10px]">
                          <span>
                            Miss:{srtResult.validationReport.missingTokenCount} · Dup:
                            {srtResult.validationReport.duplicateTokenCount} · Reord:
                            {srtResult.validationReport.reorderedTokenCount} · Ins:
                            {srtResult.validationReport.insertedTokenCount}
                          </span>
                          <span className="text-[#F4F8F5] truncate">
                            Timing: {srtResult.validationReport.timingAlignmentSummary}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                <button
                  type="button"
                  onClick={() => setShowRawSrtPreview((prev) => !prev)}
                  className="w-full flex items-center justify-between text-xs text-[#A2B5AA] hover:text-[#F4F8F5] cursor-pointer py-1"
                >
                  <span>Raw SubRip (.SRT) File Preview</span>
                  {showRawSrtPreview ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </button>

                {showRawSrtPreview && (
                  <pre className="bg-[#0B1512] border border-[#19352A] rounded-xl p-3 text-xs font-mono text-[#F4F8F5] max-h-40 overflow-y-auto whitespace-pre-wrap">
                    {liveSrtText}
                  </pre>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
