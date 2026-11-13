import { useSyncExternalStore } from 'react';
import { resolveApiUrl } from './apiClient';

export type ActivityFeatureType =
  | 'voice-loading'
  | 'audio-upload'
  | 'voice-cloning'
  | 'srt-transcription'
  | 'subtitle-processing'
  | 'export-download'
  | 'storage-cleanup';

export type ActivityOperationStatus = 'running' | 'succeeded' | 'failed' | 'cancelled';

export type ProgressDataSource =
  | 'upload-bytes'
  | 'backend-chunks'
  | 'task-steps'
  | 'indeterminate-inference'
  | 'completed'
  | 'none';

export interface ActivityStageItem {
  id: string;
  label: string;
  status: 'pending' | 'active' | 'completed' | 'failed';
  detail?: string;
}

export interface LiveActivityOperation {
  operationId: string;
  sequenceId: number;
  feature: ActivityFeatureType;
  featureLabel: string;
  title: string;
  status: ActivityOperationStatus;
  currentStage: string;
  stages: ActivityStageItem[];
  /**
   * Strictly a number (0..100) ONLY when genuine measurable progress exists
   * (e.g. XHR upload byte ratio, multi-chunk completion ratio, or 100% completion).
   * Strictly `null` during indeterminate AI inference or single-pass DSP inspection.
   */
  progressPct: number | null;
  progressSource: ProgressDataSource;
  uploadedBytes?: number;
  totalBytes?: number;
  completedUnits?: number;
  totalUnits?: number;
  startTimeMs: number;
  updatedAtMs: number;
  completedAtMs?: number;
  error?: string;
  retryCount: number;
  canCancel: boolean;
}

export interface LiveActivitySnapshot {
  primaryOperation: LiveActivityOperation | null;
  activeOperations: LiveActivityOperation[];
  recentOperations: LiveActivityOperation[];
  isOnline: boolean;
  isExpanded: boolean;
  isDismissed: boolean;
}

const FEATURE_LABELS: Record<ActivityFeatureType, string> = {
  'voice-loading': 'Voice Selection',
  'audio-upload': 'Audio Upload & DSP',
  'voice-cloning': 'Voice Cloning',
  'srt-transcription': 'Audio to SRT',
  'subtitle-processing': 'Subtitle Processing',
  'export-download': 'Export & Download',
  'storage-cleanup': 'Storage Cleanup'
};

let globalSequenceCounter = 0;
const operationsMap = new Map<string, LiveActivityOperation>();
const abortControllersMap = new Map<string, AbortController>();
const timeoutTimersMap = new Map<string, number>();
const sseStreamsMap = new Map<string, { close: () => void }>();
const latestFeatureSeqMap = new Map<ActivityFeatureType, number>();

let isExpandedState = false;
let isDismissedState = false;
let isBrowserOnline = typeof navigator !== 'undefined' ? navigator.onLine : true;
let autoHideTimer: number | null = null;

const listeners = new Set<() => void>();

let cachedSnapshot: LiveActivitySnapshot = {
  primaryOperation: null,
  activeOperations: [],
  recentOperations: [],
  isOnline: isBrowserOnline,
  isExpanded: false,
  isDismissed: false
};

function rebuildSnapshot(): void {
  const all = Array.from(operationsMap.values()).sort((a, b) => b.sequenceId - a.sequenceId);
  const activeOperations = all.filter((op) => op.status === 'running');
  const recentOperations = all.slice(0, 6);

  // Primary operation prioritizes the newest running operation, or the most recently settled one
  const primaryOperation = activeOperations[0] || all[0] || null;

  cachedSnapshot = {
    primaryOperation,
    activeOperations,
    recentOperations,
    isOnline: isBrowserOnline,
    isExpanded: isExpandedState,
    isDismissed: isDismissedState
  };
}

function notifyListeners(): void {
  rebuildSnapshot();
  for (const listener of listeners) {
    listener();
  }
}

// Monitor browser online/offline connectivity events
if (typeof window !== 'undefined') {
  window.addEventListener('offline', () => {
    isBrowserOnline = false;
    for (const op of operationsMap.values()) {
      if (op.status === 'running') {
        op.currentStage = 'Network connection interrupted — waiting for connectivity...';
        op.updatedAtMs = Date.now();
      }
    }
    notifyListeners();
  });

  window.addEventListener('online', () => {
    isBrowserOnline = true;
    for (const op of operationsMap.values()) {
      if (
        op.status === 'running' &&
        op.currentStage.startsWith('Network connection interrupted')
      ) {
        const activeStageObj = op.stages.find((s) => s.status === 'active');
        op.currentStage = activeStageObj?.label || 'Connection restored — resuming operation...';
        op.updatedAtMs = Date.now();
      }
    }
    notifyListeners();
  });
}

export function generateOperationId(prefix = 'op'): string {
  const rand = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${Date.now()}_${rand}`;
}

export interface StartOperationOptions {
  operationId?: string;
  feature: ActivityFeatureType;
  title: string;
  stages: string[];
  initialStageIndex?: number;
  initialProgressPct?: number | null;
  progressSource?: ProgressDataSource;
  canCancel?: boolean;
  timeoutMs?: number;
  cancelPreviousSameFeature?: boolean;
}

export interface StartedOperationHandle {
  operationId: string;
  sequenceId: number;
  signal: AbortSignal;
  isStale: () => boolean;
}

export function startOperation(options: StartOperationOptions): StartedOperationHandle {
  const {
    feature,
    title,
    stages,
    initialStageIndex = 0,
    initialProgressPct = null,
    progressSource = 'indeterminate-inference',
    canCancel = true,
    timeoutMs = 180_000,
    cancelPreviousSameFeature = false
  } = options;

  if (autoHideTimer !== null && typeof window !== 'undefined') {
    window.clearTimeout(autoHideTimer);
    autoHideTimer = null;
  }

  // Optionally cancel previous in-flight operation of the same feature to prevent stale race conditions
  if (cancelPreviousSameFeature) {
    for (const existing of operationsMap.values()) {
      if (existing.feature === feature && existing.status === 'running') {
        cancelOperation(existing.operationId, 'Superseded by a newer operation.');
      }
    }
  }

  globalSequenceCounter += 1;
  const sequenceId = globalSequenceCounter;
  const operationId = options.operationId || generateOperationId(feature.split('-')[0]);
  latestFeatureSeqMap.set(feature, sequenceId);

  const controller = new AbortController();
  abortControllersMap.set(operationId, controller);

  const stageItems: ActivityStageItem[] = stages.map((label, idx) => ({
    id: `stage_${idx}`,
    label,
    status:
      idx < initialStageIndex
        ? 'completed'
        : idx === initialStageIndex
        ? 'active'
        : 'pending'
  }));

  const now = Date.now();
  const op: LiveActivityOperation = {
    operationId,
    sequenceId,
    feature,
    featureLabel: FEATURE_LABELS[feature],
    title,
    status: 'running',
    currentStage: stageItems[initialStageIndex]?.label || title,
    stages: stageItems,
    progressPct: initialProgressPct,
    progressSource,
    startTimeMs: now,
    updatedAtMs: now,
    retryCount: 0,
    canCancel
  };

  operationsMap.set(operationId, op);
  isDismissedState = false;

  if (timeoutMs > 0 && typeof window !== 'undefined') {
    const timerId = window.setTimeout(() => {
      const current = operationsMap.get(operationId);
      if (current && current.status === 'running') {
        controller.abort();
        closeBackendStream(operationId);
        current.status = 'failed';
        current.error = `Operation timed out after ${Math.round(timeoutMs / 1000)}s.`;
        current.currentStage = current.error;
        current.completedAtMs = Date.now();
        current.updatedAtMs = Date.now();
        for (const s of current.stages) {
          if (s.status === 'active') s.status = 'failed';
        }
        notifyListeners();
      }
    }, timeoutMs);
    timeoutTimersMap.set(operationId, timerId);
  }

  notifyListeners();

  const isStale = () => {
    const current = operationsMap.get(operationId);
    if (!current) return true;
    if (current.status === 'cancelled' || current.status === 'failed') return true;
    if (
      cancelPreviousSameFeature &&
      (latestFeatureSeqMap.get(feature) || 0) > sequenceId
    ) {
      return true;
    }
    return false;
  };

  return {
    operationId,
    sequenceId,
    signal: controller.signal,
    isStale
  };
}

function clearOperationResources(operationId: string): void {
  const timer = timeoutTimersMap.get(operationId);
  if (timer !== undefined && typeof window !== 'undefined') {
    window.clearTimeout(timer);
    timeoutTimersMap.delete(operationId);
  }
  abortControllersMap.delete(operationId);
  closeBackendStream(operationId);
}

function closeBackendStream(operationId: string): void {
  const stream = sseStreamsMap.get(operationId);
  if (stream) {
    stream.close();
    sseStreamsMap.delete(operationId);
  }
}

/**
 * Updates genuine byte-level upload progress from XMLHttpRequest.upload.onprogress.
 */
export function updateUploadBytes(
  operationId: string,
  loadedBytes: number,
  totalBytes: number
): void {
  const op = operationsMap.get(operationId);
  if (!op || op.status !== 'running') return;

  op.uploadedBytes = loadedBytes;
  op.totalBytes = totalBytes;
  op.updatedAtMs = Date.now();

  if (totalBytes > 0 && loadedBytes <= totalBytes) {
    const pct = Math.min(100, Math.max(0, Math.round((loadedBytes / totalBytes) * 100)));
    op.progressPct = pct;
    op.progressSource = 'upload-bytes';
    const kbLoaded = (loadedBytes / 1024).toFixed(0);
    const kbTotal = (totalBytes / 1024).toFixed(0);
    const firstStage = op.stages[0];
    if (firstStage) {
      firstStage.status = pct >= 100 ? 'completed' : 'active';
      firstStage.detail = `${kbLoaded} KB / ${kbTotal} KB (${pct}%)`;
    }
    if (pct < 100) {
      op.currentStage = `Uploading audio (${kbLoaded} / ${kbTotal} KB)...`;
    } else if (op.stages.length > 1) {
      // Transition from upload bytes to backend inspection (honest indeterminate state)
      op.stages[1].status = 'active';
      op.currentStage = op.stages[1].label;
      op.progressPct = null;
      op.progressSource = 'indeterminate-inference';
    }
  } else {
    op.progressPct = null;
    op.progressSource = 'indeterminate-inference';
  }

  notifyListeners();
}

export interface StageUpdatePayload {
  stageIndex?: number;
  stageLabel?: string;
  stageDetail?: string;
  /**
   * Pass a number (0..100) ONLY when backed by real completed chunks/units.
   * Pass `null` when entering an indeterminate AI inference or DSP step.
   */
  progressPct?: number | null;
  progressSource?: ProgressDataSource;
  completedUnits?: number;
  totalUnits?: number;
}

export function updateOperationStage(
  operationId: string,
  payload: StageUpdatePayload
): void {
  const op = operationsMap.get(operationId);
  if (!op || op.status !== 'running') return;

  const {
    stageIndex,
    stageLabel,
    stageDetail,
    progressPct,
    progressSource,
    completedUnits,
    totalUnits
  } = payload;

  if (typeof stageIndex === 'number' && stageIndex >= 0 && stageIndex < op.stages.length) {
    op.stages.forEach((stage, idx) => {
      if (idx < stageIndex) {
        stage.status = 'completed';
      } else if (idx === stageIndex) {
        stage.status = 'active';
        if (stageDetail !== undefined) {
          stage.detail = stageDetail;
        }
      } else {
        stage.status = 'pending';
      }
    });
    op.currentStage = stageLabel || op.stages[stageIndex].label;
  } else if (stageLabel) {
    op.currentStage = stageLabel;
    const activeStage = op.stages.find((s) => s.status === 'active');
    if (activeStage && stageDetail !== undefined) {
      activeStage.detail = stageDetail;
    }
  }

  if (completedUnits !== undefined && totalUnits !== undefined && totalUnits > 1) {
    op.completedUnits = completedUnits;
    op.totalUnits = totalUnits;
    op.progressPct = Math.min(100, Math.max(0, Math.round((completedUnits / totalUnits) * 100)));
    op.progressSource = progressSource || 'backend-chunks';
  } else if (progressPct !== undefined) {
    op.progressPct = progressPct;
    if (progressSource) {
      op.progressSource = progressSource;
    } else if (progressPct === null) {
      op.progressSource = 'indeterminate-inference';
    }
  }

  op.updatedAtMs = Date.now();
  notifyListeners();
}

export function recordOperationRetry(operationId: string, reason: string): void {
  const op = operationsMap.get(operationId);
  if (!op || op.status !== 'running') return;

  op.retryCount += 1;
  op.currentStage = `${reason} (Retry #${op.retryCount})`;
  op.updatedAtMs = Date.now();
  notifyListeners();
}

export function completeOperation(operationId: string, summaryMessage?: string): void {
  const op = operationsMap.get(operationId);
  // Prevent stale completion from overwriting a cancelled or failed operation
  if (!op || op.status !== 'running') return;

  clearOperationResources(operationId);

  op.status = 'succeeded';
  op.progressPct = 100;
  op.progressSource = 'completed';
  op.stages.forEach((s) => {
    s.status = 'completed';
  });
  if (summaryMessage) {
    op.currentStage = summaryMessage;
  } else {
    op.currentStage = `${op.title} completed`;
  }
  op.completedAtMs = Date.now();
  op.updatedAtMs = Date.now();

  notifyListeners();
}

export function failOperation(operationId: string, errorMessage: string): void {
  const op = operationsMap.get(operationId);
  if (!op || op.status === 'cancelled' || op.status === 'succeeded') return;

  clearOperationResources(operationId);

  op.status = 'failed';
  op.error = errorMessage;
  op.currentStage = errorMessage;
  op.completedAtMs = Date.now();
  op.updatedAtMs = Date.now();

  let markedStage = false;
  for (const s of op.stages) {
    if (s.status === 'active') {
      s.status = 'failed';
      markedStage = true;
    }
  }
  if (!markedStage && op.stages.length > 0) {
    op.stages[op.stages.length - 1].status = 'failed';
  }

  isDismissedState = false;
  notifyListeners();
}

export function cancelOperation(operationId: string, reason = 'Cancelled by user'): void {
  const op = operationsMap.get(operationId);
  if (!op || op.status !== 'running') return;

  const controller = abortControllersMap.get(operationId);
  if (controller) {
    try {
      controller.abort();
    } catch {
      // Ignore abort errors
    }
  }

  clearOperationResources(operationId);

  op.status = 'cancelled';
  op.error = reason;
  op.currentStage = reason;
  op.completedAtMs = Date.now();
  op.updatedAtMs = Date.now();

  for (const s of op.stages) {
    if (s.status === 'active') {
      s.status = 'failed';
    }
  }

  notifyListeners();
}

/**
 * Connects to backend Server-Sent Events (`/api/operations/:operationId/events`)
 * with an automatic JSON polling fallback (`/api/operations/:operationId`) so real
 * backend pipeline stages and multi-chunk progress update the live activity bar.
 */
export function connectBackendOperationStream(operationId: string): () => void {
  if (typeof window === 'undefined') return () => {};
  closeBackendStream(operationId);

  let closed = false;
  let eventSource: EventSource | null = null;
  let pollInterval: number | null = null;

  const applyBackendSnapshot = (data: any) => {
    if (closed || !data) return;
    const op = operationsMap.get(operationId);
    if (!op || op.status !== 'running') return;

    updateOperationStage(operationId, {
      stageIndex: typeof data.stageIndex === 'number' ? data.stageIndex : undefined,
      stageLabel: typeof data.stageLabel === 'string' ? data.stageLabel : undefined,
      stageDetail: typeof data.stageDetail === 'string' ? data.stageDetail : undefined,
      progressPct:
        typeof data.progressPct === 'number'
          ? data.progressPct
          : data.progressPct === null
          ? null
          : undefined,
      progressSource: data.progressSource,
      completedUnits:
        typeof data.completedUnits === 'number' ? data.completedUnits : undefined,
      totalUnits: typeof data.totalUnits === 'number' ? data.totalUnits : undefined
    });
  };

  try {
    if (typeof EventSource !== 'undefined') {
      const sseUrl = resolveApiUrl(`/api/operations/${encodeURIComponent(operationId)}/events`);
      eventSource = new EventSource(sseUrl, { withCredentials: true });
      eventSource.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data);
          applyBackendSnapshot(parsed);
        } catch {
          // Ignore malformed frame
        }
      };
      eventSource.onerror = () => {
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }
      };
    }
  } catch {
    // Fallback to polling below
  }

  // Lightweight fallback poll every 650ms in case reverse proxy buffers SSE frames
  pollInterval = window.setInterval(async () => {
    if (closed) return;
    const op = operationsMap.get(operationId);
    if (!op || op.status !== 'running') {
      cleanup();
      return;
    }
    try {
      const statusUrl = resolveApiUrl(`/api/operations/${encodeURIComponent(operationId)}`);
      const res = await fetch(statusUrl, {
        method: 'GET',
        credentials: 'include',
        cache: 'no-store'
      });
      if (res.ok) {
        const data = await res.json();
        if (data?.operation) {
          applyBackendSnapshot(data.operation);
        }
      }
    } catch {
      // Ignore transient poll errors
    }
  }, 650);

  const cleanup = () => {
    if (closed) return;
    closed = true;
    if (eventSource) {
      try {
        eventSource.close();
      } catch {
        // Ignore
      }
      eventSource = null;
    }
    if (pollInterval !== null) {
      window.clearInterval(pollInterval);
      pollInterval = null;
    }
    sseStreamsMap.delete(operationId);
  };

  sseStreamsMap.set(operationId, { close: cleanup });
  return cleanup;
}

export function setActivityBarExpanded(expanded: boolean): void {
  isExpandedState = expanded;
  notifyListeners();
}

export function toggleActivityBarExpanded(): void {
  isExpandedState = !isExpandedState;
  notifyListeners();
}

export function dismissActivityBar(): void {
  isDismissedState = true;
  isExpandedState = false;
  notifyListeners();
}

export function clearSettledOperations(): void {
  for (const [id, op] of operationsMap.entries()) {
    if (op.status !== 'running') {
      operationsMap.delete(id);
    }
  }
  if (operationsMap.size === 0) {
    isDismissedState = true;
    isExpandedState = false;
  }
  notifyListeners();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): LiveActivitySnapshot {
  return cachedSnapshot;
}

export function useLiveActivityState(): LiveActivitySnapshot {
  return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}
