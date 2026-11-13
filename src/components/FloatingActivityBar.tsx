import React, { useEffect, useState } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  ChevronUp,
  ChevronDown,
  X,
  WifiOff,
  Ban,
  Clock
} from 'lucide-react';
import {
  cancelOperation,
  clearSettledOperations,
  dismissActivityBar,
  LiveActivityOperation,
  ProgressDataSource,
  toggleActivityBarExpanded,
  useLiveActivityState
} from '../utils/activityManager';

function formatProgressSourceLabel(source: ProgressDataSource): string {
  switch (source) {
    case 'upload-bytes':
      return 'Measured Upload Bytes';
    case 'backend-chunks':
      return 'Completed Backend Chunks';
    case 'task-steps':
      return 'Completed Pipeline Steps';
    case 'indeterminate-inference':
      return 'Indeterminate AI Inference';
    case 'completed':
      return 'Completed (100%)';
    default:
      return 'Live Telemetry';
  }
}

function getActiveStagePosition(op: LiveActivityOperation): string {
  if (op.stages.length === 0) return '';
  const activeIdx = op.stages.findIndex((s) => s.status === 'active');
  if (activeIdx !== -1) {
    return `${activeIdx + 1}/${op.stages.length}`;
  }
  const completedCount = op.stages.filter((s) => s.status === 'completed').length;
  return `${completedCount}/${op.stages.length}`;
}

export const FloatingActivityBar: React.FC = () => {
  const {
    primaryOperation,
    activeOperations,
    recentOperations,
    isOnline,
    isExpanded,
    isDismissed
  } = useLiveActivityState();

  const [nowMs, setNowMs] = useState<number>(() => Date.now());

  useEffect(() => {
    if (activeOperations.length === 0) return;
    const timer = window.setInterval(() => {
      setNowMs(Date.now());
    }, 250);
    return () => window.clearInterval(timer);
  }, [activeOperations.length]);

  if (!primaryOperation) {
    return null;
  }

  // Keep visible whenever any operation is running, or when the latest operation hasn't been dismissed
  if (isDismissed && activeOperations.length === 0) {
    return null;
  }

  const isRunning = primaryOperation.status === 'running';
  const isSucceeded = primaryOperation.status === 'succeeded';
  const isFailed = primaryOperation.status === 'failed';
  const isCancelled = primaryOperation.status === 'cancelled';

  const elapsedMs = Math.max(
    0,
    (primaryOperation.completedAtMs || nowMs) - primaryOperation.startTimeMs
  );
  const elapsedLabel = `${(elapsedMs / 1000).toFixed(1)}s`;

  const hasGenuinePercentage = typeof primaryOperation.progressPct === 'number';
  const stagePosition = getActiveStagePosition(primaryOperation);

  const borderColorClass = isFailed
    ? 'border-red-500/60'
    : isCancelled || !isOnline
    ? 'border-amber-500/60'
    : isSucceeded
    ? 'border-[#16A36A]/70'
    : 'border-[#42D9A0]/60';

  const progressFillClass = isFailed
    ? 'bg-red-500'
    : isCancelled || !isOnline
    ? 'bg-amber-400'
    : 'bg-gradient-to-r from-[#16A36A] to-[#42D9A0]';

  return (
    <aside
      role="status"
      aria-live="polite"
      data-testid="floating-activity-bar"
      style={{ paddingBottom: 'env(safe-area-inset-bottom, 0px)' }}
      className="fixed bottom-3 left-3 right-3 sm:left-auto sm:right-5 sm:bottom-4 sm:w-[430px] z-50 pointer-events-none select-none"
    >
      <div
        className={`pointer-events-auto rounded-xl bg-[#0B1512]/95 backdrop-blur-md border ${borderColorClass} shadow-xl overflow-hidden transition-colors duration-200`}
      >
        {/* Thin Top Progress Indicator (Determinate ONLY when genuine progress data exists; Indeterminate otherwise) */}
        <div
          className="w-full h-1 bg-[#19352A] overflow-hidden relative"
          data-testid="activity-progress-track"
        >
          {hasGenuinePercentage ? (
            <div
              data-testid="activity-progress-determinate"
              style={{
                width: `${Math.min(100, Math.max(0, primaryOperation.progressPct ?? 0))}%`
              }}
              className={`h-full ${progressFillClass} transition-[width] duration-150 ease-out`}
            />
          ) : isRunning ? (
            <div
              data-testid="activity-progress-indeterminate"
              className={`h-full rounded-full ${progressFillClass} voicelab-indeterminate-bar`}
            />
          ) : (
            <div
              style={{ width: isSucceeded ? '100%' : '100%' }}
              className={`h-full ${progressFillClass} opacity-60`}
            />
          )}
        </div>

        {/* Compact Single-Row Live Status Bar */}
        <div className="px-3 py-2.5 flex items-center justify-between gap-2.5">
          {/* Left: Status Icon + Feature & Short Live Stage Text */}
          <button
            type="button"
            onClick={toggleActivityBarExpanded}
            className="flex items-center gap-2.5 min-w-0 flex-1 text-left cursor-pointer"
          >
            <div className="shrink-0 flex items-center justify-center">
              {!isOnline && isRunning ? (
                <WifiOff className="w-4 h-4 text-amber-400" />
              ) : isRunning ? (
                <RefreshCw className="w-4 h-4 text-[#42D9A0] animate-spin" />
              ) : isSucceeded ? (
                <CheckCircle2 className="w-4 h-4 text-[#42D9A0]" />
              ) : isCancelled ? (
                <Ban className="w-4 h-4 text-amber-400" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-red-400" />
              )}
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5 text-[11px] font-mono text-[#A2B5AA] truncate">
                <span className="text-[#42D9A0] font-semibold">
                  {primaryOperation.featureLabel}
                </span>
                <span>·</span>
                <span className="truncate">{primaryOperation.title}</span>
                {activeOperations.length > 1 && (
                  <span
                    data-testid="concurrent-ops-badge"
                    className="ml-1 px-1.5 py-0.2 rounded bg-[#19352A] text-[#42D9A0] text-[10px] font-mono shrink-0"
                  >
                    +{activeOperations.length - 1} active
                  </span>
                )}
              </div>

              <div
                data-testid="activity-live-status-text"
                className={`text-xs font-medium truncate ${
                  isFailed
                    ? 'text-red-200'
                    : isCancelled || !isOnline
                    ? 'text-amber-200'
                    : 'text-[#F4F8F5]'
                }`}
              >
                {primaryOperation.currentStage}
              </div>
            </div>
          </button>

          {/* Right: Genuine Percentage OR Stage Counter + Elapsed + Cancel/Dismiss + Expand Toggle */}
          <div className="flex items-center gap-1.5 shrink-0">
            <div className="text-right font-mono tabular-nums text-[11px] leading-tight">
              {hasGenuinePercentage ? (
                <div
                  data-testid="activity-percentage-value"
                  className="text-[#42D9A0] font-semibold"
                >
                  {Math.round(primaryOperation.progressPct ?? 0)}%
                </div>
              ) : stagePosition ? (
                <div
                  data-testid="activity-stage-counter"
                  className="text-[#A2B5AA]"
                  title="Indeterminate stage progress (no fabricated percentage)"
                >
                  Stage {stagePosition}
                </div>
              ) : null}
              <div className="text-[10px] text-[#A2B5AA]/80">{elapsedLabel}</div>
            </div>

            {isRunning && primaryOperation.canCancel && (
              <button
                type="button"
                onClick={() =>
                  cancelOperation(primaryOperation.operationId, 'Operation cancelled by user.')
                }
                title="Cancel running operation"
                data-testid="activity-cancel-button"
                className="px-2 py-1 rounded-lg bg-[#12231D] hover:bg-red-950/80 border border-[#19352A] hover:border-red-500/40 text-[11px] font-medium text-[#A2B5AA] hover:text-red-200 transition-colors cursor-pointer"
              >
                Cancel
              </button>
            )}

            <button
              type="button"
              onClick={toggleActivityBarExpanded}
              aria-label={isExpanded ? 'Hide stage details' : 'Show stage details'}
              data-testid="activity-expand-toggle"
              className="p-1.5 rounded-lg bg-[#12231D] hover:bg-[#19352A] text-[#A2B5AA] hover:text-[#F4F8F5] transition-colors cursor-pointer"
            >
              {isExpanded ? (
                <ChevronDown className="w-3.5 h-3.5" />
              ) : (
                <ChevronUp className="w-3.5 h-3.5" />
              )}
            </button>

            {!isRunning && (
              <button
                type="button"
                onClick={dismissActivityBar}
                aria-label="Dismiss activity bar"
                data-testid="activity-dismiss-button"
                className="p-1.5 rounded-lg bg-[#12231D] hover:bg-[#19352A] text-[#A2B5AA] hover:text-[#F4F8F5] transition-colors cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Expandable Details Drawer: Processing Stages, Progress Source & Concurrent Operations */}
        {isExpanded && (
          <div
            data-testid="activity-expanded-details"
            className="border-t border-[#19352A] bg-[#12231D]/90 px-3.5 py-2.5 space-y-2.5 text-xs"
          >
            {/* Telemetry Source & Operation Metadata */}
            <div className="flex items-center justify-between gap-2 text-[11px] font-mono text-[#A2B5AA]">
              <span>
                Source:{' '}
                <span className="text-[#42D9A0]">
                  {formatProgressSourceLabel(primaryOperation.progressSource)}
                </span>
              </span>
              <span className="truncate">ID: {primaryOperation.operationId.slice(-10)}</span>
            </div>

            {/* Ordered Processing Stages Checklist */}
            {primaryOperation.stages.length > 0 && (
              <div className="space-y-1.5">
                {primaryOperation.stages.map((stage, idx) => {
                  const isStageActive = stage.status === 'active';
                  const isStageDone = stage.status === 'completed';
                  const isStageFailed = stage.status === 'failed';
                  return (
                    <div
                      key={stage.id}
                      className={`flex items-start justify-between gap-2 px-2 py-1 rounded-lg ${
                        isStageActive
                          ? 'bg-[#19352A]/90 text-[#F4F8F5]'
                          : isStageDone
                          ? 'text-[#A2B5AA]'
                          : isStageFailed
                          ? 'bg-red-950/40 text-red-200'
                          : 'text-[#A2B5AA]/50'
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="font-mono text-[10px] shrink-0 w-4">
                          0{idx + 1}
                        </span>
                        {isStageDone ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
                        ) : isStageActive ? (
                          <RefreshCw className="w-3.5 h-3.5 text-[#42D9A0] animate-spin shrink-0" />
                        ) : isStageFailed ? (
                          <AlertTriangle className="w-3.5 h-3.5 text-red-400 shrink-0" />
                        ) : (
                          <Clock className="w-3.5 h-3.5 text-[#A2B5AA]/40 shrink-0" />
                        )}
                        <span className="truncate text-[11px]">{stage.label}</span>
                      </div>
                      {stage.detail && (
                        <span className="font-mono text-[10px] text-[#42D9A0] shrink-0">
                          {stage.detail}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* Concurrent & Recent Operations List (When multiple operations are tracked) */}
            {recentOperations.length > 1 && (
              <div className="pt-2 border-t border-[#19352A] space-y-1">
                <div className="flex items-center justify-between text-[10px] font-mono text-[#A2B5AA]">
                  <span>Concurrent &amp; Recent Operations ({recentOperations.length})</span>
                  {activeOperations.length === 0 && (
                    <button
                      type="button"
                      onClick={clearSettledOperations}
                      className="text-[#42D9A0] hover:underline cursor-pointer"
                    >
                      Clear History
                    </button>
                  )}
                </div>
                <div className="space-y-1 max-h-28 overflow-y-auto pr-1">
                  {recentOperations.slice(0, 4).map((opItem) => (
                    <div
                      key={opItem.operationId}
                      className="flex items-center justify-between gap-2 text-[11px] bg-[#0B1512]/80 px-2 py-1 rounded border border-[#19352A]/70"
                    >
                      <div className="flex items-center gap-1.5 min-w-0">
                        {opItem.status === 'running' ? (
                          <RefreshCw className="w-3 h-3 text-[#42D9A0] animate-spin shrink-0" />
                        ) : opItem.status === 'succeeded' ? (
                          <CheckCircle2 className="w-3 h-3 text-[#42D9A0] shrink-0" />
                        ) : (
                          <AlertTriangle className="w-3 h-3 text-red-400 shrink-0" />
                        )}
                        <span className="text-[#F4F8F5] truncate">{opItem.title}</span>
                      </div>
                      <div className="flex items-center gap-1.5 shrink-0 font-mono text-[10px]">
                        <span
                          className={
                            opItem.status === 'succeeded'
                              ? 'text-[#42D9A0]'
                              : opItem.status === 'failed'
                              ? 'text-red-300'
                              : opItem.status === 'cancelled'
                              ? 'text-amber-300'
                              : 'text-[#A2B5AA]'
                          }
                        >
                          {typeof opItem.progressPct === 'number'
                            ? `${Math.round(opItem.progressPct)}%`
                            : opItem.status}
                        </span>
                        {opItem.status === 'running' && opItem.canCancel && (
                          <button
                            type="button"
                            onClick={() =>
                              cancelOperation(opItem.operationId, 'Cancelled by user.')
                            }
                            className="text-red-300 hover:underline cursor-pointer ml-1"
                          >
                            Stop
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
};
