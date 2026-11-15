import React, { useEffect, useState } from 'react';
import {
  Mic2,
  Captions,
  ArrowRight,
  Trash2,
  Check,
  Play,
  Clock,
  Download,
  Home,
  Settings2,
  ShieldCheck,
  Sparkles,
  Crown,
  Lock,
  AlertTriangle,
  LogIn,
  LogOut,
  Coins
} from 'lucide-react';
import {
  VoiceCloningWorkspace,
  type VoiceToSrtHandoff
} from './components/VoiceCloningWorkspace';
import { SrtGeneratorWorkspace } from './components/SrtGeneratorWorkspace';
import { ArchitectureAndQaView } from './components/ArchitectureAndQaView';
import { AdminOwnershipPanel } from './components/AdminOwnershipPanel';
import { UpgradeAccessModal } from './components/UpgradeAccessModal';
import { FloatingActivityBar } from './components/FloatingActivityBar';
import {
  FirebaseAuthProvider,
  useFirebaseAuth,
  type FeatureGateKey,
  type PackageFeatureKey
} from './contexts/FirebaseAuthContext';
import { ensureSessionInitialized, fetchApiJson } from './utils/apiClient';
import {
  completeOperation,
  failOperation,
  startOperation,
  updateOperationStage
} from './utils/activityManager';

export type ActiveScreen =
  | 'dashboard'
  | 'voice-cloning'
  | 'srt-generator'
  | 'architecture-qa'
  | 'admin-panel';

export interface ActivityRecord {
  id: string;
  type: 'voice-cloning' | 'srt-generator';
  title: string;
  languageLabel: string;
  timestampLabel: string;
  status: 'Completed';
  downloadUrl?: string;
  downloadFilename?: string;
  srtContent?: string;
  targetScreen: 'voice-cloning' | 'srt-generator';
}

export default function App() {
  return (
    <FirebaseAuthProvider>
      <AppContent />
    </FirebaseAuthProvider>
  );
}

function AppContent() {
  const {
    user,
    authReady,
    role,
    isOwner,
    isSuspended,
    tokenBalance,
    governanceConfig,
    featureConfigs,
    hasAnyPaidOrTokenFeature,
    myPaymentRequests,
    hasFeatureAccess,
    getFeatureTokenCost,
    signInWithGoogle,
    signOutUser
  } = useFirebaseAuth();
  const [activeScreen, setActiveScreen] = useState<ActiveScreen>('dashboard');
  const [voiceToSrtHandoff, setVoiceToSrtHandoff] = useState<VoiceToSrtHandoff | null>(null);
  const [purgeStatus, setPurgeStatus] = useState<string | null>(null);
  const activityStorageKey = `voicelab_recent_activities_v1_${user?.uid || 'guest'}`;
  const [recentActivities, setRecentActivities] = useState<ActivityRecord[]>([]);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(activityStorageKey);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          setRecentActivities(parsed.slice(0, 10));
          return;
        }
      }
      setRecentActivities([]);
    } catch {
      setRecentActivities([]);
    }
  }, [activityStorageKey]);
  const [upgradeModal, setUpgradeModal] = useState<{ open: boolean; feature: PackageFeatureKey }>({
    open: false,
    feature: 'pro_bundle'
  });

  // Block unauthorized direct URL access (?screen=admin / #admin) and enforce owner-only admin view
  useEffect(() => {
    const syncAndGuardRoute = () => {
      const params = new URLSearchParams(window.location.search);
      const screenParam = params.get('screen');
      const hash = window.location.hash.replace(/^#/, '');
      const wantsAdmin =
        screenParam === 'admin' ||
        screenParam === 'admin-panel' ||
        hash === 'admin' ||
        hash === 'admin-panel';

      if (wantsAdmin) {
        if (authReady && isOwner) {
          setActiveScreen('admin-panel');
        } else if (authReady && !isOwner) {
          setActiveScreen('dashboard');
          params.delete('screen');
          const nextSearch = params.toString();
          const cleanUrl = `${window.location.pathname}${nextSearch ? `?${nextSearch}` : ''}`;
          window.history.replaceState(null, '', cleanUrl);
        }
      }
    };

    syncAndGuardRoute();
    window.addEventListener('hashchange', syncAndGuardRoute);
    window.addEventListener('popstate', syncAndGuardRoute);
    return () => {
      window.removeEventListener('hashchange', syncAndGuardRoute);
      window.removeEventListener('popstate', syncAndGuardRoute);
    };
  }, [authReady, isOwner]);

  // Immediately close Admin Panel if user signs out or is not verified Owner
  useEffect(() => {
    if (activeScreen === 'admin-panel' && (!authReady || !isOwner)) {
      if (authReady && !isOwner) {
        setActiveScreen('dashboard');
      }
    }
  }, [activeScreen, authReady, isOwner]);

  // Initialize authenticated session on mount & keep Nginx Lua auth_token_cache (300s TTL) warm
  useEffect(() => {
    ensureSessionInitialized(true).catch(() => {});
    const keepAliveTimer = window.setInterval(() => {
      ensureSessionInitialized(true).catch(() => {});
    }, 210_000);
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        ensureSessionInitialized(false).catch(() => {});
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      window.clearInterval(keepAliveTimer);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  const handleRecordActivity = (item: Omit<ActivityRecord, 'id' | 'timestampLabel' | 'status'>) => {
    const newEntry: ActivityRecord = {
      ...item,
      id: `act_${Date.now()}`,
      timestampLabel: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      status: 'Completed'
    };
    setRecentActivities((prev) => {
      const next = [newEntry, ...prev.slice(0, 9)];
      try {
        window.localStorage.setItem(activityStorageKey, JSON.stringify(next));
      } catch {
        // Ignore storage quota errors
      }
      return next;
    });
  };

  const handleDownloadActivitySrt = (act: ActivityRecord) => {
    if (!act.srtContent) {
      setActiveScreen(act.targetScreen);
      return;
    }
    const filename = act.downloadFilename || 'voicelab_subtitles.srt';
    const op = startOperation({
      feature: 'export-download',
      title: `Export ${filename}`,
      stages: [
        'Encoding UTF-8 BOM SubRip (.srt) payload',
        'Triggering browser file download'
      ],
      initialProgressPct: 50,
      progressSource: 'task-steps',
      canCancel: false
    });
    const blob = new Blob([`\uFEFF${act.srtContent}`], { type: 'text/plain;charset=utf-8' });
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
    completeOperation(op.operationId, `Downloaded ${filename}`);
  };

  const handlePurgeTemporaryAudio = async () => {
    const op = startOperation({
      feature: 'storage-cleanup',
      title: 'Purge Ephemeral Audio',
      stages: [
        'Requesting ephemeral storage cleanup',
        'Removing temporary uploaded & generated media'
      ],
      initialProgressPct: null,
      progressSource: 'indeterminate-inference',
      canCancel: false
    });
    try {
      const data = await fetchApiJson<{ status: string; deletedFilesCount: number }>(
        '/api/storage/cleanup',
        { method: 'DELETE', operationId: op.operationId, signal: op.signal }
      );
      setPurgeStatus(`Purged (${data.deletedFilesCount})`);
      completeOperation(
        op.operationId,
        `Purged ${data.deletedFilesCount} temporary audio file(s)`
      );
      setTimeout(() => setPurgeStatus(null), 3000);
    } catch (err: any) {
      failOperation(op.operationId, err?.message || 'Storage cleanup failed.');
      setPurgeStatus('Cleaned');
      setTimeout(() => setPurgeStatus(null), 2500);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B1512] text-[#F4F8F5] flex flex-col md:flex-row overflow-x-hidden">
      {/* Desktop Left Sidebar Navigation */}
      <aside className="hidden md:flex md:w-60 lg:w-64 shrink-0 bg-[#0B1512] border-r border-[#19352A] flex-col justify-between p-4 select-none">
        <div className="space-y-6">
          {/* Brand Identity */}
          <button
            type="button"
            onClick={() => setActiveScreen('dashboard')}
            className="w-full flex items-center gap-3 px-2 py-1.5 text-left group cursor-pointer"
          >
            <div className="w-9 h-9 rounded-xl bg-[#12231D] border border-[#16A36A]/50 flex items-center justify-center shrink-0 group-hover:border-[#42D9A0] transition-colors">
              <Mic2 className="w-5 h-5 text-[#42D9A0]" />
            </div>
            <div className="min-w-0">
              <div className="text-base font-bold tracking-tight text-[#F4F8F5] truncate">
                VoiceLab AI
              </div>
              <div className="text-[11px] text-[#A2B5AA] truncate">
                Audio Workstation
              </div>
            </div>
          </button>

          {/* Primary Navigation (3 Core Sections) */}
          <nav className="space-y-1">
            <button
              type="button"
              onClick={() => setActiveScreen('dashboard')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-colors cursor-pointer whitespace-nowrap ${
                activeScreen === 'dashboard'
                  ? 'bg-[#19352A] text-[#F4F8F5] border border-[#16A36A]/50 shadow-sm'
                  : 'text-[#A2B5AA] hover:bg-[#12231D] hover:text-[#F4F8F5]'
              }`}
            >
              <Home className="w-4 h-4 text-[#42D9A0] shrink-0" />
              <span>Dashboard</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveScreen('voice-cloning')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-colors cursor-pointer whitespace-nowrap ${
                activeScreen === 'voice-cloning'
                  ? 'bg-[#19352A] text-[#F4F8F5] border border-[#16A36A]/50 shadow-sm'
                  : 'text-[#A2B5AA] hover:bg-[#12231D] hover:text-[#F4F8F5]'
              }`}
            >
              <Mic2 className="w-4 h-4 text-[#42D9A0] shrink-0" />
              <span>Voice Cloning</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveScreen('srt-generator')}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-colors cursor-pointer whitespace-nowrap ${
                activeScreen === 'srt-generator'
                  ? 'bg-[#19352A] text-[#F4F8F5] border border-[#16A36A]/50 shadow-sm'
                  : 'text-[#A2B5AA] hover:bg-[#12231D] hover:text-[#F4F8F5]'
              }`}
            >
              <Captions className="w-4 h-4 text-[#42D9A0] shrink-0" />
              <span>Audio to SRT</span>
            </button>

            {/* Secondary Technical & Governance Section */}
            <div className="pt-3 mt-3 border-t border-[#19352A]/80 space-y-1">
              <button
                type="button"
                onClick={() => setActiveScreen('architecture-qa')}
                className={`w-full flex items-center gap-3 px-3.5 py-2 rounded-xl text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
                  activeScreen === 'architecture-qa'
                    ? 'bg-[#19352A] text-[#F4F8F5] border border-[#16A36A]/50'
                    : 'text-[#A2B5AA] hover:bg-[#12231D] hover:text-[#F4F8F5]'
                }`}
              >
                <Settings2 className="w-4 h-4 text-[#A2B5AA] shrink-0" />
                <span>Architecture &amp; QA</span>
              </button>

              {isOwner && (
                <button
                  type="button"
                  onClick={() => setActiveScreen('admin-panel')}
                  className={`w-full flex items-center justify-between gap-2 px-3.5 py-2 rounded-xl text-xs font-medium transition-colors cursor-pointer whitespace-nowrap ${
                    activeScreen === 'admin-panel'
                      ? 'bg-[#19352A] text-[#F4F8F5] border border-[#16A36A]/50'
                      : 'text-[#A2B5AA] hover:bg-[#12231D] hover:text-[#F4F8F5]'
                  }`}
                >
                  <span className="flex items-center gap-3 min-w-0">
                    <Crown className="w-4 h-4 text-[#42D9A0] shrink-0" />
                    <span className="truncate">Admin Panel</span>
                  </span>
                  <span className="px-1.5 py-0.5 rounded bg-[#16A36A]/20 text-[10px] font-semibold uppercase text-[#42D9A0]">
                    Owner
                  </span>
                </button>
              )}
            </div>
          </nav>
        </div>

        {/* Bottom Sidebar Info Card */}
        <div className="space-y-3">
          {user ? (
            <div className="rounded-xl bg-[#12231D] border border-[#19352A] p-3 space-y-2">
              <div className="flex items-center justify-between gap-2 text-xs font-semibold text-[#F4F8F5]">
                <span className="truncate">{user.displayName || user.email}</span>
                <span className="text-[10px] uppercase text-[#42D9A0] shrink-0">
                  {role === 'owner' ? 'Owner' : 'Member'}
                </span>
              </div>
              <div className="text-[11px] text-[#A2B5AA] truncate font-mono">{user.email}</div>
              {(tokenBalance > 0 || hasAnyPaidOrTokenFeature) && (
                <div className="flex items-center justify-between text-[11px] pt-1 border-t border-[#19352A]">
                  <span className="text-[#A2B5AA] flex items-center gap-1">
                    <Coins className="w-3.5 h-3.5 text-[#42D9A0]" />
                    <span>Tokens</span>
                  </span>
                  <span className="font-mono font-semibold text-[#F4F8F5]">{tokenBalance}</span>
                </div>
              )}
              <div className="flex items-center justify-between gap-2 pt-1 border-t border-[#19352A]">
                {hasAnyPaidOrTokenFeature || myPaymentRequests.length > 0 ? (
                  <button
                    type="button"
                    onClick={() => setUpgradeModal({ open: true, feature: 'pro_bundle' })}
                    className="text-[11px] font-medium text-[#42D9A0] hover:underline cursor-pointer"
                  >
                    Upgrade / Access
                  </button>
                ) : (
                  <span className="text-[11px] text-[#A2B5AA]">Standard Access</span>
                )}
                <button
                  type="button"
                  onClick={() => void signOutUser()}
                  className="text-[11px] text-[#A2B5AA] hover:text-red-300 inline-flex items-center gap-1 cursor-pointer"
                >
                  <LogOut className="w-3 h-3" /> Sign Out
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => void signInWithGoogle()}
              className="w-full rounded-xl bg-white hover:bg-gray-100 text-gray-900 p-2.5 flex items-center justify-center gap-2 text-xs font-semibold transition-colors cursor-pointer shadow-sm"
            >
              <LogIn className="w-3.5 h-3.5 text-[#16A36A]" />
              <span>Continue with Google</span>
            </button>
          )}

          <div className="rounded-xl bg-[#12231D] border border-[#19352A] p-3.5 space-y-1">
            <div className="flex items-center gap-2 text-xs font-semibold text-[#F4F8F5]">
              <ShieldCheck className="w-4 h-4 text-[#42D9A0] shrink-0" />
              <span>Zero-Cost Open Source</span>
            </div>
            <p className="text-[11px] text-[#A2B5AA] leading-snug">
              Local &amp; Cloud Ready · Auto-purges temp audio every 30m
            </p>
          </div>
          <div className="text-[11px] text-[#A2B5AA]/70 text-center font-mono">
            VoiceLab AI v3.3
          </div>
        </div>
      </aside>

      {/* Main Viewport Column */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Compact Top Header Bar (3-Zone Contract) */}
        <header className="sticky top-0 z-30 bg-[#0B1512]/95 backdrop-blur-md border-b border-[#19352A] px-3 sm:px-6 py-2.5">
          <div className="max-w-6xl mx-auto flex items-center justify-between gap-3 sm:gap-6">
            {/* Zone 1: Brand / Active Section Title */}
            <button
              type="button"
              onClick={() => setActiveScreen('dashboard')}
              className="flex items-center gap-2 text-sm sm:text-base font-bold tracking-tight text-[#F4F8F5] hover:text-[#42D9A0] transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Mic2 className="w-4 h-4 text-[#42D9A0] md:hidden shrink-0" />
              <span>VoiceLab AI</span>
            </button>

            {/* Zone 2: Compact Single-Line Navigation Links (Visible on Mobile & Tablet) */}
            <nav className="flex md:hidden items-center gap-1 overflow-x-auto no-scrollbar py-0.5">
              <button
                type="button"
                onClick={() => setActiveScreen('dashboard')}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium whitespace-nowrap shrink-0 transition-colors cursor-pointer ${
                  activeScreen === 'dashboard'
                    ? 'bg-[#19352A] text-[#42D9A0]'
                    : 'text-[#A2B5AA] hover:text-[#F4F8F5]'
                }`}
              >
                Dashboard
              </button>
              <button
                type="button"
                onClick={() => setActiveScreen('voice-cloning')}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium whitespace-nowrap shrink-0 transition-colors cursor-pointer ${
                  activeScreen === 'voice-cloning'
                    ? 'bg-[#19352A] text-[#42D9A0]'
                    : 'text-[#A2B5AA] hover:text-[#F4F8F5]'
                }`}
              >
                Cloning
              </button>
              <button
                type="button"
                onClick={() => setActiveScreen('srt-generator')}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium whitespace-nowrap shrink-0 transition-colors cursor-pointer ${
                  activeScreen === 'srt-generator'
                    ? 'bg-[#19352A] text-[#42D9A0]'
                    : 'text-[#A2B5AA] hover:text-[#F4F8F5]'
                }`}
              >
                SRT
              </button>
              <button
                type="button"
                onClick={() => setActiveScreen('architecture-qa')}
                className={`px-2 py-1 rounded-lg text-xs font-medium whitespace-nowrap shrink-0 transition-colors cursor-pointer ${
                  activeScreen === 'architecture-qa'
                    ? 'bg-[#19352A] text-[#42D9A0]'
                    : 'text-[#A2B5AA] hover:text-[#F4F8F5]'
                }`}
              >
                QA
              </button>
              {isOwner && (
                <button
                  type="button"
                  onClick={() => setActiveScreen('admin-panel')}
                  className={`px-2 py-1 rounded-lg text-xs font-medium whitespace-nowrap shrink-0 transition-colors cursor-pointer ${
                    activeScreen === 'admin-panel'
                      ? 'bg-[#19352A] text-[#42D9A0]'
                      : 'text-[#A2B5AA] hover:text-[#F4F8F5]'
                  }`}
                >
                  Admin
                </button>
              )}
            </nav>

            {/* Desktop Breadcrumb Context */}
            <div className="hidden md:flex items-center gap-2 text-xs text-[#A2B5AA] whitespace-nowrap">
              <span>Workstation</span>
              <span>/</span>
              <span className="text-[#F4F8F5] font-medium">
                {activeScreen === 'dashboard'
                  ? 'Dashboard'
                  : activeScreen === 'voice-cloning'
                  ? 'Voice Cloning Studio'
                  : activeScreen === 'srt-generator'
                  ? 'Universal Audio to SRT'
                  : activeScreen === 'admin-panel' && isOwner
                  ? 'Admin Panel'
                  : 'Architecture & QA'}
              </span>
            </div>

            {/* Zone 3: Account, Token Balance & Purge Utility */}
            <div className="flex items-center gap-2 shrink-0">
              {user && (tokenBalance > 0 || hasAnyPaidOrTokenFeature) && (
                <div className="px-2.5 py-1.5 rounded-lg bg-[#12231D] border border-[#19352A] text-xs font-mono text-[#F4F8F5] flex items-center gap-1.5 whitespace-nowrap">
                  <Coins className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
                  <span>{tokenBalance}</span>
                </div>
              )}

              {user && (hasAnyPaidOrTokenFeature || myPaymentRequests.length > 0) && (
                <button
                  type="button"
                  onClick={() => setUpgradeModal({ open: true, feature: 'pro_bundle' })}
                  className="px-2.5 py-1.5 text-xs font-semibold text-[#42D9A0] bg-[#19352A] hover:bg-[#16A36A]/25 border border-[#16A36A]/40 rounded-lg transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5 cursor-pointer"
                >
                  <Sparkles className="w-3.5 h-3.5 shrink-0" />
                  <span className="hidden sm:inline">Upgrade</span>
                </button>
              )}

              {!user && (
                <button
                  type="button"
                  onClick={() => void signInWithGoogle()}
                  className="px-2.5 sm:px-3 py-1.5 text-xs font-semibold text-gray-900 bg-white hover:bg-gray-100 rounded-lg transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5 cursor-pointer"
                >
                  <LogIn className="w-3.5 h-3.5 text-[#16A36A] shrink-0" />
                  <span className="hidden sm:inline">Continue with Google</span>
                  <span className="sm:hidden">Google</span>
                </button>
              )}

              <button
                type="button"
                onClick={handlePurgeTemporaryAudio}
                title="Purge temporary uploaded and generated audio files"
                className="px-2.5 sm:px-3 py-1.5 text-xs font-medium text-[#F4F8F5] bg-[#12231D] hover:bg-[#19352A] border border-[#19352A] rounded-lg transition-colors whitespace-nowrap shrink-0 flex items-center gap-1.5 cursor-pointer"
              >
                {purgeStatus ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
                    <span>{purgeStatus}</span>
                  </>
                ) : (
                  <>
                    <Trash2 className="w-3.5 h-3.5 text-[#42D9A0] shrink-0" />
                    <span className="hidden lg:inline">Purge Temp Audio</span>
                    <span className="lg:hidden">Purge</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </header>

        {/* Main Content Viewport */}
        <main className="flex-1 max-w-6xl w-full mx-auto px-3 sm:px-6 py-5 sm:py-7 space-y-5">
          {governanceConfig.maintenanceMode && !isOwner && (
            <div className="rounded-2xl bg-amber-950/50 border border-amber-600/50 p-4 flex items-start gap-2.5 text-xs text-amber-200">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <div className="font-bold text-amber-100">Maintenance Mode Active</div>
                <div className="mt-0.5">{governanceConfig.maintenanceNotice}</div>
              </div>
            </div>
          )}

          {isSuspended && (
            <div className="rounded-2xl bg-red-950/50 border border-red-700/50 p-4 flex items-start gap-2.5 text-xs text-red-200">
              <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <div>
                <div className="font-bold text-red-100">Account Suspended</div>
                <div className="mt-0.5">
                  Your account has been suspended by an administrator. Contact the workspace administrator to restore access.
                </div>
              </div>
            </div>
          )}

          {governanceConfig.requireAuthForCompute &&
            !user &&
            (activeScreen === 'voice-cloning' || activeScreen === 'srt-generator') && (
              <div className="rounded-2xl bg-[#12231D] border border-[#16A36A]/50 p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                <div className="flex items-start gap-2.5">
                  <Lock className="w-4 h-4 text-[#42D9A0] shrink-0 mt-0.5" />
                  <div>
                    <div className="font-semibold text-[#F4F8F5]">
                      Google Sign-In Required
                    </div>
                    <div className="text-[#A2B5AA] mt-0.5">
                      Signing in with Google is required before running voice synthesis or SRT transcription jobs.
                    </div>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => void signInWithGoogle()}
                  className="px-3.5 py-2 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-[#F4F8F5] font-semibold whitespace-nowrap cursor-pointer self-start sm:self-center"
                >
                  Sign In with Google
                </button>
              </div>
            )}

          {activeScreen === 'dashboard' && (
            <div className="space-y-5">
              {/* Compact Welcome Banner */}
              <section className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-[#12231D] via-[#19352A]/90 to-[#12231D] border border-[#16A36A]/40 px-5 py-5 sm:px-7 sm:py-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="space-y-2 max-w-xl z-10">
                  <h1
                    className="text-xl sm:text-3xl font-bold text-[#F4F8F5] tracking-tight"
                    style={{ textWrap: 'balance' }}
                  >
                    Welcome to <span className="text-[#42D9A0]">VoiceLab AI</span>
                  </h1>
                  <p className="text-xs sm:text-sm text-[#A2B5AA] leading-relaxed">
                    Create studio-quality cloned voice or generate SRT subtitles with ease.
                  </p>
                  <div className="flex flex-wrap items-center gap-2.5 pt-1">
                    {!user ? (
                      <button
                        type="button"
                        onClick={() => void signInWithGoogle()}
                        className="px-4 py-2 rounded-xl bg-white hover:bg-gray-100 text-gray-900 font-semibold text-xs flex items-center gap-2 cursor-pointer shadow-sm"
                      >
                        <LogIn className="w-3.5 h-3.5 text-[#16A36A]" />
                        <span>Continue with Google</span>
                      </button>
                    ) : (
                      <>
                        <span className="px-3 py-1.5 rounded-xl bg-[#0B1512]/80 border border-[#16A36A]/50 text-xs text-[#42D9A0] font-medium">
                          {user.email}
                        </span>
                        {(tokenBalance > 0 || hasAnyPaidOrTokenFeature) && (
                          <span className="px-3 py-1.5 rounded-xl bg-[#0B1512]/80 border border-[#19352A] text-xs text-[#F4F8F5] font-mono inline-flex items-center gap-1.5">
                            <Coins className="w-3.5 h-3.5 text-[#42D9A0]" />
                            <span>{tokenBalance} Tokens</span>
                          </span>
                        )}
                        {(hasAnyPaidOrTokenFeature || myPaymentRequests.length > 0) && (
                          <button
                            type="button"
                            onClick={() => setUpgradeModal({ open: true, feature: 'pro_bundle' })}
                            className="px-3.5 py-1.5 rounded-xl bg-[#19352A] hover:bg-[#16A36A]/30 border border-[#16A36A]/50 text-[#42D9A0] font-semibold text-xs flex items-center gap-1.5 cursor-pointer"
                          >
                            <Sparkles className="w-3.5 h-3.5" />
                            <span>Upgrade / Request Access</span>
                          </button>
                        )}
                      </>
                    )}
                  </div>
                </div>

                {/* Decorative Acoustic Waveform Illustration (Desktop/Tablet) */}
                <div
                  aria-hidden="true"
                  className="hidden sm:flex items-center gap-1.5 opacity-80 shrink-0 pr-2"
                >
                  {[14, 28, 18, 38, 48, 26, 42, 20, 34, 16, 24, 12].map((h, i) => (
                    <div
                      key={i}
                      style={{ height: `${h}px` }}
                      className="w-1.5 rounded-full bg-gradient-to-t from-[#16A36A]/40 to-[#42D9A0]"
                    />
                  ))}
                  <div className="ml-3 w-12 h-12 rounded-2xl bg-[#0B1512]/80 border border-[#42D9A0]/50 flex items-center justify-center shadow-inner">
                    <Mic2 className="w-6 h-6 text-[#42D9A0]" />
                  </div>
                </div>
              </section>

              {/* Strictly Two Primary Feature Cards Side-by-Side on Desktop, Stacked on Mobile */}
              <section className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-5 items-stretch">
                {/* Primary Card 1: Voice Cloning */}
                <div className="bg-[#12231D] border border-[#19352A] hover:border-[#16A36A] rounded-2xl p-5 sm:p-6 flex flex-col justify-between gap-5 transition-colors">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start gap-3.5 min-w-0">
                      <div className="w-12 h-12 rounded-2xl bg-[#19352A] border border-[#16A36A]/50 flex items-center justify-center shrink-0">
                        <Mic2 className="w-6 h-6 text-[#42D9A0]" />
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <h2 className="text-lg sm:text-xl font-bold text-[#F4F8F5] tracking-tight">
                            Voice Cloning
                          </h2>
                          <FeatureStatusTag
                            feature="voice_cloning"
                            enabled={featureConfigs.voice_cloning.enabled}
                            isPaid={featureConfigs.voice_cloning.isPaid}
                            hasAccess={hasFeatureAccess('voice_cloning')}
                            tokenCost={getFeatureTokenCost('voice_cloning')}
                          />
                        </div>
                        <p className="text-xs sm:text-sm text-[#A2B5AA] mt-1 line-clamp-2 leading-relaxed">
                          Clone any voice in Bengali, Hindi or English (Same Language Only).
                        </p>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => setActiveScreen('voice-cloning')}
                      aria-label="Open Voice Cloning"
                      className="w-8 h-8 rounded-full bg-[#19352A]/70 hover:bg-[#16A36A] text-[#A2B5AA] hover:text-[#F4F8F5] flex items-center justify-center shrink-0 transition-colors cursor-pointer"
                    >
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={() => setActiveScreen('voice-cloning')}
                    className="w-full py-2.5 px-5 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-[#F4F8F5] font-semibold text-sm transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-sm whitespace-nowrap"
                  >
                    <Play className="w-4 h-4 fill-current" />
                    <span>Start Cloning</span>
                  </button>
                </div>

                {/* Primary Card 2: Audio to SRT */}
                <div className="bg-[#12231D] border border-[#19352A] hover:border-[#16A36A] rounded-2xl p-5 sm:p-6 flex flex-col justify-between gap-5 transition-colors">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start gap-3.5 min-w-0">
                      <div className="w-12 h-12 rounded-2xl bg-[#19352A] border border-[#16A36A]/50 flex items-center justify-center shrink-0">
                        <Captions className="w-6 h-6 text-[#42D9A0]" />
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <h2 className="text-lg sm:text-xl font-bold text-[#F4F8F5] tracking-tight">
                            Audio to SRT
                          </h2>
                          <FeatureStatusTag
                            feature="srt_generator"
                            enabled={featureConfigs.srt_generator.enabled}
                            isPaid={featureConfigs.srt_generator.isPaid}
                            hasAccess={hasFeatureAccess('srt_generator')}
                            tokenCost={getFeatureTokenCost('srt_generator')}
                          />
                        </div>
                        <p className="text-xs sm:text-sm text-[#A2B5AA] mt-1 line-clamp-2 leading-relaxed">
                          Convert audio to accurate subtitles in multiple languages.
                        </p>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => setActiveScreen('srt-generator')}
                      aria-label="Open Audio to SRT"
                      className="w-8 h-8 rounded-full bg-[#19352A]/70 hover:bg-[#16A36A] text-[#A2B5AA] hover:text-[#F4F8F5] flex items-center justify-center shrink-0 transition-colors cursor-pointer"
                    >
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>

                  <button
                    type="button"
                    onClick={() => setActiveScreen('srt-generator')}
                    className="w-full py-2.5 px-5 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-[#F4F8F5] font-semibold text-sm transition-colors flex items-center justify-center gap-2 cursor-pointer shadow-sm whitespace-nowrap"
                  >
                    <Play className="w-4 h-4 fill-current" />
                    <span>Start Transcription</span>
                  </button>
                </div>
              </section>

              {/* Compact Recent Activity Section */}
              {recentActivities.length > 0 && (
                <section className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 sm:p-5 space-y-3">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4 text-[#42D9A0] shrink-0" />
                      <h3 className="text-sm font-semibold text-[#F4F8F5]">
                        Recent Activity
                      </h3>
                    </div>
                    <button
                      type="button"
                      onClick={() => setActiveScreen('architecture-qa')}
                      className="text-xs font-medium text-[#42D9A0] hover:underline flex items-center gap-1 cursor-pointer whitespace-nowrap"
                    >
                      <span>QA &amp; Benchmarks</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div className="divide-y divide-[#19352A]">
                    {recentActivities.map((act) => (
                      <div
                        key={act.id}
                        className="py-2.5 first:pt-1 last:pb-0 flex items-center justify-between gap-3 text-xs"
                      >
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="w-7 h-7 rounded-lg bg-[#19352A] flex items-center justify-center shrink-0">
                            {act.type === 'voice-cloning' ? (
                              <Mic2 className="w-3.5 h-3.5 text-[#42D9A0]" />
                            ) : (
                              <Captions className="w-3.5 h-3.5 text-[#42D9A0]" />
                            )}
                          </div>
                          <div className="min-w-0 truncate">
                            <span className="font-medium text-[#F4F8F5]">{act.title}</span>
                            <span className="mx-1.5 text-[#A2B5AA]">·</span>
                            <span className="text-[#A2B5AA]">{act.languageLabel}</span>
                          </div>
                        </div>

                        <div className="flex items-center gap-3 sm:gap-4 shrink-0">
                          <span className="hidden sm:inline text-[#A2B5AA] font-mono tabular-nums">
                            {act.timestampLabel}
                          </span>
                          <span className="text-[#42D9A0] font-medium">
                            {act.status}
                          </span>
                          {act.downloadUrl ? (
                            <a
                              href={act.downloadUrl}
                              download={act.downloadFilename}
                              title="Download generated audio"
                              className="p-1.5 rounded-lg bg-[#0B1512] hover:bg-[#19352A] text-[#A2B5AA] hover:text-[#F4F8F5] transition-colors"
                            >
                              <Download className="w-3.5 h-3.5" />
                            </a>
                          ) : act.srtContent ? (
                            <button
                              type="button"
                              onClick={() => handleDownloadActivitySrt(act)}
                              title="Download generated SRT"
                              className="p-1.5 rounded-lg bg-[#0B1512] hover:bg-[#19352A] text-[#A2B5AA] hover:text-[#F4F8F5] transition-colors cursor-pointer"
                            >
                              <Download className="w-3.5 h-3.5" />
                            </button>
                          ) : (
                            <button
                              type="button"
                              onClick={() => setActiveScreen(act.targetScreen)}
                              title="Open studio workspace"
                              className="p-1.5 rounded-lg bg-[#0B1512] hover:bg-[#19352A] text-[#A2B5AA] hover:text-[#F4F8F5] transition-colors cursor-pointer"
                            >
                              <ArrowRight className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}

          {activeScreen === 'voice-cloning' &&
            (!featureConfigs.voice_cloning.enabled ? (
              <FeatureDisabledGate title="Voice Cloning Studio" />
            ) : (featureConfigs.voice_cloning.isPaid && !hasFeatureAccess('voice_cloning')) ||
              (getFeatureTokenCost('voice_cloning') > 0 &&
                tokenBalance < getFeatureTokenCost('voice_cloning') &&
                !isOwner) ? (
              <FeatureAccessGate
                title="Voice Cloning Studio"
                description="Access to Voice Cloning Studio requires an active plan or sufficient token balance."
                tokenCost={getFeatureTokenCost('voice_cloning')}
                tokenBalance={tokenBalance}
                isSignedIn={Boolean(user)}
                onSignIn={() => void signInWithGoogle()}
                onRequestUpgrade={() =>
                  setUpgradeModal({ open: true, feature: 'voice_cloning' })
                }
              />
            ) : (
              <div className="space-y-4">
                {featureConfigs.long_script.isPaid && !hasFeatureAccess('long_script') && (
                  <div className="rounded-xl bg-[#12231D] border border-[#16A36A]/40 px-4 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                    <div className="flex items-center gap-2 text-[#F4F8F5]">
                      <Lock className="w-4 h-4 text-[#42D9A0] shrink-0" />
                      <span>
                        <strong>Long-Script (&gt;1,000 Chars)</strong> requires premium feature
                        access.
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() => setUpgradeModal({ open: true, feature: 'long_script' })}
                      className="px-3.5 py-1.5 rounded-lg bg-[#16A36A] hover:bg-[#16A36A]/90 text-[#F4F8F5] font-semibold whitespace-nowrap cursor-pointer self-start sm:self-center"
                    >
                      Upgrade / Request Access
                    </button>
                  </div>
                )}

                <VoiceCloningWorkspace
                  onNavigateBackToDashboard={() => setActiveScreen('dashboard')}
                  onSendToSrtEditor={(handoff) => {
                    setVoiceToSrtHandoff(handoff);
                    setActiveScreen('srt-generator');
                  }}
                  onActivityComplete={(info) =>
                    handleRecordActivity({
                      type: 'voice-cloning',
                      title: 'Voice Cloning',
                      languageLabel: info.languageLabel,
                      downloadUrl: info.downloadUrl,
                      downloadFilename: info.downloadFilename,
                      targetScreen: 'voice-cloning'
                    })
                  }
                />
              </div>
            ))}

          {activeScreen === 'srt-generator' &&
            (!featureConfigs.srt_generator.enabled ? (
              <FeatureDisabledGate title="Audio to SRT Generator" />
            ) : (featureConfigs.srt_generator.isPaid && !hasFeatureAccess('srt_generator')) ||
              (getFeatureTokenCost('srt_generator') > 0 &&
                tokenBalance < getFeatureTokenCost('srt_generator') &&
                !isOwner) ? (
              <FeatureAccessGate
                title="Audio to SRT Generator"
                description="Access to Audio to SRT Generator requires an active plan or sufficient token balance."
                tokenCost={getFeatureTokenCost('srt_generator')}
                tokenBalance={tokenBalance}
                isSignedIn={Boolean(user)}
                onSignIn={() => void signInWithGoogle()}
                onRequestUpgrade={() =>
                  setUpgradeModal({ open: true, feature: 'srt_generator' })
                }
              />
            ) : (
              <SrtGeneratorWorkspace
                initialHandoff={voiceToSrtHandoff}
                onConsumeHandoff={() => setVoiceToSrtHandoff(null)}
                onActivityComplete={(info) =>
                  handleRecordActivity({
                    type: 'srt-generator',
                    title: 'SRT Generation',
                    languageLabel: info.languageLabel,
                    srtContent: info.srtContent,
                    downloadFilename: info.downloadFilename,
                    targetScreen: 'srt-generator'
                  })
                }
              />
            ))}

          {activeScreen === 'architecture-qa' && <ArchitectureAndQaView />}

          {activeScreen === 'admin-panel' && isOwner && <AdminOwnershipPanel />}
        </main>

        {/* Compact Footer */}
        <footer className="border-t border-[#19352A]/80 px-3 sm:px-6 py-3 pb-16 sm:pb-4 text-[11px] text-[#A2B5AA]">
          <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
            <div>VoiceLab AI · Same-Language Voice Cloning &amp; Multilingual SRT Workstation</div>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => setActiveScreen('voice-cloning')}
                className="hover:text-[#F4F8F5] transition-colors cursor-pointer"
              >
                Voice Cloning
              </button>
              <span>·</span>
              <button
                type="button"
                onClick={() => setActiveScreen('srt-generator')}
                className="hover:text-[#F4F8F5] transition-colors cursor-pointer"
              >
                Audio to SRT
              </button>
              <span>·</span>
              <button
                type="button"
                onClick={() => setActiveScreen('architecture-qa')}
                className="hover:text-[#F4F8F5] transition-colors cursor-pointer"
              >
                Architecture &amp; QA
              </button>
              {isOwner && (
                <>
                  <span>·</span>
                  <button
                    type="button"
                    onClick={() => setActiveScreen('admin-panel')}
                    className="hover:text-[#42D9A0] text-[#42D9A0] transition-colors cursor-pointer"
                  >
                    Admin Panel
                  </button>
                </>
              )}
            </div>
          </div>
        </footer>
      </div>

      {/* Global Floating Bottom Activity Bar */}
      <FloatingActivityBar />

      {/* Minimal Upgrade / Request Access Modal (Instructions loaded dynamically per selection) */}
      <UpgradeAccessModal
        isOpen={upgradeModal.open}
        initialFeature={upgradeModal.feature}
        onClose={() => setUpgradeModal((prev) => ({ ...prev, open: false }))}
      />
    </div>
  );
}

const FeatureStatusTag: React.FC<{
  feature: FeatureGateKey;
  enabled: boolean;
  isPaid: boolean;
  hasAccess: boolean;
  tokenCost: number;
}> = ({ enabled, isPaid, hasAccess, tokenCost }) => {
  if (!enabled) {
    return (
      <span className="px-2 py-0.5 rounded-md bg-red-950/60 border border-red-800/50 text-[10px] font-semibold text-red-300">
        Disabled
      </span>
    );
  }
  if (isPaid && !hasAccess) {
    return (
      <span className="px-2 py-0.5 rounded-md bg-amber-950/60 border border-amber-700/50 text-[10px] font-semibold text-amber-300">
        Access Required
      </span>
    );
  }
  if (tokenCost > 0) {
    return (
      <span className="px-2 py-0.5 rounded-md bg-[#0B1512] border border-[#19352A] text-[10px] font-mono text-[#42D9A0]">
        {tokenCost} Tokens / Run
      </span>
    );
  }
  return null;
};

const FeatureDisabledGate: React.FC<{ title: string }> = ({ title }) => (
  <div className="max-w-xl mx-auto bg-[#12231D] border border-red-800/40 rounded-2xl p-6 sm:p-8 space-y-3 text-center">
    <div className="w-12 h-12 mx-auto rounded-2xl bg-red-950/50 border border-red-800/50 flex items-center justify-center">
      <Lock className="w-6 h-6 text-red-300" />
    </div>
    <h2 className="text-lg font-bold text-[#F4F8F5]">{title} Temporarily Disabled</h2>
    <p className="text-xs sm:text-sm text-[#A2B5AA]">
      This feature has been temporarily disabled by the workspace administrator.
    </p>
  </div>
);

const FeatureAccessGate: React.FC<{
  title: string;
  description: string;
  tokenCost: number;
  tokenBalance: number;
  isSignedIn: boolean;
  onSignIn: () => void;
  onRequestUpgrade: () => void;
}> = ({
  title,
  description,
  tokenCost,
  tokenBalance,
  isSignedIn,
  onSignIn,
  onRequestUpgrade
}) => (
  <div className="max-w-xl mx-auto bg-[#12231D] border border-[#16A36A]/40 rounded-2xl p-6 sm:p-8 space-y-5">
    <div className="flex items-start gap-4">
      <div className="w-12 h-12 rounded-2xl bg-[#19352A] border border-[#16A36A]/50 flex items-center justify-center shrink-0">
        <Lock className="w-6 h-6 text-[#42D9A0]" />
      </div>
      <div className="space-y-1">
        <h2 className="text-lg font-bold text-[#F4F8F5]">{title} — Access Required</h2>
        <p className="text-xs sm:text-sm text-[#A2B5AA] leading-relaxed">{description}</p>
        {tokenCost > 0 && (
          <div className="text-xs font-mono text-[#42D9A0] pt-1">
            Required: {tokenCost} Tokens · Your Balance: {tokenBalance} Tokens
          </div>
        )}
      </div>
    </div>
    <div className="flex flex-wrap gap-3">
      {!isSignedIn && (
        <button
          type="button"
          onClick={onSignIn}
          className="px-4 py-2.5 rounded-xl bg-white hover:bg-gray-100 text-gray-900 font-semibold text-xs flex items-center gap-2 cursor-pointer"
        >
          <LogIn className="w-4 h-4 text-[#16A36A]" />
          <span>Continue with Google</span>
        </button>
      )}
      <button
        type="button"
        onClick={onRequestUpgrade}
        className="px-5 py-2.5 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-[#F4F8F5] font-semibold text-xs flex items-center gap-2 cursor-pointer"
      >
        <Sparkles className="w-4 h-4" />
        <span>Upgrade / Request Access</span>
      </button>
    </div>
  </div>
);
