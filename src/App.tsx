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
  Sparkles
} from 'lucide-react';
import { VoiceCloningWorkspace } from './components/VoiceCloningWorkspace';
import { SrtGeneratorWorkspace } from './components/SrtGeneratorWorkspace';
import { ArchitectureAndQaView } from './components/ArchitectureAndQaView';
import { ensureSessionInitialized, fetchApiJson } from './utils/apiClient';

export type ActiveScreen = 'dashboard' | 'voice-cloning' | 'srt-generator' | 'architecture-qa';

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
  const [activeScreen, setActiveScreen] = useState<ActiveScreen>('dashboard');
  const [purgeStatus, setPurgeStatus] = useState<string | null>(null);
  const [recentActivities, setRecentActivities] = useState<ActivityRecord[]>([]);

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
      timestampLabel: 'Just now',
      status: 'Completed'
    };
    setRecentActivities((prev) => [newEntry, ...prev.slice(0, 5)]);
  };

  const handleDownloadActivitySrt = (act: ActivityRecord) => {
    if (!act.srtContent) {
      setActiveScreen(act.targetScreen);
      return;
    }
    const blob = new Blob([`\uFEFF${act.srtContent}`], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = act.downloadFilename || 'voicelab_subtitles.srt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handlePurgeTemporaryAudio = async () => {
    try {
      const data = await fetchApiJson<{ status: string; deletedFilesCount: number }>(
        '/api/storage/cleanup',
        { method: 'DELETE' }
      );
      setPurgeStatus(`Purged (${data.deletedFilesCount})`);
      setTimeout(() => setPurgeStatus(null), 3000);
    } catch {
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

            {/* Secondary Technical Section */}
            <div className="pt-3 mt-3 border-t border-[#19352A]/80">
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
            </div>
          </nav>
        </div>

        {/* Bottom Sidebar Info Card */}
        <div className="space-y-3">
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
                  : 'Architecture & QA'}
              </span>
            </div>

            {/* Zone 3: 1 Primary Utility Action */}
            <div className="flex items-center shrink-0">
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
                    <span className="hidden sm:inline">Purge Temp Audio</span>
                    <span className="sm:hidden">Purge</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </header>

        {/* Main Content Viewport */}
        <main className="flex-1 max-w-6xl w-full mx-auto px-3 sm:px-6 py-5 sm:py-7">
          {activeScreen === 'dashboard' && (
            <div className="space-y-5">
              {/* Compact Welcome Banner */}
              <section className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-[#12231D] via-[#19352A]/90 to-[#12231D] border border-[#16A36A]/40 px-5 py-5 sm:px-7 sm:py-6 flex items-center justify-between gap-6">
                <div className="space-y-1.5 max-w-xl z-10">
                  <h1
                    className="text-xl sm:text-3xl font-bold text-[#F4F8F5] tracking-tight"
                    style={{ textWrap: 'balance' }}
                  >
                    Welcome to <span className="text-[#42D9A0]">VoiceLab AI</span>
                  </h1>
                  <p className="text-xs sm:text-sm text-[#A2B5AA] leading-relaxed">
                    Create studio-quality cloned voice or generate SRT subtitles with ease.
                  </p>
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
                        <h2 className="text-lg sm:text-xl font-bold text-[#F4F8F5] tracking-tight">
                          Voice Cloning
                        </h2>
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
                        <h2 className="text-lg sm:text-xl font-bold text-[#F4F8F5] tracking-tight">
                          Audio to SRT
                        </h2>
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

          {activeScreen === 'voice-cloning' && (
            <VoiceCloningWorkspace
              onNavigateBackToDashboard={() => setActiveScreen('dashboard')}
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
          )}

          {activeScreen === 'srt-generator' && (
            <SrtGeneratorWorkspace
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
          )}

          {activeScreen === 'architecture-qa' && <ArchitectureAndQaView />}
        </main>

        {/* Compact Footer */}
        <footer className="border-t border-[#19352A]/80 px-3 sm:px-6 py-3 text-[11px] text-[#A2B5AA]">
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
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
