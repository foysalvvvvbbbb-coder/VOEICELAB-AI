import React, { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Check,
  CheckCircle2,
  Clock,
  Coins,
  CreditCard,
  Crown,
  Database,
  KeyRound,
  Layers,
  Lock,
  LogOut,
  Plus,
  RefreshCw,
  ScrollText,
  Settings,
  Shield,
  ShieldCheck,
  Sliders,
  Trash2,
  Unlock,
  UserCheck,
  Users,
  UserX,
  XCircle
} from 'lucide-react';
import {
  useFirebaseAuth,
  type AdminOverviewPayload,
  type FeatureGateKey,
  type FeaturePackageData,
  type GovernanceConfigData,
  type PaymentMethodId,
  type PaymentRequestData,
  type PaymentSettingsData,
  type TokenPackageData
} from '../contexts/FirebaseAuthContext';

type AdminSectionId =
  | 'overview'
  | 'users'
  | 'features'
  | 'plans'
  | 'tokens'
  | 'payment-settings'
  | 'payment-verification'
  | 'entitlements'
  | 'security-logs';

function formatIsoTime(iso?: string): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export const AdminOwnershipPanel: React.FC = () => {
  const { user, authReady, isOwner, signOutUser } = useFirebaseAuth();

  if (!authReady) {
    return (
      <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-8 flex items-center justify-center gap-3 text-sm text-[#A2B5AA]">
        <RefreshCw className="w-4 h-4 text-[#42D9A0] animate-spin" />
        <span>Verifying session...</span>
      </div>
    );
  }

  // Strict fail-closed gate: never expose Admin UI, owner details, or queries to non-owners
  if (!user || !isOwner) {
    return (
      <div className="max-w-lg mx-auto bg-[#12231D] border border-[#19352A] rounded-2xl p-6 sm:p-8 text-center space-y-3">
        <div className="w-11 h-11 rounded-2xl bg-[#0B1512] border border-[#19352A] flex items-center justify-center mx-auto">
          <Lock className="w-5 h-5 text-[#A2B5AA]" />
        </div>
        <h1 className="text-base font-bold text-[#F4F8F5]">Restricted Workspace Area</h1>
        <p className="text-xs text-[#A2B5AA]">
          This section is unavailable for the current session.
        </p>
      </div>
    );
  }

  return (
    <OwnerAdminConsole
      userEmail={user.email || ''}
      userUid={user.uid}
      userPhoto={user.photoURL || ''}
      onSignOut={signOutUser}
    />
  );
};

interface OwnerAdminConsoleProps {
  userEmail: string;
  userUid: string;
  userPhoto: string;
  onSignOut: () => Promise<void>;
}

const OwnerAdminConsole: React.FC<OwnerAdminConsoleProps> = ({
  userEmail,
  userUid,
  userPhoto,
  onSignOut
}) => {
  const {
    fetchAdminOverview,
    saveAdminGovernance,
    saveAdminFeatures,
    saveAdminPlans,
    saveAdminPaymentSettings,
    reviewAdminPaymentRequest,
    adjustAdminUserTokens,
    grantAdminEntitlement,
    updateAdminUserStatus,
    saveAdminOwnership,
    purgeAdminStorage
  } = useFirebaseAuth();

  const [section, setSection] = useState<AdminSectionId>('overview');
  const [overview, setOverview] = useState<AdminOverviewPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [banner, setBanner] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Editable form mirrors initialized from authoritative backend overview
  const [govForm, setGovForm] = useState<GovernanceConfigData | null>(null);
  const [featuresForm, setFeaturesForm] = useState<
    Record<FeatureGateKey, AdminOverviewPayload['features'][FeatureGateKey]> | null
  >(null);
  const [featurePlansForm, setFeaturePlansForm] = useState<FeaturePackageData[]>([]);
  const [tokenPlansForm, setTokenPlansForm] = useState<TokenPackageData[]>([]);
  const [paymentForm, setPaymentForm] = useState<PaymentSettingsData | null>(null);
  const [orgNameForm, setOrgNameForm] = useState('VoiceLab AI');
  const [ownershipLockedForm, setOwnershipLockedForm] = useState(true);

  // Payment Verification controls
  const [rejectReasons, setRejectReasons] = useState<Record<string, string>>({});
  const [paymentFilter, setPaymentFilter] = useState<'all' | 'pending' | 'approved' | 'rejected'>(
    'pending'
  );

  // Token Management controls
  const [tokTargetUid, setTokTargetUid] = useState('');
  const [tokTargetEmail, setTokTargetEmail] = useState('');
  const [tokDelta, setTokDelta] = useState<number>(50);
  const [tokReason, setTokReason] = useState('Owner complimentary token grant');

  // Entitlements grant controls
  const [entUid, setEntUid] = useState('');
  const [entEmail, setEntEmail] = useState('');
  const [entDays, setEntDays] = useState(30);
  const [entVc, setEntVc] = useState(true);
  const [entLong, setEntLong] = useState(true);
  const [entSrt, setEntSrt] = useState(true);
  const [entBundle, setEntBundle] = useState(false);

  const loadOverview = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchAdminOverview();
      setOverview(data);
      setGovForm(data.governance);
      setFeaturesForm(data.features);
      setFeaturePlansForm(data.featurePackages);
      setTokenPlansForm(data.tokenPackages);
      setPaymentForm(data.paymentSettings);
      setOrgNameForm(data.ownership.organizationName || 'VoiceLab AI');
      setOwnershipLockedForm(data.ownership.ownershipLocked);
    } catch (err: any) {
      setBanner({
        type: 'error',
        text: err?.message || 'Failed to load admin overview from backend.'
      });
    } finally {
      setLoading(false);
    }
  }, [fetchAdminOverview]);

  useEffect(() => {
    void loadOverview();
  }, [loadOverview]);

  const runAction = async (key: string, fn: () => Promise<void>, successMsg: string) => {
    setBanner(null);
    setBusy(key);
    try {
      await fn();
      await loadOverview();
      setBanner({ type: 'success', text: successMsg });
    } catch (err: any) {
      setBanner({ type: 'error', text: err?.message || 'Action failed.' });
    } finally {
      setBusy(null);
    }
  };

  const pendingPaymentsCount =
    overview?.paymentRequests.filter((r) => r.status === 'pending').length || 0;

  const navItems: Array<{
    id: AdminSectionId;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    badge?: number;
  }> = [
    { id: 'overview', label: 'Overview', icon: Activity },
    { id: 'users', label: 'Users', icon: Users },
    { id: 'features', label: 'Premium Features', icon: Sliders },
    { id: 'plans', label: 'Plans & Pricing', icon: Layers },
    { id: 'tokens', label: 'Token Management', icon: Coins },
    { id: 'payment-settings', label: 'Payment Settings', icon: CreditCard },
    {
      id: 'payment-verification',
      label: 'Payment Verification',
      icon: CheckCircle2,
      badge: pendingPaymentsCount
    },
    { id: 'entitlements', label: 'Access & Entitlements', icon: ShieldCheck },
    { id: 'security-logs', label: 'Security & Audit Logs', icon: ScrollText }
  ];

  return (
    <div className="space-y-5">
      {/* Owner Header Bar */}
      <section className="bg-[#12231D] border border-[#16A36A]/40 rounded-2xl p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5 min-w-0">
          {userPhoto ? (
            <img
              src={userPhoto}
              alt={userEmail}
              referrerPolicy="no-referrer"
              className="w-11 h-11 rounded-xl border border-[#16A36A]/60 object-cover shrink-0"
            />
          ) : (
            <div className="w-11 h-11 rounded-xl bg-[#19352A] border border-[#16A36A]/60 flex items-center justify-center shrink-0">
              <Crown className="w-5 h-5 text-[#42D9A0]" />
            </div>
          )}
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-base sm:text-lg font-bold text-[#F4F8F5] truncate">
                {overview?.ownership.organizationName || 'VoiceLab AI'} · Owner Admin Panel
              </h1>
              <span className="px-2 py-0.5 rounded-md bg-[#16A36A]/20 border border-[#16A36A]/60 text-[11px] font-semibold uppercase tracking-wide text-[#42D9A0]">
                Owner Verified
              </span>
            </div>
            <div className="text-xs text-[#A2B5AA] truncate font-mono mt-0.5">
              {userEmail} · UID: {userUid}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            type="button"
            onClick={() => void loadOverview()}
            className="px-3 py-1.5 rounded-xl bg-[#0B1512] hover:bg-[#19352A] border border-[#19352A] text-xs font-medium text-[#F4F8F5] flex items-center gap-1.5 cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-[#42D9A0] ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
          <button
            type="button"
            onClick={() => void onSignOut()}
            className="px-3 py-1.5 rounded-xl bg-[#0B1512] hover:bg-red-950/50 border border-[#19352A] hover:border-red-700/50 text-xs font-medium text-[#A2B5AA] hover:text-red-200 flex items-center gap-1.5 cursor-pointer"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span>Sign Out</span>
          </button>
        </div>
      </section>

      {banner && (
        <div
          className={`rounded-xl px-4 py-3 text-xs flex items-center justify-between gap-3 border ${
            banner.type === 'success'
              ? 'bg-[#12231D] border-[#16A36A]/60 text-[#42D9A0]'
              : 'bg-red-950/40 border-red-700/50 text-red-200'
          }`}
        >
          <div className="flex items-center gap-2">
            {banner.type === 'success' ? (
              <Check className="w-4 h-4 shrink-0" />
            ) : (
              <AlertTriangle className="w-4 h-4 shrink-0" />
            )}
            <span>{banner.text}</span>
          </div>
          <button
            type="button"
            onClick={() => setBanner(null)}
            className="text-[11px] opacity-70 hover:opacity-100 cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* 9-Section Responsive Navigation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        <nav className="lg:col-span-3 bg-[#12231D] border border-[#19352A] rounded-2xl p-2.5 flex lg:flex-col gap-1 overflow-x-auto">
          {navItems.map((item, idx) => {
            const Icon = item.icon;
            const active = section === item.id;
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => setSection(item.id)}
                className={`px-3 py-2 rounded-xl text-xs font-medium flex items-center justify-between gap-2 whitespace-nowrap transition-colors cursor-pointer ${
                  active
                    ? 'bg-[#19352A] text-[#42D9A0] border border-[#16A36A]/50'
                    : 'text-[#A2B5AA] hover:bg-[#0B1512] hover:text-[#F4F8F5]'
                }`}
              >
                <span className="flex items-center gap-2">
                  <Icon className="w-3.5 h-3.5 shrink-0" />
                  <span>
                    {idx + 1}. {item.label}
                  </span>
                </span>
                {typeof item.badge === 'number' && item.badge > 0 && (
                  <span className="px-1.5 py-0.5 rounded bg-amber-950/80 text-amber-300 text-[10px] font-bold">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        <div className="lg:col-span-9 space-y-5 min-w-0">
          {/* 1. Overview */}
          {section === 'overview' && overview && govForm && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  {
                    label: 'Registered Users',
                    value: String(overview.users.length)
                  },
                  {
                    label: 'Pending Payments',
                    value: String(pendingPaymentsCount)
                  },
                  {
                    label: 'Temp Storage',
                    value: `${overview.runtime.totalTempStorageMb} MB`
                  },
                  {
                    label: 'Server Uptime',
                    value: `${overview.runtime.uptimeSeconds}s`
                  }
                ].map((stat) => (
                  <div
                    key={stat.label}
                    className="bg-[#12231D] border border-[#19352A] rounded-2xl p-4 space-y-1"
                  >
                    <div className="text-[11px] text-[#A2B5AA]">{stat.label}</div>
                    <div className="text-lg font-bold font-mono text-[#F4F8F5]">{stat.value}</div>
                  </div>
                ))}
              </div>

              <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Settings className="w-4 h-4 text-[#42D9A0]" />
                    <h2 className="text-sm font-bold text-[#F4F8F5]">
                      Runtime Governance &amp; System Health
                    </h2>
                  </div>
                  <button
                    type="button"
                    disabled={busy === 'purge'}
                    onClick={() =>
                      void runAction(
                        'purge',
                        async () => {
                          await purgeAdminStorage();
                        },
                        'Temporary audio storage purged.'
                      )
                    }
                    className="px-3 py-1.5 rounded-xl bg-red-950/50 hover:bg-red-900/60 border border-red-700/50 text-xs font-semibold text-red-200 flex items-center gap-1.5 cursor-pointer"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    <span>Purge Temp Media</span>
                  </button>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  {[
                    {
                      key: 'maintenanceMode' as const,
                      title: 'Maintenance Mode',
                      desc: 'Pauses public compute endpoints while Owner keeps full access.'
                    },
                    {
                      key: 'requireAuthForCompute' as const,
                      title: 'Require Google Sign-In for Compute',
                      desc: 'Requires verified Google Sign-In for Voice Cloning & SRT.'
                    },
                    {
                      key: 'allowVoiceCloning' as const,
                      title: 'Master Voice Cloning Switch',
                      desc: 'Global toggle for Voice Cloning studio availability.'
                    },
                    {
                      key: 'allowSrtGeneration' as const,
                      title: 'Master Audio-to-SRT Switch',
                      desc: 'Global toggle for Audio-to-SRT transcription.'
                    }
                  ].map((item) => (
                    <label
                      key={item.key}
                      className="flex items-start gap-2.5 p-3 rounded-xl bg-[#0B1512] border border-[#19352A] cursor-pointer"
                    >
                      <input
                        type="checkbox"
                        checked={Boolean(govForm[item.key])}
                        onChange={(e) =>
                          setGovForm((prev) =>
                            prev ? { ...prev, [item.key]: e.target.checked } : prev
                          )
                        }
                        className="mt-0.5 accent-[#16A36A]"
                      />
                      <div>
                        <div className="font-semibold text-[#F4F8F5]">{item.title}</div>
                        <div className="text-[11px] text-[#A2B5AA]">{item.desc}</div>
                      </div>
                    </label>
                  ))}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs text-[#A2B5AA] mb-1">
                      Rate Limit (/min/IP)
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={60}
                      value={govForm.rateLimitPerMinute}
                      onChange={(e) =>
                        setGovForm((prev) =>
                          prev
                            ? {
                                ...prev,
                                rateLimitPerMinute: Math.max(
                                  1,
                                  Math.min(60, Number(e.target.value) || 1)
                                )
                              }
                            : prev
                        )
                      }
                      className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] text-xs font-mono text-[#F4F8F5]"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-[#A2B5AA] mb-1">Max Script Chars</label>
                    <input
                      type="number"
                      min={1000}
                      max={25000}
                      step={500}
                      value={govForm.maxScriptChars}
                      onChange={(e) =>
                        setGovForm((prev) =>
                          prev
                            ? {
                                ...prev,
                                maxScriptChars: Math.max(
                                  1000,
                                  Math.min(25000, Number(e.target.value) || 12000)
                                )
                              }
                            : prev
                        )
                      }
                      className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] text-xs font-mono text-[#F4F8F5]"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-[#A2B5AA] mb-1">
                      Maintenance Notice
                    </label>
                    <input
                      type="text"
                      maxLength={300}
                      value={govForm.maintenanceNotice}
                      onChange={(e) =>
                        setGovForm((prev) =>
                          prev ? { ...prev, maintenanceNotice: e.target.value } : prev
                        )
                      }
                      className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] text-xs text-[#F4F8F5]"
                    />
                  </div>
                </div>

                <button
                  type="button"
                  disabled={busy === 'gov'}
                  onClick={() =>
                    void runAction(
                      'gov',
                      async () => {
                        await saveAdminGovernance(govForm);
                      },
                      'Runtime governance policy updated.'
                    )
                  }
                  className="px-4 py-2 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] cursor-pointer"
                >
                  {busy === 'gov' ? 'Saving...' : 'Save Runtime Policy'}
                </button>
              </div>
            </div>
          )}

          {/* 2. Users */}
          {section === 'users' && overview && (
            <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Users className="w-4 h-4 text-[#42D9A0]" />
                  <h2 className="text-sm font-bold text-[#F4F8F5]">
                    Registered Users ({overview.users.length})
                  </h2>
                </div>
                <span className="text-[11px] text-[#A2B5AA]">
                  Verified Google accounts registered with the server
                </span>
              </div>

              {overview.users.length === 0 ? (
                <div className="text-xs text-[#A2B5AA]">No registered users yet.</div>
              ) : (
                <div className="divide-y divide-[#19352A]">
                  {overview.users.map((u) => {
                    const ent = overview.entitlements.find((e) => e.uid === u.uid);
                    return (
                      <div
                        key={u.uid}
                        className="py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
                      >
                        <div className="min-w-0 space-y-0.5">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-semibold text-[#F4F8F5]">
                              {u.displayName || u.email}
                            </span>
                            <span className="px-1.5 py-0.5 rounded bg-[#0B1512] text-[#42D9A0] text-[10px] uppercase font-mono">
                              {u.role}
                            </span>
                            <span
                              className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-semibold ${
                                u.accountStatus === 'active'
                                  ? 'bg-[#16A36A]/20 text-[#42D9A0]'
                                  : 'bg-red-950/70 text-red-300'
                              }`}
                            >
                              {u.accountStatus}
                            </span>
                            <span className="px-1.5 py-0.5 rounded bg-[#0B1512] text-[#F4F8F5] text-[10px] font-mono">
                              {ent?.tokenBalance ?? 0} tokens
                            </span>
                          </div>
                          <div className="font-mono text-[11px] text-[#A2B5AA] truncate">
                            {u.email} · {u.uid}
                          </div>
                        </div>

                        <div className="flex flex-wrap items-center gap-1.5 shrink-0">
                          <button
                            type="button"
                            onClick={() => {
                              setTokTargetUid(u.uid);
                              setTokTargetEmail(u.email);
                              setSection('tokens');
                            }}
                            className="px-2.5 py-1 rounded-lg bg-[#0B1512] hover:bg-[#19352A] border border-[#19352A] text-[11px] text-[#42D9A0] cursor-pointer"
                          >
                            Tokens
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setEntUid(u.uid);
                              setEntEmail(u.email);
                              setSection('entitlements');
                            }}
                            className="px-2.5 py-1 rounded-lg bg-[#0B1512] hover:bg-[#19352A] border border-[#19352A] text-[11px] text-[#F4F8F5] cursor-pointer"
                          >
                            Entitlements
                          </button>
                          {u.uid !== userUid && (
                            <button
                              type="button"
                              disabled={busy === `user-${u.uid}`}
                              onClick={() =>
                                void runAction(
                                  `user-${u.uid}`,
                                  async () => {
                                    await updateAdminUserStatus({
                                      targetUid: u.uid,
                                      accountStatus:
                                        u.accountStatus === 'active' ? 'suspended' : 'active'
                                    });
                                  },
                                  `Updated ${u.email} status.`
                                )
                              }
                              className="px-2.5 py-1 rounded-lg bg-[#0B1512] hover:bg-red-950/50 border border-[#19352A] text-[11px] text-[#A2B5AA] hover:text-red-200 inline-flex items-center gap-1 cursor-pointer"
                            >
                              {u.accountStatus === 'active' ? (
                                <>
                                  <UserX className="w-3 h-3" /> Suspend
                                </>
                              ) : (
                                <>
                                  <UserCheck className="w-3 h-3" /> Activate
                                </>
                              )}
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* 3. Premium Features */}
          {section === 'features' && featuresForm && (
            <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-4">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-[#42D9A0]" />
                <h2 className="text-sm font-bold text-[#F4F8F5]">
                  Premium Feature Controls &amp; Token Requirements
                </h2>
              </div>
              <p className="text-xs text-[#A2B5AA]">
                Enable/disable features, toggle Free vs Paid access, and configure token cost per request. Enforced directly on backend compute endpoints.
              </p>

              <div className="space-y-3">
                {(['voice_cloning', 'long_script', 'srt_generator'] as FeatureGateKey[]).map(
                  (key) => {
                    const cfg = featuresForm[key];
                    return (
                      <div
                        key={key}
                        className="rounded-xl bg-[#0B1512] border border-[#19352A] p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4"
                      >
                        <div className="space-y-1">
                          <div className="text-xs font-bold text-[#F4F8F5]">{cfg.title}</div>
                          <div className="text-[11px] text-[#A2B5AA]">{cfg.description}</div>
                        </div>

                        <div className="flex flex-wrap items-center gap-3 shrink-0 text-xs">
                          <label className="flex items-center gap-1.5 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={cfg.enabled}
                              onChange={(e) =>
                                setFeaturesForm((prev) =>
                                  prev
                                    ? {
                                        ...prev,
                                        [key]: { ...prev[key], enabled: e.target.checked }
                                      }
                                    : prev
                                )
                              }
                              className="accent-[#16A36A]"
                            />
                            <span className="text-[#F4F8F5]">Enabled</span>
                          </label>

                          <label className="flex items-center gap-1.5 cursor-pointer">
                            <input
                              type="checkbox"
                              checked={cfg.isPaid}
                              onChange={(e) =>
                                setFeaturesForm((prev) =>
                                  prev
                                    ? {
                                        ...prev,
                                        [key]: { ...prev[key], isPaid: e.target.checked }
                                      }
                                    : prev
                                )
                              }
                              className="accent-[#42D9A0]"
                            />
                            <span className="text-[#42D9A0]">Paid Only</span>
                          </label>

                          <div className="flex items-center gap-1.5">
                            <span className="text-[11px] text-[#A2B5AA]">Token Cost:</span>
                            <input
                              type="number"
                              min={0}
                              max={1000}
                              value={cfg.tokenCost}
                              onChange={(e) =>
                                setFeaturesForm((prev) =>
                                  prev
                                    ? {
                                        ...prev,
                                        [key]: {
                                          ...prev[key],
                                          tokenCost: Math.max(
                                            0,
                                            Math.min(1000, Number(e.target.value) || 0)
                                          )
                                        }
                                      }
                                    : prev
                                )
                              }
                              className="w-16 px-2 py-1 rounded-lg bg-[#12231D] border border-[#19352A] text-xs font-mono text-[#F4F8F5]"
                            />
                          </div>
                        </div>
                      </div>
                    );
                  }
                )}
              </div>

              <button
                type="button"
                disabled={busy === 'features'}
                onClick={() =>
                  void runAction(
                    'features',
                    async () => {
                      await saveAdminFeatures(featuresForm);
                    },
                    'Premium feature gates and token costs saved.'
                  )
                }
                className="px-4 py-2 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] cursor-pointer"
              >
                {busy === 'features' ? 'Saving...' : 'Save Feature Controls'}
              </button>
            </div>
          )}

          {/* 4. Plans & Pricing */}
          {section === 'plans' && (
            <div className="space-y-5">
              <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-[#42D9A0]" />
                    <h2 className="text-sm font-bold text-[#F4F8F5]">Feature Packages (BDT)</h2>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      setFeaturePlansForm((prev) => [
                        ...prev,
                        {
                          id: `pkg_custom_${Date.now()}`,
                          featureKey: 'pro_bundle',
                          title: 'New Custom Plan',
                          description: 'Custom feature access package',
                          priceBdt: 299,
                          durationDays: 30,
                          enabled: true
                        }
                      ])
                    }
                    className="px-2.5 py-1 rounded-lg bg-[#0B1512] border border-[#19352A] text-xs text-[#42D9A0] flex items-center gap-1 cursor-pointer"
                  >
                    <Plus className="w-3.5 h-3.5" /> Add Plan
                  </button>
                </div>

                <div className="space-y-3">
                  {featurePlansForm.map((pkg, idx) => (
                    <div
                      key={pkg.id}
                      className="rounded-xl bg-[#0B1512] border border-[#19352A] p-3.5 grid grid-cols-1 sm:grid-cols-12 gap-2.5 items-center text-xs"
                    >
                      <input
                        type="text"
                        value={pkg.title}
                        onChange={(e) =>
                          setFeaturePlansForm((prev) =>
                            prev.map((p, i) => (i === idx ? { ...p, title: e.target.value } : p))
                          )
                        }
                        className="sm:col-span-3 px-2.5 py-1.5 rounded-lg bg-[#12231D] border border-[#19352A] text-[#F4F8F5]"
                        placeholder="Title"
                      />
                      <select
                        value={pkg.featureKey}
                        onChange={(e) =>
                          setFeaturePlansForm((prev) =>
                            prev.map((p, i) =>
                              i === idx
                                ? { ...p, featureKey: e.target.value as FeaturePackageData['featureKey'] }
                                : p
                            )
                          )
                        }
                        className="sm:col-span-3 px-2.5 py-1.5 rounded-lg bg-[#12231D] border border-[#19352A] text-[#F4F8F5]"
                      >
                        <option value="voice_cloning">Voice Cloning</option>
                        <option value="long_script">Long Script</option>
                        <option value="srt_generator">Audio to SRT</option>
                        <option value="pro_bundle">All-in-One Bundle</option>
                      </select>
                      <input
                        type="number"
                        min={1}
                        max={50000}
                        value={pkg.priceBdt}
                        onChange={(e) =>
                          setFeaturePlansForm((prev) =>
                            prev.map((p, i) =>
                              i === idx ? { ...p, priceBdt: Number(e.target.value) || 1 } : p
                            )
                          )
                        }
                        className="sm:col-span-2 px-2.5 py-1.5 rounded-lg bg-[#12231D] border border-[#19352A] font-mono text-[#F4F8F5]"
                        title="Price BDT"
                      />
                      <input
                        type="number"
                        min={0}
                        max={3650}
                        value={pkg.durationDays}
                        onChange={(e) =>
                          setFeaturePlansForm((prev) =>
                            prev.map((p, i) =>
                              i === idx ? { ...p, durationDays: Number(e.target.value) || 0 } : p
                            )
                          )
                        }
                        className="sm:col-span-2 px-2.5 py-1.5 rounded-lg bg-[#12231D] border border-[#19352A] font-mono text-[#F4F8F5]"
                        title="Days (0=Lifetime)"
                      />
                      <div className="sm:col-span-2 flex items-center justify-end gap-2">
                        <label className="flex items-center gap-1 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={pkg.enabled}
                            onChange={(e) =>
                              setFeaturePlansForm((prev) =>
                                prev.map((p, i) =>
                                  i === idx ? { ...p, enabled: e.target.checked } : p
                                )
                              )
                            }
                            className="accent-[#16A36A]"
                          />
                          <span>On</span>
                        </label>
                      </div>
                      <input
                        type="text"
                        value={pkg.description}
                        onChange={(e) =>
                          setFeaturePlansForm((prev) =>
                            prev.map((p, i) =>
                              i === idx ? { ...p, description: e.target.value } : p
                            )
                          )
                        }
                        className="sm:col-span-12 px-2.5 py-1.5 rounded-lg bg-[#12231D] border border-[#19352A] text-[11px] text-[#A2B5AA]"
                        placeholder="Package description"
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Coins className="w-4 h-4 text-[#42D9A0]" />
                    <h2 className="text-sm font-bold text-[#F4F8F5]">Token Packages (BDT)</h2>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      setTokenPlansForm((prev) => [
                        ...prev,
                        {
                          id: `tok_custom_${Date.now()}`,
                          title: 'Custom Token Pack',
                          description: 'Compute token pack',
                          tokens: 100,
                          priceBdt: 199,
                          enabled: true
                        }
                      ])
                    }
                    className="px-2.5 py-1 rounded-lg bg-[#0B1512] border border-[#19352A] text-xs text-[#42D9A0] flex items-center gap-1 cursor-pointer"
                  >
                    <Plus className="w-3.5 h-3.5" /> Add Token Pack
                  </button>
                </div>

                <div className="space-y-3">
                  {tokenPlansForm.map((tp, idx) => (
                    <div
                      key={tp.id}
                      className="rounded-xl bg-[#0B1512] border border-[#19352A] p-3.5 grid grid-cols-1 sm:grid-cols-12 gap-2.5 items-center text-xs"
                    >
                      <input
                        type="text"
                        value={tp.title}
                        onChange={(e) =>
                          setTokenPlansForm((prev) =>
                            prev.map((p, i) => (i === idx ? { ...p, title: e.target.value } : p))
                          )
                        }
                        className="sm:col-span-4 px-2.5 py-1.5 rounded-lg bg-[#12231D] border border-[#19352A] text-[#F4F8F5]"
                        placeholder="Pack Title"
                      />
                      <input
                        type="number"
                        min={1}
                        max={100000}
                        value={tp.tokens}
                        onChange={(e) =>
                          setTokenPlansForm((prev) =>
                            prev.map((p, i) =>
                              i === idx ? { ...p, tokens: Number(e.target.value) || 1 } : p
                            )
                          )
                        }
                        className="sm:col-span-3 px-2.5 py-1.5 rounded-lg bg-[#12231D] border border-[#19352A] font-mono text-[#42D9A0]"
                        placeholder="Tokens"
                      />
                      <input
                        type="number"
                        min={1}
                        max={50000}
                        value={tp.priceBdt}
                        onChange={(e) =>
                          setTokenPlansForm((prev) =>
                            prev.map((p, i) =>
                              i === idx ? { ...p, priceBdt: Number(e.target.value) || 1 } : p
                            )
                          )
                        }
                        className="sm:col-span-3 px-2.5 py-1.5 rounded-lg bg-[#12231D] border border-[#19352A] font-mono text-[#F4F8F5]"
                        placeholder="BDT"
                      />
                      <div className="sm:col-span-2 flex items-center justify-end gap-2">
                        <label className="flex items-center gap-1 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={tp.enabled}
                            onChange={(e) =>
                              setTokenPlansForm((prev) =>
                                prev.map((p, i) =>
                                  i === idx ? { ...p, enabled: e.target.checked } : p
                                )
                              )
                            }
                            className="accent-[#16A36A]"
                          />
                          <span>On</span>
                        </label>
                      </div>
                    </div>
                  ))}
                </div>

                <button
                  type="button"
                  disabled={busy === 'plans'}
                  onClick={() =>
                    void runAction(
                      'plans',
                      async () => {
                        await saveAdminPlans({
                          featurePackages: featurePlansForm,
                          tokenPackages: tokenPlansForm
                        });
                      },
                      'Plans and token packages saved.'
                    )
                  }
                  className="px-4 py-2 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] cursor-pointer"
                >
                  {busy === 'plans' ? 'Saving...' : 'Save Plans & Token Packages'}
                </button>
              </div>
            </div>
          )}

          {/* 5. Token Management */}
          {section === 'tokens' && overview && (
            <div className="space-y-5">
              <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-4">
                <div className="flex items-center gap-2">
                  <Coins className="w-4 h-4 text-[#42D9A0]" />
                  <h2 className="text-sm font-bold text-[#F4F8F5]">
                    Grant or Revoke User Tokens
                  </h2>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">Target Firebase UID</label>
                    <input
                      type="text"
                      value={tokTargetUid}
                      onChange={(e) => setTokTargetUid(e.target.value)}
                      placeholder="User UID"
                      className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] font-mono text-[#F4F8F5]"
                    />
                  </div>
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">Target Email</label>
                    <input
                      type="email"
                      value={tokTargetEmail}
                      onChange={(e) => setTokTargetEmail(e.target.value)}
                      placeholder="user@gmail.com"
                      className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] text-[#F4F8F5]"
                    />
                  </div>
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">
                      Token Delta (+ to grant, - to revoke)
                    </label>
                    <input
                      type="number"
                      value={tokDelta}
                      onChange={(e) => setTokDelta(Number(e.target.value) || 0)}
                      className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] font-mono text-[#F4F8F5]"
                    />
                  </div>
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">Ledger Reason</label>
                    <input
                      type="text"
                      value={tokReason}
                      onChange={(e) => setTokReason(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] text-[#F4F8F5]"
                    />
                  </div>
                </div>

                <button
                  type="button"
                  disabled={busy === 'tok-adjust'}
                  onClick={() =>
                    void runAction(
                      'tok-adjust',
                      async () => {
                        await adjustAdminUserTokens({
                          targetUid: tokTargetUid,
                          targetEmail: tokTargetEmail,
                          delta: tokDelta,
                          reason: tokReason
                        });
                      },
                      `Applied token adjustment (${tokDelta}) for ${tokTargetEmail}.`
                    )
                  }
                  className="px-4 py-2 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] cursor-pointer"
                >
                  {busy === 'tok-adjust' ? 'Applying...' : 'Apply Token Adjustment'}
                </button>
              </div>

              <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-3">
                <h3 className="text-xs font-bold text-[#F4F8F5]">
                  Server Token Ledger ({overview.tokenLedger.length})
                </h3>
                {overview.tokenLedger.length === 0 ? (
                  <div className="text-xs text-[#A2B5AA]">No token transactions recorded yet.</div>
                ) : (
                  <div className="divide-y divide-[#19352A] max-h-80 overflow-y-auto">
                    {overview.tokenLedger.map((entry) => (
                      <div
                        key={entry.entryId}
                        className="py-2.5 flex items-center justify-between gap-3 text-xs"
                      >
                        <div className="min-w-0">
                          <div className="text-[#F4F8F5] font-medium truncate">
                            {entry.userEmail} · <span className="text-[#A2B5AA]">{entry.reason}</span>
                          </div>
                          <div className="text-[10px] text-[#A2B5AA] font-mono">
                            {entry.type} · Balance after: {entry.balanceAfter} · {formatIsoTime(entry.createdAt)}
                          </div>
                        </div>
                        <span
                          className={`font-mono font-bold shrink-0 ${
                            entry.delta >= 0 ? 'text-[#42D9A0]' : 'text-red-300'
                          }`}
                        >
                          {entry.delta >= 0 ? `+${entry.delta}` : entry.delta}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 6. Payment Settings (bKash & Nagad) */}
          {section === 'payment-settings' && paymentForm && (
            <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-5">
              <div className="flex items-center gap-2">
                <CreditCard className="w-4 h-4 text-[#42D9A0]" />
                <h2 className="text-sm font-bold text-[#F4F8F5]">
                  bKash &amp; Nagad Payment Settings
                </h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* bKash Card */}
                <div className="rounded-xl bg-[#0B1512] border border-[#E2136E]/40 p-4 space-y-3 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-[#FF66AB] uppercase">bKash Settings</span>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={paymentForm.bkashEnabled}
                        onChange={(e) =>
                          setPaymentForm((prev) =>
                            prev ? { ...prev, bkashEnabled: e.target.checked } : prev
                          )
                        }
                        className="accent-[#E2136E]"
                      />
                      <span>Enabled</span>
                    </label>
                  </div>
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">Receiver bKash Number</label>
                    <input
                      type="tel"
                      maxLength={20}
                      placeholder="017XXXXXXXX"
                      value={paymentForm.bkashNumber}
                      onChange={(e) =>
                        setPaymentForm((prev) =>
                          prev ? { ...prev, bkashNumber: e.target.value } : prev
                        )
                      }
                      className="w-full px-3 py-2 rounded-lg bg-[#12231D] border border-[#19352A] font-mono text-[#F4F8F5]"
                    />
                  </div>
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">Account Type</label>
                    <select
                      value={paymentForm.bkashAccountType}
                      onChange={(e) =>
                        setPaymentForm((prev) =>
                          prev
                            ? {
                                ...prev,
                                bkashAccountType: e.target.value as 'Personal' | 'Merchant' | 'Agent'
                              }
                            : prev
                        )
                      }
                      className="w-full px-3 py-2 rounded-lg bg-[#12231D] border border-[#19352A] text-[#F4F8F5]"
                    >
                      <option value="Personal">Personal (Send Money)</option>
                      <option value="Merchant">Merchant (Payment)</option>
                      <option value="Agent">Agent (Cash Out)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">bKash Payment Instructions</label>
                    <textarea
                      rows={3}
                      maxLength={500}
                      value={paymentForm.bkashInstructions}
                      onChange={(e) =>
                        setPaymentForm((prev) =>
                          prev ? { ...prev, bkashInstructions: e.target.value } : prev
                        )
                      }
                      className="w-full px-3 py-2 rounded-lg bg-[#12231D] border border-[#19352A] text-[#F4F8F5]"
                    />
                  </div>
                </div>

                {/* Nagad Card */}
                <div className="rounded-xl bg-[#0B1512] border border-[#F36F21]/40 p-4 space-y-3 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-[#FF9248] uppercase">Nagad Settings</span>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={paymentForm.nagadEnabled}
                        onChange={(e) =>
                          setPaymentForm((prev) =>
                            prev ? { ...prev, nagadEnabled: e.target.checked } : prev
                          )
                        }
                        className="accent-[#F36F21]"
                      />
                      <span>Enabled</span>
                    </label>
                  </div>
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">Receiver Nagad Number</label>
                    <input
                      type="tel"
                      maxLength={20}
                      placeholder="018XXXXXXXX"
                      value={paymentForm.nagadNumber}
                      onChange={(e) =>
                        setPaymentForm((prev) =>
                          prev ? { ...prev, nagadNumber: e.target.value } : prev
                        )
                      }
                      className="w-full px-3 py-2 rounded-lg bg-[#12231D] border border-[#19352A] font-mono text-[#F4F8F5]"
                    />
                  </div>
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">Account Type</label>
                    <select
                      value={paymentForm.nagadAccountType}
                      onChange={(e) =>
                        setPaymentForm((prev) =>
                          prev
                            ? {
                                ...prev,
                                nagadAccountType: e.target.value as 'Personal' | 'Merchant' | 'Agent'
                              }
                            : prev
                        )
                      }
                      className="w-full px-3 py-2 rounded-lg bg-[#12231D] border border-[#19352A] text-[#F4F8F5]"
                    >
                      <option value="Personal">Personal (Send Money)</option>
                      <option value="Merchant">Merchant (Payment)</option>
                      <option value="Agent">Agent (Cash Out)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">Nagad Payment Instructions</label>
                    <textarea
                      rows={3}
                      maxLength={500}
                      value={paymentForm.nagadInstructions}
                      onChange={(e) =>
                        setPaymentForm((prev) =>
                          prev ? { ...prev, nagadInstructions: e.target.value } : prev
                        )
                      }
                      className="w-full px-3 py-2 rounded-lg bg-[#12231D] border border-[#19352A] text-[#F4F8F5]"
                    />
                  </div>
                </div>
              </div>

              <button
                type="button"
                disabled={busy === 'pay-settings'}
                onClick={() =>
                  void runAction(
                    'pay-settings',
                    async () => {
                      await saveAdminPaymentSettings(paymentForm);
                    },
                    'bKash & Nagad payment settings saved.'
                  )
                }
                className="px-4 py-2 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] cursor-pointer"
              >
                {busy === 'pay-settings' ? 'Saving...' : 'Save Payment Settings'}
              </button>
            </div>
          )}

          {/* 7. Payment Verification */}
          {section === 'payment-verification' && overview && (
            <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-[#42D9A0]" />
                  <h2 className="text-sm font-bold text-[#F4F8F5]">
                    bKash &amp; Nagad Payment Verification
                  </h2>
                </div>
                <div className="flex items-center gap-1.5 text-xs">
                  {(['pending', 'approved', 'rejected', 'all'] as const).map((f) => (
                    <button
                      key={f}
                      type="button"
                      onClick={() => setPaymentFilter(f)}
                      className={`px-2.5 py-1 rounded-lg uppercase text-[10px] font-semibold cursor-pointer ${
                        paymentFilter === f
                          ? 'bg-[#19352A] text-[#42D9A0] border border-[#16A36A]/50'
                          : 'bg-[#0B1512] text-[#A2B5AA]'
                      }`}
                    >
                      {f}
                    </button>
                  ))}
                </div>
              </div>

              {(() => {
                const filtered = overview.paymentRequests.filter((r) =>
                  paymentFilter === 'all' ? true : r.status === paymentFilter
                );
                if (filtered.length === 0) {
                  return (
                    <div className="text-xs text-[#A2B5AA]">
                      No {paymentFilter === 'all' ? '' : paymentFilter} payment requests found.
                    </div>
                  );
                }
                return (
                  <div className="space-y-3">
                    {filtered.map((ord: PaymentRequestData) => (
                      <div
                        key={ord.orderId}
                        className="rounded-xl bg-[#0B1512] border border-[#19352A] p-4 space-y-2.5 text-xs"
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="space-y-0.5">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span
                                className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                                  ord.paymentMethod === 'bkash'
                                    ? 'bg-[#E2136E] text-white'
                                    : 'bg-[#F36F21] text-white'
                                }`}
                              >
                                {ord.paymentMethod}
                              </span>
                              <span className="font-bold text-[#F4F8F5]">{ord.itemTitle}</span>
                              <span className="font-mono text-[#42D9A0]">৳{ord.amountBdt}</span>
                              {ord.tokensGranted > 0 && (
                                <span className="font-mono text-amber-300">
                                  (+{ord.tokensGranted} tokens)
                                </span>
                              )}
                            </div>
                            <div className="text-[11px] text-[#A2B5AA] font-mono">
                              User: {ord.userEmail} · Sender: {ord.senderNumber} · TrxID:{' '}
                              <span className="text-[#F4F8F5] font-bold">{ord.trxId}</span>
                            </div>
                            {ord.note && (
                              <div className="text-[11px] text-[#A2B5AA]">Note: {ord.note}</div>
                            )}
                          </div>

                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase inline-flex items-center gap-1 ${
                              ord.status === 'approved'
                                ? 'bg-[#16A36A]/20 text-[#42D9A0]'
                                : ord.status === 'rejected'
                                ? 'bg-red-950/60 text-red-300'
                                : 'bg-amber-950/60 text-amber-300'
                            }`}
                          >
                            {ord.status === 'approved' ? (
                              <>
                                <CheckCircle2 className="w-3 h-3" /> Approved
                              </>
                            ) : ord.status === 'rejected' ? (
                              <>
                                <XCircle className="w-3 h-3" /> Rejected
                              </>
                            ) : (
                              <>
                                <Clock className="w-3 h-3" /> Pending
                              </>
                            )}
                          </span>
                        </div>

                        {ord.status === 'pending' && (
                          <div className="flex flex-col sm:flex-row gap-2 pt-2 border-t border-[#19352A]">
                            <input
                              type="text"
                              placeholder="Optional rejection reason..."
                              value={rejectReasons[ord.orderId] || ''}
                              onChange={(e) =>
                                setRejectReasons((prev) => ({
                                  ...prev,
                                  [ord.orderId]: e.target.value
                                }))
                              }
                              className="flex-1 px-3 py-1.5 rounded-lg bg-[#12231D] border border-[#19352A] text-xs text-[#F4F8F5]"
                            />
                            <button
                              type="button"
                              disabled={busy === `rev-${ord.orderId}`}
                              onClick={() =>
                                void runAction(
                                  `rev-${ord.orderId}`,
                                  async () => {
                                    await reviewAdminPaymentRequest({
                                      orderId: ord.orderId,
                                      decision: 'approved'
                                    });
                                  },
                                  `Approved ${ord.trxId} and credited ${ord.userEmail}.`
                                )
                              }
                              className="px-3.5 py-1.5 rounded-lg bg-[#16A36A] hover:bg-[#16A36A]/90 text-white font-semibold cursor-pointer"
                            >
                              Approve &amp; Grant
                            </button>
                            <button
                              type="button"
                              disabled={busy === `rev-${ord.orderId}`}
                              onClick={() =>
                                void runAction(
                                  `rev-${ord.orderId}`,
                                  async () => {
                                    await reviewAdminPaymentRequest({
                                      orderId: ord.orderId,
                                      decision: 'rejected',
                                      rejectionReason: rejectReasons[ord.orderId]
                                    });
                                  },
                                  `Rejected ${ord.trxId}.`
                                )
                              }
                              className="px-3.5 py-1.5 rounded-lg bg-red-950/70 hover:bg-red-900 text-red-200 border border-red-700/50 font-semibold cursor-pointer"
                            >
                              Reject
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
          )}

          {/* 8. Access & Entitlements */}
          {section === 'entitlements' && overview && (
            <div className="space-y-5">
              <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-4">
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-[#42D9A0]" />
                  <h2 className="text-sm font-bold text-[#F4F8F5]">
                    Grant / Revoke Feature Entitlements
                  </h2>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                  <input
                    type="text"
                    placeholder="User UID"
                    value={entUid}
                    onChange={(e) => setEntUid(e.target.value)}
                    className="px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] font-mono text-[#F4F8F5]"
                  />
                  <input
                    type="email"
                    placeholder="User Email"
                    value={entEmail}
                    onChange={(e) => setEntEmail(e.target.value)}
                    className="px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] text-[#F4F8F5]"
                  />
                  <input
                    type="number"
                    min={0}
                    max={3650}
                    placeholder="Days (0=Lifetime)"
                    value={entDays}
                    onChange={(e) => setEntDays(Math.max(0, Number(e.target.value) || 0))}
                    className="px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] font-mono text-[#F4F8F5]"
                  />
                </div>

                <div className="flex flex-wrap gap-4 text-xs text-[#F4F8F5]">
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={entVc}
                      onChange={(e) => setEntVc(e.target.checked)}
                      className="accent-[#16A36A]"
                    />
                    <span>Voice Cloning</span>
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={entLong}
                      onChange={(e) => setEntLong(e.target.checked)}
                      className="accent-[#16A36A]"
                    />
                    <span>Long Script</span>
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={entSrt}
                      onChange={(e) => setEntSrt(e.target.checked)}
                      className="accent-[#16A36A]"
                    />
                    <span>Audio to SRT</span>
                  </label>
                  <label className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={entBundle}
                      onChange={(e) => setEntBundle(e.target.checked)}
                      className="accent-[#16A36A]"
                    />
                    <span>Pro Bundle</span>
                  </label>
                </div>

                <button
                  type="button"
                  disabled={busy === 'grant-ent'}
                  onClick={() =>
                    void runAction(
                      'grant-ent',
                      async () => {
                        await grantAdminEntitlement({
                          uid: entUid,
                          email: entEmail,
                          voiceCloningUnlocked: entVc,
                          longScriptUnlocked: entLong,
                          srtGeneratorUnlocked: entSrt,
                          proBundleUnlocked: entBundle,
                          durationDays: entDays
                        });
                      },
                      `Updated entitlements for ${entEmail}.`
                    )
                  }
                  className="px-4 py-2 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] cursor-pointer"
                >
                  {busy === 'grant-ent' ? 'Saving...' : 'Save User Entitlement'}
                </button>
              </div>

              <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-3">
                <h3 className="text-xs font-bold text-[#F4F8F5]">
                  Active User Entitlements &amp; Token Balances ({overview.entitlements.length})
                </h3>
                <div className="divide-y divide-[#19352A]">
                  {overview.entitlements.map((e) => (
                    <div
                      key={e.uid}
                      className="py-2.5 flex flex-wrap items-center justify-between gap-2 text-xs"
                    >
                      <div className="min-w-0">
                        <div className="font-mono text-[#F4F8F5]">{e.email}</div>
                        <div className="text-[11px] text-[#A2B5AA]">
                          Tokens: <span className="text-[#42D9A0]">{e.tokenBalance}</span> · VC:{' '}
                          {e.voiceCloningUnlocked ? 'Yes' : 'No'} · Long:{' '}
                          {e.longScriptUnlocked ? 'Yes' : 'No'} · SRT:{' '}
                          {e.srtGeneratorUnlocked ? 'Yes' : 'No'} · Bundle:{' '}
                          {e.proBundleUnlocked ? 'Yes' : 'No'}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          setEntUid(e.uid);
                          setEntEmail(e.email);
                          setEntVc(e.voiceCloningUnlocked);
                          setEntLong(e.longScriptUnlocked);
                          setEntSrt(e.srtGeneratorUnlocked);
                          setEntBundle(e.proBundleUnlocked);
                        }}
                        className="px-2.5 py-1 rounded-lg bg-[#0B1512] border border-[#19352A] text-[11px] text-[#42D9A0] cursor-pointer"
                      >
                        Edit
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 9. Security & Audit Logs */}
          {section === 'security-logs' && overview && (
            <div className="space-y-5">
              <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <KeyRound className="w-4 h-4 text-[#42D9A0]" />
                    <h2 className="text-sm font-bold text-[#F4F8F5]">
                      Ownership Lock &amp; Security Policy
                    </h2>
                  </div>
                  <span className="px-2 py-0.5 rounded bg-[#0B1512] text-xs font-mono text-[#42D9A0] inline-flex items-center gap-1">
                    {ownershipLockedForm ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                    {ownershipLockedForm ? 'Locked' : 'Unlocked'}
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div>
                    <label className="block text-[#A2B5AA] mb-1">Workspace Organization Name</label>
                    <input
                      type="text"
                      value={orgNameForm}
                      onChange={(e) => setOrgNameForm(e.target.value)}
                      className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] text-[#F4F8F5]"
                    />
                  </div>
                  <label className="flex items-center gap-2 p-3 rounded-xl bg-[#0B1512] border border-[#19352A] cursor-pointer">
                    <input
                      type="checkbox"
                      checked={ownershipLockedForm}
                      onChange={(e) => setOwnershipLockedForm(e.target.checked)}
                      className="accent-[#16A36A]"
                    />
                    <span className="text-[#F4F8F5]">Lock Primary Root Ownership</span>
                  </label>
                </div>

                <button
                  type="button"
                  disabled={busy === 'owner-save'}
                  onClick={() =>
                    void runAction(
                      'owner-save',
                      async () => {
                        await saveAdminOwnership({
                          organizationName: orgNameForm,
                          ownershipLocked: ownershipLockedForm
                        });
                      },
                      'Ownership settings updated.'
                    )
                  }
                  className="px-4 py-2 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 text-xs font-semibold text-[#F4F8F5] cursor-pointer"
                >
                  Save Ownership Lock
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-3">
                  <div className="flex items-center gap-2">
                    <ScrollText className="w-4 h-4 text-[#42D9A0]" />
                    <h3 className="text-xs font-bold text-[#F4F8F5]">
                      Admin Audit Logs ({overview.auditLogs.length})
                    </h3>
                  </div>
                  <div className="divide-y divide-[#19352A] max-h-80 overflow-y-auto text-xs">
                    {overview.auditLogs.map((log) => (
                      <div key={log.logId} className="py-2 space-y-0.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono font-semibold text-[#42D9A0]">
                            {log.action}
                          </span>
                          <span className="text-[10px] text-[#A2B5AA]">
                            {formatIsoTime(log.createdAt)}
                          </span>
                        </div>
                        <div className="text-[11px] text-[#F4F8F5] truncate">{log.target}</div>
                        <div className="text-[11px] text-[#A2B5AA]">{log.details}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-[#12231D] border border-[#19352A] rounded-2xl p-5 space-y-3">
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-amber-400" />
                    <h3 className="text-xs font-bold text-[#F4F8F5]">
                      Server Security Events ({overview.recentSecurityEvents.length})
                    </h3>
                  </div>
                  <div className="divide-y divide-[#19352A] max-h-80 overflow-y-auto text-xs">
                    {overview.recentSecurityEvents.map((ev, i) => (
                      <div key={i} className="py-2 space-y-0.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-mono font-semibold text-amber-300">{ev.event}</span>
                          <span className="text-[10px] text-[#A2B5AA]">
                            {formatIsoTime(ev.timestamp)}
                          </span>
                        </div>
                        <div className="text-[11px] text-[#F4F8F5]">Actor: {ev.actor}</div>
                        <div className="text-[11px] text-[#A2B5AA]">{ev.detail}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
