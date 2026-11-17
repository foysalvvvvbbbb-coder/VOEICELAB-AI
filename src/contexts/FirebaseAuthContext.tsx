import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { onAuthStateChanged, signInWithPopup, signOut, type User } from 'firebase/auth';
import {
  collection,
  deleteDoc,
  doc,
  getDoc,
  getDocs,
  increment,
  limit,
  query,
  serverTimestamp,
  setDoc,
  updateDoc,
  where
} from 'firebase/firestore';
import {
  auth,
  db,
  googleProvider,
  handleFirestoreError,
  OperationType
} from '../firebase';
import {
  ensureSessionInitialized,
  fetchApiJson,
  getCachedSessionTokenHeader,
  registerFirebaseIdTokenProvider,
  resolveApiUrl
} from '../utils/apiClient';
import type {
  CloningLanguageCode,
  SavedVoiceProfile,
  SavedVoiceSourceType
} from '../types/voicelab';

export type WorkspaceRole = 'owner' | 'user' | 'guest';
export type FeatureGateKey = 'voice_cloning' | 'long_script' | 'srt_generator';
export type PackageFeatureKey = FeatureGateKey | 'pro_bundle';
export type PaymentMethodId = 'bkash' | 'nagad';

export interface GovernanceConfigData {
  maintenanceMode: boolean;
  maintenanceNotice: string;
  allowVoiceCloning: boolean;
  allowSrtGeneration: boolean;
  requireAuthForCompute: boolean;
  maxScriptChars: number;
  rateLimitPerMinute: number;
  enforceZeroGeminiQuota: boolean;
}

export interface FeatureFlagConfigData {
  key: FeatureGateKey;
  title: string;
  description: string;
  enabled: boolean;
  isPaid: boolean;
  tokenCost: number;
}

export interface FeaturePackageData {
  id: string;
  featureKey: PackageFeatureKey;
  title: string;
  description: string;
  priceBdt: number;
  durationDays: number;
  enabled: boolean;
}

export interface TokenPackageData {
  id: string;
  title: string;
  description: string;
  tokens: number;
  priceBdt: number;
  enabled: boolean;
}

export interface PaymentSettingsData {
  bkashEnabled: boolean;
  bkashNumber: string;
  bkashAccountType: 'Personal' | 'Merchant' | 'Agent';
  bkashInstructions: string;
  nagadEnabled: boolean;
  nagadNumber: string;
  nagadAccountType: 'Personal' | 'Merchant' | 'Agent';
  nagadInstructions: string;
}

export interface PaymentInstructionsResponse {
  itemType: 'feature_package' | 'token_package';
  itemId: string;
  itemTitle: string;
  amountBdt: number;
  tokensGranted: number;
  durationDays: number;
  paymentMethod: PaymentMethodId;
  accountType: 'Personal' | 'Merchant' | 'Agent';
  receiverNumber: string;
  instructions: string;
}

export interface PaymentRequestData {
  orderId: string;
  userUid: string;
  userEmail: string;
  userName: string;
  itemType: 'feature_package' | 'token_package';
  itemId: string;
  itemTitle: string;
  featureKey?: PackageFeatureKey;
  amountBdt: number;
  tokensGranted: number;
  durationDays: number;
  paymentMethod: PaymentMethodId;
  senderNumber: string;
  trxId: string;
  receiverNumber: string;
  note: string;
  status: 'pending' | 'approved' | 'rejected';
  credited: boolean;
  reviewedByUid: string;
  rejectionReason: string;
  createdAt: string;
  updatedAt: string;
}

export interface UserEntitlementData {
  uid: string;
  email: string;
  tokenBalance: number;
  voiceCloningUnlocked: boolean;
  voiceCloningExpiresAtMs: number;
  longScriptUnlocked: boolean;
  longScriptExpiresAtMs: number;
  srtGeneratorUnlocked: boolean;
  srtGeneratorExpiresAtMs: number;
  proBundleUnlocked: boolean;
  proBundleExpiresAtMs: number;
  updatedAt?: string;
}

export interface TokenLedgerEntryData {
  entryId: string;
  userUid: string;
  userEmail: string;
  delta: number;
  balanceAfter: number;
  type: 'purchase_credit' | 'admin_grant' | 'admin_revoke' | 'feature_debit' | 'refund';
  reason: string;
  actorUid: string;
  createdAt: string;
}

export interface AdminUserRecord {
  uid: string;
  email: string;
  displayName: string;
  role: 'owner' | 'user';
  accountStatus: 'active' | 'suspended';
  createdAt: string;
  lastSeenAt: string;
}

export interface AdminAuditLogRecord {
  logId: string;
  actorUid: string;
  actorEmail: string;
  action: string;
  target: string;
  details: string;
  createdAt: string;
}

export interface AdminOverviewPayload {
  actor: { uid: string; email: string; role: string };
  ownership: {
    primaryOwnerEmail: string;
    primaryOwnerUid: string | null;
    organizationName: string;
    ownershipLocked: boolean;
  };
  governance: GovernanceConfigData;
  features: Record<FeatureGateKey, FeatureFlagConfigData>;
  featurePackages: FeaturePackageData[];
  tokenPackages: TokenPackageData[];
  paymentSettings: PaymentSettingsData;
  users: AdminUserRecord[];
  entitlements: UserEntitlementData[];
  paymentRequests: PaymentRequestData[];
  tokenLedger: TokenLedgerEntryData[];
  auditLogs: AdminAuditLogRecord[];
  runtime: {
    uptimeSeconds: number;
    ffmpegAvailable: boolean;
    ffprobeAvailable: boolean;
    sessionSecretSource: string;
    sessionSecretEntropyBits: number;
    uploadedFileCount: number;
    generatedFileCount: number;
    cachedChunkCount: number;
    totalTempStorageMb: number;
    activeRateBuckets: number;
  };
  recentSecurityEvents: Array<{
    timestamp: string;
    event: string;
    actor: string;
    detail: string;
  }>;
}

export const DEFAULT_GOVERNANCE_CONFIG: GovernanceConfigData = {
  maintenanceMode: false,
  maintenanceNotice: 'VoiceLab AI is currently undergoing scheduled owner maintenance.',
  allowVoiceCloning: true,
  allowSrtGeneration: true,
  requireAuthForCompute: false,
  maxScriptChars: 12000,
  rateLimitPerMinute: 25,
  enforceZeroGeminiQuota: true
};

export const DEFAULT_FEATURE_CONFIGS: Record<FeatureGateKey, FeatureFlagConfigData> = {
  voice_cloning: {
    key: 'voice_cloning',
    title: 'Voice Cloning Studio',
    description: 'Same-language neural voice cloning for Bengali, Hindi, and English.',
    enabled: true,
    isPaid: false,
    tokenCost: 0
  },
  long_script: {
    key: 'long_script',
    title: 'Long-Script Pro (>1,000 Chars)',
    description: 'Auto-chunked synthesis for scripts longer than 1,000 spoken characters.',
    enabled: true,
    isPaid: false,
    tokenCost: 0
  },
  srt_generator: {
    key: 'srt_generator',
    title: 'Audio to SRT Generator',
    description: 'Multilingual audio transcription and SubRip subtitle timing.',
    enabled: true,
    isPaid: false,
    tokenCost: 0
  }
};

interface FirebaseAuthContextValue {
  user: User | null;
  authReady: boolean;
  role: WorkspaceRole;
  isOwner: boolean;
  isSuspended: boolean;
  tokenBalance: number;
  governanceConfig: GovernanceConfigData;
  featureConfigs: Record<FeatureGateKey, FeatureFlagConfigData>;
  featurePackages: FeaturePackageData[];
  tokenPackages: TokenPackageData[];
  availablePaymentMethods: PaymentMethodId[];
  myEntitlement: UserEntitlementData | null;
  myPaymentRequests: PaymentRequestData[];
  myTokenLedger: TokenLedgerEntryData[];
  hasAnyPaidOrTokenFeature: boolean;
  authError: string | null;
  signInWithGoogle: () => Promise<void>;
  signOutUser: () => Promise<void>;
  hasFeatureAccess: (feature: FeatureGateKey) => boolean;
  getFeatureTokenCost: (feature: FeatureGateKey) => number;
  refreshPublicCatalog: () => Promise<void>;
  refreshUserAccount: () => Promise<void>;
  fetchPaymentInstructions: (params: {
    itemType: 'feature_package' | 'token_package';
    itemId: string;
    paymentMethod: PaymentMethodId;
  }) => Promise<PaymentInstructionsResponse>;
  submitPaymentRequest: (params: {
    itemType: 'feature_package' | 'token_package';
    itemId: string;
    paymentMethod: PaymentMethodId;
    senderNumber: string;
    trxId: string;
    note?: string;
  }) => Promise<PaymentRequestData>;
  // Persistent Saved Voice Profiles (Owner-Isolated)
  savedVoices: SavedVoiceProfile[];
  savedVoicesLoading: boolean;
  refreshSavedVoices: () => Promise<void>;
  saveVoiceProfile: (params: {
    referenceFileId: string;
    name?: string;
    language: CloningLanguageCode;
    referenceTranscript?: string;
    modelCheckpoint?: string;
    sourceType?: SavedVoiceSourceType;
    consentConfirmed: boolean;
    deduplicateByPromptHash?: boolean;
  }) => Promise<{ voice: SavedVoiceProfile; deduplicated?: boolean }>;
  renameSavedVoiceProfile: (voiceId: string, nextName: string) => Promise<SavedVoiceProfile>;
  restoreSavedVoiceConditioning: (
    voiceId: string,
    referenceFileId: string
  ) => Promise<{ voice: SavedVoiceProfile; cloudSynced?: boolean }>;
  deleteSavedVoiceProfile: (voiceId: string) => Promise<void>;
  recordSavedVoiceUsage: (voiceId: string) => Promise<void>;
  fetchSavedVoicePreviewBlobUrl: (voiceId: string) => Promise<string>;

  // Owner-only operations (never exposed to ordinary users)
  fetchAdminOverview: () => Promise<AdminOverviewPayload>;
  saveAdminGovernance: (next: GovernanceConfigData) => Promise<void>;
  saveAdminFeatures: (next: Record<FeatureGateKey, FeatureFlagConfigData>) => Promise<void>;
  saveAdminPlans: (params: {
    featurePackages: FeaturePackageData[];
    tokenPackages: TokenPackageData[];
  }) => Promise<void>;
  saveAdminPaymentSettings: (next: PaymentSettingsData) => Promise<void>;
  reviewAdminPaymentRequest: (params: {
    orderId: string;
    decision: 'approved' | 'rejected';
    rejectionReason?: string;
  }) => Promise<void>;
  adjustAdminUserTokens: (params: {
    targetUid: string;
    targetEmail: string;
    delta: number;
    reason?: string;
  }) => Promise<void>;
  grantAdminEntitlement: (params: {
    uid: string;
    email: string;
    voiceCloningUnlocked: boolean;
    longScriptUnlocked: boolean;
    srtGeneratorUnlocked: boolean;
    proBundleUnlocked: boolean;
    durationDays: number;
  }) => Promise<void>;
  updateAdminUserStatus: (params: {
    targetUid: string;
    accountStatus: 'active' | 'suspended';
  }) => Promise<void>;
  saveAdminOwnership: (params: {
    organizationName: string;
    ownershipLocked: boolean;
  }) => Promise<void>;
  purgeAdminStorage: () => Promise<number>;
}

const FirebaseAuthContext = createContext<FirebaseAuthContextValue | undefined>(undefined);

function isEntitlementTimestampActive(
  unlocked: boolean | undefined,
  expiresAtMs: number | undefined
): boolean {
  if (!unlocked) return false;
  if (!expiresAtMs || expiresAtMs === 0) return true;
  return expiresAtMs > Date.now();
}

export const FirebaseAuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [authReady, setAuthReady] = useState(false);
  const [role, setRole] = useState<WorkspaceRole>('guest');
  const [isSuspended, setIsSuspended] = useState(false);
  const [tokenBalance, setTokenBalance] = useState(0);
  const [governanceConfig, setGovernanceConfig] = useState<GovernanceConfigData>(
    DEFAULT_GOVERNANCE_CONFIG
  );
  const [featureConfigs, setFeatureConfigs] =
    useState<Record<FeatureGateKey, FeatureFlagConfigData>>(DEFAULT_FEATURE_CONFIGS);
  const [featurePackages, setFeaturePackages] = useState<FeaturePackageData[]>([]);
  const [tokenPackages, setTokenPackages] = useState<TokenPackageData[]>([]);
  const [availablePaymentMethods, setAvailablePaymentMethods] = useState<PaymentMethodId[]>([]);
  const [myEntitlement, setMyEntitlement] = useState<UserEntitlementData | null>(null);
  const [myPaymentRequests, setMyPaymentRequests] = useState<PaymentRequestData[]>([]);
  const [myTokenLedger, setMyTokenLedger] = useState<TokenLedgerEntryData[]>([]);
  const [savedVoices, setSavedVoices] = useState<SavedVoiceProfile[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  const refreshPublicCatalog = useCallback(async () => {
    try {
      const data = await fetchApiJson<{
        governance?: GovernanceConfigData;
        features?: Record<FeatureGateKey, FeatureFlagConfigData>;
        featurePackages?: FeaturePackageData[];
        tokenPackages?: TokenPackageData[];
        availablePaymentMethods?: PaymentMethodId[];
      }>('/api/catalog/public');
      if (data?.governance) {
        setGovernanceConfig((prev) => ({ ...prev, ...data.governance }));
      }
      if (data?.features) {
        setFeatureConfigs(data.features);
      }
      if (Array.isArray(data?.featurePackages)) {
        setFeaturePackages(data.featurePackages);
      }
      if (Array.isArray(data?.tokenPackages)) {
        setTokenPackages(data.tokenPackages);
      }
      if (Array.isArray(data?.availablePaymentMethods)) {
        setAvailablePaymentMethods(data.availablePaymentMethods);
      }
    } catch {
      // Keep current catalog state cleanly
    }
  }, []);

  const refreshUserAccount = useCallback(async () => {
    const current = auth.currentUser;
    if (!current || !current.emailVerified) {
      setRole(current ? 'user' : 'guest');
      setIsSuspended(false);
      setTokenBalance(0);
      setMyEntitlement(null);
      setMyPaymentRequests([]);
      setMyTokenLedger([]);
      return;
    }

    const me = await fetchApiJson<{
      authenticated: boolean;
      uid: string;
      email: string;
      isOwner: boolean;
      role: 'owner' | 'user';
      accountStatus: 'active' | 'suspended';
      entitlement: UserEntitlementData;
      tokenBalance: number;
      tokenLedger: TokenLedgerEntryData[];
      paymentRequests: PaymentRequestData[];
    }>('/api/auth/me');

    // Role and ownership are verified exclusively by the backend API
    setRole(me.isOwner ? 'owner' : 'user');
    setIsSuspended(me.accountStatus === 'suspended');
    setTokenBalance(Number(me.tokenBalance) || 0);
    setMyEntitlement(me.entitlement || null);
    setMyTokenLedger(Array.isArray(me.tokenLedger) ? me.tokenLedger : []);
    setMyPaymentRequests(Array.isArray(me.paymentRequests) ? me.paymentRequests : []);
  }, []);

  useEffect(() => {
    registerFirebaseIdTokenProvider(async () => {
      if (!auth.currentUser) return null;
      return auth.currentUser.getIdToken();
    });
    void refreshPublicCatalog();
    return () => registerFirebaseIdTokenProvider(null);
  }, [refreshPublicCatalog]);

  const mirrorSavedVoiceToFirestore = useCallback(async (v: SavedVoiceProfile) => {
    if (!auth.currentUser || !auth.currentUser.emailVerified) return;
    if (v.ownerUid !== auth.currentUser.uid) return;
    const docPath = `savedVoices/${v.voiceId}`;
    const docRef = doc(db, 'savedVoices', v.voiceId);
    try {
      const existingSnap = await getDoc(docRef);
      if (existingSnap.exists()) {
        return; // Server already persisted metadata & chunks cleanly
      }
    } catch {
      // Proceed to creation if existence pre-check could not read
    }

    const safeEmail = (auth.currentUser.email || v.ownerEmail || '').trim().slice(0, 254);
    if (!safeEmail || safeEmail.length < 5) return;

    const hex64 = /^[a-f0-9]{64}$/;
    const fallbackSha = '0'.repeat(64);
    const speakerSha = hex64.test(String(v.speakerConditioningSha256 || ''))
      ? v.speakerConditioningSha256
      : fallbackSha;
    const promptSha = hex64.test(String(v.promptConditioningSha256 || ''))
      ? v.promptConditioningSha256
      : fallbackSha;
    const cleanStorageKey = /^[a-zA-Z0-9_\-/]{5,200}$/.test(String(v.storageKey || ''))
      ? v.storageKey
      : `${v.ownerUid}/${v.voiceId}`.slice(0, 200);
    const allowedGrades = ['Studio Grade', 'Good', 'Acceptable (Noisy)', 'Poor / Corrupted'];
    const qualityGrade = allowedGrades.includes(v.qualityGrade) ? v.qualityGrade : 'Good';
    const allowedSources = ['cloned_output', 'reference_upload', 'microphone', 'studio_sample'];
    const sourceType = allowedSources.includes(v.sourceType) ? v.sourceType : 'reference_upload';

    try {
      const refDur = Math.max(0.1, Math.min(1800, Number(v.referenceDurationSec) || 1));
      const promptDur = Math.max(0.1, Math.min(60, Number(v.promptDurationSec) || 1));
      await setDoc(docRef, {
        voiceId: v.voiceId,
        ownerUid: v.ownerUid,
        ownerEmail: safeEmail,
        name: (String(v.name || 'Saved Voice').trim() || 'Saved Voice').slice(0, 80),
        language: ['bn', 'hi', 'en'].includes(v.language) ? v.language : 'bn',
        languageName: (String(v.languageName || v.language).trim() || 'Voice').slice(0, 64),
        conditioningMethod: (String(v.conditioningMethod || 'zero-shot-canonical-wav-v1').trim() || 'zero-shot-canonical-wav-v1').slice(0, 80),
        modelCheckpoint: (String(v.modelCheckpoint || 'Mode 2 Zero-Shot Neural').trim() || 'Mode 2 Zero-Shot Neural').slice(0, 160),
        speakerConditioningSha256: speakerSha,
        promptConditioningSha256: promptSha,
        referenceTranscript: String(v.referenceTranscript || '').slice(0, 1000),
        referenceDurationSec: refDur,
        promptDurationSec: promptDur,
        fundamentalFreqHz: Math.max(0, Math.min(1000, Number(v.fundamentalFreqHz) || 0)),
        pitchMinHz: Math.max(0, Math.min(1000, Number(v.pitchMinHz) || 0)),
        pitchMaxHz: Math.max(0, Math.min(2000, Number(v.pitchMaxHz) || 0)),
        spectralWarmthRatio: Math.max(0, Math.min(1000, Number(v.spectralWarmthRatio) || 0)),
        estimatedSnrDb: Math.max(-100, Math.min(200, Number(v.estimatedSnrDb) || 0)),
        qualityGrade,
        sourceType,
        sourceFileName: (String(v.sourceFileName || `${v.voiceId}.wav`).trim() || `${v.voiceId}.wav`).slice(0, 160),
        storageKey: cleanStorageKey,
        consentConfirmed: true,
        usageCount: 0,
        lastUsedAt: serverTimestamp(),
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp()
      });
    } catch (err) {
      handleFirestoreError(err, OperationType.CREATE, docPath);
    }
  }, []);

  const refreshSavedVoices = useCallback(async () => {
    const current = auth.currentUser;
    if (!current || !current.emailVerified) {
      setSavedVoices([]);
      return;
    }

    setSavedVoicesLoading(true);
    try {
      const toIso = (val: any, fallback: string): string => {
        if (val && typeof val.toDate === 'function') {
          return val.toDate().toISOString();
        }
        if (typeof val === 'string' && val.trim()) return val;
        return fallback;
      };

      let serverFetchSucceeded = false;
      let firestoreFetchSucceeded = false;

      const [serverResult, firestoreDocs] = await Promise.all([
        fetchApiJson<{ voices: SavedVoiceProfile[] }>('/api/voices')
          .then((res) => {
            serverFetchSucceeded = true;
            return res;
          })
          .catch(() => ({
            voices: [] as SavedVoiceProfile[]
          })),
        (async () => {
          try {
            const snap = await getDocs(
              query(collection(db, 'savedVoices'), where('ownerUid', '==', current.uid))
            );
            firestoreFetchSucceeded = true;
            return snap.docs.map((d) => {
              const raw = d.data() as any;
              const nowIso = new Date().toISOString();
              return {
                voiceId: String(raw.voiceId || d.id),
                ownerUid: String(raw.ownerUid || current.uid),
                ownerEmail: String(raw.ownerEmail || current.email || ''),
                name: String(raw.name || 'Saved Voice'),
                language: raw.language as CloningLanguageCode,
                languageName: String(raw.languageName || raw.language || ''),
                conditioningMethod: String(raw.conditioningMethod || 'zero-shot-canonical-wav-v1'),
                modelCheckpoint: String(raw.modelCheckpoint || ''),
                speakerConditioningSha256: String(raw.speakerConditioningSha256 || ''),
                promptConditioningSha256: String(raw.promptConditioningSha256 || ''),
                referenceTranscript: String(raw.referenceTranscript || ''),
                referenceDurationSec: Number(raw.referenceDurationSec) || 0,
                promptDurationSec: Number(raw.promptDurationSec) || 0,
                fundamentalFreqHz: Number(raw.fundamentalFreqHz) || 0,
                pitchMinHz: Number(raw.pitchMinHz) || 0,
                pitchMaxHz: Number(raw.pitchMaxHz) || 0,
                spectralWarmthRatio: Number(raw.spectralWarmthRatio) || 0,
                estimatedSnrDb: Number(raw.estimatedSnrDb) || 0,
                qualityGrade: raw.qualityGrade || 'Good',
                sourceType: raw.sourceType || 'reference_upload',
                sourceFileName: String(raw.sourceFileName || ''),
                storageKey: String(raw.storageKey || ''),
                consentConfirmed: Boolean(raw.consentConfirmed),
                usageCount: Number(raw.usageCount) || 0,
                hasConditioningAudio: false,
                lastUsedAt: toIso(raw.lastUsedAt, nowIso),
                createdAt: toIso(raw.createdAt, nowIso),
                updatedAt: toIso(raw.updatedAt, nowIso)
              } as SavedVoiceProfile;
            });
          } catch (fsErr) {
            console.warn('Firestore savedVoices query notice:', fsErr);
            return [] as SavedVoiceProfile[];
          }
        })()
      ]);

      // Do not clear already-loaded voices if both backend and Firestore had a transient hiccup
      if (!serverFetchSucceeded && !firestoreFetchSucceeded) {
        return;
      }

      const merged = new Map<string, SavedVoiceProfile>();
      const firestoreIds = new Set<string>();
      const serverIds = new Set((serverResult.voices || []).map((v) => v.voiceId));

      await Promise.all(
        firestoreDocs.map(async (fsVoice) => {
          firestoreIds.add(fsVoice.voiceId);
          if (!serverIds.has(fsVoice.voiceId)) {
            try {
              const chunkSnap = await getDocs(
                query(
                  collection(db, 'savedVoices', fsVoice.voiceId, 'conditioningChunks'),
                  where('ownerUid', '==', current.uid),
                  limit(1)
                )
              );
              fsVoice.hasConditioningAudio = !chunkSnap.empty;
            } catch {
              // Keep false if cloud chunks check failed
            }
          }
          merged.set(fsVoice.voiceId, fsVoice);
        })
      );

      for (const srvVoice of serverResult.voices || []) {
        merged.set(srvVoice.voiceId, srvVoice);
        if (!firestoreIds.has(srvVoice.voiceId)) {
          void mirrorSavedVoiceToFirestore(srvVoice).catch(() => {});
        }
      }

      const sorted = Array.from(merged.values()).sort(
        (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      );
      setSavedVoices(sorted);
    } finally {
      setSavedVoicesLoading(false);
    }
  }, [mirrorSavedVoiceToFirestore]);

  useEffect(() => {
    const unsubscribeAuth = onAuthStateChanged(auth, async (nextUser) => {
      setUser(nextUser);
      setAuthError(null);

      if (!nextUser) {
        setRole('guest');
        setIsSuspended(false);
        setTokenBalance(0);
        setMyEntitlement(null);
        setMyPaymentRequests([]);
        setMyTokenLedger([]);
        setSavedVoices([]);
        setAuthReady(true);
        return;
      }

      if (!nextUser.emailVerified || !nextUser.email) {
        setRole('user');
        setIsSuspended(false);
        setTokenBalance(0);
        setMyEntitlement(null);
        setMyPaymentRequests([]);
        setMyTokenLedger([]);
        setSavedVoices([]);
        setAuthReady(true);
        return;
      }

      try {
        await refreshUserAccount();

        // Sync non-privileged user profile record in Firestore
        const safeName = (nextUser.displayName || nextUser.email.split('@')[0] || 'User').slice(0, 120);
        const safePhoto = (nextUser.photoURL || '').slice(0, 512);
        const userRef = doc(db, 'users', nextUser.uid);
        try {
          const snap = await getDoc(userRef);
          if (!snap.exists()) {
            await setDoc(userRef, {
              uid: nextUser.uid,
              email: nextUser.email,
              displayName: safeName,
              photoURL: safePhoto,
              role: 'user',
              accountStatus: 'active',
              lastSeenAt: serverTimestamp(),
              createdAt: serverTimestamp(),
              updatedAt: serverTimestamp()
            });
          } else {
            await updateDoc(userRef, {
              displayName: safeName,
              photoURL: safePhoto,
              lastSeenAt: serverTimestamp(),
              updatedAt: serverTimestamp()
            });
          }
        } catch (err) {
          // Non-fatal if Firestore quota/rules temporarily restrict profile touch
          console.warn('User profile sync notice:', err);
        }

        void refreshSavedVoices();
        void refreshPublicCatalog();
      } catch (err: any) {
        setAuthError(err?.message || 'Failed to verify user account state.');
        setRole('user');
      } finally {
        setAuthReady(true);
      }
    });

    return () => unsubscribeAuth();
  }, [refreshPublicCatalog, refreshSavedVoices, refreshUserAccount]);

  const signInWithGoogle = useCallback(async () => {
    setAuthError(null);
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (err: any) {
      const msg = err?.message || 'Google Sign-In cancelled or blocked by popup policy.';
      setAuthError(msg);
      throw new Error(msg);
    }
  }, []);

  const signOutUser = useCallback(async () => {
    await signOut(auth);
    setRole('guest');
    setIsSuspended(false);
    setTokenBalance(0);
    setMyEntitlement(null);
    setMyPaymentRequests([]);
    setMyTokenLedger([]);
    setSavedVoices([]);
  }, []);

  const saveVoiceProfile = useCallback(
    async (params: {
      referenceFileId: string;
      name?: string;
      language: CloningLanguageCode;
      referenceTranscript?: string;
      modelCheckpoint?: string;
      sourceType?: SavedVoiceSourceType;
      consentConfirmed: boolean;
      deduplicateByPromptHash?: boolean;
    }): Promise<{ voice: SavedVoiceProfile; deduplicated?: boolean }> => {
      const resp = await fetchApiJson<{
        voice: SavedVoiceProfile;
        deduplicated?: boolean;
        cloudSynced?: boolean;
      }>('/api/voices', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      if (resp.voice) {
        if (!resp.deduplicated) {
          await mirrorSavedVoiceToFirestore(resp.voice);
        } else {
          try {
            const ref = doc(db, 'savedVoices', resp.voice.voiceId);
            const snap = await getDoc(ref);
            if (!snap.exists()) {
              await mirrorSavedVoiceToFirestore(resp.voice);
            } else {
              await updateDoc(ref, {
                usageCount: increment(1),
                lastUsedAt: serverTimestamp(),
                updatedAt: serverTimestamp()
              });
            }
          } catch {
            // Non-fatal if Firestore usage increment already happened server-side
          }
        }
      }
      await refreshSavedVoices();
      return resp;
    },
    [mirrorSavedVoiceToFirestore, refreshSavedVoices]
  );

  const renameSavedVoiceProfile = useCallback(
    async (voiceId: string, nextName: string): Promise<SavedVoiceProfile> => {
      const cleanName = nextName.trim().slice(0, 80);
      if (!cleanName) {
        throw new Error('Voice profile name cannot be empty.');
      }
      const resp = await fetchApiJson<{ voice: SavedVoiceProfile }>(
        `/api/voices/${encodeURIComponent(voiceId)}`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: cleanName })
        }
      );
      if (auth.currentUser && auth.currentUser.emailVerified) {
        try {
          const ref = doc(db, 'savedVoices', voiceId);
          const snap = await getDoc(ref);
          if (snap.exists() && snap.data()?.name !== cleanName) {
            await updateDoc(ref, {
              name: cleanName,
              updatedAt: serverTimestamp()
            });
          }
        } catch (err) {
          handleFirestoreError(err, OperationType.UPDATE, `savedVoices/${voiceId}`);
        }
      }
      await refreshSavedVoices();
      return resp.voice;
    },
    [refreshSavedVoices]
  );

  const restoreSavedVoiceConditioning = useCallback(
    async (
      voiceId: string,
      referenceFileId: string
    ): Promise<{ voice: SavedVoiceProfile; cloudSynced?: boolean }> => {
      const resp = await fetchApiJson<{
        voice: SavedVoiceProfile;
        cloudSynced?: boolean;
        cloudSyncWarning?: string;
      }>(`/api/voices/${encodeURIComponent(voiceId)}/restore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ referenceFileId })
      });
      await refreshSavedVoices();
      return { voice: resp.voice, cloudSynced: resp.cloudSynced };
    },
    [refreshSavedVoices]
  );

  const deleteSavedVoiceProfile = useCallback(
    async (voiceId: string): Promise<void> => {
      await fetchApiJson(`/api/voices/${encodeURIComponent(voiceId)}`, {
        method: 'DELETE'
      });
      if (auth.currentUser && auth.currentUser.emailVerified) {
        try {
          const ref = doc(db, 'savedVoices', voiceId);
          const snap = await getDoc(ref);
          if (snap.exists()) {
            await deleteDoc(ref);
          }
        } catch (err) {
          handleFirestoreError(err, OperationType.DELETE, `savedVoices/${voiceId}`);
        }
      }
      setSavedVoices((prev) => prev.filter((v) => v.voiceId !== voiceId));
    },
    []
  );

  const recordSavedVoiceUsage = useCallback(
    async (_voiceId: string): Promise<void> => {
      await refreshSavedVoices();
    },
    [refreshSavedVoices]
  );

  const fetchSavedVoicePreviewBlobUrl = useCallback(async (voiceId: string): Promise<string> => {
    await ensureSessionInitialized(false);
    const idToken = auth.currentUser ? await auth.currentUser.getIdToken() : null;
    if (!idToken) {
      throw new Error('Verified Google Sign-In required to preview your saved voice.');
    }
    const headers = new Headers({
      Authorization: `Bearer ${idToken}`,
      'X-Requested-With': 'VoiceLab-Web-Client'
    });
    const sessionTok = getCachedSessionTokenHeader();
    if (sessionTok) {
      headers.set('X-VoiceLab-Session', sessionTok);
    }
    const resp = await fetch(
      resolveApiUrl(`/api/voices/${encodeURIComponent(voiceId)}/preview`),
      {
        method: 'GET',
        headers,
        credentials: 'include',
        cache: 'no-store'
      }
    );
    if (!resp.ok) {
      const errBody = await resp.json().catch(() => null);
      throw new Error(
        errBody?.error || `Unable to load saved voice preview (HTTP ${resp.status}).`
      );
    }
    const blob = await resp.blob();
    return URL.createObjectURL(blob);
  }, []);

  const fetchPaymentInstructions = useCallback(
    async (params: {
      itemType: 'feature_package' | 'token_package';
      itemId: string;
      paymentMethod: PaymentMethodId;
    }): Promise<PaymentInstructionsResponse> => {
      return fetchApiJson<PaymentInstructionsResponse>('/api/payments/instructions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
    },
    []
  );

  const submitPaymentRequest = useCallback(
    async (params: {
      itemType: 'feature_package' | 'token_package';
      itemId: string;
      paymentMethod: PaymentMethodId;
      senderNumber: string;
      trxId: string;
      note?: string;
    }): Promise<PaymentRequestData> => {
      const res = await fetchApiJson<{ ok: boolean; request: PaymentRequestData }>(
        '/api/payments/submit',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params)
        }
      );

      // Mirror pending request into Firestore cleanly
      const reqDoc = res.request;
      if (auth.currentUser && reqDoc) {
        try {
          await setDoc(doc(db, 'paymentRequests', reqDoc.orderId), {
            orderId: reqDoc.orderId,
            userUid: reqDoc.userUid,
            userEmail: reqDoc.userEmail,
            userName: reqDoc.userName,
            itemType: reqDoc.itemType,
            itemId: reqDoc.itemId,
            itemTitle: reqDoc.itemTitle,
            amountBdt: reqDoc.amountBdt,
            tokensGranted: reqDoc.tokensGranted,
            durationDays: reqDoc.durationDays,
            paymentMethod: reqDoc.paymentMethod,
            senderNumber: reqDoc.senderNumber,
            trxId: reqDoc.trxId,
            receiverNumber: reqDoc.receiverNumber,
            note: reqDoc.note,
            status: 'pending',
            reviewedByUid: '',
            rejectionReason: '',
            createdAt: serverTimestamp(),
            updatedAt: serverTimestamp()
          });
        } catch (err) {
          handleFirestoreError(err, OperationType.CREATE, `paymentRequests/${reqDoc.orderId}`);
        }
      }

      await refreshUserAccount();
      return reqDoc;
    },
    [refreshUserAccount]
  );

  // Owner-Only Admin API Calls
  const fetchAdminOverview = useCallback(async () => {
    return fetchApiJson<AdminOverviewPayload>('/api/admin/overview');
  }, []);

  const saveAdminGovernance = useCallback(
    async (next: GovernanceConfigData) => {
      await fetchApiJson('/api/admin/governance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(next)
      });
      await refreshPublicCatalog();
    },
    [refreshPublicCatalog]
  );

  const saveAdminFeatures = useCallback(
    async (next: Record<FeatureGateKey, FeatureFlagConfigData>) => {
      await fetchApiJson('/api/admin/features', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(next)
      });
      await refreshPublicCatalog();
    },
    [refreshPublicCatalog]
  );

  const saveAdminPlans = useCallback(
    async (params: { featurePackages: FeaturePackageData[]; tokenPackages: TokenPackageData[] }) => {
      await fetchApiJson('/api/admin/plans', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      await refreshPublicCatalog();
    },
    [refreshPublicCatalog]
  );

  const saveAdminPaymentSettings = useCallback(
    async (next: PaymentSettingsData) => {
      await fetchApiJson('/api/admin/payment-settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(next)
      });
      await refreshPublicCatalog();
    },
    [refreshPublicCatalog]
  );

  const reviewAdminPaymentRequest = useCallback(
    async (params: {
      orderId: string;
      decision: 'approved' | 'rejected';
      rejectionReason?: string;
    }) => {
      await fetchApiJson('/api/admin/payments/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      await refreshUserAccount();
    },
    [refreshUserAccount]
  );

  const adjustAdminUserTokens = useCallback(
    async (params: {
      targetUid: string;
      targetEmail: string;
      delta: number;
      reason?: string;
    }) => {
      await fetchApiJson('/api/admin/tokens/adjust', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      await refreshUserAccount();
    },
    [refreshUserAccount]
  );

  const grantAdminEntitlement = useCallback(
    async (params: {
      uid: string;
      email: string;
      voiceCloningUnlocked: boolean;
      longScriptUnlocked: boolean;
      srtGeneratorUnlocked: boolean;
      proBundleUnlocked: boolean;
      durationDays: number;
    }) => {
      await fetchApiJson('/api/admin/entitlements/grant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      await refreshUserAccount();
    },
    [refreshUserAccount]
  );

  const updateAdminUserStatus = useCallback(
    async (params: { targetUid: string; accountStatus: 'active' | 'suspended' }) => {
      await fetchApiJson('/api/admin/users/status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
    },
    []
  );

  const saveAdminOwnership = useCallback(
    async (params: { organizationName: string; ownershipLocked: boolean }) => {
      await fetchApiJson('/api/admin/ownership', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
    },
    []
  );

  const purgeAdminStorage = useCallback(async (): Promise<number> => {
    const res = await fetchApiJson<{ ok: boolean; deletedFilesCount: number }>(
      '/api/admin/storage/purge',
      { method: 'POST' }
    );
    return res.deletedFilesCount;
  }, []);

  const hasFeatureAccess = useCallback(
    (feature: FeatureGateKey): boolean => {
      const cfg = featureConfigs[feature];
      if (role === 'owner') return true;
      if (!cfg?.enabled) return false;
      if (!cfg?.isPaid) return true;
      if (!user || !myEntitlement) return false;
      if (
        isEntitlementTimestampActive(
          myEntitlement.proBundleUnlocked,
          myEntitlement.proBundleExpiresAtMs
        )
      ) {
        return true;
      }
      if (feature === 'voice_cloning') {
        return isEntitlementTimestampActive(
          myEntitlement.voiceCloningUnlocked,
          myEntitlement.voiceCloningExpiresAtMs
        );
      }
      if (feature === 'long_script') {
        return isEntitlementTimestampActive(
          myEntitlement.longScriptUnlocked,
          myEntitlement.longScriptExpiresAtMs
        );
      }
      return isEntitlementTimestampActive(
        myEntitlement.srtGeneratorUnlocked,
        myEntitlement.srtGeneratorExpiresAtMs
      );
    },
    [featureConfigs, myEntitlement, role, user]
  );

  const getFeatureTokenCost = useCallback(
    (feature: FeatureGateKey): number => {
      if (role === 'owner') return 0;
      return Math.max(0, Number(featureConfigs[feature]?.tokenCost) || 0);
    },
    [featureConfigs, role]
  );

  const hasAnyPaidOrTokenFeature = useMemo(() => {
    return Object.values(featureConfigs).some(
      (f) => f.isPaid || f.tokenCost > 0 || !f.enabled
    );
  }, [featureConfigs]);

  const isOwner = role === 'owner';

  return (
    <FirebaseAuthContext.Provider
      value={{
        user,
        authReady,
        role,
        isOwner,
        isSuspended,
        tokenBalance,
        governanceConfig,
        featureConfigs,
        featurePackages,
        tokenPackages,
        availablePaymentMethods,
        myEntitlement,
        myPaymentRequests,
        myTokenLedger,
        hasAnyPaidOrTokenFeature,
        authError,
        signInWithGoogle,
        signOutUser,
        hasFeatureAccess,
        getFeatureTokenCost,
        refreshPublicCatalog,
        refreshUserAccount,
        fetchPaymentInstructions,
        submitPaymentRequest,
        savedVoices,
        savedVoicesLoading,
        refreshSavedVoices,
        saveVoiceProfile,
        renameSavedVoiceProfile,
        restoreSavedVoiceConditioning,
        deleteSavedVoiceProfile,
        recordSavedVoiceUsage,
        fetchSavedVoicePreviewBlobUrl,
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
      }}
    >
      {children}
    </FirebaseAuthContext.Provider>
  );
};

export function useFirebaseAuth(): FirebaseAuthContextValue {
  const ctx = useContext(FirebaseAuthContext);
  if (!ctx) {
    throw new Error('useFirebaseAuth must be used inside FirebaseAuthProvider');
  }
  return ctx;
}
