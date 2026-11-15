import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Clock,
  Coins,
  Copy,
  Lock,
  Smartphone,
  Sparkles,
  Unlock,
  X,
  XCircle
} from 'lucide-react';
import {
  useFirebaseAuth,
  type PackageFeatureKey,
  type PaymentInstructionsResponse,
  type PaymentMethodId
} from '../contexts/FirebaseAuthContext';

interface UpgradeAccessModalProps {
  isOpen: boolean;
  initialFeature?: PackageFeatureKey;
  onClose: () => void;
}

const GoogleIcon: React.FC = () => (
  <svg viewBox="0 0 24 24" className="w-4 h-4 shrink-0" aria-hidden="true">
    <path
      fill="#4285F4"
      d="M23.745 12.27c0-.7-.06-1.4-.19-2.07H12v4.51h6.6c-.29 1.52-1.14 2.82-2.4 3.68v3.05h3.88c2.27-2.09 3.665-5.17 3.665-9.17Z"
    />
    <path
      fill="#34A853"
      d="M12 24c3.24 0 5.95-1.08 7.93-2.91l-3.88-3.05c-1.08.72-2.45 1.16-4.05 1.16-3.12 0-5.77-2.11-6.72-4.96H1.29v3.14C3.26 21.3 7.31 24 12 24Z"
    />
    <path
      fill="#FBBC05"
      d="M5.28 14.24c-.24-.72-.38-1.49-.38-2.24s.14-1.52.38-2.24V6.62H1.29C.47 8.24 0 10.06 0 12s.47 3.76 1.29 5.38l3.99-3.14Z"
    />
    <path
      fill="#EA4335"
      d="M12 4.75c1.77 0 3.35.61 4.6 1.8l3.42-3.42C17.95 1.19 15.24 0 12 0 7.31 0 3.26 2.7 1.29 6.62l3.99 3.14c.95-2.85 3.6-4.96 6.72-4.96Z"
    />
  </svg>
);

export const UpgradeAccessModal: React.FC<UpgradeAccessModalProps> = ({
  isOpen,
  initialFeature = 'pro_bundle',
  onClose
}) => {
  const {
    user,
    tokenBalance,
    featureConfigs,
    featurePackages,
    tokenPackages,
    availablePaymentMethods,
    myPaymentRequests,
    myTokenLedger,
    hasFeatureAccess,
    signInWithGoogle,
    fetchPaymentInstructions,
    submitPaymentRequest
  } = useFirebaseAuth();

  const [purchaseTab, setPurchaseTab] = useState<'features' | 'tokens'>('features');
  const [selectedItemId, setSelectedItemId] = useState<string>('');
  const [paymentMethod, setPaymentMethod] = useState<PaymentMethodId>('bkash');
  const [instructionsData, setInstructionsData] = useState<PaymentInstructionsResponse | null>(null);
  const [loadingInstructions, setLoadingInstructions] = useState(false);
  const [senderNumber, setSenderNumber] = useState('');
  const [trxId, setTrxId] = useState('');
  const [note, setNote] = useState('');
  const [copiedNum, setCopiedNum] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [signingIn, setSigningIn] = useState(false);
  const [feedback, setFeedback] = useState<{ type: 'success' | 'error'; text: string } | null>(
    null
  );

  // Sync default item selection on modal open
  useEffect(() => {
    if (!isOpen) return;
    setFeedback(null);
    setInstructionsData(null);

    if (purchaseTab === 'features') {
      const matched =
        featurePackages.find((p) => p.featureKey === initialFeature) || featurePackages[0];
      if (matched) setSelectedItemId(matched.id);
    } else if (tokenPackages[0]) {
      setSelectedItemId(tokenPackages[0].id);
    }
  }, [isOpen, initialFeature, purchaseTab, featurePackages, tokenPackages]);

  useEffect(() => {
    if (availablePaymentMethods.length > 0 && !availablePaymentMethods.includes(paymentMethod)) {
      setPaymentMethod(availablePaymentMethods[0]);
    }
  }, [availablePaymentMethods, paymentMethod]);

  // Fetch configured payment instructions strictly for the chosen item & method when authenticated
  useEffect(() => {
    if (!isOpen || !user || !selectedItemId || availablePaymentMethods.length === 0) {
      setInstructionsData(null);
      return;
    }
    let cancelled = false;
    setLoadingInstructions(true);
    fetchPaymentInstructions({
      itemType: purchaseTab === 'features' ? 'feature_package' : 'token_package',
      itemId: selectedItemId,
      paymentMethod
    })
      .then((res) => {
        if (!cancelled) {
          setInstructionsData(res);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setInstructionsData(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingInstructions(false);
      });
    return () => {
      cancelled = true;
    };
  }, [
    isOpen,
    user,
    selectedItemId,
    purchaseTab,
    paymentMethod,
    availablePaymentMethods,
    fetchPaymentInstructions
  ]);

  if (!isOpen) return null;

  const isFeatureUnlocked = (fk: PackageFeatureKey) => {
    if (fk === 'pro_bundle') {
      return (
        hasFeatureAccess('voice_cloning') &&
        hasFeatureAccess('long_script') &&
        hasFeatureAccess('srt_generator')
      );
    }
    return hasFeatureAccess(fk);
  };

  const handleCopyNumber = async () => {
    if (!instructionsData?.receiverNumber) return;
    try {
      await navigator.clipboard.writeText(instructionsData.receiverNumber);
      setCopiedNum(true);
      setTimeout(() => setCopiedNum(false), 2000);
    } catch {
      // Ignore clipboard failure
    }
  };

  const handleGoogleContinue = async () => {
    setFeedback(null);
    setSigningIn(true);
    try {
      await signInWithGoogle();
    } catch (err: any) {
      setFeedback({ type: 'error', text: err?.message || 'Google Sign-In failed.' });
    } finally {
      setSigningIn(false);
    }
  };

  const handleSubmitOrder = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedItemId) return;
    setFeedback(null);
    setSubmitting(true);
    try {
      const req = await submitPaymentRequest({
        itemType: purchaseTab === 'features' ? 'feature_package' : 'token_package',
        itemId: selectedItemId,
        paymentMethod,
        senderNumber,
        trxId,
        note
      });
      setSenderNumber('');
      setTrxId('');
      setNote('');
      setFeedback({
        type: 'success',
        text: `আপনার ${paymentMethod === 'bkash' ? 'bKash' : 'Nagad'} পেমেন্ট রিকোয়েস্ট (${req.trxId}) পেন্ডিং হিসেবে জমা হয়েছে। ভেরিফিকেশন ও অনুমোদনের পর অটোমেটিক অ্যাক্সেস/টোকেন যুক্ত হবে।`
      });
    } catch (err: any) {
      setFeedback({
        type: 'error',
        text: err?.message || 'পেমেন্ট রিকোয়েস্ট জমা দেওয়া যায়নি।'
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-3 sm:p-5 overflow-y-auto">
      <div className="w-full max-w-3xl bg-[#0B1512] border border-[#19352A] rounded-2xl shadow-2xl overflow-hidden my-auto max-h-[92vh] flex flex-col">
        {/* Header */}
        <div className="px-5 py-4 bg-[#12231D] border-b border-[#19352A] flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-9 h-9 rounded-xl bg-[#19352A] border border-[#16A36A]/50 flex items-center justify-center shrink-0">
              <Sparkles className="w-5 h-5 text-[#42D9A0]" />
            </div>
            <div className="min-w-0">
              <h2 className="text-sm sm:text-base font-bold text-[#F4F8F5] truncate">
                Upgrade &amp; Request Access
              </h2>
              <p className="text-[11px] text-[#A2B5AA] truncate">
                Select a feature plan or token pack and submit payment verification
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close upgrade modal"
            className="p-1.5 rounded-lg bg-[#0B1512] hover:bg-[#19352A] text-[#A2B5AA] hover:text-[#F4F8F5] cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-5 overflow-y-auto">
          {/* Account & Token Summary */}
          {!user ? (
            <div className="rounded-xl bg-[#12231D] border border-[#16A36A]/50 p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div className="space-y-0.5">
                <div className="text-xs font-bold text-[#F4F8F5]">
                  Sign in with Google to continue
                </div>
                <p className="text-[11px] text-[#A2B5AA]">
                  Your unlocked features and token balance stay bound to your verified Google account.
                </p>
              </div>
              <button
                type="button"
                onClick={handleGoogleContinue}
                disabled={signingIn}
                className="px-4 py-2 rounded-xl bg-white hover:bg-gray-100 text-gray-900 font-semibold text-xs flex items-center justify-center gap-2 shrink-0 cursor-pointer shadow-sm"
              >
                <GoogleIcon />
                <span>{signingIn ? 'Signing in...' : 'Continue with Google'}</span>
              </button>
            </div>
          ) : (
            <div className="rounded-xl bg-[#12231D] border border-[#19352A] px-4 py-3 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-2 min-w-0">
                <CheckCircle2 className="w-4 h-4 text-[#42D9A0] shrink-0" />
                <span className="font-mono text-[#F4F8F5] truncate">{user.email}</span>
              </div>
              <div className="px-2.5 py-1 rounded-lg bg-[#0B1512] border border-[#19352A] flex items-center gap-1.5 font-mono text-[#42D9A0]">
                <Coins className="w-3.5 h-3.5" />
                <span>{tokenBalance} Tokens</span>
              </div>
            </div>
          )}

          {/* Switch between Feature Packages & Token Packages */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setPurchaseTab('features')}
              className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition-colors cursor-pointer ${
                purchaseTab === 'features'
                  ? 'bg-[#19352A] text-[#42D9A0] border border-[#16A36A]/60'
                  : 'bg-[#12231D] text-[#A2B5AA] border border-[#19352A]'
              }`}
            >
              Feature Plans ({featurePackages.length})
            </button>
            <button
              type="button"
              onClick={() => setPurchaseTab('tokens')}
              className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition-colors cursor-pointer flex items-center gap-1.5 ${
                purchaseTab === 'tokens'
                  ? 'bg-[#19352A] text-[#42D9A0] border border-[#16A36A]/60'
                  : 'bg-[#12231D] text-[#A2B5AA] border border-[#19352A]'
              }`}
            >
              <Coins className="w-3.5 h-3.5" />
              <span>Token Packages ({tokenPackages.length})</span>
            </button>
          </div>

          {/* Step 2: Package Grid */}
          {purchaseTab === 'features' ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {featurePackages.map((pkg) => {
                const isSelected = selectedItemId === pkg.id;
                const unlocked = isFeatureUnlocked(pkg.featureKey);
                const featurePaid =
                  pkg.featureKey === 'pro_bundle'
                    ? featureConfigs.voice_cloning.isPaid ||
                      featureConfigs.long_script.isPaid ||
                      featureConfigs.srt_generator.isPaid
                    : featureConfigs[pkg.featureKey]?.isPaid;
                return (
                  <button
                    key={pkg.id}
                    type="button"
                    onClick={() => setSelectedItemId(pkg.id)}
                    className={`text-left rounded-xl p-3.5 border transition-colors cursor-pointer space-y-1.5 ${
                      isSelected
                        ? 'bg-[#19352A] border-[#42D9A0] shadow-sm'
                        : 'bg-[#12231D] border-[#19352A] hover:border-[#16A36A]/50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-bold text-[#F4F8F5]">{pkg.title}</span>
                      {unlocked && featurePaid ? (
                        <span className="px-2 py-0.5 rounded bg-[#16A36A]/20 text-[#42D9A0] text-[10px] font-semibold flex items-center gap-1">
                          <Unlock className="w-3 h-3" /> Active
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded bg-[#0B1512] border border-[#19352A] text-[#42D9A0] text-[11px] font-bold font-mono flex items-center gap-1">
                          <Lock className="w-3 h-3" /> ৳{pkg.priceBdt}
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-[#A2B5AA] leading-snug">{pkg.description}</p>
                    <div className="text-[10px] font-mono text-[#A2B5AA]">
                      ৳{pkg.priceBdt} BDT · {pkg.durationDays > 0 ? `${pkg.durationDays} Days` : 'Lifetime'}
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {tokenPackages.map((tp) => {
                const isSelected = selectedItemId === tp.id;
                return (
                  <button
                    key={tp.id}
                    type="button"
                    onClick={() => setSelectedItemId(tp.id)}
                    className={`text-left rounded-xl p-3.5 border transition-colors cursor-pointer space-y-1.5 ${
                      isSelected
                        ? 'bg-[#19352A] border-[#42D9A0] shadow-sm'
                        : 'bg-[#12231D] border-[#19352A] hover:border-[#16A36A]/50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-bold text-[#F4F8F5]">{tp.title}</span>
                      <span className="px-2 py-0.5 rounded bg-[#0B1512] border border-[#19352A] text-[#42D9A0] text-[11px] font-bold font-mono">
                        ৳{tp.priceBdt}
                      </span>
                    </div>
                    <p className="text-[11px] text-[#A2B5AA] leading-snug">{tp.description}</p>
                    <div className="text-[10px] font-mono text-[#42D9A0]">+{tp.tokens} Tokens</div>
                  </button>
                );
              })}
            </div>
          )}

          {/* Step 3: Payment Method & Dynamic Instructions */}
          {availablePaymentMethods.length === 0 ? (
            <div className="rounded-xl bg-amber-950/40 border border-amber-700/50 p-4 text-xs text-amber-200 flex items-center gap-2.5">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <span>
                No payment method is currently configured or active. Please try again later.
              </span>
            </div>
          ) : !user ? (
            <div className="rounded-xl bg-[#12231D] border border-[#19352A] p-4 text-xs text-[#A2B5AA]">
              Sign in with Google above to view payment instructions for your selected package and submit your transaction ID.
            </div>
          ) : (
            <form
              onSubmit={handleSubmitOrder}
              className="rounded-2xl bg-[#12231D] border border-[#19352A] p-4 sm:p-5 space-y-4"
            >
              {/* Method selector */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#19352A]">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-[#F4F8F5]">Payment Method:</span>
                  {availablePaymentMethods.map((m) => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setPaymentMethod(m)}
                      className={`px-3 py-1 rounded-lg text-xs font-bold uppercase cursor-pointer transition-colors ${
                        paymentMethod === m
                          ? m === 'bkash'
                            ? 'bg-[#E2136E] text-white'
                            : 'bg-[#F36F21] text-white'
                          : 'bg-[#0B1512] text-[#A2B5AA] border border-[#19352A]'
                      }`}
                    >
                      {m === 'bkash' ? 'bKash' : 'Nagad'}
                    </button>
                  ))}
                </div>

                {instructionsData && (
                  <div className="text-xs font-mono text-[#42D9A0]">
                    Total: ৳{instructionsData.amountBdt} BDT
                  </div>
                )}
              </div>

              {loadingInstructions ? (
                <div className="text-xs text-[#A2B5AA]">Loading payment instructions...</div>
              ) : instructionsData ? (
                <div className="rounded-xl bg-[#0B1512] border border-[#19352A] p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="space-y-1">
                    <div className="text-xs font-bold text-[#F4F8F5] flex items-center gap-2">
                      <Smartphone className="w-3.5 h-3.5 text-[#42D9A0]" />
                      <span>
                        {instructionsData.itemTitle} — ৳{instructionsData.amountBdt} BDT
                      </span>
                    </div>
                    <p className="text-[11px] text-[#A2B5AA] leading-relaxed">
                      {instructionsData.instructions}
                    </p>
                  </div>

                  <div className="rounded-xl bg-[#12231D] border border-[#19352A] px-3.5 py-2 shrink-0 space-y-0.5">
                    <div className="text-[10px] uppercase tracking-wide text-[#A2B5AA]">
                      {instructionsData.paymentMethod === 'bkash' ? 'bKash' : 'Nagad'}{' '}
                      {instructionsData.accountType}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-bold font-mono text-[#F4F8F5]">
                        {instructionsData.receiverNumber}
                      </span>
                      <button
                        type="button"
                        onClick={handleCopyNumber}
                        className="p-1 rounded bg-[#19352A] hover:bg-[#16A36A]/40 text-[#42D9A0] cursor-pointer"
                        title="Copy payment number"
                      >
                        {copiedNum ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </div>
                </div>
              ) : null}

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-[#A2B5AA] mb-1">
                    Your {paymentMethod === 'bkash' ? 'bKash' : 'Nagad'} Number
                  </label>
                  <input
                    type="tel"
                    required
                    placeholder="017XXXXXXXX"
                    maxLength={11}
                    value={senderNumber}
                    onChange={(e) => setSenderNumber(e.target.value.replace(/[^0-9]/g, ''))}
                    className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] focus:border-[#16A36A] text-sm font-mono text-[#F4F8F5] outline-none"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-[#A2B5AA] mb-1">
                    Transaction ID (TrxID)
                  </label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. BGA7X9K2M1"
                    maxLength={16}
                    value={trxId}
                    onChange={(e) =>
                      setTrxId(e.target.value.toUpperCase().replace(/[^A-Z0-9]/g, ''))
                    }
                    className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] focus:border-[#16A36A] text-sm font-mono text-[#F4F8F5] outline-none"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-[#A2B5AA] mb-1">
                  Reference / Note (Optional)
                </label>
                <input
                  type="text"
                  maxLength={200}
                  placeholder="Optional reference details..."
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  className="w-full px-3 py-2 rounded-xl bg-[#0B1512] border border-[#19352A] focus:border-[#16A36A] text-xs text-[#F4F8F5] outline-none"
                />
              </div>

              {feedback && (
                <div
                  className={`rounded-xl px-3.5 py-2.5 text-xs flex items-start gap-2 border ${
                    feedback.type === 'success'
                      ? 'bg-[#0B1512] border-[#16A36A]/60 text-[#42D9A0]'
                      : 'bg-red-950/40 border-red-700/50 text-red-200'
                  }`}
                >
                  {feedback.type === 'success' ? (
                    <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  )}
                  <span>{feedback.text}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={submitting || !instructionsData}
                className="w-full py-2.5 px-5 rounded-xl bg-[#16A36A] hover:bg-[#16A36A]/90 disabled:opacity-50 text-[#F4F8F5] font-semibold text-xs sm:text-sm transition-colors cursor-pointer"
              >
                {submitting
                  ? 'Submitting verification request...'
                  : `Submit ${paymentMethod === 'bkash' ? 'bKash' : 'Nagad'} Verification Request`}
              </button>
            </form>
          )}

          {/* User's Payment Requests & Token Activity */}
          {user && (myPaymentRequests.length > 0 || myTokenLedger.length > 0) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {myPaymentRequests.length > 0 && (
                <div className="rounded-xl bg-[#12231D] border border-[#19352A] p-4 space-y-2.5">
                  <div className="text-xs font-bold text-[#F4F8F5] flex items-center gap-2">
                    <Clock className="w-3.5 h-3.5 text-[#42D9A0]" />
                    <span>Payment Requests ({myPaymentRequests.length})</span>
                  </div>
                  <div className="divide-y divide-[#19352A]">
                    {myPaymentRequests.slice(0, 5).map((ord) => (
                      <div key={ord.orderId} className="py-2 text-xs space-y-1">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-semibold text-[#F4F8F5] truncate">
                            {ord.itemTitle}
                          </span>
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase inline-flex items-center gap-1 shrink-0 ${
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
                        <div className="text-[11px] font-mono text-[#A2B5AA]">
                          {ord.paymentMethod.toUpperCase()} · ৳{ord.amountBdt} · TrxID: {ord.trxId}
                        </div>
                        {ord.status === 'rejected' && ord.rejectionReason && (
                          <div className="text-[11px] text-red-300">{ord.rejectionReason}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {myTokenLedger.length > 0 && (
                <div className="rounded-xl bg-[#12231D] border border-[#19352A] p-4 space-y-2.5">
                  <div className="text-xs font-bold text-[#F4F8F5] flex items-center gap-2">
                    <Coins className="w-3.5 h-3.5 text-[#42D9A0]" />
                    <span>Token History</span>
                  </div>
                  <div className="divide-y divide-[#19352A]">
                    {myTokenLedger.slice(0, 5).map((entry) => (
                      <div key={entry.entryId} className="py-2 flex items-center justify-between gap-2 text-xs">
                        <div className="min-w-0">
                          <div className="text-[#F4F8F5] truncate">{entry.reason}</div>
                          <div className="text-[10px] text-[#A2B5AA] font-mono">
                            Balance after: {entry.balanceAfter}
                          </div>
                        </div>
                        <span
                          className={`font-mono font-bold shrink-0 ${
                            entry.delta >= 0 ? 'text-[#42D9A0]' : 'text-amber-300'
                          }`}
                        >
                          {entry.delta >= 0 ? `+${entry.delta}` : entry.delta}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
