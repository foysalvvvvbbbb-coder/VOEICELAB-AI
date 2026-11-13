import type { CloningLanguageCode, CloningLanguageConfig, ModelResearchEntry } from '../types/voicelab.ts';

export const CLONING_LANGUAGES: Record<CloningLanguageCode, CloningLanguageConfig> = {
  bn: {
    code: 'bn',
    nativeName: 'বাংলা',
    englishName: 'Bengali',
    isoTag: 'bn-BD / bn-IN',
    scriptName: 'Bengali Unicode (U+0980–U+09FF)',
    unicodeRangeDescription: 'বাংলা লিপি (Bengali Script)',
    routedModelId: 'indic-vits-bn-dsp-v2',
    routedModelName: 'Bengali Acoustic Neural Synthesizer + FFmpeg F0/Formant Cloner',
    openSourceEquivalent: 'AI4Bharat IndicF5 / VITS-Bengali + OpenVoice V2 ToneColorConverter',
    sampleTexts: [
      {
        label: 'সংবাদ পাঠ (News Broadcast)',
        text: 'আজকের বিশেষ সংবাদে আপনাদের স্বাগতম। কৃত্রিম বুদ্ধিমত্তা ও আধুনিক অডিও প্রযুক্তির মাধ্যমে বাংলা ভাষায় কণ্ঠস্বর সংরক্ষণ ও গবেষণা নতুন দিগন্ত উন্মোচন করেছে।'
      },
      {
        label: 'তথ্যচিত্র বর্ণনা (Documentary Narration)',
        text: 'বাংলার নদী, প্রকৃতি এবং মানুষের কণ্ঠস্বরের নিজস্ব ছন্দ ও মাধুর্য রয়েছে। সঠিক উচ্চারণ এবং স্বাভাবিক স্বরভঙ্গি বজায় রাখা আমাদের প্রধান লক্ষ্য।'
      },
      {
        label: 'শিক্ষামূলক পাঠ (Educational Module)',
        text: 'বিজ্ঞান ও প্রযুক্তির এই যুগে মাতৃভাষায় স্পষ্ট ও নির্ভুল অডিও কন্টেন্ট তৈরি করা শিক্ষার্থীদের জন্য অত্যন্ত সহায়ক।'
      }
    ]
  },
  hi: {
    code: 'hi',
    nativeName: 'हिन्दी',
    englishName: 'Hindi',
    isoTag: 'hi-IN',
    scriptName: 'Devanagari Unicode (U+0900–U+097F)',
    unicodeRangeDescription: 'देवनागरी लिपि (Devanagari Script)',
    routedModelId: 'xtts-indic-hi-dsp-v2',
    routedModelName: 'Hindi Neural Voice Cloner + FFmpeg F0/Spectral Envelope Matcher',
    openSourceEquivalent: 'Coqui XTTS-v2 (hi) / AI4Bharat IndicF5 + FreeVC Spectral Transfer',
    sampleTexts: [
      {
        label: 'समाचार प्रसारण (News Bulletin)',
        text: 'नमस्कार, आज के मुख्य समाचारों में आपका स्वागत है। आधुनिक ध्वनि प्रौद्योगिकी और कृत्रिम बुद्धिमत्ता ने भारतीय भाषाओं में संवाद को एक नया आयाम दिया है।'
      },
      {
        label: 'वृत्तचित्र वाचन (Documentary Voiceover)',
        text: 'आवाज़ की पहचान केवल शब्दों से नहीं, बल्कि उसके उतार-चढ़ाव, लय और प्राकृतिक उच्चारण से होती है। यह प्रणाली मूल वक्ता की शैली को सुरक्षित रखती है।'
      },
      {
        label: 'शैक्षिक संवाद (Educational Narration)',
        text: 'स्पष्ट उच्चारण और स्वाभाविक गति के साथ हिन्दी भाषा में उच्च गुणवत्ता वाली ऑडियो रिकॉर्डिंग तैयार करना अब अत्यंत सरल और सटीक है।'
      }
    ]
  },
  en: {
    code: 'en',
    nativeName: 'English',
    englishName: 'English',
    isoTag: 'en-US / en-GB / en-IN',
    scriptName: 'Latin Alphabet (U+0020–U+007F)',
    unicodeRangeDescription: 'Standard English Latin Script',
    routedModelId: 'openvoice-xtts-en-dsp-v2',
    routedModelName: 'English Zero-Shot Acoustic Cloner + FFmpeg Pitch/Timbre Transfer',
    openSourceEquivalent: 'Coqui XTTS-v2 (en) + OpenVoice V2 Reference Tone Extractor',
    sampleTexts: [
      {
        label: 'Technical Keynote',
        text: 'Welcome to VoiceLab AI. Every speaker possesses a distinct acoustic fingerprint defined by fundamental frequency, harmonic warmth, and natural cadence.'
      },
      {
        label: 'Documentary Narration',
        text: 'In broadcast audio engineering, preserving vocal identity requires strict same-language phonetic alignment without artificial cross-lingual distortion.'
      },
      {
        label: 'Studio Audiobook',
        text: 'The morning mist settled quietly over the harbor as the research vessel prepared its acoustic sensors for the deep ocean survey.'
      }
    ]
  }
};

export interface ScriptValidationResult {
  isValid: boolean;
  detectedPrimaryScript: CloningLanguageCode | 'mixed' | 'unknown';
  bengaliCharCount: number;
  hindiCharCount: number;
  latinCharCount: number;
  errorMessage?: string;
}

/**
 * Validates that the generation text strictly matches the selected reference voice language.
 * Prevents accidental or forced cross-language voice cloning.
 */
export function validateSameLanguageText(
  text: string,
  expectedLang: CloningLanguageCode
): ScriptValidationResult {
  const trimmed = text.trim();
  if (!trimmed) {
    return {
      isValid: false,
      detectedPrimaryScript: 'unknown',
      bengaliCharCount: 0,
      hindiCharCount: 0,
      latinCharCount: 0,
      errorMessage: 'Please enter text in the selected language before generating speech.'
    };
  }

  let bengaliCharCount = 0;
  let hindiCharCount = 0;
  let latinCharCount = 0;

  for (const char of trimmed) {
    const code = char.codePointAt(0) || 0;
    // Shared Indic punctuation: Danda (। U+0964) and Double Danda (॥ U+0965)
    if (code === 0x0964 || code === 0x0965) {
      continue;
    }
    // Bengali Unicode Block: U+0980 to U+09FF
    if (code >= 0x0980 && code <= 0x09ff) {
      bengaliCharCount++;
    }
    // Devanagari Unicode Block: U+0900 to U+097F (excluding shared Danda punctuation)
    else if (code >= 0x0900 && code <= 0x097f) {
      hindiCharCount++;
    }
    // Basic Latin letters: A-Z, a-z
    else if ((code >= 0x0041 && code <= 0x005a) || (code >= 0x0061 && code <= 0x007a)) {
      latinCharCount++;
    }
  }

  const totalLetters = bengaliCharCount + hindiCharCount + latinCharCount;
  if (totalLetters === 0) {
    return {
      isValid: false,
      detectedPrimaryScript: 'unknown',
      bengaliCharCount,
      hindiCharCount,
      latinCharCount,
      errorMessage: 'Text must contain pronounceable words in the target language.'
    };
  }

  let detectedPrimaryScript: CloningLanguageCode | 'mixed' = 'en';
  if (bengaliCharCount >= hindiCharCount && bengaliCharCount >= latinCharCount) {
    detectedPrimaryScript = 'bn';
  } else if (hindiCharCount >= bengaliCharCount && hindiCharCount >= latinCharCount) {
    detectedPrimaryScript = 'hi';
  } else {
    detectedPrimaryScript = 'en';
  }

  if (expectedLang === 'bn') {
    if (hindiCharCount > 0 || latinCharCount > bengaliCharCount * 0.25) {
      return {
        isValid: false,
        detectedPrimaryScript,
        bengaliCharCount,
        hindiCharCount,
        latinCharCount,
        errorMessage:
          hindiCharCount > 0
            ? 'Cross-Language Blocked: Bengali (বাংলা) reference voice selected, but Hindi (Devanagari) script was detected in the text input. Same-language voice cloning requires Bengali script.'
            : 'Cross-Language Blocked: Bengali (বাংলা) reference voice selected, but English (Latin) text was detected. Never mix reference voice and text languages.'
      };
    }
    if (bengaliCharCount < 2) {
      return {
        isValid: false,
        detectedPrimaryScript,
        bengaliCharCount,
        hindiCharCount,
        latinCharCount,
        errorMessage: 'Please enter valid Bengali (বাংলা) Unicode text for Bengali voice cloning.'
      };
    }
  }

  if (expectedLang === 'hi') {
    if (bengaliCharCount > 0 || latinCharCount > hindiCharCount * 0.25) {
      return {
        isValid: false,
        detectedPrimaryScript,
        bengaliCharCount,
        hindiCharCount,
        latinCharCount,
        errorMessage:
          bengaliCharCount > 0
            ? 'Cross-Language Blocked: Hindi (हिन्दी) reference voice selected, but Bengali (বাংলা) script was detected in the text input. Same-language voice cloning requires Hindi Devanagari script.'
            : 'Cross-Language Blocked: Hindi (हिन्दी) reference voice selected, but English (Latin) text was detected. Never mix reference voice and text languages.'
      };
    }
    if (hindiCharCount < 2) {
      return {
        isValid: false,
        detectedPrimaryScript,
        bengaliCharCount,
        hindiCharCount,
        latinCharCount,
        errorMessage: 'Please enter valid Hindi (हिन्दी) Devanagari text for Hindi voice cloning.'
      };
    }
  }

  if (expectedLang === 'en') {
    if (bengaliCharCount > 0 || hindiCharCount > 0) {
      return {
        isValid: false,
        detectedPrimaryScript,
        bengaliCharCount,
        hindiCharCount,
        latinCharCount,
        errorMessage: `Cross-Language Blocked: English reference voice selected, but ${
          bengaliCharCount > 0 ? 'Bengali (বাংলা)' : 'Hindi (हिन्दी)'
        } script was detected in the text input. Same-language voice cloning requires English text.`
      };
    }
  }

  return {
    isValid: true,
    detectedPrimaryScript: expectedLang,
    bengaliCharCount,
    hindiCharCount,
    latinCharCount
  };
}

export const SRT_SUPPORTED_LANGUAGES = [
  { code: 'auto', label: 'Auto-Detect Spoken Language (Recommended)', nativeLabel: 'Automatic', priority: true },
  { code: 'bn', label: 'Bengali', nativeLabel: 'বাংলা', priority: true },
  { code: 'hi', label: 'Hindi', nativeLabel: 'हिन्दी', priority: true },
  { code: 'en', label: 'English', nativeLabel: 'English', priority: true },
  { code: 'ar', label: 'Arabic', nativeLabel: 'العربية', priority: false },
  { code: 'ur', label: 'Urdu', nativeLabel: 'اردو', priority: false },
  { code: 'ta', label: 'Tamil', nativeLabel: 'தமிழ்', priority: false },
  { code: 'te', label: 'Telugu', nativeLabel: 'తెలుగు', priority: false },
  { code: 'mr', label: 'Marathi', nativeLabel: 'मराठी', priority: false },
  { code: 'gu', label: 'Gujarati', nativeLabel: 'ગુજરાતી', priority: false },
  { code: 'es', label: 'Spanish', nativeLabel: 'Español', priority: false },
  { code: 'fr', label: 'French', nativeLabel: 'Français', priority: false },
  { code: 'de', label: 'German', nativeLabel: 'Deutsch', priority: false },
  { code: 'ja', label: 'Japanese', nativeLabel: '日本語', priority: false },
  { code: 'ko', label: 'Korean', nativeLabel: '한국어', priority: false },
  { code: 'zh', label: 'Chinese', nativeLabel: '中文', priority: false },
  { code: 'ru', label: 'Russian', nativeLabel: 'Русский', priority: false },
  { code: 'pt', label: 'Portuguese', nativeLabel: 'Português', priority: false },
  { code: 'tr', label: 'Turkish', nativeLabel: 'Türkçe', priority: false },
  { code: 'id', label: 'Indonesian', nativeLabel: 'Bahasa Indonesia', priority: false }
];

export const MODEL_RESEARCH_MATRIX: ModelResearchEntry[] = [
  {
    id: 'coqui-xtts-v2',
    name: 'Coqui XTTS-v2 (Zero-Shot GPT + Perceiver Resampler)',
    category: 'Voice Cloning',
    officialLanguages: '17 languages (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi)',
    bengaliSupport: {
      supported: false,
      statusLabel: 'Unsupported / Absent',
      pronunciationQuality: 'Fails (Bengali U+0980–U+09FF absent from 17-language BPE tokenizer)',
      notes: 'CRITICAL FINDING: XTTS-v2 supports Hindi (hi) and English (en), but Bengali (bn) is NOT supported natively. Never route Bengali jobs to stock XTTS-v2.'
    },
    hindiSupport: {
      supported: true,
      statusLabel: 'Verified Native',
      pronunciationQuality: 'High zero-shot naturalness on Devanagari text with 6–10s clean reference clip'
    },
    englishSupport: {
      supported: true,
      statusLabel: 'Verified Studio',
      pronunciationQuality: 'Studio zero-shot prosody, rhythm, and speaker conditioning via 32x1024 latents + 512-D speaker vector'
    },
    referenceAudioReq: '6s – 15s uncompressed WAV (22.05kHz / 24kHz mono)',
    speakerSimilarity: 'Zero-shot conditioned (32x1024 Perceiver latents + 512-D H/ASP speaker embedding)',
    naturalness: '4.3 / 5.0 MOS (en), 4.0 / 5.0 MOS (hi)',
    modelSize: '1.87 GB checkpoint (coqui/XTTS-v2)',
    cpuInference: 'Slow (~4.5x–8x real-time on 4-core CPU)',
    gpuInference: 'Fast (~0.25x RTF on T4 / RTX 3060 6GB VRAM)',
    ramRequirement: '4.5 GB – 6.0 GB system RAM minimum',
    license: 'Coqui Public Model License (CPML)',
    commercialUse: 'Non-commercial only under CPML (commercial use prohibited)',
    deploymentFeasibility: 'Self-Hosted Local Node (Mode 2) for Hindi & English; excluded for Bengali',
    roleInVoiceLab: 'Zero-shot Local Node option for English & Hindi (non-commercial); excluded from Bengali'
  },
  {
    id: 'openvoice-v2',
    name: 'MyShell OpenVoice V2 (ToneColorConverter)',
    category: 'Voice Cloning',
    officialLanguages: 'Base MeloTTS: en, es, fr, zh, ja, ko + Language-Agnostic 256-D ToneColorConverter',
    bengaliSupport: {
      supported: true,
      statusLabel: 'Experimental / Accent-Only',
      pronunciationQuality: 'Requires external Bengali base TTS (IndicF5 / MMS-TTS-ben); ToneColorConverter transfers 256-D vocal timbre',
      notes: 'OpenVoice V2 decouples base phonetic synthesis from reference speaker timbre conversion via normalizing flows.'
    },
    hindiSupport: {
      supported: true,
      statusLabel: 'Verified Native',
      pronunciationQuality: 'High when paired with native Devanagari base synthesizer (IndicF5 / MMS-TTS-hin)'
    },
    englishSupport: {
      supported: true,
      statusLabel: 'Verified Studio',
      pronunciationQuality: 'Native MeloTTS English base + 256-D zero-shot timbre & formant transfer'
    },
    referenceAudioReq: '5s – 30s clean speech (minimal background noise)',
    speakerSimilarity: 'Zero-shot 256-D ToneColorConverter spectral & formant transfer',
    naturalness: '4.2 / 5.0 MOS',
    modelSize: '340 MB (myshell-ai/OpenVoiceV2 converter + SE extractor)',
    cpuInference: 'Fast (~0.8x real-time on modern CPU)',
    gpuInference: 'Ultra-fast (<0.1x RTF)',
    ramRequirement: '2.0 GB – 2.5 GB RAM',
    license: 'MIT License',
    commercialUse: 'Permitted (MIT License)',
    deploymentFeasibility: 'Feasible on Local CPU/GPU when paired with native base synthesizer',
    roleInVoiceLab: 'MIT-licensed zero-shot 256-D ToneColorConverter in Local Node (Mode 2)'
  },
  {
    id: 'ai4bharat-indicf5',
    name: 'AI4Bharat IndicF5 (Zero-Shot Flow-Matching DiT)',
    category: 'Voice Cloning',
    officialLanguages: '11 Indic languages + English: Assamese, Bengali (bn), Gujarati, Hindi (hi), Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu',
    bengaliSupport: {
      supported: true,
      statusLabel: 'Verified Native',
      pronunciationQuality: 'Genuine zero-shot Bengali voice cloning via flow-matching Diffusion Transformer conditioned on reference mel + ref_text',
      notes: 'Trained on 1,417 hours of high-quality speech (Rasa, IndicTTS, LIMMITS). Solves XTTS-v2 Bengali gap with true zero-shot speaker conditioning.'
    },
    hindiSupport: {
      supported: true,
      statusLabel: 'Verified Native',
      pronunciationQuality: 'Genuine zero-shot Devanagari voice cloning with native schwa-deletion and retroflex accuracy'
    },
    englishSupport: {
      supported: true,
      statusLabel: 'High Accuracy',
      pronunciationQuality: 'Zero-shot speaker conditioning for Indian-English & standard English scripts'
    },
    referenceAudioReq: '5s – 15s 24kHz mono WAV + reference transcript',
    speakerSimilarity: 'Genuine zero-shot in-context acoustic conditioning (Diffusion Transformer + Vocos 24kHz)',
    naturalness: '4.3 / 5.0 MOS (Bengali & Hindi)',
    modelSize: '1.2 GB (ai4bharat/IndicF5 SAFETENSORS)',
    cpuInference: '~4.5x–6.0x RTF on 4-core CPU',
    gpuInference: '~0.18x RTF on T4 / RTX 3060 GPU',
    ramRequirement: '3.8 GB – 4.5 GB RAM',
    license: 'MIT License',
    commercialUse: 'Permitted (MIT License)',
    deploymentFeasibility: 'Primary zero-shot checkpoint for Bengali & Hindi on Self-Hosted Local Node (Mode 2)',
    roleInVoiceLab: 'Primary zero-shot neural voice cloning checkpoint for বাংলা (bn) & हिन्दी (hi) in Mode 2'
  },
  {
    id: 'ai4bharat-indic-vits',
    name: 'AI4Bharat Indic-TTS / Indic-VITS (Fixed Multi-Speaker TTS)',
    category: 'Voice Cloning',
    officialLanguages: '13 Indic languages including Bengali (bn) and Hindi (hi)',
    bengaliSupport: {
      supported: true,
      statusLabel: 'Experimental / Accent-Only',
      pronunciationQuality: 'Accurate Bengali G2P pronunciation, but uses fixed speaker ID lookup table (NOT zero-shot voice cloning)',
      notes: 'AUDIT FINDING: Stock Indic-VITS does NOT accept arbitrary reference audio embeddings; requires OpenVoice V2 ToneColorConverter for speaker conversion.'
    },
    hindiSupport: {
      supported: true,
      statusLabel: 'Experimental',
      pronunciationQuality: 'Fixed multi-speaker Devanagari TTS (requires external ToneColorConverter for speaker cloning)'
    },
    englishSupport: {
      supported: true,
      statusLabel: 'High Accuracy',
      pronunciationQuality: 'Fixed speaker ID Indian-English TTS'
    },
    referenceAudioReq: 'N/A natively (Fixed speaker IDs unless chained with OpenVoice V2)',
    speakerSimilarity: 'Fixed speaker lookup only (Not zero-shot unless paired with OpenVoice V2)',
    naturalness: '3.9 / 5.0 MOS',
    modelSize: '180 MB per language checkpoint',
    cpuInference: 'Fast (~0.9x RTF on CPU)',
    gpuInference: '<0.08x RTF on GPU',
    ramRequirement: '1.5 GB RAM',
    license: 'MIT / CC-BY-4.0',
    commercialUse: 'Permitted under MIT code license',
    deploymentFeasibility: 'Lightweight base phonemizer/synthesizer when paired with OpenVoice V2',
    roleInVoiceLab: 'Documented as fixed-speaker base TTS (superseded by IndicF5 for zero-shot cloning)'
  },
  {
    id: 'faster-whisper-large-v3',
    name: 'Faster-Whisper (CTranslate2 Whisper large-v3 / turbo)',
    category: 'Multilingual ASR',
    officialLanguages: '99 languages including Bengali (bn), Hindi (hi), English (en), Arabic (ar), Urdu (ur), Spanish, French, etc.',
    bengaliSupport: {
      supported: true,
      statusLabel: 'Verified Native',
      pronunciationQuality: 'High word-recognition accuracy in Bengali Unicode with word-level timestamps',
      notes: 'Preserves native Bengali script without translating to English when task="transcribe".'
    },
    hindiSupport: {
      supported: true,
      statusLabel: 'Verified Native',
      pronunciationQuality: 'High accuracy on conversational & broadcast Hindi in Devanagari script'
    },
    englishSupport: {
      supported: true,
      statusLabel: 'Verified Studio',
      pronunciationQuality: 'Industry benchmark WER (<4.5% clean speech) across diverse global accents'
    },
    referenceAudioReq: 'Any audio file (16kHz mono normalized internally via FFmpeg)',
    speakerSimilarity: 'N/A (ASR Model)',
    naturalness: 'Word-level cross-attention DTW timestamp alignment (HH:MM:SS,mmm)',
    modelSize: '809 MB (large-v3-turbo INT8) / 1.5 GB (FP16)',
    cpuInference: '4x faster than stock PyTorch Whisper with INT8 quantization',
    gpuInference: 'Real-time factor < 0.06x',
    ramRequirement: '1.8 GB – 3.2 GB RAM',
    license: 'MIT License',
    commercialUse: 'Permitted (MIT License)',
    deploymentFeasibility: 'Gold-standard open-source multilingual ASR for Local Node; paired with Gemini 3.1/3.5 Transcribe + FFmpeg VAD in Cloud Mode',
    roleInVoiceLab: 'Universal Multilingual SRT Transcription & Timestamp Alignment Engine'
  }
];

export const LOCAL_PYTHON_SERVER_SCRIPT = `#!/usr/bin/env python3
"""
VoiceLab AI — Zero-Cost Self-Hosted Local Neural Processing Node (Mode 2)
=========================================================================
Provides genuine zero-shot speaker-conditioned neural voice cloning & ECAPA-TDNN
speaker verification on local hardware (GPU or >=6 GB RAM CPU) at ৳0 cost.

Architecture:
  1. Pipeline A (POST /v1/voice-clone):
     - Bengali (bn): AI4Bharat IndicF5 (ai4bharat/IndicF5) zero-shot flow-matching DiT
                     conditioned directly on 24kHz mono reference audio + aligned ref_text
     - Hindi (hi):   AI4Bharat IndicF5 (ai4bharat/IndicF5) zero-shot flow-matching DiT
     - English (en): MyShell OpenVoice V2 (myshell-ai/OpenVoiceV2) MeloTTS-EN base TTS
                     + se_extractor.get_se (256-D speaker embedding) + ToneColorConverter.convert
     - Speaker Verification: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)
                             192-D L2-normalized speaker embedding cosine similarity.
  2. Security:
     - Bound to 127.0.0.1 (localhost) by default.
     - Requires X-VoiceLab-Node-Token header authentication.
"""
import base64
import os
import tempfile
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="VoiceLab AI Local Neural Inference Node", version="3.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPECTED_TOKEN = os.environ.get("LOCAL_NODE_AUTH_TOKEN", "voicelab-local-token")
_MODELS = {}

def verify_auth_token(token: Optional[str]):
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized local node request (invalid X-VoiceLab-Node-Token).")

def get_ecapa_encoder():
    if "ecapa" not in _MODELS:
        from speechbrain.inference.speaker import EncoderClassifier
        _MODELS["ecapa"] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="./pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": DEVICE}
        )
    return _MODELS["ecapa"]

def compute_ecapa_cosine_similarity(ref_wav_path: str, gen_wav_path: str) -> float:
    classifier = get_ecapa_encoder()
    sig_ref, sr_ref = torchaudio.load(ref_wav_path)
    sig_gen, sr_gen = torchaudio.load(gen_wav_path)
    if sr_ref != 16000:
        sig_ref = torchaudio.functional.resample(sig_ref, sr_ref, 16000)
    if sr_gen != 16000:
        sig_gen = torchaudio.functional.resample(sig_gen, sr_gen, 16000)
    emb_ref = classifier.encode_batch(sig_ref[:1]).squeeze()
    emb_gen = classifier.encode_batch(sig_gen[:1]).squeeze()
    emb_ref_norm = F.normalize(emb_ref, p=2, dim=-1)
    emb_gen_norm = F.normalize(emb_gen, p=2, dim=-1)
    cosine = torch.sum(emb_ref_norm * emb_gen_norm).item()
    return round(float(cosine), 4)

def get_indicf5_model():
    if "indicf5" not in _MODELS:
        from transformers import AutoModel
        _MODELS["indicf5"] = AutoModel.from_pretrained(
            "ai4bharat/IndicF5",
            trust_remote_code=True
        ).to(DEVICE)
    return _MODELS["indicf5"]

def get_openvoice_v2_en_pipeline():
    if "openvoice_en" not in _MODELS:
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        from melo.api import TTS
        ckpt_converter = "./checkpoints_v2/converter"
        converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=DEVICE)
        converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
        base_tts = TTS(language="EN", device=DEVICE)
        source_se = torch.load("./checkpoints_v2/base_speakers/ses/en-default.pth", map_location=DEVICE)
        _MODELS["openvoice_en"] = (base_tts, converter, se_extractor, source_se)
    return _MODELS["openvoice_en"]

class VoiceCloneRequest(BaseModel):
    language: str
    text: str
    referenceAudioBase64: str
    referenceTranscript: Optional[str] = ""
    referencePitchHz: Optional[float] = 160.0

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "local-neural-zero-shot",
        "device": DEVICE,
        "checkpoints": {
            "bn": "ai4bharat/IndicF5 (Zero-Shot Flow-Matching DiT)",
            "hi": "ai4bharat/IndicF5 (Zero-Shot Flow-Matching DiT)",
            "en": "myshell-ai/OpenVoiceV2 (MeloTTS-EN + 256-D ToneColorConverter)",
            "speaker_verification": "speechbrain/spkrec-ecapa-voxceleb (192-D L2-Normalized ECAPA-TDNN)"
        }
    }

@app.post("/v1/voice-clone")
def clone_voice(
    req: VoiceCloneRequest,
    x_voicelab_node_token: Optional[str] = Header(default=None)
):
    verify_auth_token(x_voicelab_node_token)
    if req.language not in ("bn", "hi", "en"):
        raise HTTPException(status_code=400, detail="Unsupported language code.")

    ref_bytes = base64.b64decode(req.referenceAudioBase64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_tmp:
        ref_tmp.write(ref_bytes)
        ref_path = ref_tmp.name

    base_tmp_path = ref_path.replace(".wav", "_base.wav")
    out_path = ref_path.replace(".wav", "_out.wav")
    try:
        if req.language in ("bn", "hi"):
            model = get_indicf5_model()
            audio_out = model(
                req.text,
                ref_audio_path=ref_path,
                ref_text=req.referenceTranscript or req.text
            )
            if isinstance(audio_out, np.ndarray) and audio_out.dtype == np.int16:
                audio_out = audio_out.astype(np.float32) / 32768.0
            sf.write(out_path, audio_out, 24000)
            checkpoint_used = "ai4bharat/IndicF5 (24kHz Zero-Shot Flow-Matching DiT)"
        else:
            base_tts, converter, se_extractor, source_se = get_openvoice_v2_en_pipeline()
            speaker_ids = base_tts.hps.data.spk2id
            base_tts.tts_to_file(req.text, speaker_ids["EN-Default"], base_tmp_path, speed=1.0)
            target_se, _ = se_extractor.get_se(ref_path, converter, vad=True)
            converter.convert(
                audio_src_path=base_tmp_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=out_path,
                message="@MyShell"
            )
            checkpoint_used = "myshell-ai/OpenVoiceV2 (MeloTTS-EN + 256-D ToneColorConverter)"

        ecapa_cosine = compute_ecapa_cosine_similarity(ref_path, out_path)
        with open(out_path, "rb") as f:
            wav_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "wavBase64": wav_b64,
            "audioBase64": wav_b64,
            "speakerEmbeddingDim": 192,
            "modelCheckpoint": checkpoint_used,
            "modelUsed": checkpoint_used,
            "ecapaCosineSimilarity": ecapa_cosine
        }
    finally:
        for p in (ref_path, base_tmp_path, out_path):
            if os.path.exists(p):
                os.remove(p)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
`;

