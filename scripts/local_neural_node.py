#!/usr/bin/env python3
"""
VoiceLab AI — Mode 2 Local Neural Node (Strict Loopback Server: 127.0.0.1:8001)
===============================================================================
Provides genuine zero-shot speaker-conditioned voice cloning and speaker verification:
  - Bengali (bn) & Hindi (hi): AI4Bharat IndicF5 (ai4bharat/IndicF5) zero-shot DiT
  - English (en): MeloTTS + MyShell OpenVoice V2 ToneColorConverter
  - Speaker Verification: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)

Usage (Local Workstation or Free Colab/Kaggle GPU with SSH loopback forwarding):
  pip install fastapi uvicorn torch torchaudio soundfile numpy speechbrain transformers
  python3 scripts/local_neural_node.py --host 127.0.0.1 --port 8001
"""

import argparse
import base64
import importlib.util
import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

ALLOW_TUNNEL = os.environ.get("ALLOW_TUNNEL", "0") == "1"

app = FastAPI(
    title="VoiceLab AI Mode 2 Neural Node",
    version="3.3.0",
    description="Zero-shot neural voice cloning node (IndicF5 + OpenVoice V2 + ECAPA-TDNN)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_TUNNEL else ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

EXPECTED_TOKEN = os.environ.get("LOCAL_NODE_AUTH_TOKEN", "voicelab-local-token")
_MODELS = {}


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _verify_auth_token(token: Optional[str]) -> None:
    if token != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized local node request (invalid X-VoiceLab-Node-Token).",
        )


def _check_language_readiness():
    has_torch = _has_module("torch") and _has_module("torchaudio") and _has_module("soundfile")
    indicf5_ready = bool(
        has_torch
        and _has_module("transformers")
        and (os.environ.get("INDICF5_CHECKPOINT_DIR") or os.environ.get("ENABLE_HF_INDICF5") == "1")
    )
    openvoice_dir = os.environ.get("OPENVOICE_V2_CHECKPOINT_DIR", "./checkpoints_v2")
    openvoice_ready = bool(
        has_torch
        and _has_module("openvoice")
        and _has_module("melo")
        and os.path.isdir(openvoice_dir)
    )
    return {
        "bn": indicf5_ready,
        "hi": indicf5_ready,
        "en": openvoice_ready,
    }


def _get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_ecapa_encoder():
    if "ecapa" not in _MODELS:
        from speechbrain.inference.speaker import EncoderClassifier

        _MODELS["ecapa"] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="./pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": _get_device()},
        )
    return _MODELS["ecapa"]


def _compute_ecapa_cosine_similarity(ref_wav_path: str, gen_wav_path: str) -> Optional[float]:
    if not _has_module("speechbrain"):
        return None
    try:
        import torch
        import torch.nn.functional as F
        import torchaudio

        classifier = _get_ecapa_encoder()
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
    except Exception:
        return None


def _get_indicf5_model():
    if "indicf5" not in _MODELS:
        from transformers import AutoModel

        ckpt = os.environ.get("INDICF5_CHECKPOINT_DIR") or "ai4bharat/IndicF5"
        _MODELS["indicf5"] = AutoModel.from_pretrained(
            ckpt,
            trust_remote_code=True,
        ).to(_get_device())
    return _MODELS["indicf5"]


def _get_openvoice_v2_en_pipeline():
    if "openvoice_en" not in _MODELS:
        import torch
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        from melo.api import TTS

        device = _get_device()
        ckpt_root = os.environ.get("OPENVOICE_V2_CHECKPOINT_DIR", "./checkpoints_v2")
        ckpt_converter = os.path.join(ckpt_root, "converter")
        converter = ToneColorConverter(os.path.join(ckpt_converter, "config.json"), device=device)
        converter.load_ckpt(os.path.join(ckpt_converter, "checkpoint.pth"))
        base_tts = TTS(language="EN", device=device)
        source_se = torch.load(
            os.path.join(ckpt_root, "base_speakers/ses/en-default.pth"),
            map_location=device,
        )
        _MODELS["openvoice_en"] = (base_tts, converter, se_extractor, source_se)
    return _MODELS["openvoice_en"]


class VoiceClonePayload(BaseModel):
    language: str
    text: str
    spokenText: Optional[str] = None
    chunkIndex: Optional[int] = 0
    totalChunks: Optional[int] = 1
    activeEmotionTags: Optional[List[str]] = []
    emotionStyleDirective: Optional[str] = ""
    referenceAudioBase64: str
    referenceAudioSha256: Optional[str] = None
    referenceTranscript: Optional[str] = ""
    referencePitchHz: Optional[float] = 160.0


@app.get("/health")
def health_check():
    readiness = _check_language_readiness()
    return {
        "status": "operational",
        "mode": "local-neural-zero-shot",
        "node": "VoiceLab-Local-Neural-Node",
        "readyForLanguage": readiness,
        "checkpoints": {
            "bn": "ai4bharat/IndicF5 (Zero-Shot Flow-Matching DiT)",
            "hi": "ai4bharat/IndicF5 (Zero-Shot Flow-Matching DiT)",
            "en": "myshell-ai/OpenVoiceV2 (MeloTTS-EN + 256-D ToneColorConverter)",
            "speaker_verification": "speechbrain/spkrec-ecapa-voxceleb (192-D ECAPA-TDNN)",
        },
    }


@app.post("/v1/voice-clone")
def synthesize_zero_shot(
    payload: VoiceClonePayload,
    x_voicelab_node_token: Optional[str] = Header(default=None),
):
    _verify_auth_token(x_voicelab_node_token)
    lang = payload.language.strip().lower()
    if lang not in ("bn", "hi", "en"):
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")

    target_text = (payload.spokenText or payload.text or "").strip()
    if not target_text:
        raise HTTPException(status_code=400, detail="Target spokenText must not be empty.")

    try:
        ref_bytes = base64.b64decode(payload.referenceAudioBase64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 reference audio: {exc}")

    if len(ref_bytes) < 512:
        raise HTTPException(status_code=400, detail="Reference audio buffer is too small.")

    readiness = _check_language_readiness()
    if not readiness.get(lang, False):
        if lang in ("bn", "hi"):
            raise HTTPException(
                status_code=503,
                detail=(
                    "ai4bharat/IndicF5 checkpoint or PyTorch/Transformers dependencies are not ready on this Local Neural Node. "
                    "Install torch/torchaudio/soundfile/transformers and set INDICF5_CHECKPOINT_DIR (or ENABLE_HF_INDICF5=1)."
                ),
            )
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenVoice V2 checkpoint or dependencies (openvoice, melo) are not ready on this Local Neural Node. "
                "Set OPENVOICE_V2_CHECKPOINT_DIR to your local OpenVoice V2 checkpoints directory."
            ),
        )

    import numpy as np
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_tmp:
        ref_tmp.write(ref_bytes)
        ref_path = ref_tmp.name

    base_tmp_path = ref_path.replace(".wav", "_base.wav")
    out_path = ref_path.replace(".wav", "_out.wav")
    try:
        if lang in ("bn", "hi"):
            model = _get_indicf5_model()
            audio_out = model(
                target_text,
                ref_audio_path=ref_path,
                ref_text=(payload.referenceTranscript or target_text).strip(),
            )
            if isinstance(audio_out, np.ndarray) and audio_out.dtype == np.int16:
                audio_out = audio_out.astype(np.float32) / 32768.0
            sf.write(out_path, audio_out, 24000)
            checkpoint_used = "ai4bharat/IndicF5 (24kHz Zero-Shot Flow-Matching DiT)"
        else:
            base_tts, converter, se_extractor, source_se = _get_openvoice_v2_en_pipeline()
            speaker_ids = base_tts.hps.data.spk2id
            base_tts.tts_to_file(target_text, speaker_ids["EN-Default"], base_tmp_path, speed=1.0)
            target_se, _ = se_extractor.get_se(ref_path, converter, vad=True)
            converter.convert(
                audio_src_path=base_tmp_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=out_path,
                message="@MyShell",
            )
            checkpoint_used = "myshell-ai/OpenVoiceV2 (MeloTTS-EN + 256-D ToneColorConverter)"

        ecapa_cosine = _compute_ecapa_cosine_similarity(ref_path, out_path)
        with open(out_path, "rb") as f:
            wav_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "wavBase64": wav_b64,
            "audioBase64": wav_b64,
            "chunkIndex": payload.chunkIndex,
            "speakerEmbeddingDim": 192 if ecapa_cosine is not None else 256,
            "modelCheckpoint": checkpoint_used,
            "modelUsed": checkpoint_used,
            "ecapaCosineSimilarity": ecapa_cosine,
        }
    finally:
        for p in (ref_path, base_tmp_path, out_path):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceLab AI Mode 2 Neural Node")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001)")
    parser.add_argument(
        "--allow-tunnel",
        action="store_true",
        help="Allow binding on 0.0.0.0 for HTTPS cloud tunnels (Colab/Kaggle cloudflared or ngrok)",
    )
    parser.add_argument(
        "--enable-hf-indicf5",
        action="store_true",
        help="Automatically enable ai4bharat/IndicF5 loading from HuggingFace",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Override X-VoiceLab-Node-Token auth secret",
    )
    args = parser.parse_args()

    if args.enable_hf_indicf5:
        os.environ["ENABLE_HF_INDICF5"] = "1"
    if args.token:
        EXPECTED_TOKEN = args.token.strip()
        os.environ["LOCAL_NODE_AUTH_TOKEN"] = EXPECTED_TOKEN
    if args.allow_tunnel:
        os.environ["ALLOW_TUNNEL"] = "1"
        ALLOW_TUNNEL = True

    if not ALLOW_TUNNEL and args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit(
            "Security Policy Error: Local Neural Node must bind only to a loopback interface unless --allow-tunnel is explicitly enabled."
        )

    uvicorn.run(app, host=args.host, port=args.port)
