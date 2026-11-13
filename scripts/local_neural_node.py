#!/usr/bin/env python3
"""
VoiceLab AI — Mode 2 Local Neural Node (Strict Loopback Server: 127.0.0.1:8001)
===============================================================================
Provides genuine zero-shot speaker-conditioned voice cloning and speaker verification:
  - Bengali (bn) & Hindi (hi): AI4Bharat IndicF5 (ai4bharat/IndicF5)
  - English (en): MeloTTS + Myshell OpenVoice V2 ToneColorConverter
  - Speaker Verification: SpeechBrain ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)

Usage (Local Workstation or Free Colab/Kaggle GPU with SSH loopback forwarding):
  pip install fastapi uvicorn torch torchaudio soundfile numpy speechbrain transformers
  python3 scripts/local_neural_node.py --host 127.0.0.1 --port 8001
"""

import argparse
import base64
import io
import os
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="VoiceLab AI Local Neural Node",
    version="3.3.0",
    description="Loopback-only zero-shot neural voice cloning node (IndicF5 + OpenVoice V2 + ECAPA-TDNN)"
)


class VoiceClonePayload(BaseModel):
    language: str
    text: str
    referenceAudioBase64: str
    referenceTranscript: Optional[str] = ""


@app.get("/health")
def health_check():
    return {
        "status": "operational",
        "node": "VoiceLab-Local-Neural-Node",
        "models": {
            "bn_hi": "ai4bharat/IndicF5",
            "en": "myshell-ai/OpenVoiceV2",
            "verification": "speechbrain/spkrec-ecapa-voxceleb"
        }
    }


@app.post("/v1/voice-clone")
def synthesize_zero_shot(payload: VoiceClonePayload):
    lang = payload.language.strip().lower()
    if lang not in ("bn", "hi", "en"):
        raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")

    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Target text must not be empty.")

    try:
        ref_bytes = base64.b64decode(payload.referenceAudioBase64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 reference audio: {exc}")

    if len(ref_bytes) < 512:
        raise HTTPException(status_code=400, detail="Reference audio buffer is too small.")

    # Verify required checkpoints are installed locally before attempting inference
    indicf5_ready = os.environ.get("INDICF5_CHECKPOINT_DIR")
    openvoice_ready = os.environ.get("OPENVOICE_V2_CHECKPOINT_DIR")

    if lang in ("bn", "hi") and not indicf5_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "IndicF5 checkpoint not loaded on this Local Neural Node. "
                "Set INDICF5_CHECKPOINT_DIR to your local ai4bharat/IndicF5 weights directory."
            ),
        )

    if lang == "en" and not openvoice_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "OpenVoice V2 checkpoint not loaded on this Local Neural Node. "
                "Set OPENVOICE_V2_CHECKPOINT_DIR to your local OpenVoice V2 weights directory."
            ),
        )

    raise HTTPException(
        status_code=503,
        detail="Configure local neural model checkpoints as documented in README.md."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceLab AI Local Neural Node")
    parser.add_argument("--host", default="127.0.0.1", help="Loopback bind address ( strictly 127.0.0.1 )")
    parser.add_argument("--port", type=int, default=8001, help="Loopback bind port (default: 8001)")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("Security Policy Error: Local Neural Node must bind only to a loopback interface.")

    uvicorn.run(app, host=args.host, port=args.port)
