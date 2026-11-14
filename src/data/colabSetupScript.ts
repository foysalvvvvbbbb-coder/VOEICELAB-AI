export function buildColabOneCellSetupScript(customToken = 'voicelab-local-token'): string {
  const safeToken = (customToken || 'voicelab-local-token').replace(/["'\\\n\r]/g, '').trim() || 'voicelab-local-token';
  return `# =====================================================================
# VoiceLab AI v3.3.0 — Free Google Colab T4 GPU Mode 2 Neural Node (0 Tk)
# Runtime -> Change runtime type -> T4 GPU -> Run this cell once!
# Supports Unlimited Bengali (bn) & Hindi (hi) Zero-Shot Cloning (IndicF5)
# =====================================================================
!pip install -q fastapi uvicorn pydantic soundfile numpy git+https://github.com/ai4bharat/IndicF5.git speechbrain
!wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /usr/local/bin/cloudflared

import os, re, time, threading, subprocess, base64, tempfile, importlib.util
from typing import List, Optional
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

os.environ["LOCAL_NODE_AUTH_TOKEN"] = "${safeToken}"
EXPECTED_TOKEN = os.environ["LOCAL_NODE_AUTH_TOKEN"]
_MODELS = {}

app = FastAPI(title="VoiceLab AI Colab T4 Neural Node", version="3.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_indicf5():
    if "indicf5" not in _MODELS:
        from transformers import AutoModel
        print("Loading ai4bharat/IndicF5 onto", get_device(), "...")
        _MODELS["indicf5"] = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True).to(get_device())
        print("IndicF5 ready!")
    return _MODELS["indicf5"]

def get_ecapa():
    if "ecapa" not in _MODELS and importlib.util.find_spec("speechbrain"):
        from speechbrain.inference.speaker import EncoderClassifier
        _MODELS["ecapa"] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="./pretrained_models/ecapa",
            run_opts={"device": get_device()}
        )
    return _MODELS.get("ecapa")

def compute_ecapa_cosine(ref_path, gen_path):
    try:
        clf = get_ecapa()
        if clf is None:
            return None
        import torchaudio
        import torch.nn.functional as F
        s1, sr1 = torchaudio.load(ref_path)
        s2, sr2 = torchaudio.load(gen_path)
        if sr1 != 16000: s1 = torchaudio.functional.resample(s1, sr1, 16000)
        if sr2 != 16000: s2 = torchaudio.functional.resample(s2, sr2, 16000)
        e1 = F.normalize(clf.encode_batch(s1[:1]).squeeze(), p=2, dim=-1)
        e2 = F.normalize(clf.encode_batch(s2[:1]).squeeze(), p=2, dim=-1)
        return round(float(torch.sum(e1 * e2).item()), 4)
    except Exception:
        return None

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
def health():
    return {
        "status": "operational",
        "mode": "local-neural-zero-shot",
        "node": f"VoiceLab-Colab-{get_device().upper()}",
        "readyForLanguage": {"bn": True, "hi": True, "en": False},
        "checkpoints": {"bn": "ai4bharat/IndicF5", "hi": "ai4bharat/IndicF5"}
    }

@app.post("/v1/voice-clone")
def clone_voice(payload: VoiceClonePayload, x_voicelab_node_token: Optional[str] = Header(default=None)):
    if x_voicelab_node_token != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid X-VoiceLab-Node-Token")
    lang = payload.language.strip().lower()
    if lang not in ("bn", "hi"):
        raise HTTPException(status_code=503, detail=f"Colab IndicF5 supports bn/hi. Received: {lang}")
    target_text = (payload.spokenText or payload.text or "").strip()
    ref_bytes = base64.b64decode(payload.referenceAudioBase64)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(ref_bytes)
        ref_path = f.name
    out_path = ref_path.replace(".wav", "_out.wav")
    try:
        model = get_indicf5()
        audio_out = model(target_text, ref_audio_path=ref_path, ref_text=(payload.referenceTranscript or target_text).strip())
        arr = np.array(audio_out)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        sf.write(out_path, arr.astype(np.float32), 24000)
        sim = compute_ecapa_cosine(ref_path, out_path)
        with open(out_path, "rb") as f:
            wav_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "wavBase64": wav_b64,
            "audioBase64": wav_b64,
            "chunkIndex": payload.chunkIndex,
            "modelCheckpoint": "ai4bharat/IndicF5 (Colab T4 Zero-Shot DiT)",
            "ecapaCosineSimilarity": sim
        }
    finally:
        for p in (ref_path, out_path):
            if os.path.exists(p): os.remove(p)

threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning"), daemon=True).start()
time.sleep(2)
get_indicf5()

proc = subprocess.Popen(["/usr/local/bin/cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8001"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
tunnel_url = None
for line in proc.stdout:
    m = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", line)
    if m:
        tunnel_url = m.group(0)
        print("\\n" + "="*65)
        print("VOICELAB AI NODE ONLINE! Paste this into VoiceLab AI:")
        print("Node URL   :", tunnel_url)
        print("Auth Token :", EXPECTED_TOKEN)
        print("="*65 + "\\n")
        break
proc.wait()
`;
}
