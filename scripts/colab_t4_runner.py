#!/usr/bin/env python3
"""
VoiceLab AI v3.3.0 — Free Google Colab / Kaggle T4 GPU Runner
=============================================================
Starts Mode 2 Neural Node (`scripts/local_neural_node.py`) with an official
Cloudflare Quick Tunnel (`https://*.trycloudflare.com`) so users can run
unlimited ৳0 Bengali / Hindi zero-shot voice cloning from any browser.

Usage in Colab/Kaggle:
  !python3 scripts/colab_t4_runner.py
"""

import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request


def run_cmd(cmd):
    print(f"[VoiceLab Colab Runner] $ {' '.join(cmd)}")
    subprocess.check_call(cmd)


def ensure_dependencies():
    run_cmd([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "fastapi",
        "uvicorn",
        "pydantic",
        "soundfile",
        "numpy",
        "speechbrain",
        "git+https://github.com/ai4bharat/IndicF5.git",
    ])


def ensure_cloudflared() -> str:
    existing = shutil.which("cloudflared")
    if existing:
        return existing
    target = "/usr/local/bin/cloudflared"
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    print(f"[VoiceLab Colab Runner] Downloading cloudflared from {url}...")
    urllib.request.urlretrieve(url, target)
    os.chmod(target, 0o755)
    return target


def main():
    ensure_dependencies()
    cf_bin = ensure_cloudflared()

    token = os.environ.get("LOCAL_NODE_AUTH_TOKEN") or f"vl_{secrets.token_hex(12)}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    node_script = os.path.join(script_dir, "local_neural_node.py")

    env = os.environ.copy()
    env["ALLOW_TUNNEL"] = "1"
    env["ENABLE_HF_INDICF5"] = "1"
    env["LOCAL_NODE_AUTH_TOKEN"] = token

    node_proc = subprocess.Popen(
        [
            sys.executable,
            node_script,
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
            "--allow-tunnel",
            "--enable-hf-indicf5",
            "--token",
            token,
        ],
        env=env,
    )

    time.sleep(3)
    tunnel_proc = subprocess.Popen(
        [cf_bin, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        for line in tunnel_proc.stdout:
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                print("\n" + "=" * 68)
                print("VOICELAB AI MODE 2 GPU NODE ONLINE (UNLIMITED 0-QUOTA CLONING)")
                print(f"Node URL   : {match.group(0)}")
                print(f"Auth Token : {token}")
                print("=" * 68 + "\n")
                break
        node_proc.wait()
    finally:
        tunnel_proc.terminate()
        node_proc.terminate()


if __name__ == "__main__":
    main()
