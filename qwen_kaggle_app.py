"""
Stable Kaggle launcher for Qwen3-TTS voice cloning.

Usage in Kaggle:

import urllib.request
url = "https://raw.githubusercontent.com/annapozdn/apozdn/main/qwen_kaggle_app.py"
exec(urllib.request.urlopen(url).read().decode("utf-8"))
"""

from __future__ import annotations

import importlib.metadata as md
import os
import subprocess
import sys
import tempfile
import textwrap
import time


QWEN_TTS_VERSION = "0.1.1"

REQUIRED = {
    "transformers": "4.55.4",
    "huggingface_hub": "0.34.4",
    "gradio": "5.50.0",
}

EXTRA_PACKAGES = [
    "soundfile",
    "numpy",
    "accelerate",
    "sox",
]


def _version(package: str) -> str | None:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


def _install_if_needed() -> None:
    wrong = {
        package: (_version(package), version)
        for package, version in REQUIRED.items()
        if _version(package) != version
    }
    qwen_current = _version("qwen-tts")
    if qwen_current != QWEN_TTS_VERSION:
        wrong["qwen-tts"] = (qwen_current, QWEN_TTS_VERSION)

    if not wrong:
        return

    print("Installing stable Qwen-TTS dependencies...")
    for package, (current, target) in wrong.items():
        print(f"  {package}: {current or 'not installed'} -> {target}")

    pins = [f"{package}=={version}" for package, version in REQUIRED.items()]
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--force-reinstall",
            *pins,
            *EXTRA_PACKAGES,
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            f"qwen-tts=={QWEN_TTS_VERSION}",
        ]
    )

    raise SystemExit(
        textwrap.dedent(
            """

            Dependencies were installed/changed.

            IMPORTANT:
            1. Kaggle: Draft Session -> More settings -> Restart & Clear Cell Outputs
            2. Run this same tiny loader cell again.

            This restart is required because Python may already have imported the old
            transformers/huggingface_hub modules.
            """
        ).strip()
    )


_install_if_needed()

import numpy as np
import soundfile as sf
import torch
import gradio as gr
import huggingface_hub.constants as hf_constants
from qwen_tts import Qwen3TTSModel


if not hasattr(hf_constants, "HF_HUB_ENABLE_HF_TRANSFER"):
    hf_constants.HF_HUB_ENABLE_HF_TRANSFER = False


def _cuda_report() -> str:
    if not torch.cuda.is_available():
        return "CUDA is not available. In Kaggle, enable GPU T4 x2."
    free, total = torch.cuda.mem_get_info()
    return f"CUDA available. Free memory: {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB"


print(_cuda_report())
print("Loading Qwen3-TTS Base model on GPU...")

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.float32,
    attn_implementation="eager",
)

print("Model loaded.")


def clone_voice(new_text, language, reference_audio, ref_transcript):
    try:
        if reference_audio is None:
            return None, "Please upload a reference audio file."
        if not ref_transcript or not ref_transcript.strip():
            return None, "Please provide the transcript of your reference audio."
        if not new_text or not new_text.strip():
            return None, "Please provide text to synthesize."

        wavs, sr = model.generate_voice_clone(
            text=new_text,
            language=language,
            ref_audio=reference_audio,
            ref_text=ref_transcript,
        )

        audio_data = np.array(wavs[0] if isinstance(wavs, (list, tuple)) else wavs)
        out_path = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
            dir="/kaggle/working" if os.path.isdir("/kaggle/working") else None,
        ).name
        sf.write(out_path, audio_data, int(sr))
        return out_path, f"Voice cloned successfully. Saved WAV: {out_path} | Sample rate: {sr} Hz"

    except Exception as exc:
        import traceback

        return None, f"Error: {exc}\n\n{traceback.format_exc()}"


languages = [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian",
]

interface = gr.Interface(
    fn=clone_voice,
    inputs=[
        gr.Textbox(label="New text", lines=7, value=""),
        gr.Dropdown(choices=languages, value="Russian", label="Language"),
        gr.Audio(label="Reference audio", type="filepath", sources=["upload", "microphone"]),
        gr.Textbox(label="Reference audio transcript", lines=5, value=""),
    ],
    outputs=[
        gr.Audio(label="Cloned voice output", type="filepath"),
        gr.Textbox(label="Status", lines=5),
    ],
    title="Qwen3-TTS Voice Cloning",
    description="Stable Kaggle launcher. Russian text is sent as typed; output is saved as WAV.",
)

print("Launching Gradio interface...")
interface.launch(share=True, debug=False, prevent_thread_lock=True)

while True:
    time.sleep(60)
