#!/bin/sh
# Setup for the local Unlimited-OCR server.
#
# Two-step install: transformers==4.57.1 pins huggingface-hub<1.0 while
# gradio 6 needs >=1.16 — pip cannot resolve both at once, so transformers
# goes in afterwards (pip warns about gradio's pin; the server works fine —
# this is exactly how the HF Space handles it).
set -e

python3 -m venv .venv

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install "transformers==4.57.1"
# Quantized fallback deps for GPUs with <12 GB VRAM (device_map + 8/4-bit):
.venv/bin/pip install bitsandbytes accelerate

echo
echo "Done. Start the server with:"
echo "  .venv/bin/python app.py"
echo "Then open http://127.0.0.1:7860"
