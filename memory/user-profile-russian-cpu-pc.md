---
name: user-profile-russian-cpu-pc
description: User communicates in Russian; local PC is CPU-only (no NVIDIA GPU),
  CUDA GPU lives on a separate machine
metadata:
  node_type: memory
  type: user
  originSessionId: sess_0a3c3bc7-a4da-4e36-a726-44f5d483d5aa
---

The user writes in Russian — respond in Russian. Their local PC has no NVIDIA GPU (CPU-only, 31 GB RAM, ~47 GB free disk as of 2026-08). Heavy GPU workloads are developed locally but must be run/tested on their other machine (`srs-mint-work`, Linux Mint) — its GPU has only **8 GB VRAM (7.49 GiB capacity)**, so models >7 GB in bf16 OOM there; plan on quantization (8-bit/4-bit) or offload for anything that size. They ask conceptual "what's the difference between X and Y" questions about the web stack (e.g. Gradio Interface vs FastAPI) — not a web-framework expert, so give explanatory context, not jargon. Related: [[unlimited-ocr-local-port]].
