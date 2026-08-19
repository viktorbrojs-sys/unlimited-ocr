---
name: unlimited-ocr-local-port
description: unlimited-ocr project — local port of the HF Space; quantization
  incompatible with the model, fix pushed (0a4f1c8); awaiting bf16-freed-VRAM
  or CPU test on the GPU PC
metadata:
  node_type: memory
  type: project
  originSessionId: sess_0a3c3bc7-a4da-4e36-a726-44f5d483d5aa
---

Project `/home/srs/Sync/Проекты программирования/unlimited-ocr`: a local port of the HF Space `baidu/Unlimited-OCR` — gradio 6 `Server` + the Space's own `index.html`, endpoints `/run_ocr` (token streaming via stdout interception), `/explode_pdf` (PDF→PNG), `/load_model` (lazy loading with variant picker), `/model_status`. Install is two-step (`install.sh`) because transformers==4.57.1 pins huggingface-hub<1.0 while gradio 6 needs >=1.16. Private repo: https://github.com/viktorbrojs-sys/unlimited-ocr (branch `main`).

**GPU PC (srs-mint-work, Linux Mint):** RTX 5060 8GB (Blackwell, sm_120), driver 580.173.02 / CUDA 13.0, torch cu12.8. VRAM eaters: Xorg+Cinnamon ~0.8GB, VirtualBox 385MB, RustDesk ~260MB (multiple procs), hermes-agent 114MB. GitHub auth: PAT over HTTPS (no gh CLI) — [[github-cli-setup]]. Folder: `~/unlimited-ocr`.

**Why:** Dev machine has no GPU and pip installs are blocked ([[zcode-session-python-is-appimage]]), so the server was never started there.

**How to apply — resolved issues (2026-08-17):**
- bf16 OOM on 8 GB card → auto-fallback bf16→8bit→4bit ([[user-prefers-minimal-token-solutions]]).
- 8-bit also OOM: needs both `bitsandbytes` AND `accelerate` (initially only bitsandbytes was installed).
- CPU fallback while CUDA visible → model.infer() sends inputs to cuda but model on cpu → crash "tensors on different devices". Fix: no CPU attempt when CUDA is visible; forced CPU re-execs process with `CUDA_VISIBLE_DEVICES=""`.
- Port 7860 busy (old instance not killed) → auto-increment to next free port.
- `app = Server()` placed after `@app.api` decorators → NameError at import. Fix: moved above all decorators. `py_compile` doesn't catch this — always verify decorator order with grep.
- **Lazy model loading** (commit `83ecb06`): server starts instantly, model loads on demand from UI variant selector (auto/bf16/8bit/4bit). Buttons: "Long · 640px" / "Base · 1024px". NGRAM toggle in index.html is decorative (no JS reads it); ngram prevention always-on in app.py.
- **Harmless warnings:** `torch_dtype is deprecated`, "newly initialized position_ids" — ignore.

**RESOLVED ISSUE (2026-08-17): "(no text detected)" was a quantization incompatibility, not a data problem.** bitsandbytes 8/4-bit does NOT work with this model's architecture (custom MoE decoder + R-SWA attention + SAM-ViT compressor): inference crashed with `MatMul8bitLt: Only two or three dimensional matrices are supported for argument A` and `masked_scatter_: expected self and source to have same dtypes but got Half and Float`; the web UI masked it as "(no text detected)". The model is only validated in bf16 (no quantization mentioned anywhere upstream). Fix (commit `0a4f1c8`): removed 8bit/4bit variants entirely; UI selector now `auto (bf16 → CPU) / bf16 / CPU`; CPU variant auto-restarts the server with `CUDA_VISIBLE_DEVICES=""` (os.execve from the load_model generator on a "cpu_restart" queue signal) so model.infer() keeps all tensors on CPU. GPU-PC needs `git pull` to get it.

**NEXT STEPS (pending on GPU PC):** (1) free VRAM (close VirtualBox 385MB + RustDesk ~450MB → ~7 GB free of 8) and try `bf16` (~6.7 GB weights + context + activations — tight, may still OOM; desktop Xorg+Cinnamon ~0.8 GB can't be freed); (2) if bf16 fails → `CPU` variant: 21 GB RAM fits float32 (13.3 GB weights), expect minutes per page; (3) once ANY variant actually produces text — finally evaluate Cyrillic quality on real Russian documents. Speed reference: RTX 5060, if bf16 fits it will be fast.

User's use case: documents are in **Russian**, wants **bulk processing**. Paper: no Cyrillic guarantee, no handwriting support. Batch path: repo's `infer.py` (SGLang) or custom CLI script (offered, not yet confirmed). Related: [[user-prefers-exact-reference-replica]], [[user-prefers-minimal-token-solutions]].
