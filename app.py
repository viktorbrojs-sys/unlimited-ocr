"""
Unlimited-OCR — local inference server (port of the HF Space baidu/Unlimited-OCR).

Differences from the Space version:
  • No ZeroGPU: the @spaces.GPU decorator is gone, pages can run as long as needed.
  • No runtime `pip install` of pinned deps — install requirements.txt into a venv.
  • Falls back to CPU (float32) when CUDA is unavailable (slow, but works).
  • A lock serializes model.infer() calls: one GPU, one inference at a time,
    and the stdout-interception streaming assumes a single inference in flight.

Run:  python app.py   →  http://127.0.0.1:7860
"""

import gc
import os
import sys
import queue
import tempfile
import threading
from threading import Thread
from typing import Iterator

# Reduce CUDA memory fragmentation; must be set before the first CUDA allocation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModel, AutoTokenizer
from gradio import Server
from gradio.data_classes import FileData
from fastapi.responses import HTMLResponse

# ──────────────────────────────────────────────────────────────────────────────
# Model loading (in main, not at import time, so the module can be imported
# without a GPU / without downloading the weights)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME = "baidu/Unlimited-OCR"

tokenizer = None
model = None
_infer_lock = threading.Lock()


def load_model() -> None:
    global tokenizer, model
    if model is not None:
        return

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load attempts, most preferable first. Small-VRAM GPUs (<12 GB) OOM on the
    # full bf16 weights (~6.7 GB + context + activations), so fall back to
    # bitsandbytes quantization and, as a last resort, to slow CPU float32.
    forced_cpu = os.environ.get("UNLIMITED_OCR_DEVICE", "").lower() == "cpu"
    attempts: list[tuple[str, dict, str | None]] = []  # (label, from_pretrained kwargs, .to() target)
    if forced_cpu:
        attempts.append(("CPU float32", dict(torch_dtype=torch.float32), "cpu"))
    elif torch.cuda.is_available():
        attempts.append(("CUDA bf16", dict(torch_dtype=torch.bfloat16), "cuda"))
        attempts.append(("CUDA 8-bit quantized (bitsandbytes)", dict(load_in_8bit=True, device_map={"": 0}), None))
        attempts.append(("CUDA 4-bit quantized (bitsandbytes)", dict(load_in_4bit=True, device_map={"": 0}), None))
        attempts.append(("CPU float32", dict(torch_dtype=torch.float32), "cpu"))
    else:
        print("WARNING: CUDA not available — running on CPU (this will be slow).")
        attempts.append(("CPU float32", dict(torch_dtype=torch.float32), "cpu"))

    last_err: Exception | None = None
    for name, kwargs, move_to in attempts:
        if "CPU" in name and torch.cuda.is_available():
            print("WARNING: falling back to CPU — inference will be very slow.")
        try:
            print(f"Loading model: {name}...")
            m = AutoModel.from_pretrained(
                MODEL_NAME, trust_remote_code=True, use_safetensors=True, **kwargs
            )
            m = m.eval().to(move_to) if move_to else m.eval()
            model = m
            print(f"Model ready: {name}")
            return
        except Exception as e:
            last_err = e
            print(f"  {name} failed: {e}")
            if "bitsandbytes" in str(e):
                print("  hint: .venv/bin/pip install bitsandbytes")
            model = None
            m = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"Could not load the model with any method: {last_err}")


app = Server()


# ── PDF helper — CPU only ─────────────────────────────────────────────────────
def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """Convert every page of a PDF to a PNG. Returns list of file paths."""
    import fitz
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def _collect_output(out_dir: str) -> str:
    """Read all text/markdown files written by model.infer()."""
    result = ""
    for fname in sorted(os.listdir(out_dir)):
        if fname.endswith((".txt", ".md")):
            with open(os.path.join(out_dir, fname), "r", encoding="utf-8") as f:
                result += f.read() + "\n"
    if not result:
        for fname in sorted(os.listdir(out_dir)):
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        result += f.read() + "\n"
                except Exception:
                    pass
    return result.strip()


# ── Single-page OCR — streaming generator ────────────────────────────────────
#
# Any generator decorated with @app.api() automatically streams each yielded
# value to the client via SSE. stream_every=0.1 → flushed at most every 100 ms.
#
class ThreadTargetedStdout:
    """Patch sys.stdout so print() output of the inference thread goes to a queue."""

    def __init__(self, target_thread, q, original_stdout):
        self.target_thread = target_thread
        self.q = q
        self.original_stdout = original_stdout

    def write(self, data):
        self.original_stdout.write(data)
        self.original_stdout.flush()
        if threading.current_thread() == self.target_thread:
            if data:
                lower_data = data.lower()
                if "tps:" in lower_data or "tokens/s" in lower_data:
                    return len(data)
                self.q.put(data)
        return len(data)

    def flush(self):
        self.original_stdout.flush()

    def __getattr__(self, name):
        return getattr(self.original_stdout, name)


@app.api(stream_every=0.1)
def run_ocr(
    image_path: FileData,
    mode: str = "gundam",
    prompt: str = "document parsing.",
) -> Iterator[dict]:
    """
    Stream OCR output for one image page token-by-token.

    Yields dicts: {"text": str, "done": bool}

    mode: 'gundam' — fast (640 px crop)
          'base'   — accurate (1024 px)
    """
    if model is None:
        yield {"text": "Model is not loaded yet — wait for startup to finish.", "done": True}
        return

    path    = image_path["path"]
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")

    if mode == "gundam":
        base_size, image_size, crop_mode, ngram_window = 1024, 640,  True,  128
    else:
        base_size, image_size, crop_mode, ngram_window = 1024, 1024, False, 128

    _infer_kwargs = dict(
        prompt=f"<image>{prompt}",
        image_file=path,
        output_path=out_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        max_length=8192,
        no_repeat_ngram_size=35,
        ngram_window=ngram_window,
        save_results=True,
    )

    q = queue.Queue()
    errors = []

    def _infer_thread():
        try:
            with _infer_lock:
                model.infer(tokenizer, **_infer_kwargs)
        except Exception as e:
            errors.append(str(e))

    thread = Thread(target=_infer_thread, daemon=True)

    original_stdout = sys.stdout
    targeted_stdout = ThreadTargetedStdout(thread, q, original_stdout)
    sys.stdout = targeted_stdout

    accumulated = ""
    try:
        thread.start()
        while thread.is_alive() or not q.empty():
            try:
                chunk = q.get(timeout=0.02)
                accumulated += chunk
                yield {"text": accumulated, "done": False}
            except queue.Empty:
                continue
    finally:
        sys.stdout = original_stdout
        thread.join()

    # ── Fallback/Final: read file to get clean text ───────────────────────────
    full_text = _collect_output(out_dir)

    if accumulated:
        if full_text:
            yield {"text": full_text, "done": True}
        else:
            yield {"text": accumulated, "done": True}
    else:
        if full_text:
            words = full_text.split()
            acc = ""
            for i, word in enumerate(words):
                acc += ("" if i == 0 else " ") + word
                if i % 5 == 0:
                    yield {"text": acc, "done": False}
            yield {"text": full_text, "done": True}
        else:
            if errors:
                raise RuntimeError(f"Inference failed: {', '.join(errors)}")
            yield {"text": "", "done": True}


# ── PDF explode — CPU only ───────────────────────────────────────────────────
@app.api()
def explode_pdf(pdf_file: FileData) -> dict:
    """
    Convert a PDF into per-page image paths (CPU only).
    The frontend then calls run_ocr once per page.
    """
    pages = pdf_to_images(pdf_file["path"], dpi=200)
    return {"pages": [{"path": p, "orig_name": os.path.basename(p)} for p in pages]}


# ── Static frontend ───────────────────────────────────────────────────────────
@app.get("/")
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    load_model()
    app.launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        show_error=True,
    )
