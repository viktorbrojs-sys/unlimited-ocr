"""
Unlimited-OCR — local inference server (port of the HF Space baidu/Unlimited-OCR).

Differences from the Space version:
  • No ZeroGPU: the @spaces.GPU decorator is gone, pages can run as long as needed.
  • No runtime `pip install` of pinned deps — install requirements.txt into a venv.
  • The server starts instantly; the model loads on demand from the web UI
    (variant selector in the header: auto / bf16 / 8-bit / 4-bit), with automatic
    fallback to quantized variants on small-VRAM GPUs.
  • A lock serializes model.infer() calls: one GPU, one inference at a time,
    and the stdout-interception streaming assumes a single inference in flight.

Run:  python app.py   →  http://127.0.0.1:7860 (busy ports auto-increment)
"""

import atexit
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
_model_label: str | None = None
_infer_lock = threading.Lock()
_temp_dirs: list[str] = []


def _cleanup() -> None:
    """Release model weights and remove temp directories on process exit."""
    global model, tokenizer
    model = None
    tokenizer = None
    gc.collect()
    import shutil
    for d in _temp_dirs:
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    _temp_dirs.clear()


atexit.register(_cleanup)

app = Server()

# UI variant selector → model-loading recipe. "auto" tries CUDA bf16 first,
# then falls back to CPU float32 (slow). Bitsandbytes quantization (8/4-bit)
# is NOT compatible with this model's custom MoE + R-SWA architecture.
_MODEL_VARIANTS: dict[str, str] = {
    "bf16": "CUDA bf16",
    "cpu":  "CPU float32 (slow)",
}


def _variant_kwargs(name: str) -> tuple[dict, str | None]:
    """from_pretrained kwargs and .to() target for a variant label."""
    if name == "CUDA bf16":
        return dict(dtype=torch.bfloat16), "cuda"
    return dict(dtype=torch.float32), "cpu"


def _load_model_sync(variant: str, q) -> None:
    """Load the requested variant; push ("stage"|"done"|"error", text) into q.

    The model's own infer() code sends inputs to cuda whenever
    torch.cuda.is_available(), so a CPU-resident model next to a visible GPU
    crashes with "tensors on different devices". The CPU variant therefore
    requires CUDA to be hidden — we signal this via a magic exit code so the
    caller can re-exec the whole process with CUDA_VISIBLE_DEVICES="".
    """
    global tokenizer, model, _model_label
    if model is not None:
        q.put(("stage", "Unloading current model..."))
        model = None
        _model_label = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if tokenizer is None:
        q.put(("stage", "Loading tokenizer..."))
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if not torch.cuda.is_available():
        names = ["CPU float32"]
    elif variant == "auto":
        names = ["CUDA bf16", "CPU float32 (restart)"]
    else:
        names = [_MODEL_VARIANTS.get(variant, variant)]

    last_err: Exception | None = None
    for name in names:
        # CPU next to visible CUDA is impossible — signal caller to re-exec.
        if "CPU" in name and torch.cuda.is_available():
            q.put(("stage", "CPU mode requires hiding CUDA — the server will auto-restart..."))
            q.put(("cpu_restart", ""))
            return
        try:
            q.put(("stage", f"Loading model: {name}..."))
            kwargs, move_to = _variant_kwargs(name)
            m = AutoModel.from_pretrained(
                MODEL_NAME, trust_remote_code=True, use_safetensors=True, **kwargs
            )
            m = m.eval().to(move_to) if move_to else m.eval()
            model = m
            _model_label = name
            q.put(("done", name))
            return
        except Exception as e:
            last_err = e
            q.put(("stage", f"{name} failed: {e}"))
            model = None
            m = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(
        "Could not load the model. "
        f"(last error: {last_err})"
    )


def _target_labels(variant: str) -> list[str]:
    """Resolve variant name to the list of model labels it would try."""
    if not torch.cuda.is_available():
        return ["CPU float32"]
    if variant == "auto":
        return ["CUDA bf16", "CPU float32 (restart)"]
    return [_MODEL_VARIANTS.get(variant, variant)]


@app.api(stream_every=0.5)
def load_model(variant: str = "auto") -> Iterator[dict]:
    """
    Load (or switch) the model on demand. Streams progress dicts:
    {"stage": str, "ready": bool, "label": str | None}

    For "cpu" / "auto falling back to CPU": the server auto-restarts itself
    with CUDA hidden so the model's infer() doesn't send tensors to a wrong device.
    """
    if variant != "auto" and variant not in _MODEL_VARIANTS:
        yield {"stage": f"Unknown variant: {variant}", "ready": False, "label": None}
        return

    # Skip reload if the model is already loaded with the requested variant.
    if _model_label in _target_labels(variant):
        yield {"stage": f"Model already loaded: {_model_label}", "ready": True, "label": _model_label}
        return

    if not _infer_lock.acquire(blocking=False):
        yield {"stage": "OCR in progress — try again after it finishes", "ready": False, "label": None}
        return

    q: queue.Queue = queue.Queue()

    def _worker():
        try:
            _load_model_sync(variant, q)
        except Exception as e:
            q.put(("error", str(e)))
        finally:
            q.put(None)

    Thread(target=_worker, daemon=True).start()
    try:
        while True:
            try:
                item = q.get(timeout=15)
            except queue.Empty:
                # Keep-alive: prevent SSE timeout during long from_pretrained().
                yield {"stage": "Still loading model… (download may take a while)", "ready": False, "label": None}
                continue
            if item is None:
                break
            kind, msg = item
            if kind == "stage":
                yield {"stage": msg, "ready": False, "label": None}
            elif kind == "done":
                yield {"stage": f"Model ready: {msg}", "ready": True, "label": msg}
                return
            elif kind == "cpu_restart":
                yield {"stage": "Restarting server in CPU mode...", "ready": False, "label": None}
                import time
                time.sleep(1)
                env = {**os.environ, "UNLIMITED_OCR_CPU_REEXEC": "1", "CUDA_VISIBLE_DEVICES": ""}
                os.execve(sys.executable, [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]], env)
            elif kind == "error":
                yield {"stage": f"FAILED: {msg}", "ready": False, "label": None}
                return
    finally:
        _infer_lock.release()


@app.api()
def model_status() -> dict:
    return {"loaded": model is not None, "label": _model_label}


# ── PDF helper — CPU only ─────────────────────────────────────────────────────
def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """Convert every page of a PDF to a PNG. Returns list of file paths."""
    import fitz
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    _temp_dirs.append(tmp_dir)
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
        yield {"text": "Model is not loaded — pick a variant and press LOAD in the header.", "done": True}
        return

    path    = image_path["path"]
    out_dir = tempfile.mkdtemp(prefix="ocr_out_")
    _temp_dirs.append(out_dir)

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
    # Forced-CPU mode: the model's own infer() code sends inputs to cuda
    # whenever torch.cuda.is_available(), so a CPU-resident model next to a
    # visible GPU crashes. Restart with CUDA hidden before anything initialises it.
    if (
        os.environ.get("UNLIMITED_OCR_DEVICE", "").lower() == "cpu"
        and torch.cuda.is_available()
        and os.environ.get("_UNLIMITED_OCR_CPU_REEXEC") != "1"
    ):
        print("UNLIMITED_OCR_DEVICE=cpu: restarting with CUDA hidden for consistent CPU mode...")
        env = {**os.environ, "_UNLIMITED_OCR_CPU_REEXEC": "1", "CUDA_VISIBLE_DEVICES": ""}
        os.execve(sys.executable, [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]], env)

    # The server starts immediately (no model); pick a variant in the web UI.
    port = int(os.environ.get("PORT", "7860"))
    while True:
        try:
            app.launch(
                server_name=os.environ.get("HOST", "127.0.0.1"),
                server_port=port,
                show_error=True,
            )
            break
        except OSError:
            print(f"Port {port} is busy — trying {port + 1}...")
            port += 1
