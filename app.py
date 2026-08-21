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
import shutil
import sys
import queue
import tempfile
import threading
import time
from threading import Thread
from typing import Iterator

# Reduce CUDA memory fragmentation; must be set before the first CUDA allocation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

if not torch.cuda.is_available():
    # Некоторые модели (например, remote-код baidu/Unlimited-OCR) жёстко
    # вызывают .cuda() в расчёте на GPU, игнорируя фактическое устройство
    # модели (device_map='cpu'). Делаем .cuda() безопасным no-op.
    torch.Tensor.cuda = lambda self, *args, **kwargs: self

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
_model_state_lock = threading.Lock()  # Protects model / tokenizer / _model_label
_temp_dirs: list[str] = []
_temp_dirs_lock = threading.Lock()  # Protect _temp_dirs from concurrent access


def _cleanup() -> None:
    """Release model weights and remove temp directories on process exit."""
    global model, tokenizer
    with _model_state_lock:
        model = None
        tokenizer = None
    gc.collect()
    with _temp_dirs_lock:
        for d in _temp_dirs[:]:  # Copy list to avoid modification during iteration
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
#
# AWQ (sahilchachra/Unlimited-OCR-AWQ, community requantization) was tried
# and removed in 1.2.3: `compressed-tensors` decompresses the weights to
# fp16 at inference time, so there is no actual VRAM or speed benefit at
# runtime (only a smaller download) — while still carrying the accuracy
# risk of an unofficial, third-party quantization of an OCR model, where
# accuracy is the entire point. Net negative, no upside; see CHANGELOG.
_MODEL_VARIANTS: dict[str, str] = {
    "bf16":  "CUDA bf16 (full precision)",
    "cpu":   "CPU float32 (slow)",
}


def _variant_kwargs(name: str) -> tuple[dict, str | None]:
    """from_pretrained kwargs and .to() target for a variant label."""
    if name == "CUDA bf16 (full precision)":
        return dict(dtype=torch.bfloat16), "cuda"
    # For CPU mode, explicitly ensure no CUDA device is used
    return dict(dtype=torch.float32, device_map="cpu"), "cpu"


def _load_model_sync(variant: str, q) -> None:
    """Load the requested variant; push ("stage"|"done", text) into q, or
    raise on failure (caller turns that into an "error" queue item).

    No hidden auto-fallback here: the caller (UI) picks device + precision
    explicitly, and if that exact choice fails to load, we say so plainly
    rather than silently switching to something else. The only exception is
    hardware, not choice: if there is no CUDA device at all, CPU is used
    regardless of what was requested, since there's nothing else to try.

    The model's own infer() code sends inputs to cuda whenever
    torch.cuda.is_available(), so a CPU-resident model next to a visible GPU
    crashes with "tensors on different devices". The CPU variant therefore
    requires CUDA to be hidden — we signal this via a magic exit code so the
    caller can re-exec the whole process with CUDA_VISIBLE_DEVICES="".
    """
    global tokenizer, model, _model_label
    with _model_state_lock:
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
        if variant != "cpu":
            q.put(("stage", "No CUDA GPU detected — using CPU regardless of the selected precision."))
        name = "CPU float32 (slow)"
    else:
        name = _MODEL_VARIANTS.get(variant, variant)

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
        with _model_state_lock:
            model = m
            _model_label = name
        q.put(("done", name))
    except RuntimeError as e:
        with _model_state_lock:
            model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        error_str = str(e).lower()
        if "cuda" in error_str and ("out of memory" in error_str or "oom" in error_str):
            raise RuntimeError(
                f"{name}: not enough GPU memory to load this model "
                f"(CUDA out of memory). Switch to CPU and press LOAD, or "
                f"free up VRAM and try again. Original error: {e}"
            ) from e
        raise RuntimeError(f"Could not load {name}: {e}") from e
    except Exception as e:
        with _model_state_lock:
            model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise RuntimeError(f"Could not load {name}: {e}") from e


def _target_label(variant: str) -> str:
    """Resolve variant name to the model label it would load."""
    if not torch.cuda.is_available():
        return "CPU float32 (slow)"
    return _MODEL_VARIANTS.get(variant, variant)


@app.api(stream_every=0.5)
def load_model(variant: str = "cpu") -> Iterator[dict]:
    """
    Load (or switch) the model with the exact variant requested — no hidden
    auto-fallback. Streams progress dicts:
    {"stage": str, "ready": bool, "label": str | None}

    For "cpu": the server auto-restarts itself with CUDA hidden so the
    model's infer() doesn't send tensors to a wrong device.
    """
    if variant not in _MODEL_VARIANTS:
        yield {"stage": f"Unknown variant: {variant}", "ready": False, "label": None}
        return

    # Skip reload if the model is already loaded with the requested variant.
    with _model_state_lock:
        current_label = _model_label
    if current_label == _target_label(variant):
        yield {"stage": f"Model already loaded: {current_label}", "ready": True, "label": current_label}
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
    lock_held = True
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
                time.sleep(1)
                env = {**os.environ, "UNLIMITED_OCR_CPU_REEXEC": "1", "CUDA_VISIBLE_DEVICES": ""}
                os.execve(sys.executable, [sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]], env)
            elif kind == "error":
                yield {"stage": f"FAILED: {msg}", "ready": False, "label": None}
                return
    finally:
        if lock_held:
            _infer_lock.release()


@app.api()
def model_status() -> dict:
    with _model_state_lock:
        return {"loaded": model is not None, "label": _model_label}


# ── PDF helper — CPU only ─────────────────────────────────────────────────────
def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    """Convert every page of a PDF to a PNG. Returns list of file paths."""
    import fitz
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    with _temp_dirs_lock:
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
    # Wait a bit for the model to finish writing files
    time.sleep(0.5)

    try:
        files = sorted(os.listdir(out_dir))
    except Exception:
        return ""

    result = ""
    for fname in files:
        if fname.endswith((".txt", ".md")):
            fpath = os.path.join(out_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        result += content + "\n"
            except Exception:
                pass

    # If no .txt/.md files found, try to read any file
    if not result:
        for fname in files:
            fpath = os.path.join(out_dir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if content:
                            result += content + "\n"
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
        self._lock = threading.Lock()

    def write(self, data):
        self.original_stdout.write(data)
        self.original_stdout.flush()
        with self._lock:
            if threading.current_thread() is self.target_thread:
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
    ngram_guard: bool = True,
) -> Iterator[dict]:
    """
    Stream OCR output for one image page token-by-token.

    Yields dicts: {"text": str, "done": bool}

    mode: 'gundam' — fast (640 px crop)
          'base'   — accurate (1024 px)
    ngram_guard: if True (default), forbids the model from repeating the
        same 35-token sequence back to back (no_repeat_ngram_size) — guards
        against degenerate repetition loops on long/hard documents. Disable
        for documents with legitimate long repeated sequences (e.g. tables,
        repeated boilerplate) where the guard could suppress correct output.
    """
    global model, _model_label
    with _model_state_lock:
        current_model = model
    if current_model is None:
        yield {"text": "Model is not loaded — pick a device/model and press LOAD in the header.", "done": True}
        return

    # Hold the inference lock for the whole request (not just inside the
    # worker thread) so a concurrent load_model() call — which also takes
    # this lock — cannot swap/unload the model while we're mid-inference.
    if not _infer_lock.acquire(blocking=False):
        yield {"text": "Another OCR run or model load is in progress — please wait.", "done": True}
        return

    try:
        path    = image_path["path"]
        out_dir = tempfile.mkdtemp(prefix="ocr_out_")
        with _temp_dirs_lock:
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
            no_repeat_ngram_size=(35 if ngram_guard else 0),
            ngram_window=ngram_window,
            save_results=True,
        )

        q = queue.Queue()
        errors = []
        oom = {"hit": False}

        def _infer_thread():
            # _infer_lock is already held by the outer generator (see above) —
            # do not re-acquire it here, this thread just does the actual work.
            try:
                if not torch.cuda.is_available():
                    # Веса модели уже загружены корректно (bfloat16 не
                    # подменялся на этапе импорта/загрузки). Здесь, только на
                    # время самого inference-вызова, временно подменяем
                    # torch.bfloat16 на float32 — так remote-код модели,
                    # который жёстко приводит входное изображение к
                    # .to(torch.bfloat16), получит тип, совпадающий с
                    # float32-весами CPU-модели.
                    _orig_bfloat16 = torch.bfloat16
                    torch.bfloat16 = torch.float32
                    try:
                        current_model.infer(tokenizer, **_infer_kwargs)
                    finally:
                        torch.bfloat16 = _orig_bfloat16
                else:
                    current_model.infer(tokenizer, **_infer_kwargs)
            except RuntimeError as e:
                error_str = str(e).lower()
                if "cuda" in error_str and ("out of memory" in error_str or "oom" in error_str):
                    oom["hit"] = True
                errors.append(str(e))
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

        if oom["hit"]:
            # Not enough VRAM mid-inference (as opposed to at load time —
            # that path is handled separately in _load_model_sync). The
            # model object is still technically "loaded" but its CUDA state
            # may be unreliable after an OOM, and it's holding VRAM we want
            # back — unload it explicitly instead of leaving a wounded
            # model sitting on the GPU. This is a deliberate, visible action
            # reported to the user, not a silent auto-switch to CPU: they
            # still have to pick CPU and press LOAD themselves.
            with _model_state_lock:
                model = None
                _model_label = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            full_text = ""
        else:
            # ── Fallback/Final: read file to get clean text ───────────────
            full_text = _collect_output(out_dir)
    finally:
        # Release the lock as soon as the model is actually done with this
        # request (inference thread joined, result file already read, and
        # any OOM cleanup done) — do NOT wait for this generator to reach
        # its own natural end.
        #
        # Why: everything below is just replaying already-collected text to
        # the client for the UI (no model/GPU access anymore). If we kept
        # the lock held until the generator's implicit end-of-function
        # `return`, we'd depend on the SSE-streaming wrapper calling next()
        # one extra time *after* it has already seen a payload with
        # "done": True. Some streaming implementations treat "done": True
        # as "no more messages will arrive" and simply stop driving the
        # generator at that point — the generator (and this `finally`)
        # then never actually runs, and the lock stays held forever, which
        # is exactly the "Another OCR run or model load is in progress"
        # hang seen after a run that otherwise completed successfully.
        # Releasing here removes that dependency entirely.
        _infer_lock.release()

    if oom["hit"]:
        raise RuntimeError(
            "Не хватило видеопамяти (VRAM) во время обработки этого "
            "изображения. Модель выгружена из GPU, чтобы освободить "
            "память — переключите вариант на CPU в шапке страницы и "
            f"нажмите LOAD. Исходная ошибка: {', '.join(errors)}"
        )

    if full_text and full_text.strip() != accumulated.strip():
        # The clean file-based text differs from what was already streamed
        # live from stdout (formatting cleanup, or stdout capture missed
        # something) — replay it with a real, but time-bounded, typewriter
        # effect. Step size scales with length so long documents don't take
        # forever: capped at ~200 steps / ~2.4s regardless of text length.
        total = len(full_text)
        steps = min(total, 200)
        step_size = max(1, total // steps)
        pos = 0
        while pos < total:
            pos = min(pos + step_size, total)
            yield {"text": full_text[:pos], "done": pos == total}
            time.sleep(0.012)
    elif full_text:
        # Identical to what's already on screen — no point re-flashing it.
        yield {"text": full_text, "done": True}
    elif accumulated:
        yield {"text": accumulated, "done": True}
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
    pages = pdf_to_images(pdf_file["path"], dpi=300)
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
