"""Latency profiler for the GOAT-Bench eval (diagnostics only).

Enabled ONLY when env HGR_LATENCY_PROFILE=1 and HGR_LATENCY_DIR is set.
When disabled every function is a no-op, so instrumented code behaves
identically to the uninstrumented code. Never raises: all internals are
wrapped so profiling can never crash or alter a run.

Outputs (under HGR_LATENCY_DIR):
  events.jsonl     one line per timed stage {name, dur_s, t, chan, ctx...}
  llm_calls.jsonl  one line per LLM/embedder call: kind (caller), duration,
                   usage tokens, full prompt text, per-image sha1+bytes
  images/<sha1>.png  deduped store of every image sent to the LLM
"""

import hashlib
import json
import os
import threading
import time

_ENABLED = os.environ.get("HGR_LATENCY_PROFILE", "") == "1"
_DIR = os.environ.get("HGR_LATENCY_DIR", "")
_SAVE_IMAGES = os.environ.get("HGR_LATENCY_SAVE_IMAGES", "1") == "1"

_lock = threading.Lock()
_ctx = {}
_last_tick = {}  # channel -> monotonic time
_files = {}


def enabled():
    return _ENABLED and bool(_DIR)


def _fh(name):
    f = _files.get(name)
    if f is None:
        os.makedirs(_DIR, exist_ok=True)
        f = open(os.path.join(_DIR, name), "a", buffering=1)
        _files[name] = f
    return f


def _emit(fname, rec):
    try:
        with _lock:
            rec["t"] = round(time.time(), 3)
            rec.update(_ctx)
            _fh(fname).write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def set_context(**kw):
    """Set fields (subtask_id, step, ...) attached to every later record."""
    if not enabled():
        return
    try:
        with _lock:
            _ctx.update({k: v for k, v in kw.items() if v is not None})
    except Exception:
        pass


def tick0(chan="step"):
    """Start/reset a timing channel."""
    if not enabled():
        return
    _last_tick[chan] = time.monotonic()


def tick(name, chan="step", **meta):
    """Record the elapsed time since the previous tick on `chan` as `name`."""
    if not enabled():
        return
    try:
        now = time.monotonic()
        prev = _last_tick.get(chan)
        _last_tick[chan] = now
        if prev is None:
            return
        rec = {"name": name, "chan": chan, "dur_s": round(now - prev, 4)}
        if meta:
            rec["meta"] = meta
        _emit("events.jsonl", rec)
    except Exception:
        pass


def record(name, dur_s, **meta):
    """Record an explicitly measured duration."""
    if not enabled():
        return
    rec = {"name": name, "dur_s": round(float(dur_s), 4)}
    if meta:
        rec["meta"] = meta
    _emit("events.jsonl", rec)


def _image_entry(b64):
    try:
        sha = hashlib.sha1(b64.encode("ascii", "ignore")).hexdigest()
        entry = {"sha1": sha, "b64_chars": len(b64)}
        if _SAVE_IMAGES:
            img_dir = os.path.join(_DIR, "images")
            os.makedirs(img_dir, exist_ok=True)
            path = os.path.join(img_dir, sha + ".png")
            if not os.path.exists(path):
                import base64 as _b64

                with open(path, "wb") as fh:
                    fh.write(_b64.b64decode(b64))
        return entry
    except Exception:
        return {"sha1": None, "b64_chars": len(b64) if b64 else 0}


def llm_call(kind, sys_prompt, contents, response, usage, dur_s, attempt=0, error=None):
    """Record one chat-completion call: full text prompt, image refs, usage.

    `contents` is the HGR list of (text,) / (text, b64image) tuples.
    """
    if not enabled():
        return
    try:
        text_parts = []
        images = []
        for c in contents or []:
            text_parts.append(c[0])
            if len(c) == 2 and c[1]:
                images.append(_image_entry(c[1]))
        rec = {
            "kind": kind,
            "dur_s": round(float(dur_s), 3),
            "attempt": attempt,
            "n_images": len(images),
            "prompt_text_chars": sum(len(t) for t in text_parts) + len(sys_prompt or ""),
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "sys_prompt": sys_prompt,
            "prompt_text": text_parts,
            "images": images,
            "response": response,
            "error": str(error) if error else None,
        }
        _emit("llm_calls.jsonl", rec)
    except Exception:
        pass


def embed_call(kind, dur_s, **meta):
    """Record one embedding-server call (:8002 text / :8006 VL)."""
    if not enabled():
        return
    rec = {"kind": kind, "dur_s": round(float(dur_s), 4)}
    if meta:
        rec["meta"] = meta
    _emit("llm_calls.jsonl", rec)
