import hashlib
import json
import random
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone

import config
from core import models
from core import search as web

ROLES = ("user", "assistant")


class RequestError(Exception):
    status = 400


class UserError(RequestError):
    status = 400


class Busy(RequestError):
    status = 409


class RateLimited(RequestError):
    status = 429


def tokens(text):
    return len(text) // 4 + 1  # rough; provider-reported usage replaces this when present


def estimate(messages):
    n = 0
    for m in messages:
        if isinstance(m["content"], str):
            n += tokens(m["content"])
        else:
            n += sum(tokens(p["text"]) if p["type"] == "text" else config.IMAGE_TOKENS
                     for p in m["content"])
    return n


class RequestManager:
    # shortcut: all state is per-process memory (one Flask process, as the plan requires);
    # run several workers and every limit below becomes per-worker.
    def __init__(self):
        self._lock = threading.Lock()
        self._minute = deque()  # monotonic times of calls in the last 60 s
        self._day = [None, 0]  # [UTC date, calls today]; local guard, OpenRouter keeps the real count
        self._active = set()
        self._slots = threading.BoundedSemaphore(config.MAX_CONCURRENT)
        self.stats = {"model_calls": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0, "searches": 0}

    # ---- validation and budgeting ----
    def validate(self, text, n_images, history_json, force, video=False):
        text = (text or "").strip()
        if len(text) > config.MAX_TEXT_CHARS:
            raise UserError(f"text longer than {config.MAX_TEXT_CHARS} characters")
        if not text and not n_images and not video:
            raise UserError("send a question, an image or a video")
        if n_images > config.MAX_IMAGES:
            raise UserError(f"at most {config.MAX_IMAGES} images per request")
        if video and n_images:
            raise UserError("send a video or images, not both")
        if force not in ("", "vision", "reasoning"):
            raise UserError("force must be empty, 'vision' or 'reasoning'")
        if force == "vision" and not (n_images or video):
            raise UserError("the vision model needs an image or a video")
        if force == "reasoning" and (n_images or video):
            raise UserError("the reasoning model is text-only here; remove the images or video")
        try:
            history = json.loads(history_json or "[]")
        except ValueError:
            raise UserError("history is not valid JSON")
        if not isinstance(history, list) or not all(
                isinstance(m, dict) and m.get("role") in ROLES and isinstance(m.get("content"), str)
                for m in history):
            raise UserError("history must be a list of {role: user|assistant, content: text}")
        # Rebuild each entry so extra keys (or a smuggled "system" role) never reach a model.
        history = [{"role": m["role"], "content": m["content"]} for m in history]
        frames = n_images + (config.FRAMES_PER_SEGMENT if video else 0)  # one vision call carries one segment
        return text, self.fit_history(history, tokens(text) + frames * config.IMAGE_TOKENS)

    def fit_history(self, history, used):
        """Keep the newest turns that fit the input budget."""
        room = config.MAX_INPUT_TOKENS - used - 1000  # 1000 = system prompt and findings margin
        if room < 0:
            raise UserError("request is too large for the token budget")
        kept = []
        for m in reversed(history):
            room -= tokens(m["content"])
            if room < 0:
                break
            kept.append(m)
        kept = kept[::-1]
        while kept and kept[0]["role"] == "assistant":  # trimming can cut a pair in half
            kept.pop(0)
        return kept

    def search(self, query):
        with self._lock:
            self.stats["searches"] += 1
        return web.search(query)

    # ---- duplicate protection ----
    def key(self, text, images, history, force):
        h = hashlib.sha256(json.dumps([text, history, force]).encode())
        for img in images:
            h.update(hashlib.sha256(img).digest())
        return h.hexdigest()

    @contextmanager
    def guard(self, key):
        with self._lock:
            if key in self._active:
                raise Busy("the same request is already running")
            self._active.add(key)
        try:
            yield
        finally:
            with self._lock:
                self._active.discard(key)

    # ---- rate limit, queue, retry ----
    def _take_slot(self):
        while True:
            with self._lock:
                now = time.monotonic()
                while self._minute and now - self._minute[0] >= 60:
                    self._minute.popleft()
                today = datetime.now(timezone.utc).date()
                if self._day[0] != today:
                    self._day = [today, 0]
                if self._day[1] >= config.PER_DAY:
                    raise RateLimited("daily request limit reached (resets 00:00 UTC)")
                if len(self._minute) < config.PER_MINUTE:
                    self._minute.append(now)
                    self._day[1] += 1
                    return
                wait = 60 - (now - self._minute[0])
            if wait > config.MAX_WAIT_S:
                raise RateLimited(f"per-minute limit reached, retry in {wait:.0f}s")
            time.sleep(wait)

    def call(self, model, messages, max_tokens, reasoning=False):
        """Queue, rate-limit and retry one model call; record real usage."""
        models.preflight(model)
        if not self._slots.acquire(timeout=config.QUEUE_TIMEOUT_S):
            raise Busy("server busy, try again shortly")
        try:
            for attempt in range(config.MAX_RETRIES + 1):
                self._take_slot()  # every attempt counts, retries included
                try:
                    r = models.chat(model, messages, max_tokens, reasoning)
                except models.ModelError as e:
                    with self._lock:
                        self.stats["errors"] += 1
                    delay = e.retry_after if e.retry_after is not None else 2 ** attempt + random.random()
                    if not e.transient or attempt == config.MAX_RETRIES or delay > config.MAX_WAIT_S:
                        raise
                    time.sleep(delay)
                    continue
                u = r["usage"]
                with self._lock:
                    self.stats["model_calls"] += 1
                    self.stats["prompt_tokens"] += u.get("prompt_tokens") or estimate(messages)
                    self.stats["completion_tokens"] += u.get("completion_tokens") or tokens(r["text"])
                return r
        finally:
            self._slots.release()
