import json
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import config


class ModelError(Exception):
    def __init__(self, msg, transient=False, retry_after=None):
        super().__init__(msg)
        self.transient = transient  # worth retrying
        self.retry_after = retry_after  # seconds, if the provider said


def _transient(status, msg=""):
    # A daily-quota 429 will not clear on retry, and every failed attempt costs more quota.
    if any(w in str(msg).lower() for w in ("per-day", "per day", "daily")):
        return False
    return status in (408, 429) or (status is not None and status >= 500)


def preflight(model):
    """Config checks that need no network, so they must not cost a rate-limit slot."""
    if not model.endswith(":free") and not config.ALLOW_PAID:
        raise ModelError(f"{model} is not a free model; set ALLOW_PAID=1 to allow it")
    if not config.API_KEY:
        raise ModelError("OPENROUTER_API_KEY is not set")


def chat(model, messages, max_tokens, reasoning=False):
    """One OpenRouter call. Returns {"text", "usage", "model"}; raises ModelError."""
    preflight(model)

    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": reasoning},
    }).encode()
    req = Request(config.API_URL, body, {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json",
    })
    try:
        with urlopen(req, timeout=config.TIMEOUT_S) as resp:
            raw = resp.read()
    except HTTPError as e:
        retry_after = e.headers.get("Retry-After") if e.headers else None
        try:
            msg = json.loads(e.read())["error"]["message"]
        except (ValueError, KeyError, TypeError):
            msg = e.reason
        raise ModelError(f"{model}: HTTP {e.code}: {msg}", _transient(e.code, msg),
                         float(retry_after) if retry_after and retry_after.isdigit() else None)
    except (OSError, HTTPException) as e:  # incl. URLError, timeouts and a connection dropped mid-answer
        raise ModelError(f"{model}: network error: {e}", transient=True)

    try:
        data = json.loads(raw)
    except ValueError:
        raise ModelError(f"{model}: response was not JSON")
    if not isinstance(data, dict):
        raise ModelError(f"{model}: unexpected response shape")
    if "error" in data:  # OpenRouter can return errors inside a 200
        err = data["error"]
        code = err.get("code") if isinstance(err, dict) else None
        msg = err.get("message") if isinstance(err, dict) else err
        raise ModelError(f"{model}: {msg}", _transient(code if isinstance(code, int) else None, msg))
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ModelError(f"{model}: unexpected response shape")
    if not text:
        raise ModelError(f"{model}: empty answer (thinking may have used the whole token budget)")
    return {"text": text, "usage": data.get("usage") or {}, "model": model}
