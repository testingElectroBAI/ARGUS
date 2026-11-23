import re

import config
from core import media, models, search
from core.request_manager import RequestError
from prompts import prompts

# shortcut: keyword heuristics, not a model call (a classifier call would burn free-tier quota);
# upgrade if routing mistakes show up in real use. "current" or "today" can fire when no web is needed.
_COMPLEX = re.compile(
    r"\b(why|compare|contrast|analy[sz]e|calculate|derive|prove|step[- ]by[- ]step|"
    r"trade-?offs?|pros and cons|evaluate|implications?)\b", re.I)
_WEB = re.compile(
    r"\b(search the web|web search|look(?:ing)? up|google|online|latest|current(?:ly)?|today|news|"
    r"recent(?:ly)?|price of|release date|fact[- ]check|verify)\b", re.I)
_QUERY = re.compile(r"^[ \t]*QUERY:[ \t]*(.+?)[ \t]*$", re.I | re.M)


def is_complex(text):
    return len(text) > 400 or bool(_COMPLEX.search(text))


def needs_web(text):
    return bool(_WEB.search(text))


def _user(text, images, labels=None):
    if not images:
        return {"role": "user", "content": text}
    if labels is None and len(images) > 1:
        labels = [f"Image {i}:" for i in range(1, len(images) + 1)]  # lets the answer say "image 2"
    parts = [{"type": "text", "text": text}]
    for i, u in enumerate(images):
        if labels:
            parts.append({"type": "text", "text": labels[i]})
        parts.append({"type": "image_url", "image_url": {"url": u}})
    return {"role": "user", "content": parts}


def _step(r):
    return {"model": r["model"], "tokens": r["usage"].get("total_tokens")}


def _out(answer, steps, results=None, notes=(), degraded=False):
    out = {"answer": answer, "steps": steps, "degraded": degraded or bool(notes),
           "sources": [{"title": r["title"], "url": r["url"]} for r in results or []]}
    if notes:
        out["note"] = "; ".join(notes)
    return out


def _web(rm, query, notes):
    try:
        return rm.search(query)
    except search.SearchError as e:
        notes.append(f"{e}; answered without web results")
        return None


def _split_query(text, fallback):
    found = list(_QUERY.finditer(text))
    if not found:
        return text, fallback
    last = found[-1]
    return (text[:last.start()] + text[last.end():]).strip(), last.group(1)


def run(rm, text, images, history, force="", video=None):
    """Run the cheapest workflow that answers the request.

    text-only             -> reasoning model (thinking only if the question looks complex)
    image / short video   -> vision model
    + complex ask         -> vision model (facts), then reasoning model (answer)
    long video            -> one vision call per segment, then reasoning model joins them
    + needs web           -> DuckDuckGo snippets are added before the reasoning step
    If the reasoning step fails after vision succeeded, return the vision facts, marked degraded.
    If web search fails, answer without it and say so in "note".
    """
    groups = video["segments"] if video else ([{"labels": None, "uris": images, "note": ""}] if images else [])
    text = text or ("Summarize the video." if video else "Describe the image.")
    base = [{"role": "system", "content": prompts.SYSTEM}] + history
    web = force != "vision" and needs_web(text)
    notes = []

    if not groups:
        results = _web(rm, text, notes) if web else None
        think = is_complex(text)
        prompt = prompts.synthesis(text, results=search.format_results(results)) if results else text
        r = rm.call(config.REASONING_MODEL, base + [_user(prompt, [])],
                    config.REASONING_MAX_TOKENS if think else config.MAX_OUTPUT_TOKENS, think)
        return _out(r["text"], [_step(r)], results, notes)

    multi = len(groups) > 1
    if not multi and (force == "vision" or not (web or is_complex(text))):
        g = groups[0]
        r = rm.call(config.VISION_MODEL, base + [_user(g["note"] + text, g["uris"], g["labels"])],
                    config.MAX_OUTPUT_TOKENS)
        return _out(r["text"], [_step(r)])

    facts, steps, query = [], [], text
    for g in groups:
        ask = g["note"] + prompts.FINDINGS.format(question=text) + (prompts.QUERY if web else "")
        try:
            r = rm.call(config.VISION_MODEL, base + [_user(ask, g["uris"], g["labels"])], config.MAX_OUTPUT_TOKENS)
        except (models.ModelError, RequestError) as e:
            if not facts:
                raise
            # earlier segments already cost quota: keep them rather than throw the whole video away
            notes.append(f"video analysis stopped after part {len(facts)} of {len(groups)}: {e}")
            break
        steps.append(_step(r))
        f, q = _split_query(r["text"], query)
        query = q if query == text else query  # the first segment's query wins
        facts.append(f)
    joined = "\n\n".join(f"Video {media.fmt(g['span'][0])}-{media.fmt(g['span'][1])}:\n{f}"
                         for g, f in zip(groups, facts)) if multi else facts[0]
    if force == "vision":  # long video, reasoning model not wanted: hand back the per-segment facts
        return _out(joined, steps, None, notes)

    results = _web(rm, query, notes) if web else None
    try:
        r = rm.call(config.REASONING_MODEL,
                    base + [_user(prompts.synthesis(text, joined, search.format_results(results) if results else ""), [])],
                    config.REASONING_MAX_TOKENS, True)
    except (models.ModelError, RequestError) as e:  # incl. rate limit hit between the calls
        notes.append(str(e))
        return _out(joined, steps, None, notes, True)
    return _out(r["text"], steps + [_step(r)], results, notes)
