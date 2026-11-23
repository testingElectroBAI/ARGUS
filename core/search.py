from urllib.parse import urlsplit

import config


class SearchError(Exception):
    pass


def _key(url):
    """Same page under http/https, www, a trailing slash, tracking params or a #fragment gets one key."""
    u = urlsplit(url)
    q = "&".join(sorted(p for p in u.query.split("&") if p and not p.startswith(("utm_", "fbclid", "gclid"))))
    return f"{u.netloc.lower().removeprefix('www.')}{u.path.rstrip('/')}?{q}"


def _clean(s, n):
    return " ".join((s or "").split())[:n]


def search(query):
    """Top DuckDuckGo text results as [{title, url, snippet}], deduplicated. Raises SearchError.

    shortcut: one provider, snippets only. Result pages are never fetched: a hostile result could
    point the fetch at an internal address. Swap this body for SearXNG if DuckDuckGo starts blocking us.
    """
    query = _clean(query, 300)
    if not query:
        raise SearchError("empty search query")
    try:
        from ddgs import DDGS
        raw = DDGS(timeout=config.SEARCH_TIMEOUT_S).text(
            query, max_results=config.SEARCH_RESULTS * 2, backend="duckduckgo")
    except Exception as e:  # ddgs raises its own types; any failure just means "no web this time"
        raise SearchError(f"web search failed: {e}")
    seen, out = set(), []
    for r in raw:
        url, snippet = _clean(r.get("href"), 500), _clean(r.get("body"), 400)
        if urlsplit(url).scheme not in ("http", "https") or not snippet:
            continue
        if _key(url) in seen:
            continue
        seen.add(_key(url))
        out.append({"title": _clean(r.get("title"), 200) or url, "url": url, "snippet": snippet})
        if len(out) == config.SEARCH_RESULTS:
            break
    if not out:
        raise SearchError("web search returned no usable results")
    return out


def format_results(results):
    return "\n".join(f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet']}" for i, r in enumerate(results, 1))
