import base64
import hashlib
import io
import math
import os
import re

import cv2
from PIL import Image, ImageOps

import config

# ---------------------------------------------------------------- images

# Check real bytes, not the browser-supplied filename or content type.
_SIGNATURES = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]


def _mime(data):
    for sig, mime in _SIGNATURES:
        if data.startswith(sig):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _prepare(data, mime):
    """Decode once to prove the file is real. Re-encode only if it is too big or EXIF-rotated."""
    try:
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        if w * h > config.MAX_IMAGE_PIXELS:
            raise ValueError("image has too many pixels")
        if max(w, h) <= config.MAX_IMAGE_SIDE and im.getexif().get(274, 1) == 1:
            im.load()  # a truncated file fails here, not at the provider
            return data, mime
        im = ImageOps.exif_transpose(im)  # models ignore the EXIF rotation flag
        im.thumbnail((config.MAX_IMAGE_SIDE, config.MAX_IMAGE_SIDE))
        out = io.BytesIO()
        if im.has_transparency_data:
            im.convert("RGBA").save(out, "PNG")
            return out.getvalue(), "image/png"
        im.convert("RGB").save(out, "JPEG", quality=85)
        return out.getvalue(), "image/jpeg"
    except (OSError, SyntaxError, Image.DecompressionBombError):  # incl. UnidentifiedImageError
        raise ValueError("image is corrupt or unreadable")


def _uri(data, mime):
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def to_data_uri(data):
    """Validate image bytes, shrink if needed, return a data URI. Raises ValueError."""
    if not data:
        raise ValueError("empty image file")
    if len(data) > config.MAX_IMAGE_BYTES:
        raise ValueError(f"image larger than {config.MAX_IMAGE_BYTES // 1024 // 1024} MB")
    mime = _mime(data)
    if not mime:
        raise ValueError("unsupported image type (use PNG, JPEG, GIF or WebP)")
    return _uri(*_prepare(data, mime))


# ---------------------------------------------------------------- video

_DIFF = 6  # mean grey-level change (0-255) between sampled frames that counts as "different"
_TIME = re.compile(
    r"\b(?P<a>\d{1,2}):(?P<b>\d{2})(?::(?P<c>\d{2}))?\b(?!\s*[ap]\.?m\b)"  # "10:30 AM" is a clock, not a moment
    r"|\b(?P<n>\d+(?:\.\d+)?)\s*(?P<u>min(?:ute)?s?|s(?:ec(?:ond)?s?)?)\b", re.I)
_RANGE = re.compile(r"\b(between|from|through|until)\b|\d\s*(?:[-–]|to)\s*\d", re.I)
_SPAN = re.compile(r"\b(first|last)\s+(\d+(?:\.\d+)?)\s*(min(?:ute)?s?|s(?:ec(?:ond)?s?)?)\b", re.I)


def fmt(t):
    t = int(t)
    h, m, s = t // 3600, t // 60 % 60, t % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def is_video(head):
    return (head[4:8] == b"ftyp"  # MP4, MOV
            or head[:4] == b"\x1aE\xdf\xa3"  # WebM, MKV
            or (head[:4] == b"RIFF" and head[8:12] == b"AVI "))


def save_video(stream, dirpath):
    """Copy an upload into dirpath after checking its real bytes. Returns (path, sha256 digest)."""
    head = stream.read(16)
    if not is_video(head):
        raise ValueError("unsupported video type (use MP4, MOV, WebM, MKV or AVI)")
    path = os.path.join(dirpath, "upload.bin")  # never the client's filename
    h, size = hashlib.sha256(head), len(head)
    with open(path, "wb") as out:
        out.write(head)
        while chunk := stream.read(1 << 20):
            size += len(chunk)
            if size > config.MAX_VIDEO_BYTES:
                raise ValueError(f"video larger than {config.MAX_VIDEO_BYTES // 1024 // 1024} MB")
            h.update(chunk)
            out.write(chunk)
    return path, h.digest()


def parse_times(text):
    """Seconds mentioned in a question: 1:23, 1:02:03, 90s, 2 minutes."""
    out = []
    for m in _TIME.finditer(text):
        if m["n"]:
            out.append(float(m["n"]) * (60 if m["u"].lower().startswith("m") else 1))
        elif m["c"]:
            out.append(int(m["a"]) * 3600 + int(m["b"]) * 60 + int(m["c"]))
        else:
            out.append(int(m["a"]) * 60 + int(m["b"]))
    return out


def _thin(items, limit):
    return [items[i * len(items) // limit] for i in range(limit)] if len(items) > limit else items


def _spread(lo, hi):
    return [lo + (hi - lo) * i / 5 for i in range(6)]


def _targeted(question, dur):
    """Frame times for a question that names a moment or a stretch, or None to sample the whole video."""
    limit = config.FRAMES_PER_SEGMENT
    m = _SPAN.search(question)  # "the first 30 seconds" is a stretch, not the moment 0:30
    if m:
        n = float(m[2]) * (60 if m[3].lower().startswith("m") else 1)
        return _thin(_spread(0, min(n, dur)) if m[1].lower() == "first" else _spread(max(dur - n, 0), dur), limit)
    ts = parse_times(question)[:4]
    if not ts:
        return None
    for t in ts:
        if t > dur:
            raise ValueError(f"timestamp {fmt(t)} is past the end of the video ({fmt(dur)})")
    if len(ts) == 2 and _RANGE.search(question):
        return _thin(_spread(*sorted(ts)), limit)
    around = sorted({min(max(t + d, 0), dur) for t in ts for d in (-1, 0, 1)})
    # too many moments for the frame budget: keep every named moment rather than cut the later ones
    return _thin(around if len(around) <= limit else sorted(set(ts)), limit)


def _read(cap, t, fps, n):
    """One frame, already shrunk: 24 full-size 4K frames at once would need ~600 MB."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(t * fps), int(n) - 1))
    ok, frame = cap.read()
    if not ok:
        return None
    h, w = frame.shape[:2]
    s = config.VIDEO_FRAME_SIDE / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return frame


def _distinct(frames, limit):
    """Drop frames that look like the previous kept one (also picks out scene changes), then thin evenly."""
    kept, last = [], None
    for t, f in frames:
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (32, 18))
        if last is None or cv2.absdiff(g, last).mean() > _DIFF:
            kept.append((t, f))
            last = g
    return _thin(kept, limit)


def _jpeg_uri(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("could not encode a video frame")
    return _uri(buf.tobytes(), "image/jpeg")


def sample_video(path, question=""):
    """Pick representative frames. Returns {"duration", "segments": [{"labels", "uris", "span", "note"}]}.

    A question naming a time ("what happens at 1:23?") gets frames around that moment; otherwise
    the video is cut into equal segments (one vision call each) and each gets its distinct frames.
    shortcut: native video input is not used (frames work on every model); OpenCV's frame count
    can be wrong for some WebM files, which then fail with "cannot read" instead of being guessed.
    """
    cap = cv2.VideoCapture(path)
    try:
        fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not fps > 0 or not n > 0:
            raise ValueError("cannot read this video")
        dur = n / fps
        if dur > config.MAX_VIDEO_S:
            raise ValueError(f"video longer than {fmt(config.MAX_VIDEO_S)}")
        times = _targeted(question, dur)
        k = 1 if times else min(config.MAX_SEGMENTS, math.ceil(dur / config.SEGMENT_S))
        seg_len, segments = dur / k, []
        for i in range(k):
            a, b = i * seg_len, (i + 1) * seg_len
            want = config.FRAMES_PER_SEGMENT
            ts = times or [a + (j + .5) * seg_len / (want * 3) for j in range(want * 3)]
            frames = [(t, f) for t in ts if (f := _read(cap, t, fps, n)) is not None]
            if not times:
                frames = _distinct(frames, want)
            if frames:
                segments.append({
                    "labels": [f"[{fmt(t)}]" for t, _ in frames],
                    "uris": [_jpeg_uri(f) for _, f in frames],
                    "span": (a, b),
                    "note": f"These images are frames from a video ({fmt(a)}-{fmt(b)} of {fmt(dur)}), "
                            "in time order; the text before each image is its timestamp. "})
        if not segments:
            raise ValueError("no frames could be read from this video")
        return {"duration": dur, "segments": segments}
    finally:
        cap.release()
