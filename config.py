import os

from dotenv import load_dotenv

load_dotenv()


def _int(name, default):
    return int(os.getenv(name, default))


API_KEY = os.getenv("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Verified on openrouter.ai (Nov 2026): both are free, 1M context, image input.
# Nemotron also accepts video (unused until Phase 5).
VISION_MODEL = os.getenv("VISION_MODEL", "nvidia/nemotron-3.5-super-vl:free")
REASONING_MODEL = os.getenv("REASONING_MODEL", "inclusionai/ling-3.2-flash:free")
ALLOW_PAID = os.getenv("ALLOW_PAID") == "1"  # models.py refuses non-":free" IDs without this

# Uploads
MAX_IMAGE_BYTES = _int("MAX_IMAGE_BYTES", 5 * 1024 * 1024)
MAX_IMAGES = _int("MAX_IMAGES", 4)
MAX_IMAGE_PIXELS = _int("MAX_IMAGE_PIXELS", 50_000_000)  # refuse decompression bombs before decoding
MAX_IMAGE_SIDE = _int("MAX_IMAGE_SIDE", 1568)  # larger images are shrunk to this
MAX_VIDEO_BYTES = _int("MAX_VIDEO_BYTES", 100 * 1024 * 1024)
MAX_VIDEO_S = _int("MAX_VIDEO_S", 1800)  # longest video we will sample
SEGMENT_S = _int("SEGMENT_S", 300)  # videos longer than this are split into segments
MAX_SEGMENTS = _int("MAX_SEGMENTS", 4)  # one vision call each, so this caps quota use per video
FRAMES_PER_SEGMENT = _int("FRAMES_PER_SEGMENT", 8)
VIDEO_FRAME_SIDE = _int("VIDEO_FRAME_SIDE", 768)
MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", max(MAX_VIDEO_BYTES, MAX_IMAGES * MAX_IMAGE_BYTES) + 1024 * 1024)  # whole request

# Web search (snippets only; result pages are never fetched)
SEARCH_RESULTS = _int("SEARCH_RESULTS", 5)
SEARCH_TIMEOUT_S = _int("SEARCH_TIMEOUT_S", 10)
MAX_TEXT_CHARS = _int("MAX_TEXT_CHARS", 20_000)

# Token budget (provider context is 1M; keep far below, free hosts can be tighter)
MAX_INPUT_TOKENS = _int("MAX_INPUT_TOKENS", 100_000)
IMAGE_TOKENS = _int("IMAGE_TOKENS", 1500)  # flat estimate per image
MAX_OUTPUT_TOKENS = _int("MAX_OUTPUT_TOKENS", 1000)
REASONING_MAX_TOKENS = _int("REASONING_MAX_TOKENS", 3000)  # thinking tokens count against this

# Free tier: 20/min flat, 50/day (1000/day after a $10 lifetime credit purchase)
PER_MINUTE = _int("PER_MINUTE", 20)
PER_DAY = _int("PER_DAY", 50)
MAX_CONCURRENT = _int("MAX_CONCURRENT", 2)
QUEUE_TIMEOUT_S = _int("QUEUE_TIMEOUT_S", 30)
MAX_WAIT_S = _int("MAX_WAIT_S", 30)  # longest we sleep for a rate limit or backoff
MAX_RETRIES = _int("MAX_RETRIES", 2)  # every attempt burns daily quota, so keep small
TIMEOUT_S = _int("TIMEOUT_S", 90)
