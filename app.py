import tempfile

from flask import Flask, abort, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

import config
from core import media, models, orchestrator
from core.request_manager import RequestError, RequestManager, UserError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES
rm = RequestManager()


@app.before_request
def local_only():
    # A browser tab on any other site can still POST to localhost; Host stops DNS rebinding,
    # Origin stops cross-site form posts. Both are sent by browsers and cannot be set by a web page.
    if request.host.split(":")[0] not in ("127.0.0.1", "localhost"):
        abort(403, "this app only answers on localhost")
    origin = request.headers.get("Origin")
    if origin and origin != request.host_url.rstrip("/"):
        abort(403, "cross-site request refused")


@app.after_request
def no_store(resp):
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/stats")
def stats():
    return jsonify(rm.stats)  # runtime memory only; resets on restart


@app.post("/api/ask")
def ask():
    files = [f for f in request.files.getlist("images") if f.filename]
    vf = request.files.get("video")
    has_video = bool(vf and vf.filename)
    force = request.form.get("force", "")
    text, history = rm.validate(request.form.get("text"), len(files), request.form.get("history"), force, has_video)
    raw = [f.read(config.MAX_IMAGE_BYTES + 1) for f in files]  # after the count check, so 100 files are never read
    with tempfile.TemporaryDirectory() as tmp:  # holds the upload only while this request runs
        try:
            uris = [media.to_data_uri(b) for b in raw]
            path, digest = media.save_video(vf.stream, tmp) if has_video else (None, b"")
        except ValueError as e:
            raise UserError(str(e))
        with rm.guard(rm.key(text, raw + [digest], history, force)):
            try:
                video = media.sample_video(path, text) if path else None
            except ValueError as e:
                raise UserError(str(e))
            return jsonify(orchestrator.run(rm, text, uris, history, force, video))


@app.errorhandler(RequestError)
def request_error(e):
    return jsonify(error=str(e)), e.status


@app.errorhandler(models.ModelError)
def model_error(e):
    return jsonify(error=str(e)), 502


@app.errorhandler(Exception)
def unexpected(e):
    if isinstance(e, HTTPException):  # 404, 413, ... keep their own status
        return jsonify(error=e.description), e.code
    app.logger.exception("unhandled error")
    return jsonify(error="internal error"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)  # localhost only; debug=True would expose a shell
