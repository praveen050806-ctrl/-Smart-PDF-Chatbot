"""
app.py
-------
Flask web application for the Smart PDF Chatbot.

Flow:
  1. GET  /            -> upload a PDF
  2. POST /upload      -> extract + index the PDF, redirect to /chat
  3. GET  /chat         -> chat UI
  4. POST /ask (JSON)    -> ask a question, get a cited answer back

Documents are indexed in-memory per session (see SESSION_STORE below).
This keeps the demo dependency-free, but means the index is lost if the
server restarts, and it won't scale across multiple worker processes —
see the README for how you'd swap in a real vector store for production.

Run locally with:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename

from pdf_processor import extract_pages, chunk_pages, get_document_stats, PDFProcessError
from qa_engine import DocumentIndex, answer_question

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 15 * 1024 * 1024  # 15 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

# In-memory store: session_id -> {"index": DocumentIndex, "filename": str,
#                                  "stats": dict, "history": [...]}
# NOTE: for a production deployment, replace this with a real
# database/vector store keyed by session or user id.
SESSION_STORE = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        flash("Please choose a PDF file to upload.")
        return redirect(url_for("index"))

    file = request.files["pdf"]
    if file.filename == "":
        flash("Please choose a PDF file to upload.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a PDF.")
        return redirect(url_for("index"))

    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(file_path)

    try:
        pages = extract_pages(file_path)
        chunks = chunk_pages(pages)
        stats = get_document_stats(pages, chunks)
        doc_index = DocumentIndex(chunks)
    except PDFProcessError as e:
        flash(str(e))
        return redirect(url_for("index"))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    session_id = uuid.uuid4().hex
    session["session_id"] = session_id
    SESSION_STORE[session_id] = {
        "index": doc_index,
        "filename": file.filename,
        "stats": stats,
        "history": [],
    }

    return redirect(url_for("chat"))


@app.route("/chat", methods=["GET"])
def chat():
    session_id = session.get("session_id")
    doc = SESSION_STORE.get(session_id)

    if not doc:
        flash("Please upload a PDF first.")
        return redirect(url_for("index"))

    return render_template(
        "chat.html",
        filename=doc["filename"],
        stats=doc["stats"],
        history=doc["history"],
    )


@app.route("/ask", methods=["POST"])
def ask():
    session_id = session.get("session_id")
    doc = SESSION_STORE.get(session_id)

    if not doc:
        return jsonify({"error": "No document indexed. Please upload a PDF first."}), 400

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    result = answer_question(doc["index"], question)
    doc["history"].append({"question": question, "result": result})

    return jsonify(result)


@app.route("/reset", methods=["POST"])
def reset():
    session_id = session.pop("session_id", None)
    SESSION_STORE.pop(session_id, None)
    return redirect(url_for("index"))


@app.errorhandler(413)
def file_too_large(e):
    flash("File is too large. Please upload a PDF smaller than 15 MB.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
