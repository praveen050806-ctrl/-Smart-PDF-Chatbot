# Smart PDF Chatbot

A Flask web app that lets you upload a PDF and ask questions about it in
plain English — with every answer citing the exact page it came from.
No external LLM API key is required: it uses TF-IDF retrieval + extractive
sentence selection, so it's free to run and fully explainable.

## What it does

1. Upload a PDF.
2. The app splits it into overlapping ~150-word passages, tagged with page numbers.
3. Ask a question in the chat window.
4. The app finds the most relevant passages (TF-IDF + cosine similarity),
   pulls out the sentences most likely to answer your question, and shows
   you the answer along with a **confidence level** and **page citations**.

Because it's retrieval-based rather than generative, it never invents
information that isn't in the document — if nothing relevant is found, it
says so instead of guessing.

## Project structure

```
smart-pdf-chatbot/
├── app.py               # Flask routes (upload, chat, ask API)
├── pdf_processor.py       # PDF text extraction + chunking
├── qa_engine.py             # TF-IDF retrieval + extractive answering
├── requirements.txt
├── templates/
│   ├── index.html           # Upload page
│   └── chat.html              # Chat UI (AJAX)
├── static/
│   └── style.css                # App styling
├── uploads/                       # Temp storage for PDFs (auto-cleared)
└── README.md
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/smart-pdf-chatbot.git
   cd smart-pdf-chatbot
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   python app.py
   ```

5. Open **http://127.0.0.1:5000**, upload a PDF, and start asking questions.

## How it works

- **`pdf_processor.py`**: `PyPDF2` extracts text per page, which is then
  split into overlapping word-count chunks so no answer gets cut off at a
  chunk boundary.
- **`qa_engine.py`**: builds a `TfidfVectorizer` index over all chunks for
  a document. A question is vectorized the same way, compared via cosine
  similarity, and the top-matching passages are returned. Within the best
  passage, sentences are ranked by word overlap with the question and the
  top 1-2 are returned as the answer.
- **Session storage**: each uploaded document is indexed in an in-memory
  dictionary keyed by a per-browser session id (see `SESSION_STORE` in
  `app.py`). This keeps the demo dependency-free but means the index is
  lost on server restart and won't scale across multiple worker processes.

## Important limitations (by design, for a demo project)

- **In-memory index**: not persisted to disk. Fine for local use or a demo
  deploy, not for production traffic.
- **Extractive, not generative**: answers are sentences pulled directly
  from the PDF, not a paraphrased explanation. This is a deliberate
  trade-off for accuracy/citability without needing an API key.
- **Text-based PDFs only**: scanned/image-only PDFs won't have extractable
  text; you'd need OCR (e.g. `pytesseract`) as a preprocessing step.

## Extending this project

- Swap `SESSION_STORE` for a real vector database (e.g. Chroma, FAISS,
  pgvector) to persist documents across restarts and scale horizontally.
- Plug in the Anthropic API to generate a fluent, synthesized answer from
  the retrieved passages instead of extracting raw sentences — pass the
  top-k chunks as context and ask the model to answer using only that
  context, still citing page numbers.
- Add OCR support for scanned PDFs.
- Support multi-document collections (ask questions across several PDFs
  at once).
- Add a "download conversation" export (Markdown or PDF).

## License

MIT — feel free to use this project as a portfolio piece or a starting point
for something bigger.
