# QA Chatbot

A document-based Question Answering (QA) chatbot using FAISS for vector search, SentenceTransformers for embeddings, and HuggingFace Transformers for answer generation. Supports PDF, DOCX, TXT, XLSX, and CSV file ingestion, chunking, and semantic search.


## Project Structure

```
QA_Chatbot/
│
├── app/
│   ├── ingest.py         # File parsing, chunking, and ingestion logic
│   ├── main.py           # FastAPI app entrypoint (API server)
│   ├── qa.py             # Question answering logic (prompting, model inference, logging)
│   ├── retriever.py      # FAISS-based retriever class
│   ├── faiss.index       # FAISS index for semantic search
│   ├── index_meta.json   # Metadata mapping for stored chunks and document references 
│   └── template/
│       └── index.html    # Web UI template
│   └── uploads/
│       └── sample.pdf    # uploaded pdf or any document
│
├── logs/
│   └── qa_interactions.log # Log file for user Q&A interactions
│
├── env/                  # Python virtual environment (not versioned)
│
├── requirements.txt      # Python dependencies
│
└── README.md             # Project documentation
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Hirvita6/QA_Chatbot.git
cd QA_Chatbot
```

### 2. Create and activate a virtual environment
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On Linux/macOS:
source env/bin/activate
```

### 3. Install dependencies
```bash
cd app
pip install -r requirements.txt
```

### 4. Run the FastAPI server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Access the web UI
Open your browser and go to: [http://localhost:8000](http://localhost:8000) 
OR http://127.0.0.1:8000

## Usage Workflow

1. **Upload a document** (PDF, DOCX, TXT, XLSX, CSV) via the web UI or API.
2. The file is parsed and chunked into text segments.
3. Each chunk is embedded using SentenceTransformers and indexed with FAISS.
4. On each new upload, the FAISS index and metadata are reset (overwritten) to only contain the latest file's data.
5. The user selects the desired QA model (Extractive or Generative) from a dropdown in the UI.
6. The chatbot retrieves the most relevant chunks and uses the chosen model to generate the answer.


## Key Components

- **app/ingest.py**: Handles file parsing, chunking, and ingestion.
- **app/retriever.py**: Manages the FAISS index, embeddings, and metadata. Supports reset/overwrite on new uploads.
- **app/qa.py**: Builds prompts, runs the QA model (Flan-T5 or similar) for answer generation, and logs all Q&A interactions.
- **app/main.py**: FastAPI server exposing upload and QA endpoints.
- **app/template/index.html**: Simple web interface for upload and Q&A.
- **app/logs/qa_interactions.log**: Stores all user questions, answers, and (if available) model confidence scores in JSON lines format.

## Logging of User Q&A Interactions

Every time a user asks a question, the system logs the following to `logs/qa_interactions.log`:

- Timestamp (UTC, ISO8601)
- User question
- Model-generated answer
- Confidence score (if available)

Each log entry is a single line of JSON, e.g.:
```json
{"timestamp": "2025-11-07T12:34:56Z", "question": "What is FAISS?", "answer": "FAISS is a library for efficient similarity search.", "score": 0.959384928}
```

See `app/logs/qa_interactions.log` for details and code examples for custom logging or analysis.

## Requirements
- Python 3.8+ 
- See `requirements.txt` for all dependencies

## Notes
- By default, the chatbot loads both models at startup (for quick switching).
- deepset/tinyroberta-squad2(Extractive QA) is faster and more precise for short factual queries.
- google/flan-t5-small(Generative QA) is better for conversational or summary-type answers.
- For large files or multiple users, consider using a database or persistent storage for uploads and indexes.
- You can also change or add LLM models based on your specific task (e.g., summarization, multi-turn chat, domain-specific QA) to get more relevant or detailed answers. 
- If you have a GPU, you can use larger and more accurate models such as Mistral, Llama, Falcon, etc. from Hugging Face for better answer quality and performance.
- The retriever is reset (index/metadata overwritten) on every new upload for simplicity. You can modify this for multi-file or append scenarios.
