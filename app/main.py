from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from pathlib import Path
from typing import List

try:
    import ingest, retriever, qa
    from qa import answer_with_flan_t5, answer_with_roberta
except:
    from . import ingest, retriever, qa
    from .qa import answer_with_flan_t5, answer_with_roberta


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


app = FastAPI(title="QA RAG Demo")


# Serve the uploads folder as static (optional)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# Initialize retriever singleton
retr = retriever.Retriever(index_path=str(BASE_DIR / "faiss.index"), meta_path=str(BASE_DIR / "index_meta.json"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...), reset_index: bool = Form(True)):
    if reset_index:
        retr.reset(remove_files=True)  # Clears previous data

    saved = []
    for f in files:
        dest = UPLOAD_DIR / f.filename
        with dest.open("wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved.append(str(dest))
        # ingest file and add to index
        texts = ingest.parse_and_chunk_file(str(dest))
        retr.add_documents(texts, [str(dest)] * len(texts))
    retr.save()
    return JSONResponse({"status": "ok", "files": saved})


@app.post("/ask")
async def ask(
    question: str = Form(...),
    model_name: str = Form(...),
    top_k: int = Form(3)
):
    # Retrieve top-k context
    hits = retr.search(question, top_k=top_k)
    contexts = [h['text'] for h in hits]

    # # Call QA without model_name
    # answer = qa.answer_question(question, contexts)
    # return JSONResponse({"question": question, "answer": answer, "contexts": contexts})
    if model_name == "google/flan-t5-small":
        answer = answer_with_flan_t5(question, contexts)
    else:
        answer = answer_with_roberta(question, contexts)

    return JSONResponse({"question": question, "answer": answer, "contexts": contexts})