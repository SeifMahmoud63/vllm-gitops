import os
import io
import uuid
import hmac
import logging
from typing import List, Optional

import boto3
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
from pypdf import PdfReader
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-app")

# === Configuration from Environment (ConfigMap + AWS Secrets Manager) ===
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm-service:8000/v1").rstrip("/")
VLLM_MODEL = os.getenv("VLLM_MODEL", "/models/Qwen2.5-0.5B-Instruct")
API_KEY = os.getenv("VLLM_API_KEY", "")

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-service")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")

S3_BUCKET = os.getenv("S3_BUCKET", "vllmm")
S3_UPLOAD_PREFIX = os.getenv("S3_UPLOAD_PREFIX", "uploads").strip("/")

DATABASE_URL = os.getenv("DATABASE_URL", "")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "115"))
TOP_K = int(os.getenv("TOP_K", "4"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

app = FastAPI(title="Production EKS RAG Platform", version="1.1.0")

embedder: Optional[TextEmbedding] = None
qdrant: Optional[QdrantClient] = None
s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "eu-north-1"))


def verify_api_key(
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
):
    """Protect all RAG endpoints using the API key from AWS Secrets Manager."""
    if not API_KEY:
        return
    token = x_api_key
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token or not hmac.compare_digest(token, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


def get_embedder() -> TextEmbedding:
    global embedder
    if embedder is None:
        logger.info(f"Loading FastEmbed model: {EMBEDDING_MODEL_NAME}")
        embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return embedder


def get_qdrant() -> QdrantClient:
    global qdrant
    if qdrant is None:
        qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        ensure_collection(qdrant)
    return qdrant


def ensure_collection(client: QdrantClient):
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        sample_vec = list(get_embedder().embed(["dimension probe"]))[0]
        dim = len(sample_vec)
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info(f"Created Qdrant collection '{QDRANT_COLLECTION}' with dim={dim}")


def get_db_conn():
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL)


def init_db():
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set; skipping PostgreSQL table initialization.")
        return
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    filename TEXT NOT NULL,
                    s3_key TEXT,
                    chunk_count INTEGER NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                ALTER TABLE documents ADD COLUMN IF NOT EXISTS s3_key TEXT;
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS qa_history (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_count INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
        conn.close()
        logger.info("PostgreSQL tables initialized successfully.")
    except Exception as e:
        logger.error(f"PostgreSQL init failed: {e}")


@app.on_event("startup")
def on_startup():
    init_db()
    try:
        get_embedder()
        get_qdrant()
    except Exception as e:
        logger.warning(f"Deferred Qdrant/Embedder init: {e}")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if lower.endswith((".txt", ".md")):
        return content.decode("utf-8", errors="ignore")
    raise HTTPException(status_code=400, detail="Unsupported file type. Upload .pdf, .txt, or .md")


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceChunk(BaseModel):
    filename: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    model: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "s3_uploads": f"s3://{S3_BUCKET}/{S3_UPLOAD_PREFIX}/",
        "qdrant": f"{QDRANT_HOST}:{QDRANT_PORT}",
        "db_configured": bool(DATABASE_URL),
    }


@app.post("/api/upload", dependencies=[Depends(verify_api_key)])
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit ({len(content) / (1024*1024):.2f} MB uploaded).",
        )

    text = extract_text(file.filename, content)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in file.")

    doc_id = str(uuid.uuid4())
    safe_name = os.path.basename(file.filename)
    s3_key = f"{S3_UPLOAD_PREFIX}/{doc_id}/{safe_name}"

    # 1. Store original file in AWS S3 (separate 'uploads/' folder from 'models/')
    try:
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=content)
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store file in S3: {e}")

    # 2. Embed chunks and upsert into Qdrant StatefulSet
    vectors = list(get_embedder().embed(chunks))
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.tolist(),
            payload={
                "doc_id": doc_id,
                "filename": safe_name,
                "s3_key": s3_key,
                "chunk_index": idx,
                "text": chunk,
            },
        )
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    get_qdrant().upsert(collection_name=QDRANT_COLLECTION, points=points)

    # 3. Log document metadata in AWS RDS PostgreSQL
    if DATABASE_URL:
        try:
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (id, filename, s3_key, chunk_count) VALUES (%s, %s, %s, %s)",
                    (doc_id, safe_name, s3_key, len(chunks)),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log document in RDS: {e}")

    return {
        "doc_id": doc_id,
        "filename": safe_name,
        "s3_uri": f"s3://{S3_BUCKET}/{s3_key}",
        "chunks_created": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    }


@app.post("/api/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
async def ask_question(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    k = req.top_k or TOP_K
    q_vec = list(get_embedder().embed([question]))[0].tolist()

    hits = get_qdrant().search(
        collection_name=QDRANT_COLLECTION,
        query_vector=q_vec,
        limit=k,
    )

    sources = [
        SourceChunk(
            filename=h.payload.get("filename", "unknown"),
            score=round(float(h.score), 4),
            text=h.payload.get("text", ""),
        )
        for h in hits
    ]

    context_block = "\n\n".join(
        f"[Source {i+1}: {s.filename}]\n{s.text}" for i, s in enumerate(sources)
    )

    system_prompt = (
        "You are a helpful AI assistant. Answer the user's question accurately using ONLY "
        "the provided context below. If the answer is not in the context, say you do not have enough information."
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }

    async with httpx.AsyncClient(timeout=90.0) as http_client:
        resp = await http_client.post(f"{VLLM_BASE_URL}/chat/completions", json=payload, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"vLLM error ({resp.status_code}): {resp.text}")
        data = resp.json()

    answer = data["choices"][0]["message"]["content"]

    if DATABASE_URL:
        try:
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO qa_history (question, answer, sources_count) VALUES (%s, %s, %s)",
                    (question, answer, len(sources)),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log Q&A in RDS: {e}")

    return AskResponse(answer=answer, sources=sources, model=VLLM_MODEL)


@app.get("/api/history", dependencies=[Depends(verify_api_key)])
def get_history():
    if not DATABASE_URL:
        return {"documents": [], "history": []}
    try:
        conn = get_db_conn()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT id, filename, s3_key, chunk_count, uploaded_at FROM documents ORDER BY uploaded_at DESC LIMIT 20")
            docs = cur.fetchall()
            cur.execute("SELECT id, question, answer, sources_count, created_at FROM qa_history ORDER BY created_at DESC LIMIT 20")
            hist = cur.fetchall()
        conn.close()
        return {"documents": docs, "history": hist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RDS query failed: {e}")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
