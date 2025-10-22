# app.py
import os
import uuid
import time
import redis.asyncio as aioredis

import shutil
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles
import aioredis
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ---------- Configuration ----------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")  # e.g. redis://:password@host:port/0
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")  # comma separated
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
FILE_TTL_SECONDS = int(os.getenv("FILE_TTL_SECONDS", str(60 * 60 * 2)))  # 2 hours default
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "2"))  # min seconds between queries per session

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Gemini client (sync client used inside async via asyncio.to_thread)
client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------- FastAPI app ----------
app = FastAPI(title="Academic Chat Backend (FastAPI)")

# CORS
origins = [o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Redis ----------
redis = None

async def get_redis():
    global redis
    if redis is None:
        redis = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    return redis

# ---------- Utilities ----------
def make_local_filename(original_name: str) -> str:
    ext = os.path.splitext(original_name)[1]
    return f"{uuid.uuid4().hex}{ext}"

async def cleanup_local_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

async def cleanup_remote_file_by_name(name: str):
    # Optional: If you want to delete the file from Gemini after done.
    try:
        await asyncio.to_thread(client.files.delete, name=name)
    except Exception:
        pass

# ---------- Models ----------
class ChatRequest(BaseModel):
    session_id: str
    query: str

# ---------- Endpoints ----------
@app.on_event("startup")
async def startup_event():
    await get_redis()

@app.post("/session")
async def create_session():
    """Create a new session_id for a user"""
    session_id = "sess_" + uuid.uuid4().hex
    r = await get_redis()
    await r.hset(f"session:{session_id}", mapping={"created_at": str(int(time.time()))})
    # set a TTL for the session metadata as a safety net (optional)
    await r.expire(f"session:{session_id}", FILE_TTL_SECONDS + 3600)
    return {"session_id": session_id}

@app.post("/upload")
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    """
    Upload a PDF for a session.
    Steps:
    1) save file locally to unique path
    2) upload file to Gemini via client.files.upload
    3) wait until state == 'ACTIVE'
    4) store remote file.name in Redis under session
    5) delete local file (or keep for short TTL)
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    # Prepare local file path
    local_name = make_local_filename(file.filename)
    local_path = os.path.join(UPLOAD_DIR, local_name)
    async with aiofiles.open(local_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # Upload to Gemini in a thread to avoid blocking event loop
    try:
        def upload_and_poll(path):
            # this runs in a thread
            f = client.files.upload(file=path)
            # poll
            while f.state == "PROCESSING":
                time.sleep(2)
                f = client.files.get(name=f.name)
            return f  # a FileResource object with .name and .state and .uri

        file_resource = await asyncio.to_thread(upload_and_poll, local_path)
    except Exception as e:
        # cleanup local copy then raise
        await cleanup_local_file(local_path)
        raise HTTPException(status_code=500, detail=f"Upload to Gemini failed: {e}")

    # store remote file reference in Redis session
    r = await get_redis()
    remote_name = file_resource.name
    await r.hset(f"session:{session_id}", mapping={
        "pdf_remote_name": remote_name,
        "pdf_uploaded_at": str(int(time.time())),
    })
    # set TTL for session key (extend)
    await r.expire(f"session:{session_id}", FILE_TTL_SECONDS + 3600)

    # schedule local file deletion in background
    # keep local copy short time in case you need it; delete now to save disk
    await cleanup_local_file(local_path)

    return {"success": True, "remote_name": remote_name, "state": file_resource.state}

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Chat endpoint:
    - Validates rate limit per session
    - Fetches session's remote pdf file name (if any)
    - Calls Gemini model and returns text
    """
    session_key = f"session:{req.session_id}"
    r = await get_redis()
    exists = await r.exists(session_key)
    if not exists:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id")

    # Rate limiting (simple): check last_query_ts
    last_ts = await r.hget(session_key, "last_query_ts")
    now = int(time.time())
    if last_ts and (now - int(last_ts) < RATE_LIMIT_SECONDS):
        # Too many requests too quickly
        raise HTTPException(status_code=429, detail="Too many requests; slow down a bit")

    await r.hset(session_key, "last_query_ts", str(now))
    await r.expire(session_key, FILE_TTL_SECONDS + 3600)

    # Get stored remote pdf name if present
    remote_name = await r.hget(session_key, "pdf_remote_name")

    # Build contents for the model
    if remote_name:
        contents = [remote_name, req.query]  # the genai client earlier accepted file resource objects; passing name is fine if client.files.get is used inside
        # But our client expects actual file objects; to be safe, get the file resource object:
        try:
            file_resource = await asyncio.to_thread(client.files.get, name=remote_name)
            contents = [file_resource, req.query]
        except Exception:
            # If remote file can't be used, fall back to just the query
            contents = [req.query]
    else:
        contents = [req.query]

    # System instruction (can customize)
    system_instruction = (
        "You are a helpful, professional academic assistant. "
        "If a PDF file is provided, answer strictly based on that document. "
        "If not, answer using general academic knowledge. "
        "Be concise and structured."
    )

    config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=1500,
        candidate_count=1,
        system_instruction=system_instruction,
    )

    # Use the blocking API in a thread so we don't block the event loop
    try:
        def call_gemini(contents_local):
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents_local,
                config=config
            )
            # Here we fetch the textual candidate
            if resp.candidates and resp.candidates[0].content.parts:
                return resp.candidates[0].content.parts[0].text
            else:
                return "No response from model."

        result_text = await asyncio.to_thread(call_gemini, contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model call failed: {e}")

    # Optional: store chat history (append small record)
    await r.rpush(f"history:{req.session_id}", f"{int(time.time())}:{req.query} -> {result_text[:2000]}")
    # limit history length (trim)
    await r.ltrim(f"history:{req.session_id}", -50, -1)
    await r.expire(f"history:{req.session_id}", FILE_TTL_SECONDS + 3600)

    return {"reply": result_text}

@app.post("/cleanup")
async def manual_cleanup(session_id: str):
    """Optionally allow manual cleanup of session resources"""
    r = await get_redis()
    session_key = f"session:{session_id}"
    data = await r.hgetall(session_key)
    if not data:
        return {"success": False, "detail": "session not found"}

    remote_name = data.get("pdf_remote_name")
    if remote_name:
        # delete remote file asynchronously
        asyncio.create_task(cleanup_remote_file_by_name(remote_name))

    # delete session keys
    await r.delete(session_key)
    await r.delete(f"history:{session_id}")
    return {"success": True}
