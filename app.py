# app.py
import os
import uuid
import time
import shutil
import asyncio
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles
from dotenv import load_dotenv
from google import genai
from google.genai import types
import redis.asyncio as aioredis

# Load environment variables
load_dotenv()

# ---------- Configuration ----------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
FILE_TTL_SECONDS = int(os.getenv("FILE_TTL_SECONDS", str(60 * 60 * 2)))  # default 2 hours
RATE_LIMIT_SECONDS = int(os.getenv("RATE_LIMIT_SECONDS", "2"))  # seconds per query

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY environment variable")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

# ---------- FastAPI app ----------
app = FastAPI(title="Academic Chat Backend (FastAPI)")

# CORS setup
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
    try:
        await asyncio.to_thread(client.files.delete, name=name)
    except Exception:
        pass

# ---------- Models ----------
class ChatRequest(BaseModel):
    session_id: str
    query: str

# ---------- Startup ----------
@app.on_event("startup")
async def startup_event():
    try:
        r = await get_redis()
        pong = await r.ping()
        if pong:
            print("✅ Connected to Redis successfully")
        else:
            raise RuntimeError("Redis ping failed")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        raise

# ---------- Endpoints ----------
@app.post("/session")
async def create_session():
    session_id = "sess_" + uuid.uuid4().hex
    r = await get_redis()
    await r.hset(f"session:{session_id}", mapping={"created_at": str(int(time.time()))})
    await r.expire(f"session:{session_id}", FILE_TTL_SECONDS + 3600)
    return {"session_id": session_id}

@app.post("/upload")
async def upload_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    local_name = make_local_filename(file.filename)
    local_path = os.path.join(UPLOAD_DIR, local_name)
    async with aiofiles.open(local_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    try:
        def upload_and_poll(path):
            f = client.files.upload(file=path)
            while f.state == "PROCESSING":
                time.sleep(2)
                f = client.files.get(name=f.name)
            return f

        file_resource = await asyncio.to_thread(upload_and_poll, local_path)
    except Exception as e:
        await cleanup_local_file(local_path)
        raise HTTPException(status_code=500, detail=f"Upload to Gemini failed: {e}")

    r = await get_redis()
    remote_name = file_resource.name
    await r.hset(f"session:{session_id}", mapping={
        "pdf_remote_name": remote_name,
        "pdf_uploaded_at": str(int(time.time())),
    })
    await r.expire(f"session:{session_id}", FILE_TTL_SECONDS + 3600)
    await cleanup_local_file(local_path)

    return {"success": True, "remote_name": remote_name, "state": file_resource.state}

@app.post("/chat")
async def chat(req: ChatRequest):
    r = await get_redis()
    session_key = f"session:{req.session_id}"
    exists = await r.exists(session_key)
    if not exists:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id")

    # Rate limiting
    last_ts = await r.hget(session_key, "last_query_ts")
    now = int(time.time())
    if last_ts and (now - int(last_ts) < RATE_LIMIT_SECONDS):
        raise HTTPException(status_code=429, detail="Too many requests; slow down a bit")

    await r.hset(session_key, "last_query_ts", str(now))
    await r.expire(session_key, FILE_TTL_SECONDS + 3600)

    remote_name = await r.hget(session_key, "pdf_remote_name")
    if remote_name:
        try:
            file_resource = await asyncio.to_thread(client.files.get, name=remote_name)
            contents = [file_resource, req.query]
        except Exception:
            contents = [req.query]
    else:
        contents = [req.query]

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

    try:
        def call_gemini(contents_local):
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents_local,
                config=config
            )
            if resp.candidates and resp.candidates[0].content.parts:
                return resp.candidates[0].content.parts[0].text
            else:
                return "No response from model."

        result_text = await asyncio.to_thread(call_gemini, contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model call failed: {e}")

    await r.rpush(f"history:{req.session_id}", f"{int(time.time())}:{req.query} -> {result_text[:2000]}")
    await r.ltrim(f"history:{req.session_id}", -50, -1)
    await r.expire(f"history:{req.session_id}", FILE_TTL_SECONDS + 3600)

    return {"reply": result_text}

@app.post("/cleanup")
async def manual_cleanup(session_id: str):
    r = await get_redis()
    session_key = f"session:{session_id}"
    data = await r.hgetall(session_key)
    if not data:
        return {"success": False, "detail": "session not found"}

    remote_name = data.get("pdf_remote_name")
    if remote_name:
        asyncio.create_task(cleanup_remote_file_by_name(remote_name))

    await r.delete(session_key)
    await r.delete(f"history:{session_id}")
    return {"success": True}
