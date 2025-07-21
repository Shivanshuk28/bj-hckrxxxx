from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Tuple
from datetime import datetime
import logging
import aiohttp
import asyncio
import re
import os
import io
import pdfplumber
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

app = FastAPI(title="Light Query API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def query_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{OPENAI_API_BASE}/v1/chat/completions", json=payload, headers=headers) as resp:
            if resp.status != 200:
                logger.error(await resp.text())
                raise HTTPException(status_code=500, detail="LLM API failed")
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()

def extract_pdf_text(file_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages), None
    except Exception as e:
        return None, str(e)

def extract_text_file(file_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
    try:
        return file_bytes.decode("utf-8"), None
    except Exception as e:
        return None, str(e)

def chunk_text(text: str, max_words: int = 300) -> List[str]:
    words = re.findall(r'\S+', text)
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks

async def summarize_document(text: str) -> str:
    chunks = chunk_text(text)
    results = await asyncio.gather(*[query_llm(f"Summarize: {chunk}") for chunk in chunks])
    return "\n".join(results)

def generate_report(insights: str, query: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"# Report\n\n**Generated:** {ts}\n\n**Query:** {query}\n\n**Insights:**\n{insights}"

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    user_query: str = Form(..., max_length=200)
):
    content = await file.read()
    text, err = (
        extract_pdf_text(content) if file.content_type.startswith("application/pdf")
        else extract_text_file(content)
    )
    if err or not text:
        raise HTTPException(status_code=400, detail=err or "Empty document")

    summary = await summarize_document(text)
    insights = await query_llm(f"{user_query}: {summary}")
    return {"summary": summary, "insights": insights, "report": generate_report(insights, user_query)}
