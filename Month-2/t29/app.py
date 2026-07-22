import time
import json
import uuid
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path="../.env", override=True)
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Instrumented Production RAG & LLM Application",
    description="Production App with OpenTelemetry / LangSmith structured tracing and observability logging.",
    version="1.0.0"
)

# Structured JSON Logger Setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("LLM_Tracer")

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    user_id: str = "user-101"
    prompt: str = "What are the key components of LLM observability?"

def log_llm_trace(trace_data: dict):
    """Exports structured OpenTelemetry/LangSmith trace span."""
    json_log = json.dumps(trace_data)
    logger.info(json_log)
    # Save to local trace log file
    with open("llm_traces.jsonl", "a", encoding="utf-8") as f:
        f.write(json_log + "\n")

@app.get("/health")
def health():
    return {"status": "healthy", "tracing_active": True}

@app.post("/api/v1/query")
async def process_query(req: QueryRequest):
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    
    # 1. Trace Prompt Preparation
    prompt_tokens = len(req.prompt.split())
    
    # 2. Trace LLM Call Execution
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a production assistant."},
            {"role": "user", "content": req.prompt}
        ],
        temperature=0
    )
    
    end_time = time.time()
    total_latency_ms = (end_time - start_time) * 1000
    
    completion_text = res.choices[0].message.content.strip()
    completion_tokens = res.usage.completion_tokens if hasattr(res, 'usage') and res.usage else len(completion_text.split())
    total_tokens = res.usage.total_tokens if hasattr(res, 'usage') and res.usage else prompt_tokens + completion_tokens
    
    # Estimated Cost ($0.00015 / 1K input, $0.0006 / 1K output)
    cost_est = (prompt_tokens * 0.00000015) + (completion_tokens * 0.0000006)
    
    # Construct OpenTelemetry-compatible Trace Span
    trace_span = {
        "trace_id": trace_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_id": req.user_id,
        "service": "production-rag-service",
        "span_name": "llm_completion_gpt-4o-mini",
        "status": "SUCCESS",
        "metrics": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_latency_ms": round(total_latency_ms, 2),
            "estimated_cost_usd": round(cost_est, 6)
        },
        "attributes": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "prompt_sample": req.prompt[:60],
            "response_sample": completion_text[:60]
        }
    }
    
    log_llm_trace(trace_span)
    
    return {
        "trace_id": trace_id,
        "answer": completion_text,
        "telemetry": trace_span["metrics"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
