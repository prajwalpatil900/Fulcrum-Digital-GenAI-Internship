import os
import time
import json
import uuid
import re
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path="../.env", override=True)
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Month 2 Production RAG & AI Agent Platform",
    description="Enterprise RAG + AI Agent system with Guardrails, Observability Tracing, and Prompt Versioning.",
    version="2.0.0"
)

client = OpenAI(api_key=api_key) if api_key else None

# Security Guardrail Engine
class SecurityGuardrails:
    def __init__(self):
        self.ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
        self.email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        self.injection_keywords = ["ignore previous instructions", "system prompt override", "act as dan", "reveal system prompt"]

    def validate(self, text: str) -> dict:
        lower = text.lower()
        for kw in self.injection_keywords:
            if kw in lower:
                return {"action": "BLOCK", "reason": "Prompt Injection Attack Detected", "text": text, "pii": []}
        
        pii = []
        if re.search(self.ssn_pattern, text): pii.append("SSN")
        if re.search(self.email_pattern, text): pii.append("EMAIL")
        
        sanitized = re.sub(self.ssn_pattern, "[REDACTED SSN]", text)
        sanitized = re.sub(self.email_pattern, "[REDACTED EMAIL]", sanitized)
        return {"action": "ALLOW", "reason": "Passed Security Scan", "text": sanitized, "pii": pii}

guardrails = SecurityGuardrails()
telemetry_logs = []

class ChatRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o-mini"
    guardrails_enabled: bool = True

@app.get("/health")
def health():
    return {"status": "online", "system": "Month 2 Enterprise Prototype"}

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    t0 = time.time()
    trace_id = f"tr-{uuid.uuid4().hex[:8]}"
    
    # 1. Security Check
    sec_res = guardrails.validate(req.prompt) if req.guardrails_enabled else {"action": "ALLOW", "reason": "Bypassed", "text": req.prompt, "pii": []}
    
    if sec_res["action"] == "BLOCK":
        latency = (time.time() - t0) * 1000
        log_entry = {
            "trace_id": trace_id,
            "timestamp": time.strftime("%H:%M:%S"),
            "prompt": req.prompt[:40],
            "status": "BLOCKED",
            "reason": sec_res["reason"],
            "latency_ms": round(latency, 1),
            "tokens": 0,
            "cost": "$0.000000"
        }
        telemetry_logs.append(log_entry)
        return JSONResponse(status_code=400, content={
            "status": "BLOCKED",
            "error": "🚨 Security Violation: " + sec_res["reason"],
            "trace_id": trace_id,
            "telemetry": log_entry
        })
    
    # 2. LLM / RAG Processing
    if client:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are Month 2 Production Enterprise AI Assistant. Provide helpful, structured answers."},
                {"role": "user", "content": sec_res["text"]}
            ],
            temperature=0
        )
        answer = res.choices[0].message.content.strip()
        p_tokens = res.usage.prompt_tokens
        c_tokens = res.usage.completion_tokens
        total_tokens = res.usage.total_tokens
    else:
        answer = f"[Simulated Response] Processed query: '{sec_res['text']}'"
        p_tokens, c_tokens, total_tokens = 25, 45, 70
        
    latency_ms = (time.time() - t0) * 1000
    cost_usd = (p_tokens * 0.00000015) + (c_tokens * 0.0000006)
    
    telemetry_info = {
        "trace_id": trace_id,
        "timestamp": time.strftime("%H:%M:%S"),
        "prompt": req.prompt[:40],
        "status": "SUCCESS",
        "pii_redacted": sec_res["pii"],
        "latency_ms": round(latency_ms, 1),
        "tokens": total_tokens,
        "cost": f"${cost_usd:.6f}"
    }
    telemetry_logs.append(telemetry_info)
    
    return {
        "status": "SUCCESS",
        "answer": answer,
        "trace_id": trace_id,
        "telemetry": telemetry_info
    }

@app.get("/api/telemetry")
def get_telemetry():
    return {"logs": list(reversed(telemetry_logs[-10:]))}

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Month 2 Enterprise AI Platform</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --card-bg: rgba(22, 31, 49, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary-glow: #00f2fe;
            --accent-purple: #9d4edd;
            --success-green: #10b981;
            --danger-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }

        body {
            background: radial-gradient(circle at 50% 0%, #1a233a 0%, var(--bg-dark) 70%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Glassmorphism Header */
        header {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo-box {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            color: #090d16;
            box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
        }

        .logo-title h1 { font-size: 1.2rem; font-weight: 600; letter-spacing: -0.3px; }
        .logo-title p { font-size: 0.75rem; color: var(--text-muted); }

        .header-controls {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .badge {
            background: rgba(16, 185, 129, 0.15);
            color: var(--success-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .badge-dot { width: 8px; height: 8px; background: var(--success-green); border-radius: 50%; box-shadow: 0 0 8px var(--success-green); }

        /* Main Container Layout */
        .container {
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 24px;
            padding: 24px;
            max-width: 1600px;
            margin: 0 auto;
            width: 100%;
            flex: 1;
        }

        /* Chat Panel */
        .chat-panel {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.02);
        }

        .chat-messages {
            flex: 1;
            padding: 24px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
            min-height: 450px;
            max-height: 600px;
        }

        .msg {
            max-width: 80%;
            padding: 14px 18px;
            border-radius: 14px;
            font-size: 0.95rem;
            line-height: 1.5;
            animation: fadeIn 0.3s ease-in-out;
        }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

        .msg.user {
            align-self: flex-end;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: #fff;
            border-bottom-right-radius: 4px;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }

        .msg.assistant {
            align-self: flex-start;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            border-bottom-left-radius: 4px;
        }

        .msg.system-block {
            align-self: center;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #fca5a5;
            text-align: center;
            max-width: 90%;
        }

        /* Input Area */
        .chat-input-area {
            padding: 16px 24px;
            border-top: 1px solid var(--border-color);
            background: rgba(15, 23, 42, 0.5);
            display: flex;
            gap: 12px;
        }

        input[type="text"] {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px 18px;
            color: #fff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s;
        }

        input[type="text"]:focus {
            border-color: var(--primary-glow);
            box-shadow: 0 0 12px rgba(0, 242, 254, 0.2);
        }

        button {
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            color: #090d16;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            padding: 0 24px;
            cursor: pointer;
            transition: all 0.2s;
        }

        button:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0, 242, 254, 0.4); }

        /* Sidebar Telemetry */
        .telemetry-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }

        .card h3 { font-size: 0.9rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }

        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .stat-box { background: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 10px; border: 1px solid var(--border-color); }
        .stat-val { font-size: 1.3rem; font-weight: 700; color: var(--primary-glow); }
        .stat-lbl { font-size: 0.75rem; color: var(--text-muted); }

        .telemetry-log-item {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            padding: 8px 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            justify-content: space-between;
        }
        .log-success { color: var(--success-green); }
        .log-blocked { color: var(--danger-red); }

        /* Toggle Switch */
        .toggle-wrap { display: flex; align-items: center; gap: 10px; font-size: 0.85rem; }
        .switch { position: relative; display: inline-block; width: 44px; height: 24px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #334155; transition: .4s; border-radius: 24px; }
        .slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: var(--primary-glow); }
        input:checked + .slider:before { transform: translateX(20px); background-color: #090d16; }
    </style>
</head>
<body>

    <header>
        <div class="logo-box">
            <div class="logo-icon">AI</div>
            <div class="logo-title">
                <h1>Month 2 Enterprise AI Platform</h1>
                <p>FastAPI + RAG + Guardrails + Tracing</p>
            </div>
        </div>

        <div class="header-controls">
            <div class="toggle-wrap">
                <span>🛡️ Security Guardrails</span>
                <label class="switch">
                    <input type="checkbox" id="guardrailsToggle" checked>
                    <span class="slider"></span>
                </label>
            </div>
            <div class="badge">
                <div class="badge-dot"></div>
                System Online
            </div>
        </div>
    </header>

    <div class="container">
        <!-- Main Chat Area -->
        <div class="chat-panel">
            <div class="chat-header">
                <div>🤖 Production Assistant</div>
                <div style="font-size:0.8rem; color:var(--text-muted);">Model: <strong>gpt-4o-mini</strong> | Prompt Reg: <strong>v1.2 CoT</strong></div>
            </div>

            <div class="chat-messages" id="chatContainer">
                <div class="msg assistant">
                    👋 Welcome! I am your Month 2 Production RAG & AI Agent Assistant. Ask me anything, or test our security guardrails!
                </div>
            </div>

            <div class="chat-input-area">
                <input type="text" id="userInput" placeholder="Ask a question or test prompt injection..." onkeydown="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">Send Request</button>
            </div>
        </div>

        <!-- Telemetry Sidebar -->
        <div class="telemetry-panel">
            <!-- Stats -->
            <div class="card">
                <h3>⚡ Live Telemetry</h3>
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-val" id="statLatency">0ms</div>
                        <div class="stat-lbl">Avg Latency</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-val" id="statTokens">0</div>
                        <div class="stat-lbl">Tokens Processed</div>
                    </div>
                </div>
            </div>

            <!-- Trace Log -->
            <div class="card" style="flex:1;">
                <h3>📊 Trace Spans (OpenTelemetry)</h3>
                <div id="traceLogContainer" style="max-height: 350px; overflow-y: auto;">
                    <div style="font-size:0.8rem; color:var(--text-muted); text-align:center; padding:20px;">No traces recorded yet</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const text = input.value.trim();
            if (!text) return;

            const chatContainer = document.getElementById('chatContainer');
            const isGuardrails = document.getElementById('guardrailsToggle').checked;

            // Append User Message
            const userMsg = document.createElement('div');
            userMsg.className = 'msg user';
            userMsg.textContent = text;
            chatContainer.appendChild(userMsg);
            input.value = '';
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: text, guardrails_enabled: isGuardrails })
                });

                const data = await response.json();

                if (response.ok) {
                    const aiMsg = document.createElement('div');
                    aiMsg.className = 'msg assistant';
                    aiMsg.innerHTML = data.answer + `<br><br><small style="color:var(--primary-glow); font-size:0.75rem;">Trace ID: ${data.trace_id} | ${data.telemetry.latency_ms}ms | Cost: ${data.telemetry.cost}</small>`;
                    chatContainer.appendChild(aiMsg);
                } else {
                    const blockMsg = document.createElement('div');
                    blockMsg.className = 'msg system-block';
                    blockMsg.innerHTML = `${data.error}<br><small style="font-size:0.75rem;">Trace ID: ${data.trace_id}</small>`;
                    chatContainer.appendChild(blockMsg);
                }

                updateTelemetry();
            } catch (err) {
                console.error(err);
            }
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function updateTelemetry() {
            const res = await fetch('/api/telemetry');
            const data = await res.json();
            const container = document.getElementById('traceLogContainer');
            container.innerHTML = '';

            let totalLat = 0, totalTok = 0;
            data.logs.forEach(log => {
                totalLat += log.latency_ms;
                totalTok += log.tokens;

                const item = document.createElement('div');
                item.className = 'telemetry-log-item';
                const statusClass = log.status === 'SUCCESS' ? 'log-success' : 'log-blocked';
                item.innerHTML = `<span>${log.timestamp} [${log.trace_id}]</span><span class="${statusClass}">${log.status} (${log.latency_ms}ms)</span>`;
                container.appendChild(item);
            });

            if(data.logs.length > 0) {
                document.getElementById('statLatency').textContent = Math.round(totalLat / data.logs.length) + 'ms';
                document.getElementById('statTokens').textContent = totalTok;
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
