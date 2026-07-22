import re
import os
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path="../.env", override=True)
api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title="Production RAG Application with Security Guardrails",
    description="Enterprise API with PII Redaction & Prompt Injection Mitigation.",
    version="1.0.0"
)

class SecureQueryRequest(BaseModel):
    user_id: str = "user-202"
    prompt: str

class SecurityGuardrailEngine:
    @staticmethod
    def detect_prompt_injection(text: str) -> bool:
        injection_patterns = [
            r"ignore (all )?previous instructions",
            r"system prompt",
            r"you are now (a|an)",
            r"jailbreak",
            r"bypass (all )?restrictions",
            r"reveal (your|the) secret"
        ]
        combined_pattern = "|".join(injection_patterns)
        return bool(re.search(combined_pattern, text, re.IGNORECASE))

    @staticmethod
    def redact_pii(text: str) -> tuple[str, bool]:
        pii_found = False
        
        # Email
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        if re.search(email_pattern, text):
            text = re.sub(email_pattern, "[EMAIL_REDACTED]", text)
            pii_found = True
            
        # SSN
        ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
        if re.search(ssn_pattern, text):
            text = re.sub(ssn_pattern, "[SSN_REDACTED]", text)
            pii_found = True
            
        # Phone
        phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
        if re.search(phone_pattern, text):
            text = re.sub(phone_pattern, "[PHONE_REDACTED]", text)
            pii_found = True
            
        return text, pii_found

@app.post("/api/v1/secure-query")
def secure_query(req: SecureQueryRequest):
    # 1. Input Guardrail: Prompt Injection Detection
    if SecurityGuardrailEngine.detect_prompt_injection(req.prompt):
        raise HTTPException(
            status_code=400, 
            detail="[SECURITY ALERT]: Prompt Injection attack pattern detected and blocked."
        )
        
    # 2. Input Guardrail: PII Redaction
    sanitized_prompt, pii_detected = SecurityGuardrailEngine.redact_pii(req.prompt)
    
    # 3. LLM Execution with Sanitized Input
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a secure assistant. Answer questions safely."},
            {"role": "user", "content": sanitized_prompt}
        ],
        temperature=0
    )
    
    raw_answer = res.choices[0].message.content.strip()
    
    # 4. Output Guardrail: Post-Processing Sanitization
    sanitized_answer, _ = SecurityGuardrailEngine.redact_pii(raw_answer)
    
    return {
        "user_id": req.user_id,
        "input_sanitized": sanitized_prompt,
        "pii_redacted": pii_detected,
        "answer": sanitized_answer,
        "guardrail_status": "PASSED"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
