import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Project root
ROOT = Path(__file__).resolve().parent.parent

# Allow imports from project root
sys.path.insert(0, str(ROOT))

from fraud_investigation_agent_llm import investigate_case


app = FastAPI(
    title="Fraud Investigation Agent API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "service": "fraud-investigation-agent",
    }


@app.post("/api/investigate/{case_id}")
async def investigate(case_id: str):
    case_id = case_id.upper().strip()

    if not case_id.startswith("HHG-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid case ID",
        )

    try:
        result = await investigate_case(case_id)

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc