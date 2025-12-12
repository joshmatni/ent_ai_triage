from fastapi import APIRouter
from pydantic import BaseModel

from app.ollama_client import call_ollama
from app.backend_client import save_triage_to_backend

router = APIRouter(prefix="/ai")

class TriageRequest(BaseModel):
    transcript: str
    patient_id: str # required for saving into backend DB


class TriageResponse(BaseModel):
    summary: str
    urgency: str


@router.post("/triage", response_model=TriageResponse)
async def triage(payload: TriageRequest):

    raw = await call_ollama(payload.transcript)

    summary = raw.strip()
    urgency = "unknown"
    confidence = 0.0

    await save_triage_to_backend(
        patient_id=payload.patient_id,
        transcript=payload.transcript,
        summary=summary,
        urgency=urgency,
        confidence=confidence,
    )

    return {
        "summary": summary,
        "urgency": urgency
    }