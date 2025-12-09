class TriageRequest(BaseModel):
    transcript: str
    patient_id: str  # YOU NEED THIS FOR SAVING INTO DB

class TriageResponse(BaseModel):
    summary: str
    urgency: str