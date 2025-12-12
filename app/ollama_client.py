import httpx
from app.config import settings
from app.prompts import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_PROMPT_TEMPLATE

async def call_ollama(transcript: str) -> str:
    prompt = (
        TRIAGE_SYSTEM_PROMPT.strip() +
        "\n" +
        TRIAGE_USER_PROMPT_TEMPLATE.replace("<<TRANSCRIPT>>", transcript) ## changed from strip 
    )

    payload = {
        "model": settings.OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    async with httpx.AsyncClient() as client:
        headers = {"Content-Type": "application/json"}

        resp = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            headers=headers
        )

    # if resp.status_code >= 400:
    #     print("❌ Backend /triage-cases error:", resp.status_code)
    #     print("❌ Error body:", resp.text)
    #     print("❌ Sent payload:", payload)

    resp.raise_for_status()

    data = resp.json()
    raw_text = data.get("response", "")

    return raw_text
