from fastapi import APIRouter

router = APIRouter()

@router.post("/translate")
def translate(payload: dict):
    text = payload.get("text", "")
    return {"signs": [], "caption": text, "requestId": "placeholder"}
