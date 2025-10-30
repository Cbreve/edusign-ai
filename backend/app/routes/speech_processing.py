from fastapi import APIRouter

router = APIRouter()

@router.post("/stt")
def speech_to_text():
    return {"text": ""}

@router.post("/tts")
def text_to_speech():
    return {"audioUrl": ""}
