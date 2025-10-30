from fastapi import FastAPI
from .routes import sign_translation, speech_processing, healthcheck

app = FastAPI(title="EduSign AI Backend")

app.include_router(sign_translation.router, prefix="/api")
app.include_router(speech_processing.router, prefix="/api")
app.include_router(healthcheck.router, prefix="/api")

@app.get("/")
def root():
    return {"status": "ok"}
