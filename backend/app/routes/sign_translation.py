"""
Sign Translation API Routes

Handles translation of text and video to sign language representations.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from ..services.sign_recognition import get_sign_recognition_service

router = APIRouter()


class TextTranslationRequest(BaseModel):
    """Request model for text-to-sign translation."""
    text: str
    language: Optional[str] = "en"


class SignRecognitionRequest(BaseModel):
    """Request model for video/frame sign recognition."""
    top_k: Optional[int] = 5


@router.post("/translate")
async def translate_text(request: TextTranslationRequest):
    """
    Translate text to sign language representation.
    
    Currently returns placeholder. Will integrate with text-to-sign model.
    """
    text = request.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # TODO: Implement text-to-sign translation
    # For now, return placeholder response
    return {
        "text": text,
        "signs": [],
        "caption": text,
        "requestId": "placeholder",
        "status": "not_implemented"
    }


@router.post("/recognize")
async def recognize_sign(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Recognize sign language from uploaded image or video.
    
    Accepts:
    - Image files (jpg, png, jpeg)
    - Video files (mp4, avi, mov)
    
    Returns top-k predictions with sign, meaning, and confidence.
    """
    service = get_sign_recognition_service()
    
    if not service.is_initialized():
        return JSONResponse(
            status_code=503,
            content={
                "error": "Sign recognition service not available",
                "message": "Model has not been trained yet. Please train the model first."
            }
        )
    
    # Read file content
    contents = await file.read()
    
    # Determine file type
    file_extension = file.filename.split('.')[-1].lower() if file.filename else ''
    
    try:
        if file_extension in ['jpg', 'jpeg', 'png']:
            # Process as image
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                raise HTTPException(status_code=400, detail="Failed to decode image")
            
            predictions = service.recognize_from_frame(frame, top_k=top_k)
            
            return {
                "predictions": predictions,
                "file_type": "image",
                "filename": file.filename
            }
        
        elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
            # Process as video
            # Save temporarily and process
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            try:
                # Extract frames from video
                cap = cv2.VideoCapture(tmp_path)
                frames = []
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = int(fps / 1.5)  # Extract at 1.5 fps
                frame_count = 0
                
                while len(frames) < 16:  # Limit to 16 frames
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        frames.append(frame)
                    
                    frame_count += 1
                
                cap.release()
                
                if len(frames) > 0:
                    predictions = service.recognize_from_frames(frames, top_k=top_k)
                else:
                    raise HTTPException(status_code=400, detail="No frames extracted from video")
            
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            
            return {
                "predictions": predictions,
                "file_type": "video",
                "filename": file.filename,
                "frames_processed": len(frames)
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported: jpg, png, jpeg, mp4, avi, mov, mkv"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/status")
async def get_service_status():
    """Get status of sign recognition service."""
    service = get_sign_recognition_service()
    
    return {
        "initialized": service.is_initialized(),
        "model_available": service.is_initialized(),
        "dictionary_loaded": service.dictionary is not None if service.dictionary else False,
        "num_signs": len(service.idx_to_sign) if service.is_initialized() else 0
    }
