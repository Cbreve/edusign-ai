"""
Sign Translation API Routes

Handles translation of text and video to sign language representations.
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from ..services.sign_recognition import get_sign_recognition_service

logger = logging.getLogger(__name__)
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
    
    Uses NLP-based text-to-sign mapping to convert English text
    into a sequence of GSL signs with their meanings.
    """
    from ..services.text_to_sign import get_text_to_sign_mapper
    
    text = request.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Get text-to-sign mapper
        mapper = get_text_to_sign_mapper()
        
        if not mapper.is_initialized():
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Text-to-sign mapper not available",
                    "message": "Service is initializing. Please try again in a moment."
                }
            )
        
        # Map text to signs
        result = mapper.map_sentence_to_signs(text)
        
        # Extract sign names for animation sequence
        animation_sequence = mapper.get_sign_animation_sequence(text)
        
        # Get animation paths using animation mapping service
        from ..services.animation_mapping import get_animation_mapping_service
        anim_service = get_animation_mapping_service()
        animation_paths = anim_service.get_animation_sequence(
            animation_sequence,
            include_metadata=False
        )
        
        return {
            "text": text,
            "signs": result['signs'],
            "animation_sequence": animation_sequence,
            "animation_paths": animation_paths,
            "total_signs": result['total_signs'],
            "mapped_signs": result['mapped_signs'],
            "mapping_rate": result['mapping_rate'],
            "caption": text,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error translating text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error translating text: {str(e)}")


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
    
    from ..services.animation_mapping import get_animation_mapping_service
    anim_service = get_animation_mapping_service()
    anim_stats = anim_service.get_statistics()
    
    return {
        "initialized": service.is_initialized(),
        "model_available": service.is_initialized(),
        "dictionary_loaded": service.dictionary is not None if hasattr(service, 'dictionary') else False,
        "num_signs": len(service.idx_to_sign) if service.is_initialized() else 0,
        "animation_mapping": {
            "initialized": anim_service.is_initialized(),
            "statistics": anim_stats
        }
    }


@router.get("/animations/path/{sign_name}")
async def get_animation_path(sign_name: str):
    """Get animation file path for a specific sign."""
    from ..services.animation_mapping import get_animation_mapping_service
    
    anim_service = get_animation_mapping_service()
    if not anim_service.is_initialized():
        raise HTTPException(status_code=503, detail="Animation mapping service not initialized")
    
    animation_path = anim_service.get_animation_path(sign_name)
    
    return {
        "sign_name": sign_name,
        "animation_path": animation_path,
        "exists": animation_path is not None
    }


@router.post("/animations/sequence")
async def get_animation_sequence(sign_names: List[str]):
    """Get animation sequence for multiple signs."""
    from ..services.animation_mapping import get_animation_mapping_service
    
    anim_service = get_animation_mapping_service()
    if not anim_service.is_initialized():
        raise HTTPException(status_code=503, detail="Animation mapping service not initialized")
    
    sequence = anim_service.get_animation_sequence(sign_names, include_metadata=True)
    
    return {
        "sign_names": sign_names,
        "sequence": sequence,
        "total_animations": len(sequence)
    }


@router.get("/animations/statistics")
async def get_animation_statistics():
    """Get statistics about animation mappings."""
    from ..services.animation_mapping import get_animation_mapping_service
    
    anim_service = get_animation_mapping_service()
    if not anim_service.is_initialized():
        raise HTTPException(status_code=503, detail="Animation mapping service not initialized")
    
    return anim_service.get_statistics()
