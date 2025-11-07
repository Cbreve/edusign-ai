"""
Sign Animation Sequence Mapping Service

Maps sign language names to animation file paths.
Supports multiple animation formats (GLB, GLTF, FBX) and provides
fallback mechanisms for missing animations.

Industry-standard approach:
- Sign name → Animation file mapping
- Fallback to generic animations
- Animation metadata management
- Sequence queuing and playback control
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
ANIMATIONS_DIR = PROJECT_ROOT / "frontend" / "public" / "avatar_models" / "ready-player-me" / "animations"
ANIMATION_MAPPING_FILE = PROJECT_ROOT / "backend" / "app" / "data" / "processed" / "animation_mapping.json"


class AnimationMappingService:
    """
    Service for mapping sign names to animation files.
    
    Features:
    - Sign name → Animation file mapping
    - Fallback animations for missing signs
    - Animation metadata (duration, format, etc.)
    - Sequence generation for sign phrases
    """
    
    def __init__(self):
        self.mapping: Dict[str, str] = {}
        self.metadata: Dict[str, Dict] = {}
        self.fallback_animations = ["IDLE", "NEUTRAL", "WAVE"]
        self.animation_formats = [".glb", ".gltf", ".fbx", ".dae"]
        self._initialized = False
    
    def initialize(self):
        """Initialize the animation mapping service."""
        try:
            # Load mapping file if it exists
            if ANIMATION_MAPPING_FILE.exists():
                with open(ANIMATION_MAPPING_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.mapping = data.get("mappings", {})
                    self.metadata = data.get("metadata", {})
                logger.info(f"Loaded animation mapping with {len(self.mapping)} entries")
            
            # Scan animations directory for available files
            self._scan_animations_directory()
            
            # Create default mapping if file doesn't exist
            if not ANIMATION_MAPPING_FILE.exists():
                self._create_default_mapping()
                self._save_mapping()
            
            self._initialized = True
            logger.info("Animation mapping service initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing animation mapping: {e}", exc_info=True)
            return False
    
    def _scan_animations_directory(self):
        """Scan animations directory for available animation files."""
        if not ANIMATIONS_DIR.exists():
            logger.warning(f"Animations directory not found: {ANIMATIONS_DIR}")
            return
        
        available_files = {}
        for ext in self.animation_formats:
            for file_path in ANIMATIONS_DIR.glob(f"*{ext}"):
                sign_name = file_path.stem.upper()
                available_files[sign_name] = str(file_path.relative_to(ANIMATIONS_DIR.parent))
        
        # Update mapping with available files
        for sign_name, file_path in available_files.items():
            if sign_name not in self.mapping:
                self.mapping[sign_name] = file_path
                logger.info(f"Found animation file: {sign_name} -> {file_path}")
    
    def _create_default_mapping(self):
        """Create default mapping based on GSL dictionary."""
        try:
            from .text_to_sign import get_text_to_sign_mapper
            
            mapper = get_text_to_sign_mapper()
            if mapper.is_initialized():
                # Get all sign names from dictionary
                for entry in mapper.dictionary:
                    sign_name = entry['sign'].upper()
                    # Try to find animation file
                    animation_path = self._find_animation_file(sign_name)
                    if animation_path:
                        self.mapping[sign_name] = animation_path
                    else:
                        # Use fallback or placeholder
                        self.mapping[sign_name] = f"animations/{sign_name}.glb"  # Placeholder path
        except Exception as e:
            logger.warning(f"Could not create default mapping from dictionary: {e}")
    
    def _find_animation_file(self, sign_name: str) -> Optional[str]:
        """Find animation file for a sign name."""
        if not ANIMATIONS_DIR.exists():
            return None
        
        # Try exact match first
        for ext in self.animation_formats:
            file_path = ANIMATIONS_DIR / f"{sign_name}{ext}"
            if file_path.exists():
                return f"avatar_models/ready-player-me/animations/{sign_name}{ext}"
        
        # Try case variations
        for ext in self.animation_formats:
            for file_path in ANIMATIONS_DIR.glob(f"*{sign_name}*{ext}"):
                return f"avatar_models/ready-player-me/animations/{file_path.name}"
        
        return None
    
    def _save_mapping(self):
        """Save mapping to file."""
        try:
            ANIMATION_MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(ANIMATION_MAPPING_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "mappings": self.mapping,
                    "metadata": self.metadata,
                    "fallback_animations": self.fallback_animations
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved animation mapping to {ANIMATION_MAPPING_FILE}")
        except Exception as e:
            logger.error(f"Error saving animation mapping: {e}", exc_info=True)
    
    def get_animation_path(self, sign_name: str, use_fallback: bool = True) -> Optional[str]:
        """
        Get animation file path for a sign name.
        
        Args:
            sign_name: Name of the sign (e.g., "HELLO")
            use_fallback: Whether to use fallback animation if sign not found
            
        Returns:
            Path to animation file or None
        """
        if not self._initialized:
            logger.warning("Animation mapping service not initialized")
            return None
        
        sign_name_upper = sign_name.upper().strip()
        
        # Check direct mapping
        if sign_name_upper in self.mapping:
            path = self.mapping[sign_name_upper]
            # Verify file exists if it's a relative path
            if not path.startswith("http"):
                full_path = PROJECT_ROOT / "frontend" / "public" / path
                if full_path.exists():
                    return f"/{path}"  # Return web-accessible path
                else:
                    logger.warning(f"Animation file not found: {path}")
        
        # Try to find in animations directory
        animation_path = self._find_animation_file(sign_name_upper)
        if animation_path:
            self.mapping[sign_name_upper] = animation_path
            self._save_mapping()
            return f"/{animation_path}"
        
        # Use fallback if enabled
        if use_fallback and self.fallback_animations:
            for fallback in self.fallback_animations:
                fallback_path = self.get_animation_path(fallback, use_fallback=False)
                if fallback_path:
                    logger.info(f"Using fallback animation '{fallback}' for '{sign_name}'")
                    return fallback_path
        
        # Return placeholder path
        logger.warning(f"No animation found for sign: {sign_name}")
        return f"/avatar_models/ready-player-me/animations/{sign_name_upper}.glb"  # Placeholder
    
    def get_animation_sequence(
        self,
        sign_names: List[str],
        include_metadata: bool = False
    ) -> List[Dict]:
        """
        Get animation sequence for a list of sign names.
        
        Args:
            sign_names: List of sign names (e.g., ["HELLO", "I", "GOOD"])
            include_metadata: Whether to include animation metadata
            
        Returns:
            List of animation entries with paths and metadata
        """
        sequence = []
        
        for i, sign_name in enumerate(sign_names):
            animation_path = self.get_animation_path(sign_name)
            
            # Check if file actually exists
            file_exists = False
            if animation_path:
                # Remove leading slash for path checking
                path_for_check = animation_path.lstrip('/')
                full_path = PROJECT_ROOT / "frontend" / "public" / path_for_check
                file_exists = full_path.exists()
            
            entry = {
                "sign": sign_name,
                "position": i,
                "animation_path": animation_path,
                "exists": file_exists
            }
            
            if include_metadata and sign_name in self.metadata:
                entry["metadata"] = self.metadata[sign_name]
            
            sequence.append(entry)
        
        return sequence
    
    def add_mapping(self, sign_name: str, animation_path: str, metadata: Optional[Dict] = None):
        """
        Add or update animation mapping.
        
        Args:
            sign_name: Name of the sign
            animation_path: Path to animation file
            metadata: Optional metadata (duration, format, etc.)
        """
        sign_name_upper = sign_name.upper().strip()
        self.mapping[sign_name_upper] = animation_path
        
        if metadata:
            self.metadata[sign_name_upper] = metadata
        
        self._save_mapping()
        logger.info(f"Added mapping: {sign_name_upper} -> {animation_path}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about animation mappings."""
        total_signs = len(self.mapping)
        available_animations = sum(
            1 for path in self.mapping.values()
            if (PROJECT_ROOT / "frontend" / "public" / path).exists()
        )
        
        return {
            "total_mappings": total_signs,
            "available_animations": available_animations,
            "missing_animations": total_signs - available_animations,
            "coverage_percentage": (available_animations / total_signs * 100) if total_signs > 0 else 0,
            "fallback_animations": self.fallback_animations
        }
    
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized


# Singleton instance
_animation_mapping_instance = None


def get_animation_mapping_service() -> AnimationMappingService:
    """Get singleton instance of AnimationMappingService."""
    global _animation_mapping_instance
    if _animation_mapping_instance is None:
        _animation_mapping_instance = AnimationMappingService()
        _animation_mapping_instance.initialize()
    return _animation_mapping_instance

