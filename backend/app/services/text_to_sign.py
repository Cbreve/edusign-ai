"""
Text-to-Sign Mapping Service

Maps natural language text to sign language sequences using industry-standard
NLP techniques including tokenization, normalization, and fuzzy matching.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available. Using basic text processing.")

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DICTIONARY_PATH = PROJECT_ROOT / "backend/app/data/processed/gsl_dictionary.json"


class TextToSignMapper:
    """
    Maps English text to Ghanaian Sign Language (GSL) sign sequences.
    
    Uses industry-standard NLP techniques:
    - Tokenization and normalization
    - Stopword removal
    - Lemmatization
    - Fuzzy matching for word-to-sign mapping
    - Phrase detection for compound signs
    """
    
    def __init__(self):
        self.dictionary = []
        self.sign_to_entry = {}
        self.word_to_signs = defaultdict(list)
        self.lemmatizer = None
        self.stop_words = set()
        self._initialized = False
        
    def initialize(self):
        """Initialize the text-to-sign mapper."""
        if not DICTIONARY_PATH.exists():
            logger.error(f"Dictionary not found: {DICTIONARY_PATH}")
            return False
        
        try:
            # Load dictionary
            with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
                self.dictionary = json.load(f)
            
            # Build mappings
            self._build_sign_mappings()
            
            # Initialize NLP tools if available
            if NLTK_AVAILABLE:
                try:
                    self._initialize_nltk()
                except LookupError:
                    logger.warning("NLTK data not found. Downloading required data...")
                    self._download_nltk_data()
                    self._initialize_nltk()
            else:
                logger.warning("Using basic text processing (NLTK not available)")
            
            self._initialized = True
            logger.info(f"Text-to-sign mapper initialized with {len(self.dictionary)} signs")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing text-to-sign mapper: {e}", exc_info=True)
            return False
    
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
    
    def _initialize_nltk(self):
        """Initialize NLTK components."""
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
    
    def _build_sign_mappings(self):
        """Build word-to-sign mappings from dictionary."""
        self.sign_to_entry = {}
        self.word_to_signs = defaultdict(list)
        
        for entry in self.dictionary:
            sign = entry.get('sign', '').upper().strip()
            if not sign:
                continue
            
            # Store sign entry
            self.sign_to_entry[sign] = entry
            
            # Extract keywords from sign name and meaning
            meaning = entry.get('meaning', '').lower()
            
            # Map sign name to itself
            self.word_to_signs[sign].append(sign)
            
            # Extract words from meaning for better matching
            if meaning:
                # Simple word extraction (improved with NLP later)
                words = re.findall(r'\b\w+\b', meaning.lower())
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        self.word_to_signs[word].append(sign)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize input text."""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation (keep basic sentence structure)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if NLTK_AVAILABLE and self.lemmatizer:
            try:
                tokens = word_tokenize(text.lower())
                # Lemmatize tokens
                lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
                return lemmatized
            except Exception as e:
                logger.warning(f"Error in tokenization: {e}")
        
        # Fallback: simple tokenization
        return re.findall(r'\b\w+\b', text.lower())
    
    def _find_sign_for_word(self, word: str) -> Optional[str]:
        """
        Find the best matching sign for a word.
        
        Uses fuzzy matching with multiple strategies:
        1. Exact match
        2. Word in sign name
        3. Word in meaning
        4. Partial match
        """
        word_lower = word.lower()
        word_upper = word.upper()
        
        # Strategy 1: Exact match in sign name
        if word_upper in self.sign_to_entry:
            return word_upper
        
        # Strategy 2: Word matches sign name (case-insensitive)
        for sign in self.sign_to_entry.keys():
            if sign.lower() == word_lower:
                return sign
        
        # Strategy 3: Word in word-to-sign mapping
        if word_lower in self.word_to_signs:
            # Return the first (most common) sign
            return self.word_to_signs[word_lower][0]
        
        # Strategy 4: Partial match in sign name (more restrictive)
        # Only match if word is at least 3 chars and represents a significant portion
        if len(word_lower) >= 3:
            for sign in self.sign_to_entry.keys():
                sign_lower = sign.lower()
                # Word contained in sign: require word to be at least 50% of sign length
                if word_lower in sign_lower:
                    if len(word_lower) >= len(sign_lower) * 0.5:
                        return sign
                # Sign contained in word: require sign to be at least 50% of word length
                elif sign_lower in word_lower:
                    if len(sign_lower) >= len(word_lower) * 0.5:
                        return sign
        
        # Strategy 5: Check if word appears in any sign's meaning (whole word match only)
        # Use word boundaries to avoid substring matches (e.g., "hi" in "chin")
        word_pattern = r'\b' + re.escape(word_lower) + r'\b'
        for sign, entry in self.sign_to_entry.items():
            meaning = entry.get('meaning', '').lower()
            if re.search(word_pattern, meaning):
                return sign
        
        return None
    
    def map_text_to_signs(self, text: str, include_stopwords: bool = False) -> List[Dict]:
        """
        Map text to a sequence of signs.
        
        Args:
            text: Input text to translate
            include_stopwords: Whether to include stopwords in output
            
        Returns:
            List of sign dictionaries with sign name, meaning, and position
        """
        if not self._initialized:
            logger.error("Text-to-sign mapper not initialized")
            return []
        
        if not text or not text.strip():
            return []
        
        # Normalize text
        normalized = self._normalize_text(text)
        
        # Tokenize
        tokens = self._tokenize(normalized)
        
        # Filter stopwords if needed
        if not include_stopwords and self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        # Map tokens to signs
        sign_sequence = []
        for i, token in enumerate(tokens):
            sign = self._find_sign_for_word(token)
            
            if sign:
                entry = self.sign_to_entry.get(sign)
                if entry:
                    sign_sequence.append({
                        'sign': sign,
                        'meaning': entry.get('meaning', ''),
                        'position': i,
                        'original_word': token,
                        'confidence': 1.0  # Can be improved with fuzzy matching scores
                    })
            else:
                # Word not found - add placeholder
                sign_sequence.append({
                    'sign': None,
                    'meaning': None,
                    'position': i,
                    'original_word': token,
                    'confidence': 0.0
                })
        
        return sign_sequence
    
    def map_sentence_to_signs(self, sentence: str) -> Dict:
        """
        Map a complete sentence to sign sequence with sentence structure.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Dictionary with sentence-level info and sign sequence
        """
        if not self._initialized:
            return {
                'sentence': sentence,
                'signs': [],
                'total_signs': 0,
                'mapped_signs': 0
            }
        
        # Split into sentences if multiple
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(sentence)
            except:
                sentences = [sentence]
        else:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', sentence)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        all_signs = []
        for sent in sentences:
            signs = self.map_text_to_signs(sent, include_stopwords=False)
            all_signs.extend(signs)
        
        mapped_count = sum(1 for s in all_signs if s['sign'] is not None)
        
        return {
            'sentence': sentence,
            'signs': all_signs,
            'total_words': len([s for s in all_signs]),
            'mapped_signs': mapped_count,
            'mapping_rate': mapped_count / len(all_signs) if all_signs else 0.0
        }
    
    def get_sign_animation_sequence(self, text: str) -> List[str]:
        """
        Get ordered list of sign names for animation playback.
        
        Args:
            text: Input text
            
        Returns:
            List of sign names in order
        """
        sign_sequence = self.map_text_to_signs(text)
        return [s['sign'] for s in sign_sequence if s['sign'] is not None]
    
    def is_initialized(self) -> bool:
        """Check if mapper is initialized."""
        return self._initialized


# Singleton instance
_mapper_instance = None


def get_text_to_sign_mapper() -> TextToSignMapper:
    """Get singleton instance of TextToSignMapper."""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = TextToSignMapper()
        _mapper_instance.initialize()
    return _mapper_instance

