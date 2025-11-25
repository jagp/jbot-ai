"""Text cleaning and preprocessing utilities."""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class CleaningConfig:
    """Configuration for text cleaning."""
    remove_urls: bool = True
    remove_emails: bool = True
    normalize_whitespace: bool = True
    remove_duplicates: bool = True
    min_length: int = 50
    max_length: int = 2048


class TextCleaner:
    """Clean and preprocess text documents."""
    
    def __init__(self, config: CleaningConfig = None):
        """Initialize text cleaner."""
        self.config = config or CleaningConfig()
    
    def clean(self, texts: List[str]) -> List[str]:
        """Clean a list of texts."""
        cleaned = []
        seen = set() if self.config.remove_duplicates else None
        
        for text in texts:
            if not isinstance(text, str):
                continue
            
            # Clean the text
            cleaned_text = self._clean_single(text)
            
            # Check length
            if len(cleaned_text) < self.config.min_length:
                continue
            if len(cleaned_text) > self.config.max_length:
                cleaned_text = cleaned_text[:self.config.max_length]
            
            # Check for duplicates
            if self.config.remove_duplicates:
                if cleaned_text in seen:
                    continue
                seen.add(cleaned_text)
            
            cleaned.append(cleaned_text)
        
        return cleaned
    
    def _clean_single(self, text: str) -> str:
        """Clean a single text document."""
        
        # Remove URLs
        if self.config.remove_urls:
            text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        if self.config.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
