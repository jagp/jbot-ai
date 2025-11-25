"""Data loader for processing personal texts from various sources."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import yaml
from dataclasses import dataclass


@dataclass
class TextDocument:
    """Represents a single text document."""
    content: str
    source: str
    metadata: Dict[str, Any]


class DataLoader:
    """Load personal text data from configured sources."""
    
    def __init__(self, config_path: str = "config/sources.yaml"):
        """Initialize data loader with configuration."""
        self.config = self._load_config(config_path)
        self.documents: List[TextDocument] = []
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_all_sources(self) -> List[TextDocument]:
        """Load documents from all enabled sources."""
        sources = self.config.get('sources', {})
        
        for source_name, source_config in sources.items():
            if source_config.get('enabled', False):
                print(f"Loading from {source_name}...")
                path = source_config['path']
                format_type = source_config.get('format', 'txt')
                
                if os.path.exists(path):
                    documents = self._load_source(path, format_type, source_name)
                    self.documents.extend(documents)
                    print(f"  Loaded {len(documents)} documents from {source_name}")
                else:
                    print(f"  Warning: Source path not found: {path}")
        
        return self.documents
    
    def _load_source(self, path: str, format_type: str, source_name: str) -> List[TextDocument]:
        """Load documents from a specific source."""
        documents = []
        
        if format_type == "txt":
            documents = self._load_txt_files(path, source_name)
        elif format_type == "json":
            documents = self._load_json_files(path, source_name)
        elif format_type == "csv":
            documents = self._load_csv_files(path, source_name)
        elif format_type == "mbox":
            documents = self._load_mbox_files(path, source_name)
        
        return documents
    
    def _load_txt_files(self, path: str, source_name: str) -> List[TextDocument]:
        """Load text files from directory."""
        documents = []
        for file_path in Path(path).glob("**/*.txt"):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                documents.append(TextDocument(
                    content=content,
                    source=source_name,
                    metadata={"file": str(file_path)}
                ))
        return documents
    
    def _load_json_files(self, path: str, source_name: str) -> List[TextDocument]:
        """Load JSON files from directory."""
        documents = []
        for file_path in Path(path).glob("**/*.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        documents.append(TextDocument(
                            content=item.get('content', ''),
                            source=source_name,
                            metadata=item
                        ))
                elif isinstance(data, dict):
                    documents.append(TextDocument(
                        content=data.get('content', ''),
                        source=source_name,
                        metadata=data
                    ))
        return documents
    
    def _load_csv_files(self, path: str, source_name: str) -> List[TextDocument]:
        """Load CSV files from directory."""
        documents = []
        try:
            import pandas as pd
            for file_path in Path(path).glob("**/*.csv"):
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    # Assume content is in a 'text' or 'content' column
                    content = row.get('content') or row.get('text') or str(row)
                    documents.append(TextDocument(
                        content=content,
                        source=source_name,
                        metadata=row.to_dict()
                    ))
        except ImportError:
            print("pandas not installed. CSV loading skipped.")
        return documents
    
    def _load_mbox_files(self, path: str, source_name: str) -> List[TextDocument]:
        """Load email files from mbox format."""
        documents = []
        try:
            import mailbox
            for file_path in Path(path).glob("**/*.mbox"):
                mbox = mailbox.mbox(str(file_path))
                for message in mbox:
                    content = message.get_payload()
                    if isinstance(content, str):
                        documents.append(TextDocument(
                            content=content,
                            source=source_name,
                            metadata={
                                "from": message.get('From', ''),
                                "subject": message.get('Subject', ''),
                                "date": message.get('Date', '')
                            }
                        ))
        except ImportError:
            print("mailbox module not available for mbox loading.")
        return documents
