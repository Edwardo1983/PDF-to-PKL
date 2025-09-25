# config.py - Configurare centralizată pentru PDF to Embeddings

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class EmbeddingConfig:
    """Configurare pentru embeddings"""
    model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 400
    overlap: int = 80
    batch_size: int = 32
    normalize_embeddings: bool = True
    retrieval_instruction: str = "Represent this sentence for retrieval: "
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    fallback_dimension: int = 384
    bible_chunk_size: Optional[int] = None
    bible_overlap: Optional[int] = None

@dataclass
class OCRConfig:
    """Configurare pentru OCR"""
    dpi: int = 200
    max_pages_ocr: int = 100  # Mărit pentru mai multă flexibilitate
    languages: str = "ron+eng"
    tesseract_config: str = "--psm 6 --oem 3"
    max_retries: int = 2
    retry_delay: float = 2.0

@dataclass
class DatabaseConfig:
    """Configurare pentru baza de date"""
    metric: str = "cosine"
    max_collection_name_length: int = 60
    batch_size: int = 64
    max_retries: int = 3
    retry_delay: float = 2.0

@dataclass
class ProcessingConfig:
    """Configurare pentru procesare"""
    max_workers: int = 4
    memory_threshold_percent: float = 85.0
    disk_space_threshold_gb: float = 1.0
    cleanup_interval: int = 10  # Cleanup la fiecare N fișiere
    progress_update_interval: float = 0.1  # Update progress la fiecare 10%

@dataclass
class LoggingConfig:
    """Configurare pentru logging"""
    level: str = "INFO"
    log_file: str = "pdf_converter.log"
    max_log_size_mb: int = 10
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Instanțe de configurare
EMBEDDING_CONFIG = EmbeddingConfig()
OCR_CONFIG = OCRConfig()
DB_CONFIG = DatabaseConfig()
PROCESSING_CONFIG = ProcessingConfig()
LOGGING_CONFIG = LoggingConfig()

# Cuvinte cheie pentru conținut educațional
EDUCATIONAL_KEYWORDS = [
    "elev", "învăț", "matematică", "exercițiu", "lecție", "probleme",
    "clasă", "școală", "temă", "evaluare", "test", "note", "curs",
    "manual", "exemplu", "soluție", "explicație", "definiție", "teoremă"
]

# Configurare din variabile de mediu (override pentru deployment)
def load_config_from_env():
    """Încarcă configurația din variabile de mediu"""
    global EMBEDDING_CONFIG, OCR_CONFIG, DB_CONFIG, PROCESSING_CONFIG, LOGGING_CONFIG
    
    # Override embedding config
    if os.getenv("EMBEDDING_MODEL"):
        EMBEDDING_CONFIG.model = os.getenv("EMBEDDING_MODEL")
    if os.getenv("CHUNK_SIZE"):
        try:
            EMBEDDING_CONFIG.chunk_size = int(os.getenv("CHUNK_SIZE"))
        except ValueError:
            # Valoare fallback recomandată pentru deployment-uri restricționate
            EMBEDDING_CONFIG.chunk_size = 800
    if os.getenv("CHUNK_OVERLAP"):
        try:
            EMBEDDING_CONFIG.overlap = int(os.getenv("CHUNK_OVERLAP"))
        except ValueError:
            EMBEDDING_CONFIG.overlap = 120
    if os.getenv("CHUNK_SIZE_BIBLE"):
        try:
            EMBEDDING_CONFIG.bible_chunk_size = int(os.getenv("CHUNK_SIZE_BIBLE")) or None
        except ValueError:
            EMBEDDING_CONFIG.bible_chunk_size = None
    if os.getenv("OVERLAP_BIBLE"):
        try:
            EMBEDDING_CONFIG.bible_overlap = int(os.getenv("OVERLAP_BIBLE")) or None
        except ValueError:
            EMBEDDING_CONFIG.bible_overlap = None
    if os.getenv("BATCH_SIZE"):
        EMBEDDING_CONFIG.batch_size = int(os.getenv("BATCH_SIZE"))
    
    # Override OCR config
    if os.getenv("OCR_MAX_PAGES"):
        OCR_CONFIG.max_pages_ocr = int(os.getenv("OCR_MAX_PAGES"))
    if os.getenv("OCR_DPI"):
        OCR_CONFIG.dpi = int(os.getenv("OCR_DPI"))
    
    # Override processing config
    if os.getenv("MAX_WORKERS"):
        PROCESSING_CONFIG.max_workers = int(os.getenv("MAX_WORKERS"))
    if os.getenv("MEMORY_THRESHOLD"):
        PROCESSING_CONFIG.memory_threshold_percent = float(os.getenv("MEMORY_THRESHOLD"))
    
    # Override logging config
    if os.getenv("LOG_LEVEL"):
        LOGGING_CONFIG.level = os.getenv("LOG_LEVEL")
    if os.getenv("LOG_FILE"):
        LOGGING_CONFIG.log_file = os.getenv("LOG_FILE")

# Încarcă configurația din mediu
load_config_from_env()

# Backward compatibility
EMBEDDING_CONFIG_DICT = {
    "model": EMBEDDING_CONFIG.model,
    "chunk_size": EMBEDDING_CONFIG.chunk_size,
    "overlap": EMBEDDING_CONFIG.overlap,
    "batch_size": EMBEDDING_CONFIG.batch_size,
    "normalize_embeddings": EMBEDDING_CONFIG.normalize_embeddings,
    "retrieval_instruction": EMBEDDING_CONFIG.retrieval_instruction
}

OCR_CONFIG_DICT = {
    "dpi": OCR_CONFIG.dpi,
    "max_pages_ocr": OCR_CONFIG.max_pages_ocr,
    "languages": OCR_CONFIG.languages
}

DB_CONFIG_DICT = {
    "metric": DB_CONFIG.metric,
    "max_collection_name_length": DB_CONFIG.max_collection_name_length
}