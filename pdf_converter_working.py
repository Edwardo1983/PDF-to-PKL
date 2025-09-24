import os
import json
import hashlib
import time
import re
import threading
import gc
import logging
import sys
import contextlib
import psutil
import unicodedata
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from config import (
    EMBEDDING_CONFIG, DB_CONFIG, EDUCATIONAL_KEYWORDS,
    OCR_CONFIG, PROCESSING_CONFIG, LOGGING_CONFIG
)
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Iterable, Iterator
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import snapshot_download
try:
    from huggingface_hub.utils import LocalEntryNotFoundError
except ImportError:  # Fallback for older huggingface_hub versions
    LocalEntryNotFoundError = FileNotFoundError

# Ensure console streams can emit UTF-8 characters on Windows
for _stream_name in ('stdout', 'stderr'):
    _stream = getattr(sys, _stream_name, None)
    _reconfigure = getattr(_stream, 'reconfigure', None)
    if callable(_reconfigure):
        _reconfigure(encoding='utf-8', errors='backslashreplace')

# OCR imports pentru PDF-uri dificile (imagini, scanări)
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Instrucțiunea pentru BGE/GTE în modul retrieval
RETRIEVAL_INSTRUCTION = "Represent this sentence for retrieval: "


class HashingSentenceEmbedder:
    """Fallback simplu pentru generarea de embeddings fără dependențe externe."""

    def __init__(self, dimension: int = 384):
        self.dimension = max(32, dimension)

    def encode(
        self,
        sentences: List[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        embeddings = np.zeros((len(sentences), self.dimension), dtype=np.float32)

        for row, sentence in enumerate(sentences):
            if not sentence:
                continue

            tokens = re.findall(r"\w+", sentence.lower())
            if not tokens:
                continue

            for token in tokens:
                token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
                index = token_hash % self.dimension
                embeddings[row, index] += 1.0

            if normalize_embeddings:
                norm = np.linalg.norm(embeddings[row])
                if norm > 0:
                    embeddings[row] /= norm

        return embeddings

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension


class ResourceLimitExceeded(RuntimeError):
    """Raised when memory or disk thresholds are exceeded during processing."""


# Data classes pentru monitoring
@dataclass
class ProcessingStats:
    """Statistici pentru procesarea PDF-urilor"""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_chunks: int = 0
    total_processing_time: float = 0.0
    memory_peak: float = 0.0
    disk_usage_mb: float = 0.0

@dataclass
class ProcessingResult:
    """Rezultatul procesării unui PDF"""
    file_path: str
    success: bool
    chunks_created: int = 0
    processing_time: float = 0.0
    error_message: str = ""
    memory_used_mb: float = 0.0

# Setup logging structurat cu configurare dinamică
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.level.upper()),
    format=LOGGING_CONFIG.format,
    handlers=[
        logging.FileHandler(LOGGING_CONFIG.log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Context manager pentru memory management
@contextmanager
def memory_managed_processing():
    """Context manager pentru gestionarea memoriei în timpul procesării"""
    initial_memory = psutil.virtual_memory().percent
    logger.info(f"Memory management started - Initial: {initial_memory:.1f}%")
    
    try:
        yield
    finally:
        # Cleanup agresiv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_memory = psutil.virtual_memory().percent
        logger.info(f"Memory management completed - Final: {final_memory:.1f}% (Δ: {final_memory-initial_memory:+.1f}%)")

# Context manager pentru retry logic
@contextmanager
def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Context manager pentru retry logic cu exponential backoff"""
    for attempt in range(max_retries):
        try:
            yield attempt
            return
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Final attempt failed after {max_retries} retries: {e}")
                raise
            else:
                wait_time = delay * (backoff ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

class PDFEmbeddingsConverter:
    def __init__(self, embeddings_db_path: str = "./embeddings_db"):
        logger.info("Initializing optimized PDF converter...")

        self.embeddings_db_path = embeddings_db_path
        self.processed_files_path = os.path.join(embeddings_db_path, "processed_files.json")
        self._file_lock = threading.Lock()  # Lock pentru thread safety la processed_files.json
        self._abort_processing: bool = False
        self._abort_reason: str = ""

        self.allow_online_model_download = os.getenv("ALLOW_ONLINE_MODEL_DOWNLOAD", "0") == "1"
        if not self.allow_online_model_download:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # Health checks la startup
        self._validate_environment()
        self._check_disk_space()
        self._check_runtime_resources()

        os.makedirs(embeddings_db_path, exist_ok=True)
        
        # Setari conservative pentru Intel 12 cores
        torch.set_num_threads(4)
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['MKL_NUM_THREADS'] = '4'
        
        logger.info(f"Loading {EMBEDDING_CONFIG.model} model...")
        with memory_managed_processing():
            local_model_path = os.getenv("LOCAL_SENTENCE_MODEL_PATH")
            configured_model_name = EMBEDDING_CONFIG.model
            model_source = local_model_path or configured_model_name

            try:
                resolved_model_path = model_source
                cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.getenv("HF_HOME")

                if local_model_path:
                    logger.info("Loading SentenceTransformer model from local path: %s", local_model_path)
                else:
                    repo_id = model_source if "/" in model_source else f"sentence-transformers/{model_source}"
                    logger.info(
                        "Ensuring SentenceTransformer weights are available locally for model: %s", repo_id
                    )
                    try:
                        resolved_model_path = snapshot_download(
                            repo_id=repo_id,
                            cache_dir=cache_folder,
                            local_files_only=not self.allow_online_model_download,
                        )
                        logger.debug("Using cached SentenceTransformer model path: %s", resolved_model_path)
                    except LocalEntryNotFoundError as cache_exc:
                        if self.allow_online_model_download:
                            logger.warning(
                                "Initial local cache check failed (%s). Retrying with download enabled...",
                                cache_exc,
                            )
                            resolved_model_path = snapshot_download(
                                repo_id=repo_id,
                                cache_dir=cache_folder,
                                local_files_only=False,
                            )
                        else:
                            raise FileNotFoundError(
                                "Model weights not found in local Hugging Face cache. "
                                "Provide LOCAL_SENTENCE_MODEL_PATH or enable ALLOW_ONLINE_MODEL_DOWNLOAD."
                            ) from cache_exc

                self.model = SentenceTransformer(resolved_model_path)
                self.using_fallback_model = False
                self.embedding_model_name = Path(resolved_model_path).name or configured_model_name
            except Exception as exc:
                if local_model_path:
                    logger.warning(
                        "Could not load SentenceTransformer model from provided path (%s). Falling back to hashing embeddings.",
                        exc,
                    )
                elif self.allow_online_model_download:
                    logger.warning(
                        "Could not download SentenceTransformer model (%s). Falling back to hashing embeddings.",
                        exc,
                    )
                else:
                    logger.warning(
                        "Could not load SentenceTransformer model from local cache (%s). Falling back to hashing embeddings.",
                        exc,
                    )
                self.model = HashingSentenceEmbedder(EMBEDDING_CONFIG.fallback_dimension)
                self.using_fallback_model = True
                self.embedding_model_name = f"hashing-{EMBEDDING_CONFIG.fallback_dimension}"

        logger.info("Initializing ChromaDB with cosine metric...")
        with retry_on_failure(max_retries=3):
            self.client = chromadb.PersistentClient(path=embeddings_db_path)

        self.setup_ocr_tools()
        self.processed_files = self.load_processed_files()
        
        # Initialize monitoring
        self.processing_stats = ProcessingStats()
        self._monitoring_enabled = True
        
        logger.info("Testing model functionality...")
        if self.test_model():
            logger.info("✅ Optimized converter is functional!")
        else:
            logger.error("❌ Potential model issues detected")
    
    def _validate_environment(self):
        """Validează mediul de lucru și dependențele"""
        logger.info("Validating environment...")
        
        # Verifică Python version
        import sys
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8+ required")
        
        # Verifică memoria disponibilă
        memory = psutil.virtual_memory()
        if memory.available < 2 * 1024**3:  # 2GB
            logger.warning(f"Low memory available: {memory.available / 1024**3:.1f}GB")
        
        # Verifică spațiul disk
        disk = psutil.disk_usage('.')
        if disk.free < PROCESSING_CONFIG.disk_space_threshold_gb * 1024**3:
            logger.warning(f"Low disk space: {disk.free / 1024**3:.1f}GB (threshold: {PROCESSING_CONFIG.disk_space_threshold_gb}GB)")
        
        logger.info("Environment validation completed")
    
    def _check_disk_space(self):
        """Verifică spațiul disk disponibil"""
        try:
            target_path = self.embeddings_db_path
            if not os.path.exists(target_path):
                os.makedirs(target_path, exist_ok=True)

            disk = psutil.disk_usage(target_path)
            free_gb = disk.free / (1024**3)

            if free_gb < PROCESSING_CONFIG.disk_space_threshold_gb:
                reason = (
                    f"Insufficient disk space: {free_gb:.1f}GB available "
                    f"(threshold: {PROCESSING_CONFIG.disk_space_threshold_gb}GB)"
                )
                self._abort_processing = True
                self._abort_reason = reason
                raise RuntimeError(reason)

            logger.info(f"Disk space check: {free_gb:.1f}GB available")
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            raise

    def _check_runtime_resources(self, require_disk: bool = True) -> None:
        """Aplică pragurile configurate pentru memorie și spațiu pe disc."""
        if self._abort_processing:
            raise ResourceLimitExceeded(
                self._abort_reason
                or "Processing aborted due to previously detected resource limits"
            )

        try:
            memory = psutil.virtual_memory()
            memory_threshold = PROCESSING_CONFIG.memory_threshold_percent or 0
            if memory_threshold and memory.percent >= memory_threshold:
                reason = (
                    f"Memory usage {memory.percent:.1f}% exceeded configured threshold "
                    f"{memory_threshold:.1f}%"
                )
                if not self._abort_processing:
                    self._abort_processing = True
                    self._abort_reason = reason
                raise ResourceLimitExceeded(reason)

            if require_disk:
                disk_path = self.embeddings_db_path
                if not os.path.exists(disk_path):
                    disk_path = os.path.dirname(disk_path) or "."
                disk = psutil.disk_usage(disk_path)
                free_gb = disk.free / (1024**3)
                disk_threshold = PROCESSING_CONFIG.disk_space_threshold_gb or 0
                if disk_threshold and free_gb <= disk_threshold:
                    reason = (
                        f"Available disk space {free_gb:.2f}GB is below configured threshold "
                        f"{disk_threshold:.2f}GB"
                    )
                    if not self._abort_processing:
                        self._abort_processing = True
                        self._abort_reason = reason
                    raise ResourceLimitExceeded(reason)
        except ResourceLimitExceeded:
            raise
        except Exception as exc:
            logger.error(f"Runtime resource check failed: {exc}")
            raise
    
    def setup_ocr_tools(self):
        """Configurează instrumentele OCR pentru PDF-uri dificile"""
        self.ocr_available = OCR_AVAILABLE
        self.tesseract_available = False
        self.poppler_available = False
        
        if not OCR_AVAILABLE:
            print("⚠️ OCR nu este disponibil - instalați: pip install pytesseract pdf2image Pillow")
            return
            
        # Configurează paths pentru Tesseract și Poppler
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tesseract_path = os.path.join(current_dir, "install", "Tesseract-OCR", "tesseract.exe")
        poppler_path = os.path.join(current_dir, "install", "poppler-25.07.0", "Library", "bin")
        
        # Verifică Tesseract
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self.tesseract_available = True
            print(f"✅ Tesseract găsit: {tesseract_path}")
        else:
            print(f"❌ Tesseract nu a fost găsit la: {tesseract_path}")
        
        # Configurează Poppler path
        if os.path.exists(poppler_path):
            self.poppler_path = poppler_path
            self.poppler_available = True
            print(f"✅ Poppler găsit: {poppler_path}")
        else:
            self.poppler_path = None
            print(f"❌ Poppler nu a fost găsit la: {poppler_path}")
        
        # Status final OCR
        if self.tesseract_available and self.poppler_available:
            print("🎯 OCR complet funcțional - va putea procesa toate PDF-urile!")
        else:
            print("⚠️ OCR incomplet - unele PDF-uri dificile s-ar putea să nu fie procesate")
    
    def test_model(self):
        try:
            start_time = time.time()
            # Test cu instructiunea de retrieval si normalizare
            embedding = self.model.encode(
                [RETRIEVAL_INSTRUCTION + "Test simplu pentru verificare"],
                normalize_embeddings=True
            )
            duration = time.time() - start_time
            print(f"Test embedding optimizat: {embedding.shape}, timp: {duration:.2f}s")
            
            # Estimare pentru 162 chunks
            estimated_total = duration * 162
            print(f"Estimare pentru 162 chunks: {estimated_total/60:.1f} minute")
            return True
        except Exception as e:
            print(f"Eroare test: {e}")
            return False

    def get_optimal_batch_size(self, chunk_count: Optional[int] = None) -> int:
        """Calculează batch size optim bazat pe memoria disponibilă și numărul de chunks"""
        try:
            self._check_runtime_resources(require_disk=False)
            memory_percent = psutil.virtual_memory().percent
            available_gb = psutil.virtual_memory().available / (1024**3)
            cpu_count = psutil.cpu_count()
            memory_threshold = PROCESSING_CONFIG.memory_threshold_percent or 0

            # Logică îmbunătățită pentru batch sizing
            base_batch = 16

            if chunk_count is None:
                chunk_count = 0

            # Ajustare bazată pe memorie
            if memory_threshold and memory_percent >= memory_threshold:
                logger.warning(
                    "Memory usage %.1f%% reached configured threshold %.1f%% - forcing minimal batch",
                    memory_percent,
                    memory_threshold,
                )
                base_batch = 1
            elif memory_percent > 85:
                base_batch = 4   # Foarte conservativ
            elif memory_percent > 75:
                base_batch = 8   # Conservativ
            elif memory_percent > 60:
                base_batch = 16  # Normal
            else:
                base_batch = 32  # Agresiv
            
            # Ajustare bazată pe numărul de chunks
            if chunk_count > 1000:
                base_batch = min(base_batch, 16)  # Pentru fișiere foarte mari
            elif chunk_count > 500:
                base_batch = min(base_batch, 24)  # Pentru fișiere mari
            
            # Ajustare bazată pe CPU cores
            base_batch = min(base_batch, cpu_count * 4)
            
            # Ajustare bazată pe RAM disponibil
            if available_gb < 1:
                base_batch = 4
            elif available_gb < 2:
                base_batch = min(base_batch, 8)
            
            logger.info(f"Optimal batch size: {base_batch} (memory: {memory_percent:.1f}%, chunks: {chunk_count}, RAM: {available_gb:.1f}GB)")
            return max(1, base_batch)  # Minimum 1
            
        except Exception as e:
            logger.error(f"Error calculating batch size: {e}")
            return 8  # Fallback conservativ

    def extract_text_with_ocr(self, pdf_path: str, max_pages: int = None) -> str:
        """Extrage text din PDF folosind OCR pentru PDF-uri scanate sau cu imagini"""
        if not self.ocr_available or not self.tesseract_available or not self.poppler_available:
            logger.error("OCR not fully available")
            return ""
        
        with memory_managed_processing():
            with retry_on_failure(max_retries=2, delay=2.0):
                try:
                    logger.info(f"Processing with OCR: {os.path.basename(pdf_path)}")

                    # Convertește PDF în imagini cu retry
                    self._check_runtime_resources(require_disk=False)
                    images = convert_from_path(
                        pdf_path,
                        dpi=OCR_CONFIG.dpi,
                        poppler_path=self.poppler_path,
                        first_page=1,
                        last_page=max_pages if max_pages else OCR_CONFIG.max_pages_ocr
                    )
                    
                    logger.info(f"Converted to {len(images)} images")
                    
                    if not images:
                        return ""
                    
                    extracted_text = []

                    # Procesează fiecare pagină cu OCR și memory management
                    for i, image in enumerate(tqdm(images, desc="OCR pages")):
                        self._check_runtime_resources(require_disk=False)
                        try:
                            with memory_managed_processing():
                                # Optimizare imagine pentru OCR
                                if image.mode != 'RGB':
                                    image = image.convert('RGB')
                                
                                # OCR cu limba română și retry
                                with retry_on_failure(max_retries=OCR_CONFIG.max_retries, delay=OCR_CONFIG.retry_delay):
                                    page_text = pytesseract.image_to_string(
                                        image, 
                                        lang=OCR_CONFIG.languages,
                                        config=OCR_CONFIG.tesseract_config
                                    )
                                
                                if page_text.strip():
                                    extracted_text.append(f"\n--- Page {i+1} ---\n{page_text.strip()}")
                                
                                # Cleanup agresiv pentru fiecare imagine
                                del image
                                
                        except Exception as e:
                            logger.warning(f"OCR error on page {i+1}: {e}")
                            continue
                        
                        # Cleanup la fiecare 3 pagini pentru a preveni memory leaks
                        if i % 3 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    final_text = "\n".join(extracted_text)
                    logger.info(f"OCR completed: {len(final_text)} characters from {len(images)} pages")
                    
                    return final_text
                    
                except Exception as e:
                    logger.error(f"General OCR error: {e}")
                    return ""
    
    def is_pdf_text_extractable(self, pdf_path: str) -> Tuple[bool, str]:
        """Verifică dacă PDF-ul conține text extractabil și returnează motivul"""
        try:
            # Test rapid cu PyMuPDF
            doc = fitz.open(pdf_path)
            total_chars = 0
            pages_checked = min(3, len(doc))
            
            for page_num in range(pages_checked):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                total_chars += len(page_text.strip())
            
            doc.close()
            
            chars_per_page = total_chars / pages_checked if pages_checked > 0 else 0
            
            if chars_per_page < 50:
                return False, f"PDF scanat/imagini (doar {chars_per_page:.1f} chars/pagină)"
            else:
                return True, f"Text extractabil ({chars_per_page:.1f} chars/pagină)"
                
        except Exception as e:
            return False, f"Eroare verificare: {e}"

    # -------- Text extraction helpers (stream friendly) --------

    def _clean_page_text(self, text: str) -> str:
        """Normalizează textul extras pentru a reduce caracterele inutile."""
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.replace("\x00", " ").replace("\xa0", " ")
        text = text.replace("\r", "\n")
        text = re.sub(r"-\s*\n", "", text)  # Unește cuvintele despărțite la capăt de linie
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"[\t\u200b]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        cleaned = "".join(ch for ch in text if ch.isprintable() or ch in "\n ")
        return cleaned.strip()

    def _stream_text_fitz(self, pdf_path: str) -> Iterator[Tuple[str, int]]:
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            logger.warning(f"PyMuPDF open failed: {exc}")
            return iter([])

        try:
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                text = page.get_text("text") or ""
                cleaned = self._clean_page_text(text)
                if cleaned:
                    yield cleaned, page_index + 1
        finally:
            doc.close()

    def _stream_text_pdfplumber(self, pdf_path: str) -> Iterator[Tuple[str, int]]:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    cleaned = self._clean_page_text(text)
                    if cleaned:
                        yield cleaned, page_index + 1
        except Exception as exc:
            logger.warning(f"pdfplumber failed: {exc}")
            return iter([])

    def _stream_text_pypdf2(self, pdf_path: str) -> Iterator[Tuple[str, int]]:
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page_index, page in enumerate(reader.pages):
                    try:
                        text = page.extract_text() or ""
                    except Exception:
                        text = ""
                    cleaned = self._clean_page_text(text)
                    if cleaned:
                        yield cleaned, page_index + 1
        except Exception as exc:
            logger.warning(f"PyPDF2 failed: {exc}")
            return iter([])

    def _stream_text_ocr(self, pdf_path: str, max_pages: int = 50) -> Iterator[Tuple[str, int]]:
        if not (self.ocr_available and self.tesseract_available and self.poppler_available):
            return iter([])

        raw_text = self.extract_text_with_ocr(pdf_path, max_pages=max_pages)
        if not raw_text:
            return iter([])

        page_number = None
        buffer: List[str] = []

        page_pattern = re.compile(r"--- Page (\d+) ---")
        for line in raw_text.splitlines():
            match = page_pattern.match(line.strip())
            if match:
                if buffer and page_number is not None:
                    cleaned = self._clean_page_text(" ".join(buffer))
                    if cleaned:
                        yield cleaned, page_number
                page_number = int(match.group(1))
                buffer = []
            else:
                buffer.append(line)

        if buffer and page_number is not None:
            cleaned = self._clean_page_text(" ".join(buffer))
            if cleaned:
                yield cleaned, page_number

    def stream_text_with_pages(self, pdf_path: str) -> Iterator[Tuple[str, int]]:
        """Generează textul pagină cu pagină folosind prima metodă care reușește."""
        self._check_runtime_resources(require_disk=False)
        is_extractable, reason = self.is_pdf_text_extractable(pdf_path)
        logger.info(f"Text analysis for {os.path.basename(pdf_path)}: {reason}")

        extraction_methods: List[Tuple[str, Callable[[str], Iterable[Tuple[str, int]]]]] = []
        if is_extractable:
            extraction_methods.extend([
                ("PyMuPDF", self._stream_text_fitz),
                ("pdfplumber", self._stream_text_pdfplumber),
                ("PyPDF2", self._stream_text_pypdf2),
            ])
        else:
            logger.info("Direct text extraction likely to fail, trying OCR fallback")

        extraction_methods.append(("OCR", self._stream_text_ocr))

        for method_name, method in extraction_methods:
            yielded = False
            try:
                self._check_runtime_resources(require_disk=False)
                for page_text, page_number in method(pdf_path):
                    if not yielded:
                        logger.info(f"Using {method_name} for {os.path.basename(pdf_path)}")
                    yielded = True
                    self._check_runtime_resources(require_disk=False)
                    yield page_text, page_number
            except Exception as exc:
                logger.warning(f"{method_name} extraction failed: {exc}")
                yielded = False

            if yielded:
                return

        logger.error(f"All extraction methods failed for {os.path.basename(pdf_path)}")

    def extract_text_with_pages(self, pdf_path: str) -> Tuple[str, List[Tuple[str, int]]]:
        """Extrage textul integral folosind fluxul de pagini."""
        text_pages = list(self.stream_text_with_pages(pdf_path))
        if not text_pages:
            return "", []

        full_text = "\n".join(page_text for page_text, _ in text_pages)
        return full_text.strip(), text_pages

    def stream_chunks(
        self,
        pdf_path: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        min_words: int = 40,
    ) -> Iterator[Tuple[str, Dict[str, int]]]:
        """Generează chunk-uri din PDF fără a încărca întregul fișier în memorie."""

        if chunk_size is None:
            chunk_size = EMBEDDING_CONFIG.chunk_size
        if overlap is None:
            overlap = EMBEDDING_CONFIG.overlap

        if chunk_size <= 0:
            raise ValueError("chunk_size trebuie să fie > 0")

        word_buffer: List[Tuple[str, int]] = []

        for page_text, page_number in self.stream_text_with_pages(pdf_path):
            self._check_runtime_resources(require_disk=False)
            words = page_text.split()
            if not words:
                continue

            word_buffer.extend((word, page_number) for word in words)

            while len(word_buffer) >= chunk_size:
                self._check_runtime_resources(require_disk=False)
                chunk_slice = word_buffer[:chunk_size]
                chunk_words = [word for word, _ in chunk_slice]

                if len(chunk_words) < min_words:
                    break

                chunk_text = " ".join(chunk_words)
                chunk_metadata = {
                    "page_from": chunk_slice[0][1],
                    "page_to": chunk_slice[-1][1],
                    "sentence_count": max(1, len(re.split(r"[.!?]", chunk_text)) - 1),
                    "word_count": len(chunk_words),
                }

                yield chunk_text, chunk_metadata

                if overlap > 0:
                    word_buffer = word_buffer[chunk_size - overlap :]
                else:
                    word_buffer = []

        # Procesează cuvintele rămase
        if len(word_buffer) >= max(min_words, 1):
            self._check_runtime_resources(require_disk=False)
            chunk_words = [word for word, _ in word_buffer]
            chunk_text = " ".join(chunk_words)
            chunk_metadata = {
                "page_from": word_buffer[0][1],
                "page_to": word_buffer[-1][1],
                "sentence_count": max(1, len(re.split(r"[.!?]", chunk_text)) - 1),
                "word_count": len(chunk_words),
            }
            yield chunk_text, chunk_metadata

    def advanced_chunk_text(self, text: str, text_pages: List[Tuple[str, int]],
                       chunk_size: int = None, overlap: int = None) -> Tuple[List[str], List[Dict]]:
        """Chunking îmbunătățit cu LangChain pentru conținut educațional"""

        # Folosește configurația dinamică dacă nu sunt specificate parametrii
        if chunk_size is None:
            chunk_size = EMBEDDING_CONFIG.chunk_size
        if overlap is None:
            overlap = EMBEDDING_CONFIG.overlap
        
        # Folosește LangChain pentru chunking mai inteligent
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True
        )
        
        chunks = splitter.split_text(text)
        
        # Filtrează chunk-urile prea scurte
        meaningful_chunks = [chunk for chunk in chunks if len(chunk.split()) >= 10]
        
        # Generează metadata simplificată pentru LangChain chunks
        chunk_metadata = []
        for i, chunk in enumerate(meaningful_chunks):
            # Estimează pagina bazată pe poziția în text
            char_position = text.find(chunk)
            estimated_page = 1
            if text_pages and char_position != -1:
                cumulative_chars = 0
                for page_text, page_num in text_pages:
                    if cumulative_chars + len(page_text) > char_position:
                        estimated_page = page_num
                        break
                    cumulative_chars += len(page_text)
            
            chunk_metadata.append({
                "page_from": estimated_page,
                "page_to": estimated_page,
                "sentence_count": len([s for s in chunk.split('.') if s.strip()]),
                "word_count": len(chunk.split())
            })
        
        print(f"LangChain chunking: {len(meaningful_chunks)} bucăți create")
        return meaningful_chunks, chunk_metadata
    
    def generate_embeddings_streaming(self, chunks: List[str]) -> np.ndarray:
        """Generează embeddings cu streaming și dynamic batch sizing pentru optimizare memorie"""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        # Calculează batch size optim dinamic
        optimal_batch_size = self.get_optimal_batch_size(len(chunks))
        logger.info(f"Using dynamic batch size: {optimal_batch_size}")

        all_embeddings = []
        successful = 0
        failed = 0

        with memory_managed_processing():
            for i in range(0, len(chunks), optimal_batch_size):
                self._check_runtime_resources()
                batch_end = min(i + optimal_batch_size, len(chunks))
                batch_chunks = chunks[i:batch_end]

                try:
                    batch_num = i//optimal_batch_size + 1
                    total_batches = (len(chunks) + optimal_batch_size - 1)//optimal_batch_size
                    
                    logger.info(f"Processing batch {batch_num}/{total_batches}: chunks {i+1}-{batch_end}/{len(chunks)}")
                    
                    with retry_on_failure(max_retries=3, delay=1.0):
                        # Adaugă instrucțiunea de retrieval și normalizează
                        prefixed_chunks = [RETRIEVAL_INSTRUCTION + chunk for chunk in batch_chunks]
                        batch_embeddings = self.model.encode(
                            prefixed_chunks,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        
                        all_embeddings.append(batch_embeddings)
                        successful += len(batch_chunks)
                        
                        # Cleanup memorie după fiecare batch
                        del prefixed_chunks
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
                    failed += len(batch_chunks)
                    
                    # Încearcă să proceseze chunk-urile individual
                    logger.info(f"Attempting individual chunk processing for batch {batch_num}")
                    individual_embeddings = []
                    
                    for chunk in batch_chunks:
                        try:
                            with retry_on_failure(max_retries=2, delay=0.5):
                                individual_embedding = self.model.encode(
                                    [RETRIEVAL_INSTRUCTION + chunk],
                                    normalize_embeddings=True,
                                    show_progress_bar=False,
                                    convert_to_numpy=True
                                )
                                individual_embeddings.append(individual_embedding[0])
                                successful += 1
                                failed -= 1
                        except Exception as individual_error:
                            logger.warning(f"Individual chunk failed: {individual_error}")
                            # Embedding dummy pentru consistency
                            dummy_dim = self.model.get_sentence_embedding_dimension()
                            individual_embeddings.append(np.zeros(dummy_dim))
                    
                    if individual_embeddings:
                        all_embeddings.append(np.array(individual_embeddings))
        
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
            logger.info(f"Embeddings generation completed: {final_embeddings.shape} (successful: {successful}, failed: {failed})")
            return final_embeddings
        else:
            logger.error("No embeddings generated")
            return np.array([])
    
    def get_collection_name(self, file_path: str) -> str:
        """Generează nume colecție consistente"""
        path_parts = Path(file_path).parts
        
        if "materiale_didactice" in path_parts:
            idx = path_parts.index("materiale_didactice")
            relevant_parts = path_parts[idx+1:]
        else:
            relevant_parts = path_parts[:-1]
        
        clean_parts = []
        for part in relevant_parts:
            clean_part = part.replace(" ", "_").replace("-", "_").lower()
            clean_parts.append(clean_part)
        
        collection_name = "_".join(clean_parts)
        collection_name = "".join(c for c in collection_name if c.isalnum() or c == "_")
        
        if len(collection_name) > 60:  # Păstrează 3 caractere pentru siguranță
        # Scurtează inteligent
            parts = collection_name.split("_")
            short_name = "_".join(parts[:3])  # Păstrează primele 3 părți
            if len(short_name) > 60:
                short_name = short_name[:60]
            collection_name = short_name

        return collection_name or "general"
    
    def get_file_hash(self, file_path: str) -> str:
        """Calculează hash MD5 pentru detectarea modificărilor"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def load_processed_files(self) -> Dict[str, str]:
        """Încarcă registry-ul fișierelor procesate cu thread safety"""
        with self._file_lock:
            if os.path.exists(self.processed_files_path):
                try:
                    with open(self.processed_files_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Eroare citire processed_files.json: {e}. Creez registry nou.")
                    return {}
            return {}
    
    def save_processed_files_unsafe(self):
        """Salvează fără lock - pentru uz intern când lock-ul e deja achiziționat"""
        try:
            # Scriere atomică prin temp file
            temp_path = self.processed_files_path + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, ensure_ascii=False, indent=2)
            
            # Rename atomic (pe majoritatea sistemelor)
            if os.name == 'nt':  # Windows
                if os.path.exists(self.processed_files_path):
                    os.remove(self.processed_files_path)
                os.rename(temp_path, self.processed_files_path)
            else:  # Unix-like
                os.rename(temp_path, self.processed_files_path)
                
        except (IOError, OSError) as e:
            print(f"Eroare salvare processed_files.json: {e}")
            # Cleanup temp file dacă există
            temp_path = self.processed_files_path + '.tmp'
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def save_processed_files(self):
        """Salvează registry-ul fișierelor procesate cu thread safety"""
        with self._file_lock:
            self.save_processed_files_unsafe()
    
    def generate_stable_id(self, file_path: str, chunk_index: int, file_hash: str) -> str:
        """Generează ID-uri stabile pentru evitarea coliziunilor"""
        collection_name = self.get_collection_name(file_path)
        return f"{collection_name}_{file_hash[:8]}_{chunk_index}"

    def _persist_embeddings_batch(
        self,
        collection,
        chunk_texts: List[str],
        metadatas: List[Dict],
        ids: List[str],
    ) -> None:
        """Cod reutilizabil pentru a salva un batch de embeddings."""
        if not chunk_texts:
            return

        with retry_on_failure(max_retries=3, delay=1.0):
            self._check_runtime_resources()
            prefixed_chunks = [RETRIEVAL_INSTRUCTION + chunk for chunk in chunk_texts]
            embeddings = self.model.encode(
                prefixed_chunks,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self._check_runtime_resources()

            collection.add(
                embeddings=embeddings.tolist(),
                documents=chunk_texts,
                metadatas=metadatas,
                ids=ids,
            )

        # Cleanup memorie
        del prefixed_chunks, embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_pdf(self, pdf_path: str, progress_callback=None) -> bool:
        """Procesează PDF cu toate optimizările, progress tracking și resume capability"""
        start_time = time.time()

        if not pdf_path:
            logger.error('PDF path is required for processing')
            if progress_callback:
                progress_callback(0.0, 'PDF path is missing')
            return False

        if not os.path.exists(pdf_path):
            logger.error(f'File not found: {pdf_path}')
            if progress_callback:
                progress_callback(0.0, 'PDF file not found')
            return False

        if not os.path.isfile(pdf_path):
            logger.error(f'Provided path is not a file: {pdf_path}')
            if progress_callback:
                progress_callback(0.0, 'Invalid PDF path')
            return False

        try:
            self._check_runtime_resources()
        except ResourceLimitExceeded as exc:
            logger.error(f"Resource constraints prevent processing {pdf_path}: {exc}")
            if progress_callback:
                progress_callback(0.0, str(exc))
            return False

        try:
            file_hash = self.get_file_hash(pdf_path)
        except (OSError, IOError) as exc:
            logger.error(f'Failed to read PDF for hashing: {exc}')
            if progress_callback:
                progress_callback(0.0, 'Unable to read PDF contents')
            return False

        # Verifică dacă e deja procesat
        if pdf_path in self.processed_files and self.processed_files[pdf_path] == file_hash:
            logger.info(f"File already processed - skipping: {os.path.basename(pdf_path)}")
            return True

        logger.info(f"Processing optimized: {os.path.basename(pdf_path)}")

        try:
            self._check_runtime_resources()
            with memory_managed_processing():
                if progress_callback:
                    progress_callback(0.05, "Analyzing PDF structure...")

                collection_name = self.get_collection_name(pdf_path)
                with retry_on_failure(max_retries=3, delay=2.0):
                    collection = self.client.get_or_create_collection(
                        name=collection_name,
                        metadata={
                            "hnsw:space": "cosine",
                            "description": f"Optimized embeddings for {pdf_path}",
                            "created_at": datetime.now().isoformat(),
                            "chunk_count": 0,
                            "model": self.embedding_model_name,
                            "retrieval_instruction": True,
                            "normalized": True,
                            "file_hash": file_hash,
                        },
                    )

                optimal_batch_size = max(
                    1, min(DB_CONFIG.batch_size, self.get_optimal_batch_size())
                )
                chunk_buffer: List[str] = []
                metadata_buffer: List[Dict] = []
                id_buffer: List[str] = []
                chunk_count = 0
                words_total = 0

                for chunk_text, chunk_meta in self.stream_chunks(pdf_path):
                    if progress_callback and chunk_count == 0:
                        progress_callback(0.25, "Chunking text and generating embeddings...")

                    metadata = {
                        "source_file": pdf_path,
                        "chunk_index": chunk_count,
                        "processed_at": datetime.now().isoformat(),
                        "page_from": chunk_meta["page_from"],
                        "page_to": chunk_meta["page_to"],
                        "sentence_count": chunk_meta["sentence_count"],
                        "word_count": chunk_meta["word_count"],
                        "file_hash": file_hash,
                    }

                    chunk_buffer.append(chunk_text)
                    metadata_buffer.append(metadata)
                    id_buffer.append(self.generate_stable_id(pdf_path, chunk_count, file_hash))

                    chunk_count += 1
                    words_total += chunk_meta["word_count"]

                    self._check_runtime_resources(require_disk=False)
                    if len(chunk_buffer) >= optimal_batch_size:
                        self._check_runtime_resources()
                        self._persist_embeddings_batch(
                            collection, chunk_buffer, metadata_buffer, id_buffer
                        )
                        chunk_buffer.clear()
                        metadata_buffer.clear()
                        id_buffer.clear()

                        if progress_callback:
                            progress_callback(
                                min(0.9, 0.25 + chunk_count * 0.01),
                                f"Stored {chunk_count} chunks...",
                            )

                if chunk_buffer:
                    self._check_runtime_resources()
                    self._persist_embeddings_batch(
                        collection, chunk_buffer, metadata_buffer, id_buffer
                    )

                if chunk_count == 0:
                    logger.error("Could not create any chunks from PDF")
                    return False

                try:
                    updated_metadata = {
                        **(collection.metadata or {}),
                        "description": f"Optimized embeddings for {pdf_path}",
                        "chunk_count": chunk_count,
                        "model": self.embedding_model_name,
                        "retrieval_instruction": True,
                        "normalized": True,
                        "file_hash": file_hash,
                        "words_total": words_total,
                        "updated_at": datetime.now().isoformat(),
                    }
                    collection.modify(metadata=updated_metadata)
                except Exception as exc:
                    logger.warning(f"Could not update collection metadata: {exc}")

                self.processed_files[pdf_path] = file_hash
                self.save_processed_files()

                processing_time = time.time() - start_time
                logger.info(
                    f"Processing completed successfully in {processing_time:.2f}s - {chunk_count} chunks"
                )

                if progress_callback:
                    progress_callback(1.0, "Processing completed!")

                return True

        except ResourceLimitExceeded as exc:
            logger.error(f"Processing aborted for {pdf_path}: {exc}")
            if progress_callback:
                progress_callback(0.0, str(exc))
            return False
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False
    
    def process_pdfs_parallel(self, pdf_paths: List[str], max_workers: int = None,
                            progress_callback: Callable = None) -> List[ProcessingResult]:
        """Procesează multiple PDF-uri în paralel cu monitoring"""
        if max_workers is None:
            max_workers = min(PROCESSING_CONFIG.max_workers, psutil.cpu_count())

        logger.info(f"Starting parallel processing of {len(pdf_paths)} PDFs with {max_workers} workers")

        results = []
        start_time = time.time()

        try:
            self._check_runtime_resources()
        except ResourceLimitExceeded as exc:
            logger.error(f"Cannot start parallel processing due to resource limits: {exc}")
            return []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_pdf_with_monitoring, pdf_path): pdf_path
                for pdf_path in pdf_paths
            }
            abort_triggered = False

            # Process completed tasks with progress tracking
            with tqdm(total=len(pdf_paths), desc="Processing PDFs") as pbar:
                for future in as_completed(future_to_path):
                    pdf_path = future_to_path[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # Update stats
                        self.processing_stats.total_files += 1
                        if result.success:
                            self.processing_stats.successful += 1
                            self.processing_stats.total_chunks += result.chunks_created
                        else:
                            self.processing_stats.failed += 1

                        self.processing_stats.total_processing_time += result.processing_time
                        self.processing_stats.memory_peak = max(
                            self.processing_stats.memory_peak,
                            result.memory_used_mb
                        )

                        # Progress callback
                        if progress_callback:
                            progress = len(results) / len(pdf_paths)
                            progress_callback(progress, f"Processed {len(results)}/{len(pdf_paths)} files")

                        pbar.update(1)
                        pbar.set_postfix({
                            'Success': self.processing_stats.successful,
                            'Failed': self.processing_stats.failed,
                            'Success Rate': f"{self.processing_stats.successful/self.processing_stats.total_files*100:.1f}%"
                        })

                        if self._abort_processing:
                            abort_triggered = True
                            break

                    except ResourceLimitExceeded as exc:
                        logger.error(f"Resource limits exceeded while processing {pdf_path}: {exc}")
                        if not self._abort_processing:
                            self._abort_processing = True
                        if not self._abort_reason:
                            self._abort_reason = str(exc)
                        self.processing_stats.total_files += 1
                        self.processing_stats.failed += 1
                        results.append(ProcessingResult(
                            file_path=pdf_path,
                            success=False,
                            error_message=str(exc)
                        ))
                        abort_triggered = True
                        break
                    except Exception as e:
                        logger.error(f"Error processing {pdf_path}: {e}")
                        results.append(ProcessingResult(
                            file_path=pdf_path,
                            success=False,
                            error_message=str(e)
                        ))
                        self.processing_stats.failed += 1
                        pbar.update(1)

                        if self._abort_processing:
                            abort_triggered = True
                            break

                if abort_triggered:
                    for pending_future in future_to_path:
                        if not pending_future.done():
                            pending_future.cancel()

        total_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {total_time:.2f}s")
        logger.info(f"Results: {self.processing_stats.successful} successful, {self.processing_stats.failed} failed")

        if self._abort_processing:
            logger.warning(
                "Processing aborted due to resource constraints: %s",
                self._abort_reason or "threshold reached"
            )

        return results
    
    def _process_single_pdf_with_monitoring(self, pdf_path: str) -> ProcessingResult:
        """Procesează un singur PDF cu monitoring detaliat"""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            try:
                self._check_runtime_resources()
            except ResourceLimitExceeded as exc:
                return ProcessingResult(
                    file_path=pdf_path,
                    success=False,
                    processing_time=time.time() - start_time,
                    error_message=str(exc),
                    memory_used_mb=0,
                )

            # Verifică dacă e deja procesat
            file_hash = self.get_file_hash(pdf_path)
            if pdf_path in self.processed_files and self.processed_files[pdf_path] == file_hash:
                return ProcessingResult(
                    file_path=pdf_path,
                    success=True,
                    processing_time=time.time() - start_time,
                    memory_used_mb=0
                )

            # Procesează cu monitoring
            success = self.process_pdf(pdf_path)
            error_message = ""
            if not success and self._abort_processing:
                error_message = self._abort_reason or "Processing aborted due to resource constraints"

            processing_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory

            # Estimează numărul de chunks (simplificat)
            chunks_created = 0
            if success:
                try:
                    collection_name = self.get_collection_name(pdf_path)
                    collection = self.client.get_collection(collection_name)
                    chunks_created = collection.count()
                except:
                    chunks_created = 0
            
            return ProcessingResult(
                file_path=pdf_path,
                success=success,
                chunks_created=chunks_created,
                processing_time=processing_time,
                memory_used_mb=memory_used,
                error_message=error_message,
            )

        except ResourceLimitExceeded as exc:
            self._abort_processing = True
            if not self._abort_reason:
                self._abort_reason = str(exc)
            raise
        except Exception as e:
            return ProcessingResult(
                file_path=pdf_path,
                success=False,
                processing_time=time.time() - start_time,
                error_message=str(e),
                memory_used_mb=psutil.Process().memory_info().rss / 1024 / 1024 - initial_memory
            )
    
    def get_processing_stats(self) -> ProcessingStats:
        """Returnează statisticile de procesare"""
        # Update disk usage
        try:
            embeddings_size = sum(
                f.stat().st_size for f in Path(self.embeddings_db_path).rglob('*') if f.is_file()
            )
            self.processing_stats.disk_usage_mb = embeddings_size / (1024**2)
        except:
            pass
        
        return self.processing_stats
    
    def reset_processing_stats(self):
        """Resetează statisticile de procesare"""
        self.processing_stats = ProcessingStats()
        self._abort_processing = False
        self._abort_reason = ""
        logger.info("Processing statistics reset")
    
    def enhanced_search_educational(self, query: str, top_k: int = 5, collection_name: str = None):
        """Căutare îmbunătățită cu scoring special pentru conținut educațional"""
        
        # Căutare inițială cu mai multe rezultate
        initial_results = self.search_similar(query, top_k=top_k*2, collection_name=collection_name)
        
        if not initial_results or not initial_results.get('documents') or not initial_results['documents'][0]:
            return initial_results
        
        docs = initial_results['documents'][0]
        metadatas = initial_results.get('metadatas', [[]])[0] or [{} for _ in docs]
        similarities = initial_results.get('similarities', [[]])[0] or [0.0 for _ in docs]
        collections = initial_results.get('collections', [[]])[0] or ["unknown" for _ in docs]
        
        # Re-scoring cu boost pentru termeni educaționali
        enhanced_items = []
        
        for doc, meta, sim, col in zip(docs, metadatas, similarities, collections):
            # Scor de bază
            score = sim
            
            # Boost pentru termeni educaționali
            educational_boost = 0
            doc_lower = doc.lower()
            query_lower = query.lower()
            
            for keyword in EDUCATIONAL_KEYWORDS:
                if keyword in doc_lower:
                    educational_boost += 0.05
                if keyword in query_lower and keyword in doc_lower:
                    educational_boost += 0.1
            
            # Boost pentru coincidențe exacte de cuvinte
            query_words = set(query_lower.split())
            doc_words = set(doc_lower.split())
            word_overlap = len(query_words & doc_words)
            if word_overlap > 0:
                educational_boost += word_overlap * 0.02
            
            # Scor final
            final_score = min(score + educational_boost, 1.0)
            
            enhanced_items.append({
                "document": doc,
                "metadata": meta,
                "similarity": final_score,
                "collection": col,
                "original_score": sim,
                "educational_boost": educational_boost
            })
        
        # Sortare după scorul îmbunătățit
        enhanced_items.sort(key=lambda x: x["similarity"], reverse=True)
        top_items = enhanced_items[:top_k]
        
        # Format compatibil cu sistemul existent
        return {
            "documents": [[item["document"] for item in top_items]],
            "metadatas": [[item["metadata"] for item in top_items]],
            "similarities": [[item["similarity"] for item in top_items]],
            "collections": [[item["collection"] for item in top_items]]
        }

    def search_similar(self, query: str, top_k: int = 5, collection_name: str = None):
        """Căutare optimizată cu sortare globală și scoruri"""
        try:
            # Encodează query cu instrucțiunea de retrieval și normalizare
            query_embedding = self.model.encode(
                [RETRIEVAL_INSTRUCTION + query],
                normalize_embeddings=True
            )
            
            collections_to_search = []
            if collection_name:
                try:
                    collections_to_search = [self.client.get_collection(collection_name)]
                except:
                    print(f"Colecția {collection_name} nu există")
                    return None
            else:
                collections_to_search = self.client.list_collections()
            
            # Colectează rezultate din toate colecțiile cu scoruri
            all_items = []
            
            for collection in collections_to_search:
                try:
                    results = collection.query(
                        query_embeddings=query_embedding.tolist(),
                        n_results=min(top_k, 5)  # Ia mai puține din fiecare colecție
                    )
                    
                    if results and results.get('documents') and results['documents'][0]:
                        docs = results["documents"][0]
                        metadatas = results.get("metadatas", [[]])[0] or [{} for _ in docs]
                        distances = results.get("distances", [[]])[0] or [1.0 for _ in docs]
                        
                        # Pentru cosine similarity, convertește distanțele în similarități
                        similarities = [1 - dist for dist in distances]
                        
                        for doc, meta, sim in zip(docs, metadatas, similarities):
                            all_items.append({
                                "document": doc,
                                "metadata": meta,
                                "similarity": sim,
                                "collection": collection.name
                            })
                            
                except Exception as e:
                    print(f"Eroare căutare în {collection.name}: {e}")
            
            # Sortare globală după similaritate (descrescător)
            all_items.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Returnează top_k global cu format compatibil
            top_items = all_items[:top_k]
            
            if top_items:
                result = {
                    "documents": [[item["document"] for item in top_items]],
                    "metadatas": [[item["metadata"] for item in top_items]],
                    "similarities": [[item["similarity"] for item in top_items]],
                    "collections": [[item["collection"] for item in top_items]]
                }
                return result
            else:
                return {"documents": [[]], "metadatas": [[]], "similarities": [[]], "collections": [[]]}
                
        except Exception as e:
            print(f"Eroare căutare optimizată: {e}")
            return None
    
    def cleanup_resources(self):
        """Cleanup manual pentru resurse, temp files și optimizare disk usage"""
        try:
            print("🧹 Cleanup resurse...")
            
            # Force garbage collection pentru Python
            import gc
            gc.collect()
            
            # Cleanup CUDA cache daca existe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Cleanup temp files Windows specifice pentru OCR și PDF
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_patterns = ['tmp', 'pdf2image', 'tesseract', 'chroma', 'sentence-transformers']
            
            cleaned_count = 0
            for pattern in temp_patterns:
                try:
                    temp_files = Path(temp_dir).glob(f"*{pattern}*")
                    for temp_file in temp_files:
                        try:
                            if temp_file.is_file():
                                temp_file.unlink()
                                cleaned_count += 1
                            elif temp_file.is_dir():
                                import shutil
                                shutil.rmtree(temp_file, ignore_errors=True)
                                cleaned_count += 1
                        except (PermissionError, FileNotFoundError):
                            continue  # Skip files in use
                except Exception:
                    continue
            
            if cleaned_count > 0:
                print(f"🗑️ Șters {cleaned_count} fișiere temporare")
                      # Reconnect ChromaDB fără stop forțat
            try:
                # Testează conexiunea și reconectează dacă e nevoie
                test_collections = self.client.list_collections()
                print(f"ChromaDB activ cu {len(test_collections)} colecții")
            except Exception as e:
                print(f"Reconectez ChromaDB după eroare: {e}")
                self.client = chromadb.PersistentClient(path=self.embeddings_db_path)
                
                print("✅ Cleanup resurse completat")
            
        except Exception as e:
            print(f"⚠️ Warning cleanup general: {e}")
    
    def get_disk_usage_stats(self):
        """Returnează statistici despre utilizarea disk-ului"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.embeddings_db_path)
            
            # Calculează dimensiunea embeddings_db
            embeddings_size = sum(
                f.stat().st_size for f in Path(self.embeddings_db_path).rglob('*') if f.is_file()
            )
            
            return {
                "total_disk_gb": round(total / (1024**3), 2),
                "used_disk_gb": round(used / (1024**3), 2), 
                "free_disk_gb": round(free / (1024**3), 2),
                "disk_usage_percent": round((used/total)*100, 1),
                "embeddings_db_mb": round(embeddings_size / (1024**2), 2)
            }
        except Exception as e:
            return {"error": str(e)}
        
    def list_collections(self):  
        """Listează colecții cu informații detaliate"""
        collections = self.client.list_collections()
        print(f"Colecții disponibile ({len(collections)}):")
        
        for collection in collections:
            try:
                count = collection.count()
                metadata = collection.metadata or {}
                model = metadata.get("model", "necunoscut")
                optimized = "✓" if metadata.get("retrieval_instruction") else "✗"
                
                print(f"  - {collection.name}: {count} documente")
                print(f"    Model: {model}, Optimizat: {optimized}")
                
                if "created_at" in metadata:
                    print(f"    Creat: {metadata['created_at'][:16]}")
                    
            except Exception as e:
                print(f"  - {collection.name}: eroare - {e}")

if __name__ == "__main__":
    converter = PDFEmbeddingsConverter()
