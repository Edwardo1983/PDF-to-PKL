"""
API optimizat pentru deployment Railway cu PDF to Embeddings
Compatibil cu OpenAI și optimizat pentru performanță
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import convertorul optimizat
from pdf_converter_working import PDFEmbeddingsConverter

# Configurare logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurare app
app = FastAPI(
    title="AI Educational - PDF Embeddings API",
    description="API optimizat pentru conversie PDF în embeddings compatibile OpenAI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware pentru limitarea dimensiunii request-urilor
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """Middleware pentru limitarea dimensiunii upload-urilor"""
    MAX_SIZE = 50 * 1024 * 1024  # 50MB
    
    # Verifică Content-Length header
    content_length = request.headers.get('content-length')
    if content_length:
        try:
            size = int(content_length)
            if size > MAX_SIZE:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Request too large (max {MAX_SIZE//1024//1024}MB)"}
                )
        except ValueError:
            pass  # Continuă dacă header-ul nu e valid
    
    response = await call_next(request)
    return response

# CORS pentru integrare frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # În producție, specifică domeniile
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models pentru API
class QueryRequest(BaseModel):
    query: str = Field(..., description="Întrebarea sau textul de căutat")
    top_k: int = Field(5, ge=1, le=20, description="Numărul de rezultate dorite")
    collection_name: Optional[str] = Field(None, description="Nume colecție specifică (opțional)")
    grade: Optional[str] = Field(None, description="Clasa pentru filtrare contextuală")
    subject: Optional[str] = Field(None, description="Materia pentru filtrare contextuală")

class SearchResponse(BaseModel):
    success: bool
    results: List[Dict]
    total_found: int
    query_time: float
    collections_searched: List[str]

class ProcessResponse(BaseModel):
    success: bool
    message: str
    file_name: Optional[str] = None
    collection_name: Optional[str] = None
    chunks_created: Optional[int] = None
    processing_time: Optional[float] = None

class CollectionInfo(BaseModel):
    name: str
    document_count: int
    created_at: Optional[str] = None
    model: Optional[str] = None
    optimized: bool = False

class StatsResponse(BaseModel):
    total_collections: int
    total_documents: int
    optimized_collections: int
    collections: List[CollectionInfo]
    storage_size_mb: Optional[float] = None

# Inițializare converter global
converter = None

@app.on_event("startup")
async def startup_event():
    """Inițializează convertorul la startup"""
    global converter
    try:
        logger.info("🚀 Inițializez PDF Embeddings Converter optimizat...")
        
        # Configurare pentru Railway
        embeddings_path = os.getenv("EMBEDDINGS_PATH", "./embeddings_db")
        
        converter = PDFEmbeddingsConverter(embeddings_db_path=embeddings_path)
        logger.info("✅ Converter inițializat cu succes!")
        
    except Exception as e:
        logger.error(f"❌ Eroare inițializare converter: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI Educational PDF Embeddings API",
        "status": "running",
        "version": "2.0.0",
        "optimized": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test basic converter functionality
        if converter is None:
            raise Exception("Converter not initialized")
        
        # Test model
        test_result = converter.test_model()
        
        return {
            "status": "healthy",
            "converter": "initialized",
            "model": "BAAI/bge-m3",
            "model_test": "passed" if test_result else "failed",
            "optimizations": {
                "retrieval_instruction": True,
                "normalized_embeddings": True,
                "cosine_similarity": True,
                "chunking_with_overlap": True,
                "page_tracking": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_embeddings(request: QueryRequest):
    """Căutare optimizată în embeddings"""
    if converter is None:
        raise HTTPException(status_code=503, detail="Converter not initialized")
    
    start_time = datetime.now()
    
    try:
        # Construiește collection_name din grade și subject dacă sunt specificate
        target_collection = request.collection_name
        if request.grade and request.subject and not target_collection:
            # Construiește numele colecției bazat pe grade și subject
            target_collection = f"scoala_normala_clasa_{request.grade}_{request.subject.lower()}"
        
        # Efectuează căutarea optimizată
        results = converter.search_similar(
            query=request.query,
            top_k=request.top_k,
            collection_name=target_collection
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return SearchResponse(
                success=True,
                results=[],
                total_found=0,
                query_time=processing_time,
                collections_searched=[]
            )
        
        # Formatează rezultatele pentru OpenAI compatibility
        formatted_results = []
        docs = results['documents'][0]
        metadatas = results.get('metadatas', [[]])[0] or [{} for _ in docs]
        similarities = results.get('similarities', [[]])[0] or [0.0 for _ in docs]
        collections_used = results.get('collections', [[]])[0] or ["unknown" for _ in docs]
        
        for i, (doc, meta, sim, collection) in enumerate(zip(docs, metadatas, similarities, collections_used)):
            formatted_result = {
                "index": i,
                "content": doc,
                "similarity": round(sim, 4),
                "collection": collection,
                "metadata": {
                    "source_file": meta.get("source_file", "").split("/")[-1] if meta.get("source_file") else "",
                    "page_from": meta.get("page_from"),
                    "page_to": meta.get("page_to"),
                    "chunk_index": meta.get("chunk_index"),
                    "word_count": meta.get("word_count"),
                    "processed_at": meta.get("processed_at")
                }
            }
            formatted_results.append(formatted_result)
        
        return SearchResponse(
            success=True,
            results=formatted_results,
            total_found=len(formatted_results),
            query_time=round(processing_time, 3),
            collections_searched=list(set(collections_used))
        )
        
    except Exception as e:
        logger.error(f"Eroare căutare: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/upload", response_model=ProcessResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to process")
):
    """Upload și procesare PDF în background cu limite robuste"""
    if converter is None:
        raise HTTPException(status_code=503, detail="Converter not initialized")
    
    # Validare fișier
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Limită de dimensiune robustă (50MB)
    MAX_SIZE = 50 * 1024 * 1024  # 50MB
    
    try:
        # Salvează fișierul cu verificare dimensiune în timp real
        upload_dir = "./uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            total_size = 0
            async for chunk in file.iter_chunks():
                total_size += len(chunk)
                
                # Verifică limita în timp real
                if total_size > MAX_SIZE:
                    # Cleanup și eroare
                    buffer.close()
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    raise HTTPException(
                        status_code=413, 
                        detail=f"File too large (>{MAX_SIZE//1024//1024}MB limit)"
                    )
                
                buffer.write(chunk)
        
        logger.info(f"File uploaded successfully: {file.filename} ({total_size} bytes)")
        
        # Procesează în background
        background_tasks.add_task(process_pdf_background, file_path, file.filename)
        
        return ProcessResponse(
            success=True,
            message="File uploaded successfully. Processing in background.",
            file_name=file.filename,
            collection_name=converter.get_collection_name(file_path)
        )
        
    except HTTPException:
        # Re-ridică HTTPException-urile
        raise
    except Exception as e:
        logger.error(f"Eroare upload: {e}")
        # Cleanup în caz de eroare
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

async def process_pdf_background(file_path: str, filename: str):
    """Procesează PDF în background"""
    try:
        start_time = datetime.now()
        logger.info(f"🔄 Începe procesarea în background: {filename}")
        
        success = converter.process_pdf(file_path)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if success:
            logger.info(f"✅ Procesare completă pentru {filename} în {processing_time:.2f}s")
        else:
            logger.error(f"❌ Procesare eșuată pentru {filename}")
        
        # Cleanup
        try:
            os.remove(file_path)
        except:
            pass
            
    except Exception as e:
        logger.error(f"Eroare procesare background {filename}: {e}")

@app.get("/collections", response_model=List[CollectionInfo])
async def list_collections():
    """Listează toate colecțiile disponibile"""
    if converter is None:
        raise HTTPException(status_code=503, detail="Converter not initialized")
    
    try:
        collections = converter.client.list_collections()
        
        result = []
        for collection in collections:
            try:
                metadata = collection.metadata or {}
                
                info = CollectionInfo(
                    name=collection.name,
                    document_count=collection.count(),
                    created_at=metadata.get("created_at"),
                    model=metadata.get("model", "unknown"),
                    optimized=bool(metadata.get("retrieval_instruction", False))
                )
                result.append(info)
                
            except Exception as e:
                logger.warning(f"Eroare citire colecție {collection.name}: {e}")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Statistici detaliate despre embeddings"""
    if converter is None:
        raise HTTPException(status_code=503, detail="Converter not initialized")
    
    try:
        collections = converter.client.list_collections()
        
        total_documents = 0
        optimized_count = 0
        collection_infos = []
        
        for collection in collections:
            try:
                count = collection.count()
                metadata = collection.metadata or {}
                is_optimized = bool(metadata.get("retrieval_instruction", False))
                
                total_documents += count
                if is_optimized:
                    optimized_count += 1
                
                info = CollectionInfo(
                    name=collection.name,
                    document_count=count,
                    created_at=metadata.get("created_at"),
                    model=metadata.get("model", "unknown"),
                    optimized=is_optimized
                )
                collection_infos.append(info)
                
            except Exception as e:
                logger.warning(f"Eroare statistici colecție {collection.name}: {e}")
        
        # Calculează dimensiunea storage-ului dacă posibil
        storage_size_mb = None
        try:
            if os.path.exists(converter.embeddings_db_path):
                size_bytes = sum(
                    os.path.getsize(os.path.join(root, file))
                    for root, dirs, files in os.walk(converter.embeddings_db_path)
                    for file in files
                )
                storage_size_mb = round(size_bytes / (1024 * 1024), 2)
        except:
            pass
        
        return StatsResponse(
            total_collections=len(collections),
            total_documents=total_documents,
            optimized_collections=optimized_count,
            collections=collection_infos,
            storage_size_mb=storage_size_mb
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@app.get("/openai-compatible/{collection_name}")
async def openai_compatible_context(
    collection_name: str,
    query: str,
    max_tokens: int = 1000
):
    """
    Endpoint compatibil OpenAI pentru a obține context din embeddings
    Returnează context formatat pentru a fi folosit în prompt-uri OpenAI
    """
    if converter is None:
        raise HTTPException(status_code=503, detail="Converter not initialized")
    
    try:
        # Căutare optimizată
        results = converter.search_similar(
            query=query,
            top_k=3,
            collection_name=collection_name
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return {
                "context": "Nu am găsit informații relevante în materialele didactice.",
                "sources": [],
                "query": query,
                "collection": collection_name
            }
        
        # Construiește contextul optimizat pentru LLM
        context_parts = []
        sources = []
        
        docs = results['documents'][0]
        metadatas = results.get('metadatas', [[]])[0] or [{} for _ in docs]
        similarities = results.get('similarities', [[]])[0] or [0.0 for _ in docs]
        
        token_count = 0
        for i, (doc, meta, sim) in enumerate(zip(docs, metadatas, similarities)):
            if token_count >= max_tokens:
                break
            
            # Estimează tokens (aproximativ 4 caractere = 1 token)
            doc_tokens = len(doc) // 4
            if token_count + doc_tokens > max_tokens:
                # Trunchiază documentul
                remaining_tokens = max_tokens - token_count
                doc = doc[:remaining_tokens * 4] + "..."
            
            context_parts.append(doc)
            token_count += len(doc) // 4
            
            # Adaugă sursă
            source_info = {
                "similarity": round(sim, 3),
                "source_file": meta.get("source_file", "").split("/")[-1] if meta.get("source_file") else "",
                "pages": f"{meta.get('page_from', '')}-{meta.get('page_to', '')}" if meta.get('page_from') else ""
            }
            sources.append(source_info)
        
        context = "\n\n---\n\n".join(context_parts)
        
        return {
            "context": context,
            "sources": sources,
            "query": query,
            "collection": collection_name,
            "token_estimate": token_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting context: {str(e)}")

# Pentru development local
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "railway_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # False pentru producție
        log_level="info"
    )