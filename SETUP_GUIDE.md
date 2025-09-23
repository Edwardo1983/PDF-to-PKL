# 🎓 AI Educational - PDF to Embeddings Converter

## 📋 Descriere
Sistem de conversie PDF în embeddings folosind modelul BAAI/bge-m3, care rulează local și păstrează structura folderelor tale.

## 🔧 Instalare și Setup

### Pas 1: Pregătirea folderului
```bash
# Creează un folder nou pentru proiect
mkdir PDF_to_Embeddings
cd PDF_to_Embeddings
```

### Pas 2: Instalarea dependențelor
```bash
# Instalează Python packages necesare
pip install -r requirements.txt
```

**NOTĂ**: Prima rulare va descărca modelul BAAI/bge-m3 (~2GB). Asigură-te că ai conexiune la internet.

### Pas 3: Structura proiectului
```
PDF_to_Embeddings/
├── requirements.txt
├── pdf_converter_working.py
├── test_pdf_working.py
├── SETUP_GUIDE.md
├── embeddings_db/          # Se creează automat - aici se salvează embeddings-urile
├── new_pdfs/              # Se creează automat - pune PDF-urile noi aici
├── processed_pdfs/        # Se creează automat - PDF-urile procesate se mută aici
└── materiale_didactice/   # Copiază aici folderul tău cu PDF-uri (opțional)
```

## Utilizare:
```bash
python test_pdf_working.py
```

Aceasta va deschide meniul interactiv cu opțiunile:
1. **Procesează un director întreg** - Procesează toate PDF-urile dintr-un folder
2. **Procesează un singur fișier** - Procesează un PDF specific  
3. **Procesează materiale_didactice** - Procesează folderul tău de materiale
4. **Listează colecțiile** - Vezi ce embeddings ai creat
5. **Testează căutarea** - Caută în embeddings-uri
6. **Ieșire**

### Opțiunea 2: Procesare automată (Batch)
```bash
python batch_converter.py
```

Aceasta monitorizează folderul `new_pdfs/` și procesează automat orice PDF nou adăugat.

### Opțiunea 3: Import direct în cod
```python
from pdf_embeddings_converter import PDFEmbeddingsConverter

# Inițializează convertorul
converter = PDFEmbeddingsConverter()

# Procesează un director
converter.process_directory("./materiale_didactice")

# Procesează un fișier
converter.process_single_file("./test.pdf")

# Caută în embeddings
results = converter.search_similar("ce este un cerc", top_k=5)
```

## 🏗️ Cum funcționează

### 1. Structura colecțiilor
Sistemul creează automat colecții bazate pe structura de foldere:
```
materiale_didactice/Scoala_Normala/clasa_2/Matematica/matematica.pdf
↓
Colecția: scoala_normala_clasa_2_matematica
```

### 2. Procesarea PDF-urilor
- Extrage text din fiecare PDF
- Împarte textul în bucăți de ~1000 cuvinte cu overlap de 100
- Generează embeddings cu BAAI/bge-m3
- Salvează în ChromaDB cu metadata

### 3. Evitarea duplicatelor
- Calculează hash MD5 pentru fiecare fișier
- Ține evidența fișierelor procesate în `processed_files.json`
- Reprocesează doar fișierele modificate

## 📁 Organizarea fișierelor

### Pentru materiale existente:
1. Copiază folderul `materiale_didactice` în `PDF_to_Embeddings/`
2. Rulează `python main.py` și alege opțiunea 3
3. Sistemul va procesa toate PDF-urile păstrând structura

### Pentru PDF-uri noi:
1. Pune PDF-urile noi în folderul `new_pdfs/`
2. Rulează `python batch_converter.py`
3. PDF-urile vor fi procesate și mutate în `processed_pdfs/`

## 🔍 Căutare în embeddings

```python
# Exemplu de căutare
converter = PDFEmbeddingsConverter()

# Caută în toate colecțiile
results = converter.search_similar("ce este un cerc")

# Caută într-o colecție specifică
results = converter.search_similar(
    "ce este un cerc", 
    collection_name="scoala_normala_clasa_2_matematica"
)
```

## 🛠️ Configurări avansate

### Modificarea dimensiunii bucăților
```python
# În pdf_embeddings_converter.py, linia ~85
chunks = self.chunk_text(text, chunk_size=1500, overlap=150)
```

### Schimbarea modelului
```python
# În __init__, linia ~25
self.model = SentenceTransformer('alt-model-name')
```

### Configurarea ChromaDB
```python
# Pentru deployment cloud, modifică în __init__:
self.client = chromadb.HttpClient(host="your-host", port=8000)
```

## 🐛 Troubleshooting

### Erori comune:

**1. "No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers
```

**2. "CUDA out of memory"**
```python
# În __init__, forțează CPU:
self.model = SentenceTransformer('BAAI/bge-m3', device='cpu')
```

**3. "Cannot read PDF"**
- Verifică dacă PDF-ul nu este corupt
- Încearcă să deschizi PDF-ul manual
- Unele PDF-uri scanate nu conțin text extractabil

**4. Procesare lentă**
- Prima rulare descarcă modelul (~2GB)
- Modelul BGE-M3 este optimizat pentru CPU
- Pentru GPU: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Verificarea statusului
```bash
# Vezi ce colecții ai creat
python -c "from pdf_embeddings_converter import PDFEmbeddingsConverter; PDFEmbeddingsConverter().list_collections()"

# Vezi câte fișiere ai procesat
cat embeddings_db/processed_files.json
```

## 🔗 Integrare cu proiectul principal

Pentru a integra embeddings-urile în proiectul tău "AI Educational":

### 1. Copierea bazei de date
```bash
# Copiază folderul embeddings_db în proiectul principal
cp -r embeddings_db/ /path/to/AISchool/embeddings_db/
```

### 2. Cod pentru căutare în proiectul principal
```python
import chromadb
from sentence_transformers import SentenceTransformer

class AITeacherWithEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-m3')
        self.client = chromadb.PersistentClient(path="./embeddings_db")
    
    def get_context_for_query(self, user_question: str, grade: str, subject: str):
        """Găsește context relevant pentru întrebarea utilizatorului"""
        
        # Construiește numele colecției bazat pe clasă și materie
        collection_name = f"scoala_normala_clasa_{grade}_{subject.lower()}"
        
        try:
            collection = self.client.get_collection(collection_name)
            
            # Caută în embeddings
            query_embedding = self.model.encode([user_question])
            results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=3
            )
            
            # Returnează contextul găsit
            if results['documents']:
                context = "\n\n".join(results['documents'][0])
                return context
            else:
                return "Nu am găsit informații relevante în materialele didactice."
                
        except Exception as e:
            return f"Eroare la căutarea în embeddings: {e}"
    
    def answer_with_context(self, user_question: str, grade: str, subject: str):
        """Răspunde la întrebare folosind contextul din embeddings"""
        context = self.get_context_for_query(user_question, grade, subject)
        
        # Aici integrezi cu LLM-ul tău (OpenAI, Claude, etc.)
        prompt = f"""
        Ești un profesor AI care ajută copiii cu temele. 
        
        Context din materialele didactice:
        {context}
        
        Întrebarea elevului: {user_question}
        
        Răspunde simplu și pedagogic, adaptând explicația pentru clasa {grade}.
        Folosește doar informațiile din context și explică pas cu pas.
        """
        
        # Returnează prompt-ul pentru LLM sau procesează direct
        return prompt
```

### 3. API pentru Railway deployment
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI Educational API")

class QuestionRequest(BaseModel):
    question: str
    grade: str
    subject: str
    school_type: str = "normala"  # sau "muzica"

teacher = AITeacherWithEmbeddings()

@app.post("/ask")
async def ask_teacher(request: QuestionRequest):
    try:
        context = teacher.get_context_for_query(
            request.question, 
            request.grade, 
            request.subject
        )
        
        return {
            "context": context,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """Lista toate colecțiile disponibile"""
    collections = teacher.client.list_collections()
    return {
        "collections": [
            {
                "name": col.name,
                "count": col.count()
            } for col in collections
        ]
    }
```

## 📊 Monitorizare și statistici

### Script pentru statistici
```python
# stats.py
from pdf_embeddings_converter import PDFEmbeddingsConverter
import json

def generate_stats():
    converter = PDFEmbeddingsConverter()
    collections = converter.client.list_collections()
    
    stats = {
        "total_collections": len(collections),
        "total_documents": sum(col.count() for col in collections),
        "collections_detail": []
    }
    
    for col in collections:
        stats["collections_detail"].append({
            "name": col.name,
            "document_count": col.count(),
            "metadata": col.metadata
        })
    
    # Salvează statisticile
    with open("embeddings_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Statistici generate:")
    print(f"   - {stats['total_collections']} colecții")
    print(f"   - {stats['total_documents']} documente totale")
    
    return stats

if __name__ == "__main__":
    generate_stats()
```

## 🚀 Deploy pe Railway

### 1. Pregătirea pentru production
```python
# config.py
import os

class Config:
    # Pentru development
    EMBEDDINGS_PATH = "./embeddings_db"
    MODEL_NAME = "BAAI/bge-m3"
    
    # Pentru production
    if os.getenv("RAILWAY_ENVIRONMENT"):
        EMBEDDINGS_PATH = "/app/embeddings_db"
        # Folosește un model mai mic pentru production
        MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
```

### 2. Dockerfile pentru Railway
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalează dependențele de sistem
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiază requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiază codul aplicației
COPY . .

# Copiază embeddings-urile (dacă nu sunt prea mari)
COPY embeddings_db ./embeddings_db

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Alternative pentru embeddings mari
Pentru că embeddings-urile pot fi mari pentru Railway:

**Opțiunea A: S3/Cloud Storage**
```python
import boto3

class CloudEmbeddingsManager:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.bucket = "your-embeddings-bucket"
    
    def download_embeddings(self):
        # Descarcă embeddings-urile la startup
        local_path = "./embeddings_db"
        # ... cod pentru download din S3
```

**Opțiunea B: Embeddings on-demand**
```python
class OnDemandEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('BAAI/bge-m3')
        self.pdf_cache = {}  # Cache pentru PDF-uri procesate
    
    def get_context(self, query, pdf_content):
        # Generează embeddings în timp real
        if pdf_content not in self.pdf_cache:
            chunks = self.chunk_text(pdf_content)
            embeddings = self.model.encode(chunks)
            self.pdf_cache[pdf_content] = (chunks, embeddings)
        
        chunks, embeddings = self.pdf_cache[pdf_content]
        # Caută cel mai relevant chunk
        query_embedding = self.model.encode([query])
        # ... logica de căutare
```

## 🔧 Comenzi utile

```bash
# Verifică dimensiunea bazei de date
du -sh embeddings_db/

# Șterge toate embeddings-urile (ATENȚIE!)
rm -rf embeddings_db/

# Reprocesează doar fișierele modificate
rm embeddings_db/processed_files.json

# Creează backup
tar -czf embeddings_backup.tar.gz embeddings_db/

# Restaurează backup
tar -xzf embeddings_backup.tar.gz
```

## 📝 To-Do pentru integrare

1. ✅ **Crează sistemul de embeddings local**
2. ✅ **Testează pe materialele tale didactice**
3. 🔲 **Integrează cu proiectul AI Educational**
4. 🔲 **Optimizează pentru Railway deployment**
5. 🔲 **Conectează cu Bubble.io frontend**
6. 🔲 **Adaugă cache pentru performanță**
7. 🔲 **Implementează logging și monitoring**

## 📞 Support

Pentru probleme sau întrebări:
1. Verifică secțiunea Troubleshooting
2. Rulează `python -c "from pdf_embeddings_converter import PDFEmbeddingsConverter; PDFEmbeddingsConverter().list_collections()"`
3. Verifică log-urile în terminal
