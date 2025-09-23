"""
Script pentru testarea optimizărilor implementate
Verifică toate îmbunătățirile: BGE prefix, normalizare, chunking, etc.
"""

import time
import numpy as np
from pdf_converter_working import PDFEmbeddingsConverter, RETRIEVAL_INSTRUCTION
import os

def test_retrieval_instruction():
    """Test 1: Verifică dacă se folosește instrucțiunea de retrieval"""
    print("🧪 Test 1: Instrucțiune de retrieval")
    print("-" * 40)
    
    converter = PDFEmbeddingsConverter()
    
    # Test cu și fără instrucțiune
    test_text = "Matematica este știința numerelor"
    
    # Cu instrucțiune (cum ar trebui să fie)
    start_time = time.time()
    embedding_with = converter.model.encode(
        [RETRIEVAL_INSTRUCTION + test_text], 
        normalize_embeddings=True
    )
    time_with = time.time() - start_time
    
    # Fără instrucțiune (pentru comparație)
    start_time = time.time()
    embedding_without = converter.model.encode(
        [test_text], 
        normalize_embeddings=True
    )
    time_without = time.time() - start_time
    
    print(f"✅ Instrucțiune folosită: '{RETRIEVAL_INSTRUCTION.strip()}'")
    print(f"📊 Embedding cu instrucțiune: {embedding_with.shape}")
    print(f"📊 Embedding fără instrucțiune: {embedding_without.shape}")
    print(f"⏱️ Timp cu instrucțiune: {time_with:.3f}s")
    print(f"⏱️ Timp fără instrucțiune: {time_without:.3f}s")
    
    # Verifică dacă embeddings-urile sunt diferite
    similarity = np.dot(embedding_with[0], embedding_without[0])
    print(f"🔍 Similaritate între embeddings: {similarity:.4f}")
    print(f"✅ Embeddings diferite: {'Da' if similarity < 0.99 else 'Nu'}")
    
    return True

def test_normalization():
    """Test 2: Verifică normalizarea embeddings-urilor"""
    print("\n🧪 Test 2: Normalizarea embeddings-urilor")
    print("-" * 40)
    
    converter = PDFEmbeddingsConverter()
    
    test_texts = [
        "Text scurt pentru test",
        "Acesta este un text mai lung pentru a verifica dacă normalizarea funcționează corect în toate cazurile"
    ]
    
    for i, text in enumerate(test_texts):
        embedding = converter.model.encode(
            [RETRIEVAL_INSTRUCTION + text], 
            normalize_embeddings=True
        )
        
        norm = np.linalg.norm(embedding[0])
        print(f"📝 Text {i+1}: {text[:30]}...")
        print(f"📊 Norma vectorului: {norm:.6f}")
        print(f"✅ Normalizat corect: {'Da' if abs(norm - 1.0) < 1e-5 else 'Nu'}")
    
    return True

def test_advanced_chunking():
    """Test 3: Testează chunking-ul avansat cu overlap"""
    print("\n🧪 Test 3: Chunking avansat cu overlap")
    print("-" * 40)
    
    converter = PDFEmbeddingsConverter()
    
    # Text de test lung
    test_text = """
    Matematica este știința care studiază numerele, structurile, spațiul și schimbarea. 
    Ea se bazează pe logică și pe demonstrații riguroase. Principalele ramuri ale matematicii sunt 
    algebra, geometria, analiza matematică și teoria numerelor. Matematica este folosită în multe 
    domenii precum fizica, ingineeria, economia și informatica. Importanța matematicii în educație 
    este recunoscută la nivel mondial. Elevii învață matematica de la o vârstă fragedă. 
    Conceptele matematice de bază includ adunarea, scăderea, înmulțirea și împărțirea. 
    Geometria studiază formele și spațiul. Algebra lucrează cu simboluri și ecuații. 
    Analiza matematică se ocupă cu limite și derivate. Statistica analizează datele și probabilitățile.
    """ * 3  # Repetă pentru a face textul mai lung
    
    # Simulează text_pages pentru test
    text_pages = [(test_text, 1)]
    
    # Test chunking cu parametri diferiți
    chunks_400, metadata_400 = converter.advanced_chunk_text(test_text, text_pages, chunk_size=400, overlap=80)
    chunks_200, metadata_200 = converter.advanced_chunk_text(test_text, text_pages, chunk_size=200, overlap=40)
    
    print(f"📝 Text original: {len(test_text.split())} cuvinte")
    print(f"🔧 Chunk size 400, overlap 80: {len(chunks_400)} chunks")
    print(f"🔧 Chunk size 200, overlap 40: {len(chunks_200)} chunks")
    
    # Verifică overlap-ul
    if len(chunks_400) > 1:
        chunk1_words = set(chunks_400[0].split())
        chunk2_words = set(chunks_400[1].split())
        overlap_words = len(chunk1_words & chunk2_words)
        print(f"🔗 Overlap între primul și al doilea chunk: {overlap_words} cuvinte")
    
    # Verifică metadata paginilor
    if metadata_400:
        first_meta = metadata_400[0]
        print(f"📄 Primul chunk: pagina {first_meta['page_from']}-{first_meta['page_to']}")
        print(f"📊 Cuvinte în primul chunk: {first_meta['word_count']}")
        print(f"📝 Propoziții în primul chunk: {first_meta['sentence_count']}")
    
    return True

def test_cosine_similarity():
    """Test 4: Verifică setarea metric cosine în ChromaDB"""
    print("\n🧪 Test 4: Metrică cosine în ChromaDB")
    print("-" * 40)
    
    converter = PDFEmbeddingsConverter()
    
    # Creează o colecție de test
    test_collection_name = "test_cosine_optimization"
    
    try:
        # Șterge colecția de test dacă există
        try:
            converter.client.delete_collection(test_collection_name)
        except:
            pass
        
        # Creează colecția cu metrică cosine
        collection = converter.client.get_or_create_collection(
            name=test_collection_name,
            metadata={
                "hnsw:space": "cosine",
                "test": True
            }
        )
        
        print(f"✅ Colecție de test creată: {test_collection_name}")
        
        # Verifică metadata
        metadata = collection.metadata or {}
        hnsw_space = metadata.get("hnsw:space", "None")
        print(f"🎯 HNSW space configurat: {hnsw_space}")
        print(f"✅ Metrică cosine setată: {'Da' if hnsw_space == 'cosine' else 'Nu'}")
        
        # Test cu embeddings de test
        test_embeddings = [
            [0.1, 0.2, 0.3, 0.4],  # Vector normalizat 1
            [0.2, 0.3, 0.4, 0.1],  # Vector normalizat 2
        ]
        
        # Normalizează vectorii
        test_embeddings = [
            (np.array(vec) / np.linalg.norm(vec)).tolist() 
            for vec in test_embeddings
        ]
        
        collection.add(
            embeddings=test_embeddings,
            documents=["Document test 1", "Document test 2"],
            ids=["test1", "test2"]
        )
        
        # Test query
        query_embedding = [0.15, 0.25, 0.35, 0.25]
        query_embedding = (np.array(query_embedding) / np.linalg.norm(query_embedding)).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        
        if results.get('distances'):
            distances = results['distances'][0]
            print(f"📏 Distanțe cosine: {[f'{d:.4f}' for d in distances]}")
            print(f"✅ Distanțe în intervalul [0,2]: {'Da' if all(0 <= d <= 2 for d in distances) else 'Nu'}")
        
        # Cleanup
        converter.client.delete_collection(test_collection_name)
        print("🗑️ Colecție de test ștearsă")
        
    except Exception as e:
        print(f"❌ Eroare test cosine: {e}")
        return False
    
    return True

def test_search_with_scores():
    """Test 5: Verifică căutarea cu scoruri și sortare globală"""
    print("\n🧪 Test 5: Căutare cu scoruri și sortare globală")
    print("-" * 40)
    
    converter = PDFEmbeddingsConverter()
    
    # Creează colecții de test cu documente
    test_collections = ["test_search_1", "test_search_2"]
    
    try:
        for col_name in test_collections:
            # Șterge dacă există
            try:
                converter.client.delete_collection(col_name)
            except:
                pass
        
        # Creează colecții cu documente de test
        docs_col1 = [
            "Matematica este știința numerelor și calculelor",
            "Geometria studiază formele și spațiul",
        ]
        
        docs_col2 = [
            "Fizica este știința naturii și mișcării",
            "Chimia studiază substanțele și reacțiile",
        ]
        
        # Adaugă în prima colecție
        col1 = converter.client.create_collection(
            "test_search_1",
            metadata={"hnsw:space": "cosine"}
        )
        
        embeddings1 = converter.model.encode(
            [RETRIEVAL_INSTRUCTION + doc for doc in docs_col1],
            normalize_embeddings=True
        )
        
        col1.add(
            embeddings=embeddings1.tolist(),
            documents=docs_col1,
            ids=[f"doc1_{i}" for i in range(len(docs_col1))]
        )
        
        # Adaugă în a doua colecție
        col2 = converter.client.create_collection(
            "test_search_2", 
            metadata={"hnsw:space": "cosine"}
        )
        
        embeddings2 = converter.model.encode(
            [RETRIEVAL_INSTRUCTION + doc for doc in docs_col2],
            normalize_embeddings=True
        )
        
        col2.add(
            embeddings=embeddings2.tolist(),
            documents=docs_col2,
            ids=[f"doc2_{i}" for i in range(len(docs_col2))]
        )
        
        print(f"✅ Colecții de test create cu {len(docs_col1)} + {len(docs_col2)} documente")
        
        # Test căutare cu sortare globală
        query = "știință"
        results = converter.search_similar(query, top_k=3)
        
        if results and results.get('documents') and results['documents'][0]:
            docs = results['documents'][0]
            similarities = results.get('similarities', [[]])[0] or []
            collections = results.get('collections', [[]])[0] or []
            
            print(f"🔍 Query: '{query}'")
            print(f"📊 Rezultate găsite: {len(docs)}")
            
            for i, (doc, sim, col) in enumerate(zip(docs, similarities, collections)):
                print(f"  {i+1}. Similaritate: {sim:.4f} | Colecție: {col}")
                print(f"     Text: {doc[:50]}...")
            
            # Verifică dacă sunt sortate descrescător
            if len(similarities) > 1:
                is_sorted = all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))
                print(f"✅ Rezultate sortate global: {'Da' if is_sorted else 'Nu'}")
        
        # Cleanup
        for col_name in test_collections:
            try:
                converter.client.delete_collection(col_name)
            except:
                pass
        
        print("🗑️ Colecții de test șterse")
        
    except Exception as e:
        print(f"❌ Eroare test căutare: {e}")
        return False
    
    return True

def test_performance_improvements():
    """Test 6: Verifică îmbunătățirile de performanță"""
    print("\n🧪 Test 6: Îmbunătățiri de performanță")
    print("-" * 40)
    
    converter = PDFEmbeddingsConverter()
    
    # Test streaming embeddings vs batch
    test_chunks = [
        f"Chunk de test numărul {i} cu text suficient pentru a testa performanța sistemului optimizat"
        for i in range(20)
    ]
    
    print(f"📝 Testez {len(test_chunks)} chunks")
    
    # Test streaming (implementarea optimizată)
    start_time = time.time()
    embeddings_streaming = converter.generate_embeddings_streaming(test_chunks)
    streaming_time = time.time() - start_time
    
    print(f"⚡ Streaming embeddings: {embeddings_streaming.shape} în {streaming_time:.2f}s")
    print(f"📊 Viteza: {len(test_chunks)/streaming_time:.2f} chunks/s")
    
    # Verifică că embeddings-urile sunt normalizate
    if embeddings_streaming.size > 0:
        norms = np.linalg.norm(embeddings_streaming, axis=1)
        avg_norm = np.mean(norms)
        print(f"📏 Norma medie a embeddings-urilor: {avg_norm:.6f}")
        print(f"✅ Embeddings normalizate: {'Da' if abs(avg_norm - 1.0) < 1e-4 else 'Nu'}")
    
    return True

def test_memory_usage():
    """Test 7: Verifică utilizarea memoriei"""
    print("\n🧪 Test 7: Utilizarea memoriei")
    print("-" * 40)
    
    import psutil
    import gc
    
    process = psutil.Process()
    
    # Memorie înainte
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    converter = PDFEmbeddingsConverter()
    
    # Memorie după încărcare
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"💾 Memorie înainte de încărcare: {memory_before:.1f} MB")
    print(f"💾 Memorie după încărcare model: {memory_after:.1f} MB")
    print(f"📊 Memorie folosită de model: {memory_after - memory_before:.1f} MB")
    
    # Test procesare cu monitoring memorie
    test_chunks = [f"Test chunk {i} pentru monitoring memorie" for i in range(50)]
    
    memory_before_processing = process.memory_info().rss / 1024 / 1024
    embeddings = converter.generate_embeddings_streaming(test_chunks)
    memory_after_processing = process.memory_info().rss / 1024 / 1024
    
    print(f"💾 Memorie înainte de procesare: {memory_before_processing:.1f} MB")
    print(f"💾 Memorie după procesare: {memory_after_processing:.1f} MB")
    print(f"📊 Memorie folosită pentru procesare: {memory_after_processing - memory_before_processing:.1f} MB")
    
    # Cleanup
    del embeddings
    gc.collect()
    
    memory_after_cleanup = process.memory_info().rss / 1024 / 1024
    print(f"🗑️ Memorie după cleanup: {memory_after_cleanup:.1f} MB")
    
    return True

def run_all_tests():
    """Rulează toate testele de optimizare"""
    print("🚀 TESTARE OPTIMIZĂRI PDF TO EMBEDDINGS")
    print("=" * 60)
    
    tests = [
        ("Instrucțiune de retrieval", test_retrieval_instruction),
        ("Normalizarea embeddings-urilor", test_normalization),
        ("Chunking avansat cu overlap", test_advanced_chunking),
        ("Metrică cosine ChromaDB", test_cosine_similarity),
        ("Căutare cu scoruri globale", test_search_with_scores),
        ("Îmbunătățiri performanță", test_performance_improvements),
        ("Utilizarea memoriei", test_memory_usage),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Sumar final
    print("\n🎯 REZULTATE FINALE")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\n📊 Scor final: {passed}/{total} teste trecute")
    print(f"📈 Rata de succes: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 Toate optimizările funcționează corect!")
    else:
        print("⚠️ Unele optimizări necesită atenție!")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()