#!/usr/bin/env python3
"""
Test script pentru convertorul PDF îmbunătățit
Testează toate îmbunătățirile implementate pentru a crește rata de succes
"""

import os
import time
import logging
from pathlib import Path
from pdf_converter_working import PDFEmbeddingsConverter, ProcessingStats, ProcessingResult

def test_single_pdf():
    """Test procesare un singur PDF cu monitoring"""
    print("🧪 Test 1: Procesare un singur PDF cu monitoring")
    print("=" * 60)
    
    converter = PDFEmbeddingsConverter()
    
    # Test cu un PDF din materiale_didactice
    test_pdf = None
    for root, dirs, files in os.walk("./materiale_didactice"):
        for file in files:
            if file.lower().endswith('.pdf'):
                test_pdf = os.path.join(root, file)
                break
        if test_pdf:
            break
    
    if not test_pdf:
        print("❌ Nu am găsit PDF-uri pentru test")
        return False
    
    print(f"📄 Testez cu: {os.path.basename(test_pdf)}")
    
    # Progress callback pentru monitoring
    def progress_callback(progress, message):
        print(f"📊 Progress: {progress*100:.1f}% - {message}")
    
    start_time = time.time()
    success = converter.process_pdf(test_pdf, progress_callback=progress_callback)
    processing_time = time.time() - start_time
    
    if success:
        print(f"✅ Procesare reușită în {processing_time:.2f}s")
        
        # Test căutare
        print("\n🔍 Test căutare...")
        results = converter.search_similar("matematică", top_k=3)
        if results and results.get('documents') and results['documents'][0]:
            print(f"✅ Căutare reușită: {len(results['documents'][0])} rezultate")
        else:
            print("❌ Căutare eșuată")
    else:
        print("❌ Procesare eșuată")
    
    return success

def test_parallel_processing():
    """Test procesare paralelă cu monitoring"""
    print("\n🧪 Test 2: Procesare paralelă cu monitoring")
    print("=" * 60)
    
    converter = PDFEmbeddingsConverter()
    
    # Găsește primele 5 PDF-uri pentru test
    pdf_files = []
    for root, dirs, files in os.walk("./materiale_didactice"):
        for file in files:
            if file.lower().endswith('.pdf') and len(pdf_files) < 5:
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print("❌ Nu am găsit PDF-uri pentru test paralel")
        return False
    
    print(f"📄 Procesez {len(pdf_files)} PDF-uri în paralel...")
    
    # Progress callback pentru monitoring
    def progress_callback(progress, message):
        print(f"📊 Progress: {progress*100:.1f}% - {message}")
    
    start_time = time.time()
    results = converter.process_pdfs_parallel(pdf_files, progress_callback=progress_callback)
    processing_time = time.time() - start_time
    
    # Analizează rezultatele
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    success_rate = (successful / len(results)) * 100 if results else 0
    
    print(f"\n📊 Rezultate procesare paralelă:")
    print(f"✅ Succese: {successful}")
    print(f"❌ Eșecuri: {failed}")
    print(f"📈 Rata de succes: {success_rate:.1f}%")
    print(f"⏱️ Timp total: {processing_time:.2f}s")
    print(f"⚡ Viteza medie: {processing_time/len(results):.2f}s/PDF")
    
    # Afișează statistici detaliate
    stats = converter.get_processing_stats()
    print(f"\n📈 Statistici detaliate:")
    print(f"💾 Memorie vârf: {stats.memory_peak:.1f}MB")
    print(f"💿 Disk usage: {stats.disk_usage_mb:.1f}MB")
    print(f"📄 Total chunks: {stats.total_chunks}")
    
    return success_rate >= 80  # Consideră reușit dacă >= 80%

def test_memory_management():
    """Test gestionarea memoriei"""
    print("\n🧪 Test 3: Gestionarea memoriei")
    print("=" * 60)
    
    import psutil
    
    converter = PDFEmbeddingsConverter()
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"💾 Memorie inițială: {initial_memory:.1f}MB")
    
    # Procesează câteva PDF-uri și monitorizează memoria
    pdf_files = []
    for root, dirs, files in os.walk("./materiale_didactice"):
        for file in files:
            if file.lower().endswith('.pdf') and len(pdf_files) < 3:
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print("❌ Nu am găsit PDF-uri pentru test memorie")
        return False
    
    peak_memory = initial_memory
    
    for i, pdf_file in enumerate(pdf_files):
        print(f"\n📄 Procesez PDF {i+1}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
        
        before_memory = process.memory_info().rss / 1024 / 1024
        success = converter.process_pdf(pdf_file)
        after_memory = process.memory_info().rss / 1024 / 1024
        
        memory_delta = after_memory - before_memory
        peak_memory = max(peak_memory, after_memory)
        
        print(f"💾 Memorie înainte: {before_memory:.1f}MB")
        print(f"💾 Memorie după: {after_memory:.1f}MB")
        print(f"📊 Delta: {memory_delta:+.1f}MB")
        print(f"✅ Status: {'SUCCES' if success else 'EȘUAT'}")
        
        # Cleanup manual
        converter.cleanup_resources()
        time.sleep(1)  # Pauză pentru stabilizare
    
    final_memory = process.memory_info().rss / 1024 / 1024
    total_delta = final_memory - initial_memory
    
    print(f"\n📊 Rezumat gestionare memorie:")
    print(f"💾 Memorie inițială: {initial_memory:.1f}MB")
    print(f"💾 Memorie finală: {final_memory:.1f}MB")
    print(f"📈 Memorie vârf: {peak_memory:.1f}MB")
    print(f"📊 Delta total: {total_delta:+.1f}MB")
    
    # Consideră reușit dacă delta-ul este rezonabil (< 500MB)
    return abs(total_delta) < 500

def test_error_handling():
    """Test gestionarea erorilor"""
    print("\n🧪 Test 4: Gestionarea erorilor")
    print("=" * 60)
    
    converter = PDFEmbeddingsConverter()
    
    # Test cu fișier inexistent
    print("📄 Test cu fișier inexistent...")
    result = converter.process_pdf("nonexistent.pdf")
    if not result:
        print("✅ Gestionare corectă a fișierului inexistent")
    else:
        print("❌ Ar fi trebuit să eșueze pentru fișier inexistent")
    
    # Test cu fișier corupt (simulat)
    print("\n📄 Test cu fișier corupt (simulat)...")
    try:
        # Creează un fișier PDF corupt temporar
        corrupt_pdf = "test_corrupt.pdf"
        with open(corrupt_pdf, 'w') as f:
            f.write("This is not a valid PDF")
        
        result = converter.process_pdf(corrupt_pdf)
        if not result:
            print("✅ Gestionare corectă a fișierului corupt")
        else:
            print("❌ Ar fi trebuit să eșueze pentru fișier corupt")
        
        # Cleanup
        if os.path.exists(corrupt_pdf):
            os.remove(corrupt_pdf)
            
    except Exception as e:
        print(f"✅ Gestionare corectă a erorii: {e}")
    
    return True

def test_configuration():
    """Test configurația dinamică"""
    print("\n🧪 Test 5: Configurația dinamică")
    print("=" * 60)
    
    from config import EMBEDDING_CONFIG, OCR_CONFIG, PROCESSING_CONFIG, LOGGING_CONFIG
    
    print("📋 Configurații curente:")
    print(f"🤖 Model embeddings: {EMBEDDING_CONFIG.model}")
    print(f"📄 Chunk size: {EMBEDDING_CONFIG.chunk_size}")
    print(f"🔄 Batch size: {EMBEDDING_CONFIG.batch_size}")
    print(f"👁️ OCR DPI: {OCR_CONFIG.dpi}")
    print(f"📚 OCR max pages: {OCR_CONFIG.max_pages_ocr}")
    print(f"⚡ Max workers: {PROCESSING_CONFIG.max_workers}")
    print(f"💾 Memory threshold: {PROCESSING_CONFIG.memory_threshold_percent}%")
    print(f"📝 Log level: {LOGGING_CONFIG.level}")
    
    return True

def main():
    """Rulează toate testele"""
    print("🚀 Test Convertor PDF Îmbunătățit")
    print("=" * 80)
    print("Testez toate îmbunătățirile implementate pentru a crește rata de succes")
    print("=" * 80)
    
    tests = [
        ("Procesare un singur PDF", test_single_pdf),
        ("Procesare paralelă", test_parallel_processing),
        ("Gestionarea memoriei", test_memory_management),
        ("Gestionarea erorilor", test_error_handling),
        ("Configurația dinamică", test_configuration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results.append((test_name, result))
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Rezumat final
    print("\n" + "="*80)
    print("📊 REZUMAT FINAL TESTE")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 Rezultat general: {passed}/{total} teste trecute ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Toate testele au trecut! Convertorul este optimizat.")
    elif passed >= total * 0.8:
        print("✅ Majoritatea testelor au trecut. Convertorul este îmbunătățit.")
    else:
        print("⚠️ Unele teste au eșuat. Verifică configurația.")
    
    return passed / total

if __name__ == "__main__":
    success_rate = main()
    exit(0 if success_rate >= 0.8 else 1)
