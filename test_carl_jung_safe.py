import os
import time

import pytest

from pdf_converter_working import PDFEmbeddingsConverter

def test_with_memory_limits():
    test_dir = r'C:\Users\Opaop\Desktop\Conversie PDF to PKL\PDF_to_Embeddings\materiale_didactice\Scoala_de_Muzica_George_Enescu\clasa_2\Dezvoltare_personala\Prof_Carl_Jung'

    if not os.path.exists(test_dir):
        pytest.skip("Materialele Carl Jung nu sunt disponibile în mediul de test")

    converter = PDFEmbeddingsConverter()
    
    # Gaseste PDF-urile
    pdf_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith('.pdf')]
    pdf_files.sort(key=lambda x: os.path.getsize(x))  # Sorteaza dupa dimensiune
    
    print(f"Testez {len(pdf_files)} PDF-uri (de la cel mai mic)")
    
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files):
        file_size_mb = os.path.getsize(pdf_file) / (1024*1024)
        
        print(f"\n--- PDF {i+1}/{len(pdf_files)} ---")
        print(f"Fisier: {os.path.basename(pdf_file)} ({file_size_mb:.1f}MB)")
        
        # Skip fisiere prea mari pentru OCR
        if file_size_mb > 20:
            print("Skipat - fisier prea mare pentru OCR sigur")
            continue
        
        try:
            if converter.process_pdf(pdf_file):
                successful += 1
                print("SUCCES")
            else:
                failed += 1
                print("ESUAT")
        except Exception as e:
            failed += 1
            print(f"EROARE: {e}")
        
        # Cleanup agresiv dupa fiecare PDF
        converter.cleanup_resources()
        time.sleep(3)  # Pauza pentru stabilizare memoria
        
        # Verifica memoria (basic check)
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:
            print(f"ATENTIE: Memory la {memory_percent}% - opresc procesarea")
            break
    
    print(f"\nRezultat: {successful} succese, {failed} esec")

if __name__ == "__main__":
    test_with_memory_limits()
