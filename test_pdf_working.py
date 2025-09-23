import shutil
import os
import time
from pdf_converter_working import PDFEmbeddingsConverter

def print_search_results_with_scores(results, query):
    """Afișează rezultate căutare cu scoruri și metadata îmbunătățite"""
    if not results or not results.get('documents') or not results['documents'][0]:
        print("Nu am gasit rezultate")
        return
    
    docs = results['documents'][0]
    metadatas = results.get('metadatas', [[]])[0] or [{} for _ in docs]
    similarities = results.get('similarities', [[]])[0] or [0.0 for _ in docs]
    collections = results.get('collections', [[]])[0] or ["necunoscut" for _ in docs]
    
    print(f"\n🔍 Gasit {len(docs)} rezultate pentru: '{query}'")
    print("=" * 60)
    
    for i, (doc, meta, sim, collection) in enumerate(zip(docs, metadatas, similarities, collections)):
        print(f"\n--- Rezultatul {i+1} (Similaritate: {sim:.3f}) ---")
        print(f"📚 Colectie: {collection}")
        
        # Afișează informații despre pagini dacă sunt disponibile
        if 'page_from' in meta and 'page_to' in meta:
            if meta['page_from'] == meta['page_to']:
                print(f"📄 Pagina: {meta['page_from']}")
            else:
                print(f"📄 Pagini: {meta['page_from']}-{meta['page_to']}")
        
        # Afișează informații despre chunk dacă sunt disponibile
        if 'word_count' in meta:
            print(f"📝 Cuvinte: {meta['word_count']}")
        
        if 'source_file' in meta:
            source = os.path.basename(meta['source_file'])
            print(f"📁 Fisier: {source}")
        
        # Afișează textul (truncat)
        display_text = doc[:300] + "..." if len(doc) > 300 else doc
        print(f"📖 Text:\n{display_text}")
        print("-" * 40)

def main():
    print("🚀 Test PDF to Embeddings - Versiunea Optimizata")
    print("=" * 55)
    
    converter = PDFEmbeddingsConverter()
    
    while True:
        print("\n📋 Optiuni optimizate:")
        print("1. Proceseaza un PDF (optimizat)")
        print("2. Cauta in embeddings (cu scoruri)")
        print("3. Listeaza colectii (detaliat)")
        print("4. Test pe Biblia Romania")
        print("5. Proceseaza TOATE materialele didactice")
        print("6. Monitorizare folder new_pdfs")
        print("7. Cautare avansata cu filtru colectie")
        print("8. Statistici embeddings")
        print("9. Test director specific")
        print("10. Test complet convertor îmbunătățit")
        print("11. Iesire")
        
        choice = input("\n🎯 Alege optiunea (1-11): ").strip()
        
        if choice == "1":
            file_path = input("📁 Calea catre PDF: ").strip()
            if file_path and os.path.exists(file_path):
                print(f"\n🔄 Procesez optimizat: {os.path.basename(file_path)}")
                success = converter.process_pdf(file_path)
                if success:
                    print("✅ Procesare optimizata cu succes!")
                else:
                    print("❌ Procesare esuata!")
            else:
                print("❌ Fisier inexistent!")
        
        elif choice == "2":
            query = input("🔍 Ce cauti: ").strip()
            if query:
                print(f"\n🔄 Cautare optimizata pentru: '{query}'")
                results = converter.search_similar(query, top_k=5)
                print_search_results_with_scores(results, query)
        
        elif choice == "3":
            print("\n📚 Colectii disponibile:")
            converter.list_collections()
        
        elif choice == "4":
            bible_path = r".\materiale_didactice\director_pedagogie\Biblia_Romania.pdf"
            if os.path.exists(bible_path):
                print(f"📖 Procesez Biblia Romania optimizat...")
                success = converter.process_pdf(bible_path)
                if success:
                    print("✅ Biblia procesata cu optimizari complete!")
                    
                    # Test căutare optimizată
                    print("\n🔍 Test cautare optimizata...")
                    results = converter.search_similar("Dumnezeu", top_k=3)
                    print_search_results_with_scores(results, "Dumnezeu")
                else:
                    print("❌ Procesare esuata")
            else:
                print("❌ Biblia_Romania.pdf nu a fost gasita")    
        
        elif choice == "5":
            materiale_path = "./materiale_didactice"
            if os.path.exists(materiale_path):
                print("📚 Procesez TOATE materialele didactice cu optimizari complete...")
                
                # Găsește toate PDF-urile recursiv
                pdf_files = []
                for root, dirs, files in os.walk(materiale_path):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            pdf_files.append(os.path.join(root, file))
                
                print(f"📄 Gasit {len(pdf_files)} PDF-uri in structura educationala")
                
                # Afișează structura optimizată care va fi procesată
                collections = {}
                for pdf_file in pdf_files:
                    collection_name = converter.get_collection_name(pdf_file)
                    if collection_name not in collections:
                        collections[collection_name] = []
                    collections[collection_name].append(os.path.basename(pdf_file))
                
                print("\n🏗️ Structura colectiilor optimizate care vor fi create:")
                for collection, files in collections.items():
                    print(f"  📁 {collection}: {len(files)} fisiere")
                
                # Opțiuni de procesare
                print("\n🎯 Optiuni procesare:")
                print("1. Procesare secvențială (original)")
                print("2. Procesare paralelă (îmbunătățită)")
                print("3. Test cu primele 10 PDF-uri")
                
                processing_choice = input("Alege modul de procesare (1-3): ").strip()
                
                if processing_choice == "2":
                    # Procesare paralelă îmbunătățită
                    confirm = input("\n❓ Continui cu procesarea paralelă optimizată? (y/n): ").strip().lower()
                    if confirm == 'y':
                        def progress_callback(progress, message):
                            print(f"📊 Progress: {progress*100:.1f}% - {message}")
                        
                        print("\n🚀 Încep procesarea paralelă...")
                        results = converter.process_pdfs_parallel(pdf_files, progress_callback=progress_callback)
                        
                        # Analizează rezultatele
                        successful = sum(1 for r in results if r.success)
                        failed = len(results) - successful
                        success_rate = (successful / len(results)) * 100 if results else 0
                        
                        print(f"\n🎉 === REZULTAT FINAL PARALEL OPTIMIZAT ===")
                        print(f"✅ Procesate cu succes: {successful}")
                        print(f"❌ Eșuate: {failed}")
                        print(f"📊 Total: {len(pdf_files)}")
                        print(f"📈 Rata de succes: {success_rate:.1f}%")
                        
                        # Afișează statistici detaliate
                        stats = converter.get_processing_stats()
                        print(f"\n📈 Statistici detaliate:")
                        print(f"💾 Memorie vârf: {stats.memory_peak:.1f}MB")
                        print(f"💿 Disk usage: {stats.disk_usage_mb:.1f}MB")
                        print(f"📄 Total chunks: {stats.total_chunks}")
                        print(f"⏱️ Timp total procesare: {stats.total_processing_time:.1f}s")
                    else:
                        print("🚫 Procesare anulată")
                        
                elif processing_choice == "3":
                    # Test cu primele 10 PDF-uri
                    test_files = pdf_files[:10]
                    print(f"\n🧪 Test cu primele {len(test_files)} PDF-uri...")
                    
                    def progress_callback(progress, message):
                        print(f"📊 Progress: {progress*100:.1f}% - {message}")
                    
                    results = converter.process_pdfs_parallel(test_files, progress_callback=progress_callback)
                    
                    successful = sum(1 for r in results if r.success)
                    failed = len(results) - successful
                    success_rate = (successful / len(results)) * 100 if results else 0
                    
                    print(f"\n📊 Rezultat test:")
                    print(f"✅ Succese: {successful}")
                    print(f"❌ Eșecuri: {failed}")
                    print(f"📈 Rata de succes: {success_rate:.1f}%")
                    
                    if success_rate >= 80:
                        print("✅ Test reușit! Poți continua cu toate PDF-urile.")
                    else:
                        print("⚠️ Test cu probleme. Verifică configurația.")
                        
                else:
                    # Procesare secvențială originală
                    confirm = input("\n❓ Continui cu procesarea secvențială optimizată? (y/n): ").strip().lower()
                    if confirm == 'y':
                        # Procesează toate cu progres optimizat
                        successful = 0
                        failed = 0
                        start_time = time.time()
                        
                        for i, pdf_file in enumerate(pdf_files):
                            print(f"\n🔄 --- PDF {i+1}/{len(pdf_files)} ---")
                            print(f"📄 Fisier: {os.path.basename(pdf_file)}")
                            print(f"📁 Colectie: {converter.get_collection_name(pdf_file)}")
                            
                            def progress_callback(progress, message):
                                print(f"  📊 {progress*100:.1f}% - {message}")
                            
                            if converter.process_pdf(pdf_file, progress_callback=progress_callback):
                                successful += 1
                                print("✅ Status: SUCCES OPTIMIZAT")
                            else:
                                failed += 1
                                print("❌ Status: ESUAT")
                            
                            # Cleanup automat la fiecare 10 PDF-uri pentru a preveni disk full
                            if (i + 1) % 10 == 0:
                                print("🧹 Cleanup automat la fiecare 10 PDF-uri...")
                                converter.cleanup_resources()
                                
                                # Afișează statistici disk
                                disk_stats = converter.get_disk_usage_stats()
                                if 'disk_usage_percent' in disk_stats:
                                    print(f"💾 Disk usage: {disk_stats['disk_usage_percent']}% | "
                                          f"Free: {disk_stats['free_disk_gb']}GB | "
                                          f"Embeddings DB: {disk_stats['embeddings_db_mb']}MB")
                                
                                # Pauza pentru stabilizare
                                time.sleep(2)
                        
                        duration = time.time() - start_time
                        
                        print(f"\n🎉 === REZULTAT FINAL SECVENȚIAL OPTIMIZAT ===")
                        print(f"✅ Procesate cu succes: {successful}")
                        print(f"❌ Esuat: {failed}")
                        print(f"📊 Total: {len(pdf_files)}")
                        print(f"📈 Rata de succes: {(successful/len(pdf_files)*100):.1f}%")
                        print(f"⏱️ Timp total: {duration/60:.1f} minute")
                        print(f"⚡ Viteza medie: {duration/len(pdf_files):.1f}s/PDF")
                    else:
                        print("🚫 Procesare anulată")
            else:
                print("❌ Folderul materiale_didactice nu exista")
                print("💡 Creaza folderul si pune PDF-urile in structura:")
                print("  📁 materiale_didactice/")
                print("    📁 Scoala_Normala/")
                print("      📁 clasa_1/matematica/")
                print("      📁 clasa_2/romana/")
                print("    📁 Scoala_de_Muzica_George_Enescu/")
                print("      📁 clasa_1/muzica/")

        elif choice == "6":
            # Monitorizare folder new_pdfs optimizată
            new_pdfs_path = "./new_pdfs"
            processed_path = "./processed_pdfs"
            
            os.makedirs(new_pdfs_path, exist_ok=True)
            os.makedirs(processed_path, exist_ok=True)
            
            pdf_files = [f for f in os.listdir(new_pdfs_path) if f.lower().endswith('.pdf')]
            
            if pdf_files:
                print(f"📄 Gasit {len(pdf_files)} PDF-uri noi in new_pdfs/")
                processed_count = 0
                
                for pdf_file in pdf_files:
                    pdf_path = os.path.join(new_pdfs_path, pdf_file)
                    print(f"\n🔄 Procesez optimizat: {pdf_file}")
                    
                    if converter.process_pdf(pdf_path):
                        # Mută fișierul procesat
                        shutil.move(pdf_path, os.path.join(processed_path, pdf_file))
                        processed_count += 1
                        print(f"✅ Procesat optimizat și mutat in processed_pdfs/")
                    else:
                        print(f"❌ Eroare la procesare - fisierul ramane in new_pdfs/")
                
                print(f"\n📊 Sumar: {processed_count}/{len(pdf_files)} procesate cu succes")
            else:
                print("📝 Nu am gasit PDF-uri noi in new_pdfs/")
                print("💡 Pune fisiere PDF in folderul new_pdfs/ pentru procesare automata optimizata")

        elif choice == "7":
            # Căutare avansată cu filtru colecție
            print("\n📚 Colectii disponibile:")
            collections = converter.client.list_collections()
            for i, col in enumerate(collections):
                print(f"  {i+1}. {col.name} ({col.count()} documente)")
            
            collection_choice = input("\n🎯 Alege colectia (nume sau numar, enter pentru toate): ").strip()
            
            if collection_choice:
                # Încearcă să interpreteze ca număr
                try:
                    col_index = int(collection_choice) - 1
                    if 0 <= col_index < len(collections):
                        selected_collection = collections[col_index].name
                    else:
                        selected_collection = collection_choice
                except ValueError:
                    selected_collection = collection_choice
                
                query = input(f"🔍 Cauta in '{selected_collection}': ").strip()
                if query:
                    results = converter.search_similar(query, top_k=5, collection_name=selected_collection)
                    print_search_results_with_scores(results, query)
            else:
                query = input("🔍 Cauta in toate colectiile: ").strip()
                if query:
                    results = converter.enhanced_search_educational(query, top_k=5)
                    print_search_results_with_scores(results, query)

        elif choice == "8":
            # Statistici embeddings
            print("\n📊 STATISTICI EMBEDDINGS OPTIMIZATE")
            print("=" * 50)
            
            collections = converter.client.list_collections()
            total_docs = 0
            optimized_collections = 0
            
            stats_by_model = {}
            stats_by_type = {}
            
            for collection in collections:
                try:
                    count = collection.count()
                    metadata = collection.metadata or {}
                    
                    total_docs += count
                    
                    # Verifică dacă colecția este optimizată
                    if metadata.get("retrieval_instruction"):
                        optimized_collections += 1
                    
                    # Statistici pe model
                    model = metadata.get("model", "necunoscut")
                    if model not in stats_by_model:
                        stats_by_model[model] = {"collections": 0, "documents": 0}
                    stats_by_model[model]["collections"] += 1
                    stats_by_model[model]["documents"] += count
                    
                    # Statistici pe tip (din numele colecției)
                    col_parts = collection.name.split("_")
                    if len(col_parts) >= 2:
                        col_type = col_parts[0] + "_" + col_parts[1]
                    else:
                        col_type = "general"
                    
                    if col_type not in stats_by_type:
                        stats_by_type[col_type] = {"collections": 0, "documents": 0}
                    stats_by_type[col_type]["collections"] += 1
                    stats_by_type[col_type]["documents"] += count
                    
                except Exception as e:
                    print(f"❌ Eroare colectia {collection.name}: {e}")
            
            print(f"📚 Total colectii: {len(collections)}")
            print(f"📄 Total documente: {total_docs}")
            print(f"⚡ Colectii optimizate: {optimized_collections}/{len(collections)} ({optimized_collections/len(collections)*100:.1f}%)")
            
            if stats_by_model:
                print(f"\n🤖 Statistici pe model:")
                for model, stats in stats_by_model.items():
                    print(f"  {model}: {stats['collections']} colectii, {stats['documents']} documente")
            
            if stats_by_type:
                print(f"\n🏫 Statistici pe tip institutie:")
                for col_type, stats in stats_by_type.items():
                    print(f"  {col_type}: {stats['collections']} colectii, {stats['documents']} documente")
            
            # Statistici fișiere procesate
            processed_count = len(converter.processed_files)
            print(f"\n📁 Fisiere procesate: {processed_count}")
            
            if processed_count > 0:
                avg_docs_per_file = total_docs / processed_count
                print(f"📊 Medie documente/fisier: {avg_docs_per_file:.1f}")

        elif choice == "9":
            # Test director specific
            test_dir = input("📁 Calea catre director pentru test: ").strip()
            if os.path.exists(test_dir) and os.path.isdir(test_dir):
                print(f"🔄 Testez toate PDF-urile din: {test_dir}")
                
                # Găsește toate PDF-urile din director (nu recursiv)
                pdf_files = []
                for file in os.listdir(test_dir):
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(test_dir, file))
                
                if pdf_files:
                    print(f"📄 Găsit {len(pdf_files)} PDF-uri în director")
                    
                    successful = 0
                    failed = 0
                    
                    for i, pdf_file in enumerate(pdf_files):
                        print(f"\n--- PDF {i+1}/{len(pdf_files)} ---")
                        print(f"📄 Procesez: {os.path.basename(pdf_file)}")
                        
                        if converter.process_pdf(pdf_file):
                            successful += 1
                            print("✅ SUCCES")
                        else:
                            failed += 1
                            print("❌ EȘUAT")
                        
                        # Cleanup la fiecare 5 PDF-uri
                        if (i + 1) % 5 == 0:
                            converter.cleanup_resources()
                    
                    print(f"\n📊 REZULTAT FINAL:")
                    print(f"✅ Succese: {successful}")
                    print(f"❌ Eșuate: {failed}")
                    print(f"📈 Rata succes: {(successful/len(pdf_files)*100):.1f}%")
                else:
                    print("❌ Nu am găsit PDF-uri în director")
            else:
                print("❌ Directorul nu există")

        elif choice == "10":
            # Test complet convertor îmbunătățit
            print("\n🧪 Test complet convertor îmbunătățit")
            print("=" * 50)
            
            try:
                # Import și rulează testul complet
                import subprocess
                import sys
                
                print("🚀 Rulez testul complet...")
                result = subprocess.run([sys.executable, "test_improved_converter.py"], 
                                      capture_output=True, text=True)
                
                print("📊 Rezultat test:")
                print(result.stdout)
                
                if result.stderr:
                    print("⚠️ Erori:")
                    print(result.stderr)
                
                if result.returncode == 0:
                    print("✅ Test complet reușit!")
                else:
                    print("❌ Test complet cu probleme")
                    
            except Exception as e:
                print(f"❌ Eroare la rularea testului: {e}")
                print("💡 Rulează manual: python test_improved_converter.py")

        elif choice == "11":
            print("👋 La revedere! Embeddings-urile optimizate sunt salvate.")
            break

        else:
            print("❌ Optiune invalida! Alege 1-11.")

if __name__ == "__main__":
    main()