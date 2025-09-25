# Migrarea metadatelor existente

Utilitarul `tools/migrate_metadata.py` aliniază colecțiile Chroma existente la noua schemă de metadate (`grade`, `subject`, `institution`, `tags`, `source_file`, `page`, `chunk_index`). Scriptul nu re-generează embeddings; actualizează doar metadatele.

## Pași recomandați

1. Oprește orice proces care scrie în baza de date de embeddings.
2. Creează un backup (opțional dar recomandat):
   ```bash
   python pdf_converter_working.py --snapshot-db
   ```
3. Rulează migrarea (dry-run pentru verificare, apoi aplicație reală):
   ```bash
   python tools/migrate_metadata.py --db-path ./embeddings_db --dry-run
   python tools/migrate_metadata.py --db-path ./embeddings_db
   ```
4. Verifică logurile pentru numărul de colecții/documente actualizate.

Scriptul raportează câte colecții au fost inspectate, câte documente au primit metadatele noi și câte au rămas neschimbate.
