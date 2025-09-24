# PDF-to-PKL

Instrument pentru conversia materialelor educaționale PDF în embeddings Chroma, cu accent pe rularea în medii restricționate precum Railway.

## Variabile de mediu cheie

| Variabilă | Implicit | Descriere |
|-----------|----------|-----------|
| `EMBEDDINGS_DB_PATH` | `./embeddings_db` | Directorul persistent pentru baza Chroma. Este creat automat dacă nu există. |
| `CHUNK_SIZE` | configurarea din `config.py` (fallback 800 când valoarea introdusă nu este validă) | Dimensiunea chunk-urilor generate din PDF. |
| `CHUNK_OVERLAP` | configurarea din `config.py` (fallback 120 când valoarea introdusă nu este validă) | Suprapunerea dintre chunk-uri consecutive. |
| `CHROMA_TELEMETRY` | `0` | Setează `0` pentru a opri telemetria Posthog/Chroma. Orice altă valoare reactivează raportarea anonimă. |
| `KEYWORD_SWEEP_OUTPUT` | `keyword_sweep.csv` | (Opțional) Path implicit pentru exportul utilitarului de sweep semantic. |

> **Notă:** Valorile invalide pentru `CHUNK_SIZE`/`CHUNK_OVERLAP` sunt înlocuite la runtime cu fallback-uri conservatoare (800/120) pentru a evita erori la deploy.

## CLI & întreținere index

Scriptul principal oferă un mic CLI pentru operațiuni de mentenanță:

```bash
python pdf_converter_working.py --prune-collections        # Șterge colecțiile goale și cele prefixate cu tmp_
python pdf_converter_working.py --prune-collections --dry-run
python pdf_converter_working.py --snapshot-db              # Creează un snapshot în ./snapshots/
python pdf_converter_working.py --snapshot-db --snapshot-dest /var/backups
```

### Sweep semantic rapid

Utilitarul `tools/keyword_sweep.py` e gândit pentru verificare rapidă a calității embeddings-urilor fără a modifica indexul:

```bash
python tools/keyword_sweep.py \
  --keywords "copil, programă, exercițiu, Piaget" \
  --output data/keyword_sweep.csv \
  --top-k 3
```

CSV-ul rezultat conține coloanele: `keyword`, `collection`, `hit_count`, `first_hit_id`, `first_hit_page`, `latency_ms`.

## Deploy pe Railway

1. **Construiește imaginea** folosind `Dockerfile` furnizat sau `railway up`.
2. **Configurare variabile de mediu** din dashboard:
   - `EMBEDDINGS_DB_PATH=/data/embeddings_db`
   - `CHROMA_TELEMETRY=0`
   - (opțional) `CHUNK_SIZE` și `CHUNK_OVERLAP` pentru ajustări runtime.
3. **Persistă datele** montând un volum la `EMBEDDINGS_DB_PATH` (ex: `/data`).
4. **Rulează probele** după deploy: `python pdf_converter_working.py --prune-collections --dry-run` și `python tools/keyword_sweep.py --top-k 2` pentru a verifica conectivitatea către baza de embeddings.
5. **Monitorizare**: logurile vor raporta configurarea chunk-urilor la startup și eventualele clamp-uri `top_k` atunci când dimensiunea colecției este depășită.

## Testare

Rularea testelor unitare se face cu:

```bash
pytest
```

Acoperirea include probele modelului, verificarea dezactivării telemetriei și a faptului că interogările nu introduc duplicate în rezultate.
