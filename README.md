# PDF-to-PKL

Instrument pentru conversia materialelor educaționale PDF în embeddings Chroma, cu accent pe rularea în medii restricționate precum Railway.

## Variabile de mediu cheie

| Variabilă | Implicit | Descriere |
|-----------|----------|-----------|
| `EMBEDDINGS_DB_PATH` | `./embeddings_db` | Directorul persistent pentru baza Chroma. Este creat automat dacă nu există. |
| `CHUNK_SIZE` | configurarea din `config.py` (fallback 800 când valoarea introdusă nu este validă) | Dimensiunea chunk-urilor generate din PDF. |
| `CHUNK_OVERLAP` | configurarea din `config.py` (fallback 120 când valoarea introdusă nu este validă) | Suprapunerea dintre chunk-uri consecutive. |
| `CHUNK_SIZE_BIBLE` | `2 * CHUNK_SIZE` (implicit) | Dimensiune alternativă a chunk-urilor pentru fișiere ce conțin `Biblia`/`Bible` în nume. |
| `OVERLAP_BIBLE` | `CHUNK_OVERLAP // 2` (implicit) | Suprapunere dedicată corpusului biblic atunci când este activată optimizarea. |
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

## Schema metadatelor

Toate documentele noi și colecțiile migrate respectă schema standard de mai jos:

| Câmp | Exemplu | Observații |
|------|---------|------------|
| `grade` | `clasa_2` | întotdeauna în forma `clasa_X`; `clasa_0` pentru materiale generale. |
| `subject` | `dezvoltare_personala` | slugificat din directorul disciplinei sau din numele fișierului. |
| `institution` | `Scoala_de_Muzica_George_Enescu` | extras din arborele de directoare ori din colecție. |
| `tags` | `['dezvoltare_personala', 'clasa_2', 'scoala_de_muzica_george_enescu']` | include cel puțin subiectul și clasa, valorile sunt slugificate. |
| `source_file` | `A976.pdf` | doar numele fișierului PDF. |
| `page` | `12` | pagina de start a chunk-ului (pe lângă `page_from`/`page_to`). |
| `chunk_index` | `42` | index incremental pentru fiecare chunk din fișier. |

Textul este normalizat Unicode (NFC), cu diacritice românești corectate (`ș`, `ț`, `ă`, `î`, `â`), spații invizibile eliminate și spațiere uniformizată înainte de vectorizare. Logurile raportează sumar câte caractere au fost ajustate pentru fiecare PDF procesat.

### Filtrare în `search_similar`

Funcția `search_similar` acceptă un parametru opțional `where`, transmis mai departe către Chroma. Pentru cazurile educaționale, helperul `where_from(grade, subject)` construiește filtre consistente:

```python
from pdf_converter_working import where_from

results = converter.search_similar(
    "responsabilitate și managementul timpului",
    top_k=5,
    where=where_from("clasa_2", "dezvoltare_personala"),
)
```

Poți furniza filtre personalizate, de exemplu:

```python
search_similar(
  "responsabilitate și managementul timpului",
  top_k=5,
  where={"grade": "clasa_2", "tags": {"$contains": "dezvoltare_personala"}}
)
```

Rezultatele sunt deduplicate pe ID, iar `top_k` este „clamp-uit” la dimensiunea fiecărei colecții (eveniment logat automat).

### Corpusul biblic – reducerea zgomotului

Pentru fișierele care conțin `Biblia`/`Bible` în nume există două abordări complementare:

1. **Opțiunea A (implicită)** – folosește `CHUNK_SIZE_BIBLE` și `OVERLAP_BIBLE` pentru a genera chunk-uri mai mari și mai puțin redundante.
2. **Opțiunea B** – separă corpusul într-o colecție proprie (prin structura directoarelor) și folosește filtre `where` pe `grade`/`tags` pentru a-l exclude din interogările educaționale.

Logurile indică momentul în care este aplicată configurația specială pentru corpusul biblic.

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
