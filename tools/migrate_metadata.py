#!/usr/bin/env python3
"""Actualizează metadatele existente pentru a respecta schema unificată."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from pdf_converter_working import ensure_standard_metadata


def _load_processed_registry(db_path: Path) -> Dict[str, str]:
    registry_path = db_path / "processed_files.json"
    if not registry_path.exists():
        return {}

    try:
        with registry_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        logging.warning("Nu am putut citi processed_files.json din %s", registry_path)
        return {}

    return {str(Path(path)): value for path, value in data.items()}


def _create_client(db_path: Path) -> chromadb.PersistentClient:
    telemetry_env = os.getenv("CHROMA_TELEMETRY")
    anonymized_telemetry = telemetry_env not in {"0", "false", "False", ""} if telemetry_env is not None else False
    settings = Settings(anonymized_telemetry=anonymized_telemetry)
    return chromadb.PersistentClient(path=str(db_path), settings=settings)


def _invert_registry(registry: Dict[str, str]) -> Dict[str, str]:
    inverted: Dict[str, str] = {}
    for path, file_hash in registry.items():
        if file_hash and file_hash not in inverted:
            inverted[file_hash] = path
    return inverted


def _resolve_source_path(
    collection_metadata: Optional[Dict[str, Any]],
    item_metadata: Dict[str, Any],
    registry_by_hash: Dict[str, str],
) -> str:
    for key in ("source_path", "pdf_path", "path"):
        value = item_metadata.get(key)
        if isinstance(value, str) and value:
            return value

    source_file = item_metadata.get("source_file")
    if isinstance(source_file, str) and os.path.sep in source_file:
        return source_file

    description = (collection_metadata or {}).get("description", "")
    if isinstance(description, str) and " for " in description:
        candidate = description.split(" for ", 1)[-1].strip()
        if candidate:
            return candidate

    file_hash = item_metadata.get("file_hash") or (collection_metadata or {}).get("file_hash")
    if isinstance(file_hash, str) and file_hash in registry_by_hash:
        return registry_by_hash[file_hash]

    if isinstance(source_file, str) and source_file:
        return source_file

    return (collection_metadata or {}).get("name") or "material_didactic"


def _infer_chunk_index(metadata: Dict[str, Any], fallback_index: int, doc_id: Optional[str]) -> int:
    chunk_index = metadata.get("chunk_index")
    if isinstance(chunk_index, int) and chunk_index >= 0:
        return chunk_index

    if isinstance(chunk_index, str) and chunk_index.isdigit():
        return int(chunk_index)

    if doc_id:
        match = re.search(r"_(\d+)$", doc_id)
        if match:
            return int(match.group(1))

    return fallback_index


def _batched(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def migrate_metadata(db_path: Path, dry_run: bool = False) -> Tuple[int, int, int]:
    client = _create_client(db_path)
    processed_registry = _load_processed_registry(db_path)
    registry_by_hash = _invert_registry(processed_registry)

    total_collections = 0
    total_updated = 0
    total_skipped = 0

    for collection in client.list_collections():
        total_collections += 1
        metadata = getattr(collection, "metadata", {}) or {}

        try:
            collection_size = collection.count()
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Nu pot determina dimensiunea colecției %s: %s", collection.name, exc)
            total_skipped += 1
            continue

        if collection_size == 0:
            logging.info("Colecția %s este goală - ignor", collection.name)
            total_skipped += 1
            continue

        records = collection.get(include=["metadatas", "ids"], limit=collection_size)
        metadatas = records.get("metadatas") or []
        ids = records.get("ids") or []

        update_metadatas: List[Dict[str, Any]] = []
        update_ids: List[str] = []
        updated_here = 0

        for position, (doc_id, item_metadata) in enumerate(zip(ids, metadatas)):
            current_meta = dict(item_metadata or {})
            pdf_reference = _resolve_source_path(metadata, current_meta, registry_by_hash)
            chunk_index = _infer_chunk_index(current_meta, position, doc_id)

            enriched = ensure_standard_metadata(pdf_reference, chunk_index, current_meta)

            if enriched != item_metadata:
                update_metadatas.append(enriched)
                update_ids.append(doc_id)
                updated_here += 1

        if updated_here:
            logging.info(
                "Colecția %s: actualizez %d/%d documente",
                collection.name,
                updated_here,
                collection_size,
            )
            if not dry_run:
                for batch_ids, batch_metas in zip(
                    _batched(update_ids, 500), _batched(update_metadatas, 500)
                ):
                    collection.update(ids=batch_ids, metadatas=batch_metas)
        else:
            logging.info("Colecția %s are deja metadatele aliniate", collection.name)

        total_updated += updated_here
        total_skipped += collection_size - updated_here

    logging.info(
        "Migrare completă: %d colecții inspectate, %d documente actualizate, %d fără schimbări",
        total_collections,
        total_updated,
        total_skipped,
    )

    return total_collections, total_updated, total_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrare metadate Chroma la schema standardizată")
    parser.add_argument(
        "--db-path",
        default="./embeddings_db",
        help="Calea către directorul bazei de date Chroma",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Rulează migrarea fără a scrie modificările",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"Baza de date nu există la {db_path}")

    migrate_metadata(db_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
