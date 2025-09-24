"""Utility pentru sweep semantic rapid pe colecțiile Chroma."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional

from pdf_converter_working import PDFEmbeddingsConverter

DEFAULT_KEYWORDS = [
    "copil",
    "programă",
    "exercițiu",
    "Piaget",
    "Feuerstein",
    "muzică",
    "ritm",
    "fracții",
    "evaluare",
    "lectură",
    "creativitate",
    "emoție",
    "STEM",
    "istorie",
    "geografie",
    "chimie",
    "literatură",
    "soft skills",
    "arte vizuale",
    "sport",
]


def _parse_keyword_source(keyword_source: Optional[str]) -> List[str]:
    if not keyword_source:
        return DEFAULT_KEYWORDS

    path = Path(keyword_source)
    if path.exists():
        content = path.read_text(encoding="utf-8")
    else:
        content = keyword_source

    separators = [",", "\n", "\r"]
    for separator in separators:
        content = content.replace(separator, " ")

    keywords = [token.strip() for token in content.split(" ") if token.strip()]
    return keywords or DEFAULT_KEYWORDS


def _extract_first_page(metadata: Optional[dict]) -> Optional[int]:
    if not metadata:
        return None

    for key in ("page_from", "page", "page_start", "page_number"):
        value = metadata.get(key)
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def keyword_sweep(
    keywords: Iterable[str],
    output_path: Path,
    top_k: int = 3,
) -> Path:
    converter = PDFEmbeddingsConverter()
    collections = converter.client.list_collections()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["keyword", "collection", "hit_count", "first_hit_id", "first_hit_page", "latency_ms"])

        for collection in collections:
            collection_name = getattr(collection, "name", "unknown")
            for keyword in keywords:
                start_time = time.time()
                results = converter.search_similar(keyword, top_k=top_k, collection_name=collection_name)
                latency_ms = round((time.time() - start_time) * 1000, 2)

                docs = results.get("documents", [[]])[0] if results else []
                metadatas = results.get("metadatas", [[]])[0] if results else []
                ids = results.get("ids", [[]])[0] if results else []

                first_hit_id = ids[0] if ids else ""
                first_hit_page = _extract_first_page(metadatas[0] if metadatas else None)

                writer.writerow([
                    keyword,
                    collection_name,
                    len(docs),
                    first_hit_id,
                    first_hit_page if first_hit_page is not None else "",
                    latency_ms,
                ])

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rulează un keyword sweep pe toate colecțiile Chroma.")
    parser.add_argument(
        "--keywords",
        help="Fișier sau listă separată prin virgulă/spații cu cuvinte cheie.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("KEYWORD_SWEEP_OUTPUT", "keyword_sweep.csv"),
        help="Path-ul fișierului CSV de ieșire.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Numărul de rezultate căutate pentru fiecare keyword.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    keywords = _parse_keyword_source(args.keywords)
    output_path = Path(args.output)

    keyword_sweep(keywords, output_path, top_k=args.top_k)
    print(f"Keyword sweep completed. Results saved to {output_path}")


if __name__ == "__main__":
    main()
