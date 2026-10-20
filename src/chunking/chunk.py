"""
2. Chunking con Docling HybridChunker

Objetivo: Leer los DoclingDocument guardados en data/extracted/ y producir
una lista de "chunks" listos para embeber: texto + metadata (archivo,
páginas, sección, idioma, tema). Se guardan en data/chunks/ (ej. un JSON
por PDF, o un único JSONL con todos).
"""

import json
import sys
from pathlib import Path
from typing import Any

import tiktoken

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

# from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
# from transformers import AutoTokenizer

from src.conf import CHUNKS_DIR, EMBEDDING_MODEL, PDF_METADATA, EXTRACTED_DIR, ensure_dirs

def build_chunker() -> HybridChunker:
    """
    Construye y devuelve un HybridChunker configurado para EMBEDDING_MODEL

    El tokenizer es del modelo de embeddings real (EMBEDDING_MODEL), así, el límite
    de tokens por chunk queda atado al modelo que se usará en la fase de embeddings
    si se cambia el EMBEDDING_MODEL, el chunking se ajusta solo
    """
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model(EMBEDDING_MODEL), # cl100k_base
        max_tokens=512,
    )

    return HybridChunker(
        tokenizer=tokenizer,
        # junta chunks chicos contiguos que comparten heading
        merge_peers=True,
        # cada trozo de una tabla larga repite el encabezado; sino, un fragmento
        # suelto de specs pierde el significado de las columnas
        repeat_table_header=True,
    )

def chunk_document(chunker: HybridChunker, doc_json_path: Path) -> list[dict[str, Any]]:
    """
    Carga un DoclingDocument desde JSON, lo chunkea y arma la metadata

    Devuelve una lista de dicts, uno por chunk, con el texto a embeber y la metadata
    que después se guarda junto al vector
    """
    doc = DoclingDocument.load_from_json(doc_json_path)

    # El nombre del PDF original viaja dentro del propio JSON (origin), que es la clave
    # con la que EQUIPMENT_METADATA está indexado
    file = doc.origin.filename if doc.origin else f"{doc_json_path.stem}.pdf"

    pdf_metadata = PDF_METADATA.get(file)

    if pdf_metadata is None:
        print(f" AVISO: '{file} no está en PDF_METADATA (conf.py)'")
        pdf_metadata = {}

    registers: list[dict[str, Any]] = []

    for chunk in chunker.chunk(dl_doc=doc):
        # Un chunk puede abarcar varios doc_items y por tanto varias páginas
        pages = [prov.page_no for item in chunk.meta.doc_items for prov in item.prov]

        registers.append(
            {
                # contextualize() antepone los headings al texto: el título
                # de la sección queda dentro del vector y mejora el recall
                "text": chunker.contextualize(chunk),
                "file": file,
                "start_page": min(pages) if pages else None,
                "end_page": max(pages) if pages else None,
                "section": " > ".join(chunk.meta.headings) if chunk.meta.headings else None,
                "pdf_topic": pdf_metadata.get("topic"),
                "doc_type": pdf_metadata.get("doc_type"),
                "language": pdf_metadata.get("language")
            }
        )

    return registers

def main() -> None:
    """
    Recorre todos los *.json de EXTRACTED_DIR, chunkea y guarda el resultado.

    Escribe un JSONL por PDF en CHUNKS_DIR (un chunk por línea, fácil de inspeccionar en un
    editor de texto)
    """
    ensure_dirs()

    doc_paths = sorted(EXTRACTED_DIR.glob("*.json"))

    if not doc_paths:
        sys.exit(f"No hay documentos en {EXTRACTED_DIR}. Extrae primero")

    # Instanciarlo descarga/carga el tokenizer, así se reutiliza
    chunker = build_chunker()

    for doc_path in doc_paths:
        print(f"Chunking {doc_path.name} ...")
        chunks = chunk_document(chunker, doc_path)

        out_path = CHUNKS_DIR / f"{doc_path.stem}.jsonl"

        with out_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                # ensure_ascii=False para que µ, ±, ° se lean tal cual
                f.write(json.dumps(chunk, ensure_ascii=False)+"\n")
        print(f"  -> {len(chunks)} chunks en {out_path}")

if __name__=="__main__":
    main()