"""
Orquesta el pipeline completo de ingesta: extracción -> chunking ->
embeddings -> carga a Postgres, para todos los PDFs en pdfs/
"""

import json
import sys

from src.conf import CHUNKS_DIR, PDF_METADATA, ensure_dirs
from src.embedding.embedder import embed_texts
from src.vectorstore.db import get_connection, insert_chunks, upsert_document

def ingest_all() -> None:
    """
    Recorre todos los PDFs y los deja completamente cargados en Postgres
    """

    ensure_dirs()

    # Opción b: este orquestador NO re-extrae ni re-chunkea; consume lo que
    # ya dejaron extract.py y chunk.py en data/chunks/
    chunk_paths = sorted(CHUNKS_DIR.glob("*.jsonl"))

    if not chunk_paths:
        sys.exit(f"No hay chunks en {CHUNKS_DIR}. Ejecuta antes:\n python -m src.extraction.extract\n python -m src.chunking.chunk")

    conn = get_connection()
    total_docs = 0
    total_chunks = 0

    for chunk_path in chunk_paths:
        chunks = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        if not chunks:
            print(f"{chunk_path.name}: sin chunks, se omite")

        # Un .jsonl == un PDF, así que basta con mirar el primer chunk
        file = chunks[0]["file"]

        # La metadata se relee de conf.py en vez de usar la que quedó
        # congelada en el .jsonl: así, corregir PDF_METADATA solo
        # exige re-ingestar, no re-chunkear todo el PDF
        meta = PDF_METADATA.get(file)
        if meta is None:
            print(f" AVISO: '{file}' mp está en PDF_METADATA; uso la metadata del .jsonl")
            meta = chunks[0]

        print(f"Ingestando {file} ({len(chunks)} chunks) ...")

        doc_id = upsert_document(conn, file, meta.get("topic"), meta.get("doc_type"), meta.get("language"))
        embeddings = embed_texts([chunk["text"] for chunk in chunks])
        insert_chunks(conn, doc_id, chunks, embeddings)

        # Commit por documento: si un PDF posterior falla, los ya cargados
        # quedan en la base y se puede reintentar sin repetir todo
        conn.commit()
        
        total_docs+=1
        total_chunks+=len(chunks)
        
        print(f" -> documento id={doc_id}, {len(chunks)} chunks guardados")
    
    print(f"\nList: {total_docs} documentos, {total_chunks} chunks en total")

if __name__=="__main__":
    ingest_all()
