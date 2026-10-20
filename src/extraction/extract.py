"""
1. Extracción de PDFs con Docling

Objetivo: convertir cada PDF de pdfs/ en un DoclingDocument (que conserva heading, tablas y páginas)
y guardarlo en data/extracted/, para no tener que re-procesar el PDF cada vez que cambie chunking/embeddings
"""

import sys
from pathlib import Path

from docling.document_converter import DocumentConverter

from src.conf import EXTRACTED_DIR, PDFS_DIR, ensure_dirs

def extract_pdf(converter: DocumentConverter, pdf_path: Path) -> Path:
    """
    Convierte un único PDF y guarda el resultado en data/extracted

    Guarda 2 archivos con el mismo nombre base que el PDF:

    - `<stem>.json`: el DoclingDocument completo, fuente de verdad para el
    chunking (conserva headings, tablas y página por bloque).
    - `<stem>.md`: export en markdown, solo para revisar a ojo que las tablas
    salieron bien.

    Devuelve la ruta del .json generado.
    """

    # Documento convertido
    document = converter.convert(pdf_path).document

    # Archivo JSON convertido
    # Se usa el .stem para quitar la extensión del archivo y poderlo
    # guardar con distintas extensiones
    json_path = EXTRACTED_DIR / f"{pdf_path.stem}.json"
    document.save_as_json(json_path)

    # Guardado también el archivo markdown del documento
    md_path = EXTRACTED_DIR / f"{pdf_path.stem}.md"

    # Uso de utf-8 para guardar símbolos
    md_path.write_text(document.export_to_markdown(), encoding="utf-8")

    return json_path

def main() -> None:
    """
    Recorre todos los PDFs de PDFS_DIR y llama extract_pdf en cada uno
    """
    ensure_dirs()

    # sorted() para que el orden de proceso no dependa del sistema de archivos
    pdf_paths = sorted(PDFS_DIR.glob("*.pdf"))

    if not pdf_paths:
        sys.exit(f"No hay PDFs en {PDFS_DIR}")

    # Un único converter: instanciarlo carga los modelos de layout/tablas,
    # así que se reutiliza para todos los PDFs
    converter = DocumentConverter()

    for pdf_path in pdf_paths:
        print(f"Extrayendo {pdf_path.name} ...")
        json_path = extract_pdf(converter, pdf_path)
        print(f" -> {json_path}")

if __name__=="__main__":
    main()
