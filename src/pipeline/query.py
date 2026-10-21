"""
Consulta a la base de conocimiento: pregunta -> retrieval -> generación,
e imprime la respuesta junto con las fuentes citadas

Uso:
    python -m src.pipeline.query "¿Qué es FBI con HH-XRF?"
    python -m src.pipeline.query            (modo interactivo)
    python -m src.pipeline.query --topic "..." --top-k 8 "pregunta"
"""

import argparse

from src.conf import TOP_K
from src.generation.generator import format_pages, generate_answer
from src.retrieval.retriever import retrieve

def ask(question: str, top_k: int = TOP_K, pdf_topic: str | None = None,
        debug: bool = False) -> None:
    """
    Responde una pregunta e imprime la respuesta y sus fuentes
    """

    chunks = retrieve(question, top_k, pdf_topic)

    if debug:
        # La distancia coseno ayuda a ver si el retrieval trae algo útil:
        # valores cercanos a 0 son muy similares, cercanos a 1 casi nada
        print("\nChunks recuperados:")
        for i, chunk in enumerate(chunks, start=1):
            print(f"  [{i}] {chunk['distance']:.3f}  {chunk['file']}, {format_pages(chunk)}")

    result = generate_answer(question, chunks)

    print(f"\n{result['answer']}\n")

    if result["sources"]:
        print("Fuentes:")
        for source in result["sources"]:
            line = f"  [Fragmento {source['fragment']}] {source['file']}, {format_pages(source)}"
            if source.get("section"):
                line += f" ({source['section']})"
            print(line)

def main() -> None:
    parser = argparse.ArgumentParser(description="Pregunta a la base de conocimiento")
    parser.add_argument("question", nargs="*", help="pregunta; si se omite, entra en modo interactivo")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="chunks a recuperar")
    parser.add_argument("--topic", default=None, help="filtrar por pdf_topic")
    parser.add_argument("--debug", action="store_true", help="mostrar chunks recuperados y su distancia")
    args = parser.parse_args()

    if args.question:
        ask(" ".join(args.question), args.top_k, args.topic, args.debug)
        return

    print("Escribe tu pregunta (vacío o Ctrl+C para salir)")
    try:
        while question := input("\n> ").strip():
            ask(question, args.top_k, args.topic, args.debug)
    except (KeyboardInterrupt, EOFError):
        pass

if __name__ == "__main__":
    main()
