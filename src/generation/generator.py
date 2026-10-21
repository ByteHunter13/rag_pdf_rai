"""
5b. Generación de respuestas con OpenAI API

Objetivo: armar un prompt RAG a partir de la pregunta + los chunks
recuperados, mandarlo a la API de OpenAI y devolver la respuesta junto con las
fuentes citadas
"""

import re
from typing import Any
from src.conf import CHAT_MODEL
from src.openai_api import post

# Acepta [Fragmento 3] y también variantes como [fragmento 3] o [Fragmento  3]
FRAGMENT_RE = re.compile(r"\[\s*fragmento\s+(\d+)\s*\]", re.IGNORECASE)

SYSTEM_PROMPT = """\
Eres un asistente técnico que responde preguntas sobre equipos que vende la
empresa, basándote ÚNICAMENTE en los fragmentos de documentación
proporcionados.

Reglas:
- Responde en el mismo idioma en que está escrita la pregunta.
- Si los fragmentos no contienen información suficiente para responder,
  dilo explícitamente en vez de inventar datos.
- Cada dato que afirmes debe salir de un fragmento; no agregues información
  de tu propio conocimiento.

Cómo citar: escribe [Fragmento N] justo después de cada dato que tomes de
ese fragmento, dentro de la misma oración. Eso es todo: no armes una lista
de fuentes al final, ni escribas nombres de archivo o números de página —de
eso se encarga el programa—.

Ejemplo:

El detector opera en un rango de 10 a 40 kV [Fragmento 2] y pesa 1.5 kg
[Fragmento 5].

Cita solo los fragmentos de los que sacaste un dato concreto. Si ninguno
responde la pregunta, dilo en una oración y no cites ninguno.

Sé breve: responde lo que se preguntó y detente. No repitas el contenido de
los fragmentos ni agregues resúmenes ni recomendaciones que nadie pidió.
"""

def format_pages(chunk: dict[str, Any]) -> str:
    """
    Formatea el rango de páginas de un chunk para citarlo.

    Lo usan tanto el prompt como la lista de fuentes que imprime query.py,
    para que la cita que vel modelo y la que ve el usuario coincidan
    """

    start, end = chunk["start_page"], chunk["end_page"]

    if start is None:
        # pasa con el texto que va antes del primer heading del PDF
        return "página desconocida"

    if start == end:
        return f"pág. {start}"
    return f"págs. {start}-{end}"

def quoted_fragments(answer: str, n_chunks: int) -> list[int]:
    """
    Extrae los números de fragmento que el modelo citó en `answer`.

    Devuelve cada número una sola vez, en el orden en que aparecen. Descarta
    los que estén fuera de rango: el modelo puede escribir [Fragmento 9]
    cuando solo le pasamos 6, y esa cita no debe llegar a la lista de
    fuentes (chunks[8] sería un IndexError, o peor, un chunk equivocado).
    """

    numbers: list[int] = []

    for match in FRAGMENT_RE.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= n_chunks and n not in numbers:
            numbers.append(n)

    return numbers

def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """
    Arma el prompt final (contexto + pregunta) a partir de los chunks
    recuperados por retriever.retrieve().
    """

    # Los fragmentos se numeran desde 1 porque así los cita el modelo;
    # quoted_fragments() deshace ese desfase al mapear de vuelta a `chunks`
    fragments = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[Fragmento {i}] {chunk['file']}, {format_pages(chunk)}"
        if chunk.get("section"):
            header += f", sección: {chunk['section']}"
        fragments.append(f"{header}\n{chunk['text'].strip()}")

    context = "\n\n---\n\n".join(fragments)

    return (
        f"Fragmentos de documentación:\n\n{context}\n\n"
        f"Pregunta: {question.strip()}"
    )

def call_openai(messages: list[dict[str, str]], model: str = CHAT_MODEL) -> str:
    """
    Manda `messages` a la API de Chat Completions y devuelve el texto de la
    respuesta del modelo
    """

    response = post("/chat/completions", {"model": model, "messages": messages})

    message = response["choices"][0]["message"]

    # `content` viene en null si el modelo se niega a responder
    return message.get("content") or message.get("refusal") or ""

def generate_answer(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Genera la respuesta final

    Devuelve un dict con:
    - `answer`: el texto del modelo, con las citas [Fragmento N]
    - `sources`: los chunks citados, en el orden en que aparecen en la
      respuesta; cada uno lleva además `fragment` (su N en el prompt)
    """

    if not chunks:
        # Sin contexto no tiene sentido gastar una llamada: el modelo solo
        # podría contestar con su propio conocimiento, que es lo que evitamos
        return {
            "answer": "No encontré documentación relacionada con esa pregunta.",
            "sources": [],
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, chunks)},
    ]

    answer = call_openai(messages)

    sources = [
        {**chunks[n - 1], "fragment": n}
        for n in quoted_fragments(answer, len(chunks))
    ]

    return {"answer": answer, "sources": sources}