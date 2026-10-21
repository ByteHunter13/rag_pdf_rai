-- Esquema del vector store. Ejecutar una vez contra la base de datos
-- (con pgvector ya instalado con `CREATE EXTENSION IF NOT EXISTS vector;`).
--
-- Uso sugerido:
--   psql "$DATABASE_URL" -f src/vectorstore/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL PRIMARY KEY,
    filename      TEXT NOT NULL UNIQUE,
    pdf_topic     TEXT,           -- nombre/modelo del equipo (metadata manual)
    doc_type      TEXT,           -- application_note | datasheet | manual | ...
    language      TEXT,           -- 'es' | 'en'
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- IMPORTANTE: 1024 debe coincidir con EMBEDDING_DIM en tu .env
-- (BAAI/bge-m3 produce vectores de 1024 dimensiones).
CREATE TABLE IF NOT EXISTS chunks (
    id             SERIAL PRIMARY KEY,
    document_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text           TEXT NOT NULL,
    start_page     INTEGER,
    end_page       INTEGER,
    section        TEXT,           -- ruta de headings, ej. "Especificaciones > Sensores"
    embedding      vector(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);

-- Índice aproximado HNSW para búsqueda por similitud coseno.
-- Se puede crear después de cargar datos (o antes; con pocos miles de filas
-- no hay diferencia de rendimiento notable en la creación).
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
