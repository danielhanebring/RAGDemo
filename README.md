# Retrieval-demo (RAG – R)

Detta är en enkel demo av **retrieval-steget** i ett RAG-system (Retrieval-Augmented Generation).

## Vad den gör

Applikationen visar hur man kan:
- omvandla text till embeddings (vektorer)
- lagra dessa i en vector-databas
- ställa en fråga
- hämta det mest relevanta dokumentet baserat på semantisk likhet

## Hur det fungerar

1. Dokument (dummydata) omvandlas till embeddings
2. Embeddings lagras i en vector-databas (Chroma)
3. Användaren ställer en fråga
4. Frågan omvandlas också till en embedding
5. Systemet jämför frågan med dokumenten och returnerar det mest relevanta


## Kör lokalt

Installera beroenden:

```bash
pip install -r requirements.txt