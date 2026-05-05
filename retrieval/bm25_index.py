"""
Retriever por keywords usando BM25.
Complementa a busca semântica do Chroma com matching exato de termos.
"""
import json
import logging
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

INDEX_FILENAME = "bm25_index.pkl"


def _tokenize(text: str) -> list[str]:
    """Tokenização simples para BM25 (lowercase, split por espaço/pontuação)."""
    import re
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def build_bm25_index(
    kb_path: str = "./data/knowledge_base.json",
    output_dir: str = "./data",
) -> int:
    """
    Constrói índice BM25 a partir da knowledge base.

    Salva o índice em pickle para não precisar recalcular.

    Returns:
        Número de documentos indexados
    """
    logger.info(f"Construindo índice BM25 a partir de: {kb_path}")

    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)

    if not kb:
        logger.warning("Knowledge base vazia")
        return 0

    # Construir corpus de documentos (mesmo formato do Chroma)
    documents = []
    doc_ids = []

    for entry in kb:
        # Texto completo para BM25 (incluindo todos os campos searcháveis)
        parts = []
        if entry.get("title"):
            parts.append(entry["title"])
        if entry.get("symptoms"):
            parts.extend(entry["symptoms"])
        if entry.get("recommended_response"):
            parts.append(entry["recommended_response"])
        if entry.get("examples"):
            parts.extend(entry["examples"])
        if entry.get("steps"):
            parts.extend(entry["steps"])
        if entry.get("intent"):
            parts.append(entry["intent"])

        doc_text = " ".join(parts)
        documents.append(doc_text)
        doc_ids.append(entry["id"])

    # Tokenizar
    tokenized_corpus = [_tokenize(doc) for doc in documents]

    # Criar BM25
    bm25 = BM25Okapi(tokenized_corpus)

    # Salvar
    index_data = {
        "bm25": bm25,
        "documents": documents,
        "doc_ids": doc_ids,
        "kb": kb,
        "tokenized_corpus": tokenized_corpus,
    }

    output_path = Path(output_dir) / INDEX_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(index_data, f)

    logger.info(f"Índice BM25 salvo: {output_path} ({len(documents)} documentos)")
    return len(documents)


def load_bm25_index(data_dir: str = "./data") -> dict:
    """Carrega índice BM25 do disco."""
    index_path = Path(data_dir) / INDEX_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(
            f"Índice BM25 não encontrado em {index_path}. Execute build_bm25_index() primeiro."
        )

    with open(index_path, "rb") as f:
        return pickle.load(f)


def bm25_search(
    query: str,
    top_k: int = 5,
    data_dir: str = "./data",
) -> list[dict]:
    """
    Busca por keywords usando BM25.

    Returns:
        Lista de resultados com kb_id, content, metadata, keyword_score
    """
    index_data = load_bm25_index(data_dir)
    bm25 = index_data["bm25"]
    documents = index_data["documents"]
    doc_ids = index_data["doc_ids"]
    kb = index_data["kb"]

    # Tokenizar query
    query_tokens = _tokenize(query)

    if not query_tokens:
        return []

    # Buscar
    scores = bm25.get_scores(query_tokens)

    # Rankear
    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue

        # Normalizar score para 0-1 range (aproximado)
        max_score = float(scores.max()) if scores.max() > 0 else 1.0
        normalized_score = score / max_score

        kb_entry = kb[idx]
        results.append({
            "kb_id": doc_ids[idx],
            "content": documents[idx],
            "metadata": {
                "category": kb_entry.get("category", ""),
                "intent": kb_entry.get("intent", ""),
                "confidence": kb_entry.get("confidence", 0.0),
                "needs_human_review": kb_entry.get("needs_human_review", False),
            },
            "keyword_score": round(normalized_score, 4),
        })

    return results


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    data_dir = os.getenv("DATA_DIR", "./data")
    kb_path = f"{data_dir}/knowledge_base.json"

    count = build_bm25_index(kb_path, data_dir)
    print(f"\n{count} documentos indexados no BM25")

    # Teste rápido
    if count > 0:
        results = bm25_search("tirzec preço", top_k=3, data_dir=data_dir)
        print(f"\nBusca BM25 'tirzec preço': {len(results)} resultados")
        for r in results:
            print(f"  [{r['keyword_score']:.3f}] {r['kb_id']}: {r['content'][:80]}...")
