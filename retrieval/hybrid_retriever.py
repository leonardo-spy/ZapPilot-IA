"""
Retriever híbrido: combina busca semântica (Chroma) + keywords (BM25) com reranking.
"""
import logging
from dataclasses import dataclass
from retrieval.chroma_index import search_chroma
from retrieval.bm25_index import bm25_search

logger = logging.getLogger(__name__)

# Pesos para combinação de scores
SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35


@dataclass
class RetrievedDocument:
    kb_id: str
    content: str
    metadata: dict
    semantic_score: float
    keyword_score: float
    final_score: float

    def to_dict(self) -> dict:
        return {
            "kb_id": self.kb_id,
            "content": self.content,
            "metadata": self.metadata,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "final_score": self.final_score,
        }


def hybrid_search(
    query: str,
    top_k: int = 5,
    chroma_path: str = "./data/chroma_db",
    data_dir: str = "./data",
    embed_model: str = "all-MiniLM-L6-v2",
    semantic_weight: float = SEMANTIC_WEIGHT,
    keyword_weight: float = KEYWORD_WEIGHT,
    category_filter: str = None,
) -> list[RetrievedDocument]:
    """
    Busca híbrida: Chroma (semântica) + BM25 (keyword).

    1. Busca no Chroma
    2. Busca no BM25
    3. Merge + dedup por kb_id
    4. Score combinado ponderado
    5. Ordena e retorna top_k

    Returns:
        Lista de RetrievedDocument ordenada por final_score
    """
    # Buscar em ambos com margem extra para merge
    search_k = top_k * 2

    # Busca semântica (Chroma)
    try:
        semantic_results = search_chroma(
            query, top_k=search_k, chroma_path=chroma_path,
            embed_model=embed_model, category_filter=category_filter
        )
    except Exception as e:
        logger.warning(f"Erro na busca Chroma: {e}")
        semantic_results = []

    # Busca keyword (BM25)
    try:
        keyword_results = bm25_search(query, top_k=search_k, data_dir=data_dir)
    except Exception as e:
        logger.warning(f"Erro na busca BM25: {e}")
        keyword_results = []

    # Merge por kb_id
    merged: dict[str, dict] = {}

    for r in semantic_results:
        kb_id = r["kb_id"]
        merged[kb_id] = {
            "kb_id": kb_id,
            "content": r["content"],
            "metadata": r["metadata"],
            "semantic_score": r["semantic_score"],
            "keyword_score": 0.0,
        }

    for r in keyword_results:
        kb_id = r["kb_id"]
        if kb_id in merged:
            merged[kb_id]["keyword_score"] = r["keyword_score"]
        else:
            merged[kb_id] = {
                "kb_id": kb_id,
                "content": r["content"],
                "metadata": r["metadata"],
                "semantic_score": 0.0,
                "keyword_score": r["keyword_score"],
            }

    # Calcular score final
    documents = []
    for data in merged.values():
        final_score = (
            semantic_weight * data["semantic_score"]
            + keyword_weight * data["keyword_score"]
        )

        documents.append(RetrievedDocument(
            kb_id=data["kb_id"],
            content=data["content"],
            metadata=data["metadata"],
            semantic_score=data["semantic_score"],
            keyword_score=data["keyword_score"],
            final_score=round(final_score, 4),
        ))

    # Ordenar por score final decrescente
    documents.sort(key=lambda x: x.final_score, reverse=True)

    result = documents[:top_k]

    logger.info(
        f"Hybrid search '{query[:50]}': "
        f"{len(semantic_results)} semantic + {len(keyword_results)} keyword "
        f"→ {len(merged)} merged → {len(result)} retornados"
    )

    return result


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    data_dir = os.getenv("DATA_DIR", "./data")
    chroma_path = f"{data_dir}/chroma_db"
    embed_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    queries = [
        "quanto custa tirzec",
        "como aplicar a dose",
        "efeitos colaterais",
        "entrega em sjc",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        results = hybrid_search(q, top_k=3, chroma_path=chroma_path, data_dir=data_dir, embed_model=embed_model)
        for r in results:
            print(f"  [{r.final_score:.3f}] (sem:{r.semantic_score:.3f} kw:{r.keyword_score:.3f}) {r.kb_id}: {r.content[:80]}...")
