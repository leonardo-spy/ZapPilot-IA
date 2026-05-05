"""
Indexação semântica da knowledge base no ChromaDB.
Usa o EmbeddingProvider centralizado (Google Gemini Embedding 2 / fallback local).
"""
import json
import logging
import os
from pathlib import Path
import chromadb

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "android_box_suporte")


def build_document_text(entry: dict) -> str:
    """Cria texto rico para indexação a partir de uma entrada da KB."""
    parts = []

    if entry.get("title"):
        parts.append(f"Título: {entry['title']}")

    if entry.get("category"):
        parts.append(f"Categoria: {entry['category']}")

    if entry.get("intent"):
        parts.append(f"Intenção: {entry['intent']}")

    if entry.get("symptoms"):
        parts.append(f"Sintomas: {'; '.join(entry['symptoms'][:5])}")

    if entry.get("steps"):
        steps_text = "\n".join(f"  - {s}" for s in entry["steps"][:6])
        parts.append(f"Passos:\n{steps_text}")

    if entry.get("recommended_response"):
        parts.append(f"Resposta recomendada: {entry['recommended_response']}")

    if entry.get("examples"):
        examples_text = "; ".join(entry["examples"][:5])
        parts.append(f"Exemplos de perguntas: {examples_text}")

    return "\n\n".join(parts)


def index_knowledge_base(
    kb_path: str = "./data/knowledge_base.json",
    chroma_path: str = "./data/chroma_db",
    embed_model: str = "all-MiniLM-L6-v2",
) -> int:
    """
    Lê knowledge_base.json e indexa no ChromaDB.

    Returns:
        Número de documentos indexados
    """
    logger.info(f"Carregando KB: {kb_path}")
    with open(kb_path, encoding="utf-8") as f:
        kb = json.load(f)

    if not kb:
        logger.warning("Knowledge base vazia")
        return 0

    # Usar provider centralizado (Google ou fallback local)
    from llm.embeddings import get_embedding_provider
    provider = get_embedding_provider()
    logger.info(f"Embedding provider para indexação: {provider.name()}")

    # Inicializar Chroma
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)

    # Deletar collection existente e recriar
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' existente removida")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Gerar documentos e embeddings
    documents = []
    ids = []
    metadatas = []

    for entry in kb:
        doc_text = build_document_text(entry)
        documents.append(doc_text)
        ids.append(entry["id"])
        metadatas.append({
            "kb_id": entry["id"],
            "category": entry.get("category", ""),
            "intent": entry.get("intent", ""),
            "confidence": entry.get("confidence", 0.0),
            "needs_human_review": entry.get("needs_human_review", False),
            "cluster_id": entry.get("cluster_id", -1),
        })

    logger.info(f"Gerando embeddings para {len(documents)} documentos...")
    embeddings = provider.encode(documents, task_type="retrieval_document", show_progress_bar=True, batch_size=32)

    # Inserir em batches (Chroma tem limite)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end = min(i + batch_size, len(documents))
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings[i:end].tolist(),
            metadatas=metadatas[i:end],
        )

    logger.info(f"Indexados {len(documents)} documentos no ChromaDB ({chroma_path})")
    return len(documents)


def search_chroma(
    query: str,
    top_k: int = 5,
    chroma_path: str = "./data/chroma_db",
    embed_model: str = "all-MiniLM-L6-v2",
    category_filter: str = None,
) -> list[dict]:
    """
    Busca semântica no ChromaDB.

    Returns:
        Lista de resultados com content, metadata, score
    """
    from llm.embeddings import get_embedding_provider
    provider = get_embedding_provider()
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = provider.encode(query, task_type="retrieval_query").flatten().tolist()

    where_filter = None
    if category_filter:
        where_filter = {"category": category_filter}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for i in range(len(results["ids"][0])):
        # Chroma retorna distância (menor = melhor), converter para score
        distance = results["distances"][0][i]
        score = 1 - distance  # cosine similarity = 1 - cosine distance

        output.append({
            "kb_id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "semantic_score": round(score, 4),
        })

    return output


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    data_dir = os.getenv("DATA_DIR", "./data")
    kb_path = f"{data_dir}/knowledge_base.json"
    chroma_path = f"{data_dir}/chroma_db"
    embed_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    count = index_knowledge_base(kb_path, chroma_path, embed_model)
    print(f"\n{count} documentos indexados no Chroma")

    # Teste rápido
    if count > 0:
        results = search_chroma("quanto custa", top_k=3, chroma_path=chroma_path, embed_model=embed_model)
        print(f"\nBusca 'quanto custa': {len(results)} resultados")
        for r in results:
            print(f"  [{r['semantic_score']:.3f}] {r['kb_id']}: {r['content'][:80]}...")
