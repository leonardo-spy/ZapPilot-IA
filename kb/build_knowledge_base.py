"""
Módulo de construção da Knowledge Base.
Usa embeddings + clustering para gerar base de conhecimento estruturada a partir dos turns.
"""
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

logger = logging.getLogger(__name__)


def build_knowledge_base(
    turns: list[dict],
    embed_model: str = "all-MiniLM-L6-v2",
    eps: float = 0.35,
    min_samples: int = 3,
    output_path: str = "./data/knowledge_base.json",
) -> list[dict]:
    """
    Constrói knowledge_base.json a partir de turns semânticos.

    1. Gera embeddings dos problemas dos clientes
    2. Clusteriza por similaridade (DBSCAN)
    3. Para cada cluster, extrai: intent, resposta canônica, sintomas, exemplos
    4. Salva em JSON estruturado

    Returns:
        Lista de entradas da knowledge base
    """
    if not turns:
        logger.warning("Nenhum turn para processar")
        return []

    logger.info(f"Gerando KB a partir de {len(turns)} turns...")

    # Usar provider centralizado para embeddings
    from llm.embeddings import get_embedding_provider
    provider = get_embedding_provider()
    logger.info(f"Embedding provider: {provider.name()}")

    # Gerar embeddings dos problemas dos clientes (task: clustering)
    problems = [t["client_context"] for t in turns]
    logger.info("Gerando embeddings...")
    embeddings = provider.encode(problems, task_type="clustering", show_progress_bar=True, batch_size=64)

    # Clustering com DBSCAN
    logger.info(f"Clusterizando (eps={eps}, min_samples={min_samples})...")
    distance_matrix = 1 - cosine_similarity(embeddings)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.clip(distance_matrix, 0, 2)  # Fix float precision negatives

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(distance_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"Clusters: {n_clusters}, Noise: {n_noise}/{len(labels)}")

    # Agrupar turns por cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters[label].append(idx)

    # Gerar knowledge base
    kb_entries = []

    for cluster_id, indices in sorted(clusters.items()):
        cluster_turns = [turns[i] for i in indices]
        cluster_embeddings = embeddings[indices]

        # Encontrar turn mais representativo (mais próximo do centroide)
        centroid = cluster_embeddings.mean(axis=0)
        similarities = cosine_similarity([centroid], cluster_embeddings)[0]
        representative_idx = similarities.argmax()
        representative_turn = cluster_turns[representative_idx]

        # Extrair problemas exemplo (variações únicas)
        sample_problems = _get_unique_samples(
            [t["client_context"] for t in cluster_turns], max_samples=8
        )

        # Resposta canônica: a mais longa/completa do representativo ou mediana
        responses = [t["assistant_context"] for t in cluster_turns]
        canonical_response = _select_best_response(responses)

        # Categorizar (heurística simples)
        category = _categorize(sample_problems, canonical_response)

        # Gerar intent simples a partir do problema representativo
        intent = _generate_intent(representative_turn["client_context"])

        # Confiança baseada na coesão do cluster
        confidence = float(similarities.mean())

        entry = {
            "id": f"kb_{cluster_id:03d}",
            "cluster_id": cluster_id,
            "category": category,
            "intent": intent,
            "title": representative_turn["client_context"][:100].replace("\n", " "),
            "symptoms": sample_problems[:5],
            "recommended_response": canonical_response,
            "steps": _extract_steps(canonical_response),
            "examples": sample_problems,
            "source_turn_count": len(cluster_turns),
            "confidence": round(confidence, 3),
            "needs_human_review": confidence < 0.5 or len(cluster_turns) < 4,
        }
        kb_entries.append(entry)

    # Ordenar por confiança decrescente
    kb_entries.sort(key=lambda x: x["confidence"], reverse=True)

    # Converter tipos numpy para nativos Python
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    for entry in kb_entries:
        for key, val in entry.items():
            entry[key] = _convert(val)

    # Salvar
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(kb_entries, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Knowledge base gerada: {len(kb_entries)} entradas salvas em {output_path}"
    )
    return kb_entries


def _get_unique_samples(texts: list[str], max_samples: int = 8) -> list[str]:
    """Retorna amostras únicas (sem duplicatas exatas), priorizando mais curtas."""
    seen = set()
    unique = []
    for t in sorted(texts, key=len):
        normalized = t.strip().lower()[:200]
        if normalized not in seen:
            seen.add(normalized)
            unique.append(t.strip())
        if len(unique) >= max_samples:
            break
    return unique


def _select_best_response(responses: list[str]) -> str:
    """Seleciona a melhor resposta: mais completa mas não excessivamente longa."""
    if not responses:
        return ""

    # Filtrar respostas muito curtas
    viable = [r for r in responses if len(r) > 20]
    if not viable:
        viable = responses

    # Pegar a de comprimento mediano-alto (percentil 75)
    viable.sort(key=len)
    idx = min(int(len(viable) * 0.75), len(viable) - 1)
    return viable[idx]


def _categorize(problems: list[str], response: str) -> str:
    """Categoriza heuristicamente como venda, suporte ou info usando config do domínio."""
    text = " ".join(problems).lower() + " " + response.lower()

    try:
        from agent.prompts import get_domain_config
        domain = get_domain_config()
        sale_keywords = domain.get("sale_keywords", [])
        support_keywords = domain.get("support_keywords", [])
    except Exception:
        sale_keywords = ["quanto custa", "preço", "valor", "comprar", "pagar", "pix"]
        support_keywords = ["não funcionou", "problema", "ajuda", "como fazer"]

    sale_score = sum(1 for kw in sale_keywords if kw in text)
    support_score = sum(1 for kw in support_keywords if kw in text)

    if sale_score > support_score:
        return "venda"
    elif support_score > sale_score:
        return "suporte"
    return "info"


def _generate_intent(problem_text: str) -> str:
    """Gera um identificador de intent simples a partir do texto."""
    text = problem_text.lower().strip()[:150]

    # Mapeamento de padrões para intents
    intent_map = [
        (["preço", "valor", "quanto custa", "quanto é"], "consulta_preco"),
        (["comprar", "quero", "interesse", "adquirir"], "intencao_compra"),
        (["desconto", "parcela", "negociar", "1400", "1300"], "negociacao_preco"),
        (["como usar", "como aplica", "aplicar", "dose"], "instrucoes_uso"),
        (["efeito", "colateral", "enjoo", "dor"], "efeitos_colaterais"),
        (["resultado", "perdi", "emagreci", "funcionou", "não perdi"], "resultado_tratamento"),
        (["entrega", "envio", "frete", "chegar"], "logistica_entrega"),
        (["diferença", "qual melhor", "tg", "tirzec", "monjaro"], "comparacao_produto"),
        (["falsa", "original", "confiança", "anvisa"], "autenticidade"),
        (["receita", "médico", "prescrição"], "requisitos_compra"),
        (["renovar", "mais", "outra", "de novo"], "recompra"),
    ]

    for keywords, intent in intent_map:
        if any(kw in text for kw in keywords):
            return intent

    return "geral"


def _extract_steps(response: str) -> list[str]:
    """Extrai passos de uma resposta (se houver estrutura de lista)."""
    lines = response.split("\n")
    steps = []

    for line in lines:
        line = line.strip()
        # Detectar linhas que parecem passos (começam com *, -, número, etc)
        if line and (
            line.startswith(("- ", "* ", "• "))
            or (len(line) > 2 and line[0].isdigit() and line[1] in (".", ")", " "))
        ):
            steps.append(line.lstrip("-*• 0123456789.) "))

    # Se não encontrou passos formatados, quebrar resposta em sentenças relevantes
    if not steps and len(response) > 50:
        sentences = [s.strip() for s in response.replace("\n", ". ").split(". ") if len(s.strip()) > 15]
        steps = sentences[:5]

    return steps[:8]


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.whatsapp_loader import load_whatsapp_auto
    from preprocessing.turns import build_turns_from_messages

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    output_path = os.getenv("DATA_DIR", "./data") + "/knowledge_base.json"

    messages = load_whatsapp_auto()
    turns = build_turns_from_messages(messages)

    print(f"\nTotal turns para clustering: {len(turns)}")
    kb = build_knowledge_base(turns, output_path=output_path)
    print(f"Knowledge base: {len(kb)} entradas")
