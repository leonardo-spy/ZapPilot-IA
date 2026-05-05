"""
Geração de Knowledge Base complementar via LLM + Embeddings.
Gera entradas de KB para problemas/perguntas comuns do domínio
que podem NÃO estar nas conversas do WhatsApp mas são conhecimento padrão.

IMPORTANTE: Todas as entradas geradas têm needs_human_review=True.
O operador deve revisar antes de confiar nestas informações.
"""
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


def generate_domain_kb(
    quantity: int = 10,
    output_path: str = None,
    merge_with_existing: bool = True,
) -> list[dict]:
    """
    Usa LLM para gerar entradas de KB complementares baseadas no domínio.

    Estas entradas cobrem conhecimento padrão do domínio que pode não estar
    nas conversas exportadas, mas são perguntas/problemas comuns.

    Args:
        quantity: Número de entradas a gerar
        output_path: Onde salvar (default: data/domain_kb.json)
        merge_with_existing: Se True, não gera entradas duplicadas de intents já existentes

    Returns:
        Lista de entradas geradas (marcadas como needs_human_review=True)
    """
    from agent.prompts import get_domain_kb_generation_prompt
    from llm.providers import get_default_provider

    data_dir = os.getenv("DATA_DIR", "./data")
    if output_path is None:
        output_path = f"{data_dir}/domain_kb.json"

    # Carregar intents existentes para evitar duplicatas
    existing_intents = set()
    if merge_with_existing:
        kb_path = f"{data_dir}/knowledge_base.json"
        if os.path.exists(kb_path):
            with open(kb_path, encoding="utf-8") as f:
                existing_kb = json.load(f)
            existing_intents = {e.get("intent", "") for e in existing_kb}
            logger.info(f"Intents existentes: {len(existing_intents)}")

    # Gerar via LLM
    llm = get_default_provider()
    prompt = get_domain_kb_generation_prompt(quantity)

    logger.info(f"Gerando {quantity} entradas complementares via LLM ({llm.name()})...")

    messages = [
        {"role": "system", "content": "Você gera JSON válido. Responda APENAS com o array JSON."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm.chat(messages, temperature=0.7, max_tokens=4096)
        entries = _parse_json_array(response)
    except Exception as e:
        logger.error(f"Erro ao gerar KB complementar: {e}")
        return []

    if not entries:
        logger.warning("Nenhuma entrada gerada (resposta não era JSON válido)")
        return []

    # Processar e marcar entradas
    processed = []
    for i, entry in enumerate(entries):
        # Pular se intent já existe
        intent = entry.get("intent", "")
        if intent in existing_intents:
            logger.debug(f"Skipping duplicata: {intent}")
            continue

        processed_entry = {
            "id": f"domain_{i:03d}",
            "cluster_id": -1,  # Indica que é gerado, não clusterizado
            "category": entry.get("category", "info"),
            "intent": intent,
            "title": entry.get("title", ""),
            "symptoms": entry.get("symptoms", []),
            "recommended_response": entry.get("recommended_response", ""),
            "steps": entry.get("steps", []),
            "examples": entry.get("examples", []),
            "source_turn_count": 0,  # Gerado, não vem de turns
            "confidence": 0.5,  # Confiança média (gerado por LLM)
            "needs_human_review": True,  # SEMPRE True para geradas
            "generated": True,  # Flag de que é gerado
        }
        processed.append(processed_entry)

    # Salvar
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    logger.info(f"KB complementar: {len(processed)} entradas salvas em {output_path}")
    return processed


def merge_knowledge_bases(
    primary_path: str = None,
    domain_path: str = None,
    output_path: str = None,
) -> list[dict]:
    """
    Merge a KB primária (dos dados) com a KB complementar (gerada por generate_domain_kb)
    + KB entries do domain config (gerada por generate_domain_config, se aprovadas).
    A KB primária sempre tem prioridade.

    Returns:
        KB merged
    """
    data_dir = os.getenv("DATA_DIR", "./data")
    primary_path = primary_path or f"{data_dir}/knowledge_base.json"
    domain_path = domain_path or f"{data_dir}/domain_kb.json"
    output_path = output_path or f"{data_dir}/knowledge_base_full.json"

    # Carregar primária
    primary_kb = []
    if os.path.exists(primary_path):
        with open(primary_path, encoding="utf-8") as f:
            primary_kb = json.load(f)

    # Carregar complementar (generate_domain_kb)
    domain_kb = []
    if os.path.exists(domain_path):
        with open(domain_path, encoding="utf-8") as f:
            domain_kb = json.load(f)

    # Carregar KB entries do domain config (generate_domain_config, se aprovadas)
    config_kb = []
    try:
        from kb.generate_domain_config import get_generated_kb_entries
        config_kb = get_generated_kb_entries(only_approved=True)
        if config_kb:
            # Dar IDs únicos
            for i, entry in enumerate(config_kb):
                if "id" not in entry:
                    entry["id"] = f"config_{i:03d}"
    except Exception:
        pass

    # Merge: primária primeiro, domain depois, config por último
    merged = primary_kb + domain_kb + config_kb

    # Salvar
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info(
        f"KB merged: {len(primary_kb)} primária + {len(domain_kb)} complementar "
        f"+ {len(config_kb)} config = {len(merged)} total"
    )
    return merged


def index_full_kb(
    kb_path: str = None,
    use_domain_kb: bool = True,
):
    """
    Indexa a KB completa (primária + complementar) no Chroma e BM25.
    Usa o novo embedding provider (Google ou local).
    """
    data_dir = os.getenv("DATA_DIR", "./data")

    if use_domain_kb:
        # Merge primeiro
        merged = merge_knowledge_bases()
        kb_path = f"{data_dir}/knowledge_base_full.json"
    else:
        kb_path = kb_path or f"{data_dir}/knowledge_base.json"

    # Indexar Chroma
    from retrieval.chroma_index import index_knowledge_base
    chroma_path = f"{data_dir}/chroma_db"
    embed_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chroma_count = index_knowledge_base(kb_path, chroma_path, embed_model)

    # Indexar BM25
    from retrieval.bm25_index import build_bm25_index
    bm25_count = build_bm25_index(kb_path, data_dir)

    logger.info(f"Indexação completa: Chroma={chroma_count}, BM25={bm25_count}")
    return chroma_count, bm25_count


def _parse_json_array(text: str) -> list:
    """Extrai JSON array da resposta do LLM."""
    text = text.strip()

    # Tentar parse direto
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Tentar extrair array de dentro do texto
    import re
    # Procurar por [ ... ] (pode ser multi-line)
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Gerando KB complementar para o domínio ===\n")
    entries = generate_domain_kb(quantity=10)

    if entries:
        print(f"\n{len(entries)} entradas geradas:")
        for e in entries[:3]:
            print(f"  [{e['category']}] {e['intent']}: {e['title']}")
            print(f"    Symptoms: {e['symptoms'][:3]}")
            print(f"    ⚠️  needs_human_review: {e['needs_human_review']}")
            print()

        # Opcionalmente fazer merge e re-indexar
        print("\nPara merge + indexação, rode:")
        print("  python -c \"from kb.generate_domain_kb import index_full_kb; index_full_kb()\"")
