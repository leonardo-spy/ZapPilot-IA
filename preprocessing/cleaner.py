"""
Módulo de limpeza e merge de mensagens.
Remove spam, normaliza texto e agrupa mensagens consecutivas do mesmo autor.
Suporta detecção semântica de noise/spam via embeddings do domínio.
"""
import re
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Padrões de texto a remover/filtrar
URL_PATTERN = re.compile(r'https?://\S+', re.IGNORECASE)
EXCESSIVE_NEWLINES = re.compile(r'\n{3,}')
EXCESSIVE_EMOJIS = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]{5,}')
ONLY_LINK_PATTERN = re.compile(r'^\s*https?://\S+\s*$', re.IGNORECASE)


def _load_short_noise() -> set[str]:
    """Carrega SHORT_NOISE do domain YAML (não mais hardcoded)."""
    try:
        from config import load_domain_config
        domain = load_domain_config()
        terms = domain.get("short_noise", [])
        if terms:
            return set(t.lower() for t in terms)
    except Exception:
        pass
    # Fallback mínimo caso config não carregue
    return {
        "ok", "sim", "não", "nao", "boa", "vlw", "blz", "top",
        "beleza", "certo", "entendi", "hmm", "hm", "ah", "uhum",
        "👍", "👌", "🙏", "kkk", "kkkk", "kkkkk", "rs", "rsrs",
        "haha", "hahaha", "oi", "olá", "ola", "bom dia", "boa tarde",
        "boa noite", "obg", "obrigado", "obrigada", "valeu",
    }


# Lazy-load para não importar config no nível do módulo em circular imports
SHORT_NOISE: set[str] = None  # type: ignore


def get_short_noise() -> set[str]:
    """Retorna SHORT_NOISE (lazy loaded do YAML)."""
    global SHORT_NOISE
    if SHORT_NOISE is None:
        SHORT_NOISE = _load_short_noise()
    return SHORT_NOISE


# ==================== SEMANTIC NOISE DETECTION ====================

_noise_embeddings = None
_spam_embeddings = None
_feedback_pos_embeddings = None
_feedback_neg_embeddings = None
_embedding_provider = None


def _get_embedding_provider():
    """Lazy load do embedding provider."""
    global _embedding_provider
    if _embedding_provider is None:
        from llm.embeddings import get_embedding_provider
        _embedding_provider = get_embedding_provider()
    return _embedding_provider


def _get_noise_embeddings():
    """Gera embeddings dos termos de noise: base + gerados + extraídos do WhatsApp."""
    global _noise_embeddings
    if _noise_embeddings is not None:
        return _noise_embeddings

    try:
        from kb.extract_patterns import get_all_noise_terms
        noise_terms = get_all_noise_terms()
    except Exception:
        try:
            from kb.generate_domain_config import get_expanded_noise_terms
            noise_terms = get_expanded_noise_terms()
        except Exception:
            from agent.prompts import get_domain_config
            domain = get_domain_config()
            noise_terms = domain.get("noise_terms", [])

    if not noise_terms:
        _noise_embeddings = np.array([])
        return _noise_embeddings

    provider = _get_embedding_provider()
    _noise_embeddings = provider.encode(noise_terms, task_type="classification")
    logger.info(f"Noise embeddings gerados: {len(noise_terms)} termos")
    return _noise_embeddings


def _get_spam_indicators_from_domain() -> list[str]:
    """Pega indicadores de spam: base + gerados + extraídos do WhatsApp."""
    try:
        from kb.extract_patterns import get_all_noise_terms
        return get_all_noise_terms()
    except Exception:
        try:
            from kb.generate_domain_config import get_expanded_noise_terms
            return get_expanded_noise_terms()
        except Exception:
            from agent.prompts import get_domain_config
            domain = get_domain_config()
            return domain.get("noise_terms", [])


def is_semantic_noise(text: str, threshold: float = None) -> bool:
    """
    Verifica se texto é noise/spam via similaridade semântica com termos do domínio.
    Mais inteligente que lista fixa — detecta variações e paráfrases.
    """
    global _noise_embeddings

    if threshold is None:
        from config import get_setting
        threshold = get_setting("semantic", "noise_threshold", 0.75)
    noise_embs = _get_noise_embeddings()
    if noise_embs.size == 0:
        return False

    provider = _get_embedding_provider()
    text_emb = provider.encode(text, task_type="classification")

    # Verificar dimensão consistente (fallback pode mudar provider entre calls)
    if text_emb.shape[-1] != noise_embs.shape[-1]:
        # Regenerar noise embeddings com mesmo provider
        _noise_embeddings = None
        noise_embs = _get_noise_embeddings()
        if noise_embs.size == 0:
            return False

    # Calcular similaridade com todos os termos de noise
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(text_emb.reshape(1, -1), noise_embs)[0]

    max_sim = float(similarities.max())
    return max_sim >= threshold


def is_semantic_feedback(text: str, threshold: float = None) -> dict:
    """
    Detecta feedback via embeddings.
    Usa termos expandidos (base + gerados via LLM, se aprovados).
    Returns: {"is_feedback": bool, "type": "positive"|"negative"|"neutral", "score": float}
    """
    global _feedback_pos_embeddings, _feedback_neg_embeddings

    if threshold is None:
        from config import get_setting
        threshold = get_setting("semantic", "feedback_threshold", 0.7)

    provider = _get_embedding_provider()

    # Gerar embeddings de feedback (cached) — usa termos combinados (base + LLM + WhatsApp)
    if _feedback_pos_embeddings is None:
        try:
            from kb.extract_patterns import get_all_feedback_terms
            terms = get_all_feedback_terms()
            pos_terms = terms["positive"]
            neg_terms = terms["negative"]
        except Exception:
            try:
                from kb.generate_domain_config import get_expanded_feedback_terms
                terms = get_expanded_feedback_terms()
                pos_terms = terms["positive"]
                neg_terms = terms["negative"]
            except Exception:
                from agent.prompts import get_domain_config
                domain = get_domain_config()
                pos_terms = domain.get("feedback_positive", [])
                neg_terms = domain.get("feedback_negative", [])

        _feedback_pos_embeddings = provider.encode(pos_terms, task_type="classification") if pos_terms else np.array([])
        _feedback_neg_embeddings = provider.encode(neg_terms, task_type="classification") if neg_terms else np.array([])

    text_emb = provider.encode(text, task_type="classification").reshape(1, -1)

    # Verificar dimensão consistente
    if _feedback_pos_embeddings.size > 0 and text_emb.shape[-1] != _feedback_pos_embeddings.shape[-1]:
        _feedback_pos_embeddings = None
        _feedback_neg_embeddings = None
        return is_semantic_feedback(text, threshold)  # Retry com cache limpo

    from sklearn.metrics.pairwise import cosine_similarity

    pos_score = 0.0
    neg_score = 0.0

    if _feedback_pos_embeddings.size > 0:
        pos_score = float(cosine_similarity(text_emb, _feedback_pos_embeddings)[0].max())

    if _feedback_neg_embeddings.size > 0:
        neg_score = float(cosine_similarity(text_emb, _feedback_neg_embeddings)[0].max())

    if pos_score >= threshold and pos_score > neg_score:
        return {"is_feedback": True, "type": "positive", "score": pos_score}
    elif neg_score >= threshold and neg_score > pos_score:
        return {"is_feedback": True, "type": "negative", "score": neg_score}

    return {"is_feedback": False, "type": "neutral", "score": max(pos_score, neg_score)}


def clean_text(text: str) -> str:
    """Limpa texto individual: normaliza espaços e quebras."""
    # Normalizar quebras de linha excessivas
    text = EXCESSIVE_NEWLINES.sub('\n\n', text)
    # Remover espaços em branco no início/fim de cada linha
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def is_noise_message(text: str, is_isolated: bool = True) -> bool:
    """Verifica se a mensagem é ruído (sem valor informativo)."""
    cleaned = text.strip().lower()

    # Mensagem vazia
    if not cleaned:
        return True

    # Só link
    if ONLY_LINK_PATTERN.match(text):
        return True

    # Referência de mídia sem texto útil
    if cleaned.startswith("<media") and cleaned.endswith(">"):
        return True

    # Mensagens curtas sem contexto (apenas se isoladas)
    # Usa SHORT_NOISE expandido: YAML + LLM + extraído do WhatsApp
    try:
        from kb.extract_patterns import get_all_short_noise
        expanded_noise = get_all_short_noise()
    except Exception:
        try:
            from kb.generate_domain_config import get_expanded_short_noise
            expanded_noise = get_expanded_short_noise()
        except Exception:
            expanded_noise = get_short_noise()
    if is_isolated and cleaned in expanded_noise:
        return True

    # Excesso de emojis sem texto real
    without_emojis = EXCESSIVE_EMOJIS.sub('', cleaned)
    if len(without_emojis.strip()) < 3 and len(cleaned) > 5:
        return True

    return False


def is_spam(text: str) -> bool:
    """Detecta spam/broadcast por heurísticas + termos do domínio."""
    lower = text.lower()
    spam_indicators = _get_spam_indicators_from_domain()
    score = sum(1 for indicator in spam_indicators if indicator in lower)
    # Se 2+ indicadores, provavelmente spam
    return score >= 2


def clean_messages(messages: list[dict], remove_urls: bool = False, use_semantic: bool = False) -> list[dict]:
    """
    Limpa lista de mensagens.

    Args:
        messages: Lista de dicts com 'text', 'from_me', etc.
        remove_urls: Se True, remove URLs do texto (default False pois URLs podem ser relevantes)
        use_semantic: Se True, usa embeddings para detectar spam semanticamente (mais lento)

    Returns:
        Lista filtrada e limpa
    """
    cleaned = []
    total = len(messages)

    for i, msg in enumerate(messages):
        text = msg["text"]

        # Limpar texto
        text = clean_text(text)

        # Remover referências de mídia inline
        text = re.sub(r'<Media/No Text>', '', text)
        text = re.sub(r'<Media: [^>]+>', '', text).strip()

        if not text:
            continue

        # Remover URLs se configurado
        if remove_urls:
            text = URL_PATTERN.sub('', text).strip()
            if not text:
                continue

        # Detectar spam (heurístico)
        if is_spam(text):
            continue

        # Detecção semântica de noise (opcional, mais lento mas mais preciso)
        if use_semantic and len(text) > 20:
            if is_semantic_noise(text, threshold=0.8):
                continue

        # Verificar se é ruído (checar se isolada - sem msgs do mesmo autor ao redor)
        is_isolated = True
        if i > 0 and messages[i-1].get("from_me") == msg.get("from_me"):
            is_isolated = False
        if i < total - 1 and messages[i+1].get("from_me") == msg.get("from_me"):
            is_isolated = False

        if is_noise_message(text, is_isolated=is_isolated):
            continue

        cleaned_msg = {**msg, "text": text}
        cleaned.append(cleaned_msg)

    removed = total - len(cleaned)
    logger.info(f"Limpeza: {total} → {len(cleaned)} mensagens ({removed} removidas)")
    return cleaned


def merge_consecutive(messages: list[dict]) -> list[dict]:
    """
    Agrupa mensagens consecutivas do mesmo autor.

    Entrada: lista de msgs ordenadas por timestamp
    Saída: lista de blocos merged com 'author' ('me' ou 'client'), 'text', 'timestamp_start', 'timestamp_end'
    """
    if not messages:
        return []

    merged = []

    for msg in messages:
        author = "me" if msg["from_me"] else "client"
        text = msg["text"].strip()

        if not text:
            continue

        if merged and merged[-1]["author"] == author:
            merged[-1]["text"] += "\n" + text
            merged[-1]["timestamp_end"] = msg["timestamp"]
            merged[-1]["message_ids"].append(msg.get("message_id", ""))
        else:
            merged.append({
                "author": author,
                "text": text,
                "timestamp_start": msg["timestamp"],
                "timestamp_end": msg["timestamp"],
                "message_ids": [msg.get("message_id", "")],
                "chat_id": msg.get("chat_id", ""),
                "subject": msg.get("subject", ""),
            })

    logger.info(f"Merge: {len(messages)} msgs → {len(merged)} blocos")
    return merged


if __name__ == "__main__":
    import os, json
    from dotenv import load_dotenv
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.whatsapp_loader import load_whatsapp_auto

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    messages = load_whatsapp_auto()
    cleaned = clean_messages(messages)
    merged = merge_consecutive(cleaned)

    print(f"\nResultado: {len(merged)} blocos merged")
    if merged:
        print(f"Primeiro bloco ({merged[0]['author']}): {merged[0]['text'][:100]}...")
