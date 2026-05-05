"""
Extração de padrões (noise, spam, feedback) diretamente dos dados brutos do WhatsApp.

Usa embeddings para:
1. Identificar quais conversas pertencem ao domínio configurado
2. Extrair SHORT_NOISE (msgs curtas sem valor)
3. Extrair SPAM_INDICATORS (padrões de spam reais)
4. Extrair feedback_positive/negative (como clientes reais expressam satisfação/insatisfação)

Todas as extrações passam por embeddings para classificação semântica.
"""
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def extract_patterns_from_whatsapp(
    json_path: str = None,
    db_path: str = None,
    output_path: str = None,
    min_domain_score: int = 1,
) -> dict:
    """
    Extrai padrões de noise, spam e feedback das conversas reais do WhatsApp
    que pertencem ao domínio configurado.

    Fontes suportadas:
    - json_path: JSON exportado (formato direto ou bot-zdg)
    - db_path: msgstore.db decriptado (SQLite)
    - Auto-detect via env vars WHATSAPP_SOURCE, WHATSAPP_DB, WHATSAPP_JSON

    Fluxo:
    1. Filtra chats do domínio (por keywords + embeddings)
    2. Classifica msgs curtas como SHORT_NOISE
    3. Detecta padrões de SPAM/broadcast
    4. Identifica feedback positivo/negativo via embeddings

    Returns:
        Dict com os padrões extraídos (needs_human_review=True)
    """
    from agent.prompts import get_domain_config

    data_dir = os.getenv("DATA_DIR", "./data")
    output_path = output_path or f"{data_dir}/extracted_patterns.json"

    domain = get_domain_config()

    # Resolver fonte de dados
    all_chats = _load_chats_data(json_path=json_path, db_path=db_path)
    logger.info(f"Extraindo padrões do domínio '{domain['name']}' ({len(all_chats)} chats)")

    # 1. Carregar e filtrar chats do domínio
    domain_chats = _filter_domain_chats(all_chats, domain, min_domain_score)
    logger.info(f"Chats do domínio: {len(domain_chats)} de {len(all_chats)} total")

    if not domain_chats:
        logger.warning("Nenhum chat do domínio encontrado!")
        return {}

    # 2. Extrair todas as mensagens do domínio
    all_msgs = []
    for chat in domain_chats:
        for msg in chat.get("messages", []):
            text = (msg.get("text") or "").strip()
            if text:
                all_msgs.append({
                    "text": text,
                    "from_me": msg.get("from_me", False),
                    "chat_id": chat.get("jid", ""),
                })

    logger.info(f"Total de mensagens no domínio: {len(all_msgs)}")

    # 3. Extrair padrões
    short_noise = _extract_short_noise(all_msgs)
    spam_indicators = _extract_spam_indicators(all_msgs)
    feedback_pos, feedback_neg = _extract_feedback_patterns(all_msgs, domain)

    result = {
        "domain": domain["name"],
        "source": "whatsapp_extraction",
        "needs_human_review": True,
        "approved": False,
        "total_chats_analyzed": len(domain_chats),
        "total_msgs_analyzed": len(all_msgs),
        "short_noise": short_noise,
        "spam_indicators": spam_indicators,
        "feedback_positive": feedback_pos,
        "feedback_negative": feedback_neg,
    }

    # Salvar
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Padrões extraídos e salvos em {output_path}:\n"
        f"  short_noise: {len(short_noise)}\n"
        f"  spam_indicators: {len(spam_indicators)}\n"
        f"  feedback_positive: {len(feedback_pos)}\n"
        f"  feedback_negative: {len(feedback_neg)}\n"
        f"  ⚠️  PRECISA DE REVISÃO (approved=False)"
    )
    return result


# ==================== CARREGAMENTO DE DADOS ====================

def _load_chats_data(json_path: str = None, db_path: str = None) -> list[dict]:
    """
    Carrega dados de chats em formato agrupado (lista de chats com messages).

    Prioridade:
    1. Se db_path fornecido e existe → msgstore.db
    2. Se json_path fornecido e existe → JSON
    3. Auto-detect via env vars

    Retorna lista no formato: [{id, jid, subject, messages: [{text, timestamp, from_me, sender}]}]
    """
    source_mode = os.getenv("WHATSAPP_SOURCE", "auto").lower()

    if not db_path:
        db_path = os.getenv("WHATSAPP_DB", "./input/msgstore.db")
    if not json_path:
        json_path = os.getenv("WHATSAPP_JSON", "./input/whatsapp_chats.json")

    # Tentar msgstore.db
    if source_mode == "db" or (source_mode == "auto" and os.path.exists(db_path)):
        if os.path.exists(db_path):
            logger.info(f"Fonte: msgstore.db ({db_path})")
            return _load_chats_from_db(db_path)

    # Fallback: JSON
    if os.path.exists(json_path):
        logger.info(f"Fonte: JSON ({json_path})")
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)

        # Normalizar formato bot-zdg para formato direto
        if isinstance(raw, dict) and "conversations" in raw:
            return _normalize_botzdg_to_chats(raw)
        return raw

    raise FileNotFoundError(
        f"Nenhuma fonte encontrada: JSON ({json_path}) ou DB ({db_path})"
    )


def _load_chats_from_db(db_path: str) -> list[dict]:
    """Carrega chats agrupados do msgstore.db."""
    from ingestion.msgstore_loader import MsgStoreReader

    reader = MsgStoreReader(db_path)
    if not reader.connect():
        raise FileNotFoundError(f"Não foi possível abrir: {db_path}")

    try:
        chats = reader.list_chats()
        result = []

        for chat in chats:
            messages = reader.get_messages(chat["id"], chat["jid"])
            if not messages:
                continue

            result.append({
                "id": chat["jid"],
                "jid": chat["jid"],
                "subject": chat["subject"],
                "messages": [
                    {
                        "text": msg["text"],
                        "timestamp": msg["timestamp"],
                        "from_me": msg["from_me"],
                        "sender": msg["sender"],
                    }
                    for msg in messages
                    if msg.get("text")
                ],
            })

        return result
    finally:
        reader.close()


def _normalize_botzdg_to_chats(raw: dict) -> list[dict]:
    """Converte formato bot-zdg para formato direto de chats."""
    result = []
    for conv in raw.get("conversations", []):
        messages = []
        for msg in conv.get("messages", []):
            text = (msg.get("body") or "").strip()
            if not text:
                continue
            messages.append({
                "text": text,
                "timestamp": msg.get("timestamp", 0),
                "from_me": msg.get("fromMe", False),
                "sender": msg.get("from", ""),
            })
        if messages:
            result.append({
                "id": conv.get("id", ""),
                "jid": conv.get("id", ""),
                "subject": conv.get("name", ""),
                "messages": messages,
            })
    return result


# ==================== FILTRO DE DOMÍNIO ====================

def _filter_domain_chats(all_chats: list, domain: dict, min_score: int = 1) -> list:
    """Filtra chats que pertencem ao domínio configurado."""
    # Keywords do domínio para filtro rápido
    keywords = [p.lower() for p in domain.get("products", [])]
    keywords += [k.lower() for k in domain.get("sale_keywords", [])[:10]]
    keywords += [k.lower() for k in domain.get("support_keywords", [])[:10]]

    # Filtro por keyword (rápido)
    domain_chats = []
    for chat in all_chats:
        # Concatenar primeiras N mensagens para classificação
        msgs = chat.get("messages", [])
        sample_text = " ".join(
            (m.get("text") or "")[:200] for m in msgs[:100]
        ).lower()

        score = sum(1 for kw in keywords if kw in sample_text)
        if score >= min_score:
            domain_chats.append(chat)

    return domain_chats


# ==================== EXTRAÇÃO DE SHORT_NOISE ====================

def _extract_short_noise(msgs: list[dict], max_len: int = None) -> list[str]:
    """
    Extrai mensagens curtas frequentes que são noise (sem valor informativo).
    Analisa msgs curtas (< max_len chars) e usa frequência + embeddings para classificar.
    """
    from config import get_setting, load_domain_config

    if max_len is None:
        max_len = get_setting("extraction", "short_noise_max_len", 15)
    min_freq = get_setting("extraction", "short_noise_min_frequency", 3)
    noise_threshold = get_setting("extraction", "noise_similarity_threshold", 0.5)
    max_results = get_setting("extraction", "max_short_noise", 50)

    domain = load_domain_config()
    refs = domain.get("references", {})

    # Coletar msgs curtas dos CLIENTES (não do bot)
    short_msgs = Counter()
    for msg in msgs:
        text = msg["text"].strip().lower()
        # Mensagens curtas de clientes
        if len(text) <= max_len and not msg["from_me"]:
            # Normalizar
            normalized = re.sub(r'[.!?,;:]+$', '', text).strip()
            if normalized and len(normalized) >= 1:
                short_msgs[normalized] += 1

    # Filtrar por frequência
    frequent_short = {text for text, count in short_msgs.items() if count >= min_freq}

    if not frequent_short:
        return list(short_msgs.keys())[:max_results]

    # Usar embeddings para classificar quais são realmente noise vs pergunta curta válida
    from llm.embeddings import get_embedding_provider

    provider = get_embedding_provider()
    candidates = list(frequent_short)

    # Referências do YAML
    noise_reference = refs.get("noise", [
        "ok", "sim", "não", "beleza", "certo", "entendi",
        "👍", "obrigado", "valeu", "blz", "haha", "kkk",
    ])
    valid_reference = refs.get("valid_short", [
        "quanto custa", "como funciona", "tem disponível",
        "qual o preço", "preciso de ajuda", "não funciona",
    ])

    try:
        noise_embs = provider.encode(noise_reference, task_type="classification")
        valid_embs = provider.encode(valid_reference, task_type="classification")
        candidate_embs = provider.encode(candidates, task_type="classification")

        # Verificar consistência de dimensão
        if noise_embs.shape[1] != candidate_embs.shape[1]:
            # Provider mudou entre calls (rate limit), retry tudo com o mesmo
            all_texts = noise_reference + valid_reference + candidates
            all_embs = provider.encode(all_texts, task_type="classification")
            n1 = len(noise_reference)
            n2 = n1 + len(valid_reference)
            noise_embs = all_embs[:n1]
            valid_embs = all_embs[n1:n2]
            candidate_embs = all_embs[n2:]

        from sklearn.metrics.pairwise import cosine_similarity

        noise_sims = cosine_similarity(candidate_embs, noise_embs).max(axis=1)
        valid_sims = cosine_similarity(candidate_embs, valid_embs).max(axis=1)

        # É noise se mais similar a noise do que a msgs válidas
        extracted_noise = []
        for i, text in enumerate(candidates):
            if noise_sims[i] > valid_sims[i] and noise_sims[i] > noise_threshold:
                extracted_noise.append(text)

        return sorted(extracted_noise)[:max_results]

    except Exception as e:
        logger.warning(f"Falha no embedding para short_noise, usando frequência: {e}")
        # Fallback: retornar os mais frequentes
        return sorted(frequent_short)[:max_results]


# ==================== EXTRAÇÃO DE SPAM ====================

def _extract_spam_indicators(msgs: list[dict], min_len: int = None) -> list[str]:
    """
    Extrai padrões de spam/broadcast das conversas.
    Spam geralmente: msgs longas, repetidas em múltiplos chats, com links, emojis excessivos.
    """
    from config import get_setting

    if min_len is None:
        min_len = get_setting("extraction", "spam_min_len", 30)
    max_results = get_setting("extraction", "max_spam_indicators", 30)
    # Coletar msgs longas de NÃO-bot (spam vem de outros no grupo)
    long_msgs = []
    msg_texts = Counter()

    for msg in msgs:
        text = msg["text"].strip()
        if len(text) >= min_len and not msg["from_me"]:
            long_msgs.append(text)
            # Contar repetições (normalizado)
            normalized = text[:100].lower().strip()
            msg_texts[normalized] += 1

    # Indicadores heurísticos de spam
    spam_patterns = []
    url_pattern = re.compile(r'https?://\S+')
    forward_pattern = re.compile(r'(encaminh|compartilh|repass|divulg)', re.IGNORECASE)
    promo_pattern = re.compile(r'(promoção|desconto|grátis|sorteio|prêmio|ganhe)', re.IGNORECASE)

    seen_indicators = set()

    for text in long_msgs:
        score = 0
        lower = text.lower()

        # Links excessivos
        urls = url_pattern.findall(text)
        if len(urls) >= 2:
            score += 2

        # Padrão de forward/compartilhamento
        if forward_pattern.search(text):
            score += 2

        # Promoção não relacionada ao domínio
        if promo_pattern.search(text) and not any(
            p.lower() in lower for p in ["android box", "iptv", "xciptv", "tirzepatida", "tirzec"]
        ):
            score += 2

        # Emojis excessivos (5+)
        emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', text))
        if emoji_count >= 5:
            score += 1

        # Msg repetida em múltiplos contextos
        if msg_texts.get(text[:100].lower().strip(), 0) >= 3:
            score += 2

        if score >= 3:
            # Extrair o "indicador" (substring chave)
            indicator = _extract_spam_indicator(text)
            if indicator and indicator not in seen_indicators:
                seen_indicators.add(indicator)
                spam_patterns.append(indicator)

    # Usar embeddings para validar
    if spam_patterns:
        spam_patterns = _validate_spam_with_embeddings(spam_patterns)

    return spam_patterns[:max_results]


def _extract_spam_indicator(text: str) -> str:
    """Extrai a substring-chave que indica spam de um texto longo."""
    lower = text.lower()

    # Procurar frases típicas de spam
    patterns = [
        r'(encaminhe\s+(?:para|essa|esta).{0,30})',
        r'(compartilhe\s+(?:com|essa|esta).{0,30})',
        r'(clique\s+(?:aqui|no link).{0,30})',
        r'(promoção\s+.{0,40})',
        r'(sorteio\s+.{0,30})',
        r'(ganhe\s+.{0,30})',
        r'(você\s+foi\s+(?:sorteado|selecionado).{0,30})',
    ]

    for p in patterns:
        match = re.search(p, lower)
        if match:
            return match.group(1).strip()[:60]

    # Fallback: primeiros 60 chars se não encontrou padrão
    clean = re.sub(r'\s+', ' ', lower[:60]).strip()
    return clean if len(clean) >= 20 else ""


def _validate_spam_with_embeddings(candidates: list[str]) -> list[str]:
    """Valida se os candidatos são realmente spam via embeddings."""
    try:
        from config import get_setting, load_domain_config
        from llm.embeddings import get_embedding_provider

        provider = get_embedding_provider()
        spam_threshold = get_setting("extraction", "spam_similarity_threshold", 0.4)

        domain = load_domain_config()
        refs = domain.get("references", {})

        spam_reference = refs.get("spam", [
            "encaminhe essa mensagem para 10 pessoas",
            "clique no link para ganhar prêmio",
            "promoção imperdível de celular grátis",
            "você foi sorteado",
            "compartilhe com seus amigos",
        ])

        # Encode tudo em uma chamada para garantir consistência de dimensão
        all_texts = candidates + spam_reference
        all_embs = provider.encode(all_texts, task_type="classification")
        candidate_embs = all_embs[:len(candidates)]
        ref_embs = all_embs[len(candidates):]

        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(candidate_embs, ref_embs).max(axis=1)

        # Manter apenas os que são similares a spam de referência
        validated = [candidates[i] for i in range(len(candidates)) if sims[i] > spam_threshold]
        return validated if validated else candidates[:10]

    except Exception as e:
        logger.warning(f"Falha na validação de spam via embeddings: {e}")
        return candidates


# ==================== EXTRAÇÃO DE FEEDBACK ====================

def _extract_feedback_patterns(msgs: list[dict], domain: dict) -> tuple[list[str], list[str]]:
    """
    Extrai padrões de feedback positivo e negativo das conversas reais.
    
    Identifica msgs de clientes que:
    - Vêm DEPOIS de uma resposta do bot/atendente (from_me=True)
    - Expressam satisfação ou insatisfação
    
    Usa embeddings para classificar como positivo ou negativo.
    """
    # Coletar candidatos a feedback:
    # Msgs de cliente que aparecem logo após resposta do bot
    feedback_candidates = []

    # Agrupar por chat
    chats_msgs = {}
    for msg in msgs:
        chat_id = msg["chat_id"]
        if chat_id not in chats_msgs:
            chats_msgs[chat_id] = []
        chats_msgs[chat_id].append(msg)

    for chat_id, chat_msgs in chats_msgs.items():
        for i in range(1, len(chat_msgs)):
            prev = chat_msgs[i - 1]
            curr = chat_msgs[i]

            # Cliente responde após bot
            if prev["from_me"] and not curr["from_me"]:
                text = curr["text"].strip()
                # Msgs de 3-80 chars são bons candidatos a feedback
                if 3 <= len(text) <= 80:
                    feedback_candidates.append(text.lower())

    if not feedback_candidates:
        logger.warning("Nenhum candidato a feedback encontrado")
        return [], []

    # Desduplicar e pegar os mais frequentes
    from config import get_setting
    max_candidates = get_setting("extraction", "max_feedback_candidates", 200)
    counter = Counter(feedback_candidates)
    # Pegar top N únicos para classificar via embeddings
    unique_candidates = [text for text, _ in counter.most_common(max_candidates)]

    logger.info(f"Candidatos a feedback: {len(unique_candidates)} únicos (de {len(feedback_candidates)} total)")

    # Classificar via embeddings
    return _classify_feedback_with_embeddings(unique_candidates, domain)


def _classify_feedback_with_embeddings(
    candidates: list[str], domain: dict
) -> tuple[list[str], list[str]]:
    """Classifica candidatos como feedback positivo ou negativo via embeddings."""
    try:
        from config import get_setting, load_domain_config
        from llm.embeddings import get_embedding_provider

        provider = get_embedding_provider()
        feedback_threshold = get_setting("extraction", "feedback_similarity_threshold", 0.55)
        max_per_type = get_setting("extraction", "max_feedback_per_type", 30)

        domain_cfg = load_domain_config()
        refs = domain_cfg.get("references", {})

        # Referências do YAML
        pos_reference = refs.get("feedback_positive", [
            "resolveu meu problema", "deu certo", "funcionou", "obrigado ajudou",
            "perfeito", "show de bola", "tá funcionando agora", "muito bom",
            "excelente atendimento", "voltou ao normal", "maravilha",
        ])
        neg_reference = refs.get("feedback_negative", [
            "não resolveu", "continua com problema", "piorou", "não funcionou",
            "mesmo erro", "ainda não funciona", "não adiantou", "voltou o problema",
            "péssimo", "horrível", "não ajudou nada",
        ])
        neutral_reference = refs.get("neutral", [
            "quanto custa", "como funciona", "quero comprar",
            "tem disponível", "qual o prazo", "bom dia",
        ])

        # Encode tudo junto para garantir consistência de dimensão
        all_texts = candidates + pos_reference + neg_reference + neutral_reference
        all_embs = provider.encode(all_texts, task_type="classification")

        n_cand = len(candidates)
        n_pos = len(pos_reference)
        n_neg = len(neg_reference)

        candidate_embs = all_embs[:n_cand]
        pos_embs = all_embs[n_cand:n_cand + n_pos]
        neg_embs = all_embs[n_cand + n_pos:n_cand + n_pos + n_neg]
        neutral_embs = all_embs[n_cand + n_pos + n_neg:]

        from sklearn.metrics.pairwise import cosine_similarity

        pos_sims = cosine_similarity(candidate_embs, pos_embs).max(axis=1)
        neg_sims = cosine_similarity(candidate_embs, neg_embs).max(axis=1)
        neutral_sims = cosine_similarity(candidate_embs, neutral_embs).max(axis=1)

        feedback_positive = []
        feedback_negative = []

        for i, text in enumerate(candidates):
            # Threshold: deve ser mais similar a feedback do que a neutro
            max_feedback = max(pos_sims[i], neg_sims[i])
            if max_feedback <= neutral_sims[i]:
                continue  # É neutro, não é feedback

            if pos_sims[i] > neg_sims[i] and pos_sims[i] > feedback_threshold:
                feedback_positive.append(text)
            elif neg_sims[i] > pos_sims[i] and neg_sims[i] > feedback_threshold:
                feedback_negative.append(text)

        return feedback_positive[:max_per_type], feedback_negative[:max_per_type]

    except Exception as e:
        logger.warning(f"Falha na classificação de feedback via embeddings: {e}")
        # Fallback heurístico
        return _heuristic_feedback_classification(candidates, domain)


def _heuristic_feedback_classification(
    candidates: list[str], domain: dict
) -> tuple[list[str], list[str]]:
    """Fallback heurístico se embeddings falharem."""
    pos_keywords = {"deu certo", "funcionou", "resolveu", "obrigado", "valeu", "top", "show", "perfeito"}
    neg_keywords = {"não resolveu", "não funcionou", "continua", "piorou", "mesmo problema", "não deu"}

    pos_results = []
    neg_results = []

    for text in candidates:
        if any(kw in text for kw in pos_keywords):
            pos_results.append(text)
        elif any(kw in text for kw in neg_keywords):
            neg_results.append(text)

    return pos_results[:30], neg_results[:30]


# ==================== INTEGRAÇÃO COM SISTEMA ====================

def load_extracted_patterns(only_approved: bool = True) -> dict | None:
    """Carrega padrões extraídos do WhatsApp (se existirem e estiverem aprovados)."""
    data_dir = os.getenv("DATA_DIR", "./data")
    path = f"{data_dir}/extracted_patterns.json"

    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        patterns = json.load(f)

    if only_approved and not patterns.get("approved", False):
        return None

    return patterns


def get_all_noise_terms() -> list[str]:
    """
    Retorna TODOS os noise terms combinados:
    1. Base do DOMAIN_CONFIG
    2. Gerados pelo LLM (generate_domain_config, se aprovado)
    3. Extraídos do WhatsApp (este módulo, se aprovado)
    """
    from agent.prompts import get_domain_config
    domain = get_domain_config()
    terms = list(domain.get("noise_terms", []))

    # Do LLM
    try:
        from kb.generate_domain_config import load_generated_config
        generated = load_generated_config(only_approved=True)
        if generated:
            terms += generated.get("noise_terms", [])
            terms += generated.get("spam_indicators", [])
    except Exception:
        pass

    # Do WhatsApp
    extracted = load_extracted_patterns(only_approved=True)
    if extracted:
        terms += extracted.get("spam_indicators", [])

    return list(set(terms))


def get_all_short_noise() -> set[str]:
    """
    Retorna SHORT_NOISE combinado: base (YAML) + gerado + extraído.
    """
    from preprocessing.cleaner import get_short_noise
    result = set(get_short_noise())

    # Do LLM
    try:
        from kb.generate_domain_config import load_generated_config
        generated = load_generated_config(only_approved=True)
        if generated:
            result.update(s.lower() for s in generated.get("short_noise", []))
    except Exception:
        pass

    # Do WhatsApp
    extracted = load_extracted_patterns(only_approved=True)
    if extracted:
        result.update(s.lower() for s in extracted.get("short_noise", []))

    return result


def get_all_feedback_terms() -> dict:
    """
    Retorna feedback terms combinados de todas as fontes.
    Returns: {"positive": [...], "negative": [...]}
    """
    from agent.prompts import get_domain_config
    domain = get_domain_config()

    positive = set(domain.get("feedback_positive", []))
    negative = set(domain.get("feedback_negative", []))

    # Do LLM
    try:
        from kb.generate_domain_config import load_generated_config
        generated = load_generated_config(only_approved=True)
        if generated:
            positive.update(generated.get("feedback_positive", []))
            negative.update(generated.get("feedback_negative", []))
    except Exception:
        pass

    # Do WhatsApp
    extracted = load_extracted_patterns(only_approved=True)
    if extracted:
        positive.update(extracted.get("feedback_positive", []))
        negative.update(extracted.get("feedback_negative", []))

    return {"positive": sorted(positive), "negative": sorted(negative)}


# ==================== CLI ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "approve":
        data_dir = os.getenv("DATA_DIR", "./data")
        path = f"{data_dir}/extracted_patterns.json"

        if not os.path.exists(path):
            print("❌ Nenhum padrão extraído. Execute sem argumentos primeiro.")
            sys.exit(1)

        with open(path, encoding="utf-8") as f:
            patterns = json.load(f)

        if patterns.get("approved"):
            print("✓ Padrões já aprovados.")
            sys.exit(0)

        print(f"=== Padrões extraídos do WhatsApp ({patterns['domain']}) ===")
        print(f"  Chats analisados: {patterns.get('total_chats_analyzed', '?')}")
        print(f"  Msgs analisadas: {patterns.get('total_msgs_analyzed', '?')}")
        print()

        print(f"short_noise ({len(patterns.get('short_noise', []))}):")
        for t in patterns.get("short_noise", [])[:15]:
            print(f"  - {t}")
        if len(patterns.get("short_noise", [])) > 15:
            print(f"  ... +{len(patterns['short_noise'])-15} mais")

        print(f"\nspam_indicators ({len(patterns.get('spam_indicators', []))}):")
        for t in patterns.get("spam_indicators", [])[:10]:
            print(f"  - {t}")

        print(f"\nfeedback_positive ({len(patterns.get('feedback_positive', []))}):")
        for t in patterns.get("feedback_positive", [])[:10]:
            print(f"  - {t}")

        print(f"\nfeedback_negative ({len(patterns.get('feedback_negative', []))}):")
        for t in patterns.get("feedback_negative", [])[:10]:
            print(f"  - {t}")

        print("\n" + "=" * 60)
        resp = input("Aprovar estes padrões extraídos? (s/n): ").strip().lower()

        if resp in ("s", "sim", "y", "yes"):
            patterns["approved"] = True
            patterns["needs_human_review"] = False
            with open(path, "w", encoding="utf-8") as f:
                json.dump(patterns, f, ensure_ascii=False, indent=2)
            print("✓ Padrões APROVADOS. Serão usados na detecção semântica.")
        else:
            print("✗ Não aprovado. Edite manualmente ou re-execute.")

    else:
        print("=== Extraindo padrões do WhatsApp ===\n")
        result = extract_patterns_from_whatsapp()
        if result:
            print(f"\n✓ Extração concluída!")
            print(f"  short_noise: {len(result.get('short_noise', []))}")
            print(f"  spam_indicators: {len(result.get('spam_indicators', []))}")
            print(f"  feedback_positive: {len(result.get('feedback_positive', []))}")
            print(f"  feedback_negative: {len(result.get('feedback_negative', []))}")
            print(f"\n⚠️  Execute 'python -m kb.extract_patterns approve' para revisar e aprovar.")
        else:
            print("❌ Nenhum padrão extraído.")
