"""
Geração de configurações de domínio via LLM.

Gera automaticamente:
- noise_terms / spam_indicators
- feedback_positive / feedback_negative
- KB entries (symptoms, steps, examples, recommended_response)

Tudo é gerado com needs_human_review=True e salvo em data/generated_domain_config.json.
Após revisão/aprovação, o sistema usa esses termos expandidos para detecção semântica.
"""
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

GENERATED_CONFIG_FILE = "generated_domain_config.json"


def generate_domain_terms(output_path: str = None) -> dict:
    """
    Usa o LLM para gerar termos de noise, spam, feedback e KB entries
    específicos do domínio configurado.

    Tudo é marcado com needs_human_review=True.

    Returns:
        Dict com: noise_terms, spam_indicators, feedback_positive, 
        feedback_negative, short_noise, kb_entries
    """
    from agent.prompts import get_domain_config
    from llm.providers import get_default_provider

    data_dir = os.getenv("DATA_DIR", "./data")
    if output_path is None:
        output_path = f"{data_dir}/{GENERATED_CONFIG_FILE}"

    domain = get_domain_config()
    llm = get_default_provider()

    prompt = _build_generation_prompt(domain)

    logger.info(f"Gerando termos do domínio '{domain['name']}' via LLM ({llm.name()})...")

    messages = [
        {"role": "system", "content": "Você é um especialista em configuração de chatbots. Responda APENAS com JSON válido, sem texto antes ou depois."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = llm.chat(messages, temperature=0.7, max_tokens=4096)
        generated = _parse_json_object(response)
    except Exception as e:
        logger.error(f"Erro ao gerar termos do domínio: {e}")
        return {}

    if not generated:
        logger.warning("Nenhum termo gerado (resposta não era JSON válido)")
        return {}

    # Estruturar resultado com review flag
    result = {
        "domain": domain["name"],
        "needs_human_review": True,
        "approved": False,
        "noise_terms": generated.get("noise_terms", []),
        "spam_indicators": generated.get("spam_indicators", []),
        "short_noise": generated.get("short_noise", []),
        "feedback_positive": generated.get("feedback_positive", []),
        "feedback_negative": generated.get("feedback_negative", []),
        "feedback_responses": generated.get("feedback_responses", {}),
        "kb_entries": generated.get("kb_entries", []),
    }

    # Marcar cada KB entry individualmente
    for entry in result["kb_entries"]:
        entry["needs_human_review"] = True
        entry["generated"] = True
        entry["confidence"] = 0.5

    # Salvar
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Configuração gerada e salva em {output_path}:\n"
        f"  noise_terms: {len(result['noise_terms'])}\n"
        f"  spam_indicators: {len(result['spam_indicators'])}\n"
        f"  short_noise: {len(result['short_noise'])}\n"
        f"  feedback_positive: {len(result['feedback_positive'])}\n"
        f"  feedback_negative: {len(result['feedback_negative'])}\n"
        f"  kb_entries: {len(result['kb_entries'])}\n"
        f"  ⚠️  PRECISA DE REVISÃO HUMANA (approved=False)"
    )
    return result


def load_generated_config(only_approved: bool = True) -> dict | None:
    """
    Carrega configuração gerada (se existir e estiver aprovada).
    
    Args:
        only_approved: Se True, retorna None se não estiver aprovado.
                       Se False, retorna mesmo sem aprovação (para review).
    """
    data_dir = os.getenv("DATA_DIR", "./data")
    config_path = f"{data_dir}/{GENERATED_CONFIG_FILE}"

    if not os.path.exists(config_path):
        return None

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if only_approved and not config.get("approved", False):
        return None

    return config


def get_expanded_noise_terms() -> list[str]:
    """
    Retorna noise_terms expandidos: base do DOMAIN_CONFIG + gerados (se aprovados).
    """
    from agent.prompts import get_domain_config
    domain = get_domain_config()
    base_terms = domain.get("noise_terms", [])

    generated = load_generated_config(only_approved=True)
    if generated:
        expanded = generated.get("noise_terms", []) + generated.get("spam_indicators", [])
        return list(set(base_terms + expanded))

    return base_terms


def get_expanded_short_noise() -> set[str]:
    """
    Retorna SHORT_NOISE expandido: base (YAML) + gerado (se aprovado).
    """
    from preprocessing.cleaner import get_short_noise
    base = set(get_short_noise())

    generated = load_generated_config(only_approved=True)
    if generated:
        extra = generated.get("short_noise", [])
        base.update(s.lower().strip() for s in extra)

    return base


def get_expanded_feedback_terms() -> dict:
    """
    Retorna termos de feedback expandidos: base + gerados (se aprovados).
    Returns: {"positive": [...], "negative": [...]}
    """
    from agent.prompts import get_domain_config
    domain = get_domain_config()

    positive = list(domain.get("feedback_positive", []))
    negative = list(domain.get("feedback_negative", []))

    generated = load_generated_config(only_approved=True)
    if generated:
        positive = list(set(positive + generated.get("feedback_positive", [])))
        negative = list(set(negative + generated.get("feedback_negative", [])))

    return {"positive": positive, "negative": negative}


def get_feedback_responses() -> dict:
    """
    Retorna respostas de feedback geradas (se aprovadas).
    Returns: {"positive": "...", "negative_with_docs": "...", "negative_no_docs": "...", "neutral": "..."}
    """
    generated = load_generated_config(only_approved=True)
    if generated and generated.get("feedback_responses"):
        return generated["feedback_responses"]

    # Defaults
    return {
        "positive": "Que bom que deu certo! 😊 Se precisar de algo mais, é só chamar.",
        "negative_with_docs": "Entendo que não resolveu. Deixa eu tentar de outra forma.\n\nPode me descrever melhor o que está acontecendo para eu verificar outras opções?",
        "negative_no_docs": "Entendo que não resolveu. Vou encaminhar para suporte humano para verificar melhor seu caso.",
        "neutral": "Entendi! Posso ajudar com mais alguma coisa?",
    }


def get_generated_kb_entries(only_approved: bool = True) -> list[dict]:
    """
    Retorna KB entries geradas pelo LLM (para merge com a KB principal).
    """
    generated = load_generated_config(only_approved=only_approved)
    if not generated:
        return []
    return generated.get("kb_entries", [])


# ==================== PROMPT DE GERAÇÃO ====================

def _build_generation_prompt(domain: dict) -> str:
    """Constrói o prompt para o LLM gerar termos e KB entries do domínio."""
    products = ", ".join(domain["products"])
    
    return f"""Você é especialista no domínio: **{domain['name']}** ({domain['description']}).

Gere um JSON com termos e configurações para um chatbot de atendimento deste domínio.
O chatbot precisa detectar spam, noise, feedback, e responder perguntas comuns.

Produtos/termos do domínio: {products}

Gere o seguinte JSON (todos os campos obrigatórios):

{{
  "noise_terms": [
    "lista de 15-20 frases que indicam SPAM ou mensagens irrelevantes neste domínio",
    "ex: mensagens de corrente, propaganda não relacionada, links aleatórios",
    "devem ser frases/padrões que aparecem em grupos de WhatsApp mas não são sobre {domain['name']}"
  ],
  "spam_indicators": [
    "lista de 10-15 indicadores de spam/broadcast específicos",
    "ex: 'encaminhe para 10 amigos', 'clique no link', 'promoção imperdível de [outro produto]'"
  ],
  "short_noise": [
    "lista de 20-30 mensagens curtas sem valor (1-2 palavras)",
    "ex: 'ok', 'sim', 'blz', emojis isolados, risadas"
  ],
  "feedback_positive": [
    "lista de 15-20 frases que indicam que o problema do cliente FOI resolvido",
    "específicas para {domain['name']}",
    "ex: 'voltou a funcionar', 'canal voltou', 'app rodou'"
  ],
  "feedback_negative": [
    "lista de 15-20 frases que indicam que o problema NÃO foi resolvido",
    "específicas para {domain['name']}",
    "ex: 'continua travando', 'canal não voltou', 'mesma tela preta'"
  ],
  "feedback_responses": {{
    "positive": "resposta curta quando cliente diz que resolveu (max 2 frases)",
    "negative_with_docs": "resposta quando não resolveu mas temos alternativas na base (pedir mais detalhes)",
    "negative_no_docs": "resposta quando não resolveu e não temos alternativa (encaminhar para humano)",
    "neutral": "resposta genérica para feedback neutro"
  }},
  "kb_entries": [
    {{
      "category": "support",
      "intent": "identificador_sem_espacos",
      "title": "Título do problema comum",
      "symptoms": ["sintoma 1", "sintoma 2", "sintoma 3"],
      "recommended_response": "Resposta completa que o atendente deve dar",
      "steps": ["Passo 1 diagnóstico", "Passo 2", "Passo 3", "Se não resolver, encaminhar humano"],
      "examples": ["exemplo de como cliente pergunta 1", "exemplo 2", "exemplo 3"]
    }}
  ]
}}

REGRAS IMPORTANTES:
1. noise_terms devem ser frases/padrões de SPAM em grupos WhatsApp (não relacionados a {domain['name']})
2. feedback deve ser específico ao domínio (ex: "canal voltou" para IPTV, não genérico)
3. kb_entries: gere 10-15 problemas REAIS e COMUNS do domínio
4. NÃO invente preços
5. steps devem ser práticos e acionáveis
6. examples devem ser como um cliente real falaria no WhatsApp (informal, abreviado)
7. Todos os textos em português brasileiro

Responda APENAS com o JSON, sem markdown, sem explicação."""


# ==================== JSON PARSING ====================

def _parse_json_object(text: str) -> dict:
    """Extrai JSON object da resposta do LLM."""
    text = text.strip()

    # Remover markdown code blocks se existirem
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove primeira e última linha (```json e ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    # Tentar parse direto
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Tentar extrair {...} do texto
    import re
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return {}


# ==================== INTERACTIVE REVIEW ====================

def _review_list_items(items: list[str], category: str) -> list[str]:
    """
    Interactive review of a list of items.
    Returns only accepted items.
    """
    if not items:
        return []

    print(f"\n{'─' * 50}")
    print(f"  {category} ({len(items)} itens)")
    print(f"{'─' * 50}")

    accepted = []
    i = 0
    while i < len(items):
        item = items[i]
        print(f"\n  [{i+1}/{len(items)}] \"{item}\"")
        resp = input("    (a)ceitar | (r)ejeitar | (e)ditar | (A)ceitar TODOS restantes | (R)ejeitar TODOS | (q)uit: ").strip()

        if resp in ("a", ""):
            accepted.append(item)
            i += 1
        elif resp == "r":
            i += 1  # skip
        elif resp.lower() == "e":
            new_val = input(f"    Novo valor: ").strip()
            if new_val:
                accepted.append(new_val)
            else:
                accepted.append(item)
            i += 1
        elif resp == "A" or resp.lower() == "all":
            # Accept all remaining
            accepted.extend(items[i:])
            break
        elif resp == "R":
            # Reject all remaining
            break
        elif resp.lower() == "q":
            break
        else:
            print("    Opção inválida. Use: a/r/e/A/R/q")

    print(f"  → {len(accepted)}/{len(items)} aceitos")
    return accepted


def _review_kb_entries(entries: list[dict]) -> list[dict]:
    """Interactive review of KB entries."""
    if not entries:
        return []

    print(f"\n{'─' * 50}")
    print(f"  kb_entries ({len(entries)} entradas)")
    print(f"{'─' * 50}")

    accepted = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        print(f"\n  [{i+1}/{len(entries)}] [{entry.get('category','?')}] {entry.get('title','?')}")
        print(f"    symptoms: {entry.get('symptoms', [])[:3]}")
        print(f"    response: {entry.get('recommended_response', '')[:80]}...")
        print(f"    examples: {entry.get('examples', [])[:2]}")
        resp = input("    (a)ceitar | (r)ejeitar | (A)ceitar TODOS restantes | (R)ejeitar TODOS | (q)uit: ").strip()

        if resp in ("a", ""):
            accepted.append(entry)
            i += 1
        elif resp == "r":
            i += 1
        elif resp == "A" or resp.lower() == "all":
            accepted.extend(entries[i:])
            break
        elif resp == "R":
            break
        elif resp.lower() == "q":
            break
        else:
            print("    Opção inválida.")

    print(f"  → {len(accepted)}/{len(entries)} aceitos")
    return accepted


def _review_feedback_responses(responses: dict) -> dict:
    """Interactive review of feedback response templates."""
    if not responses:
        return {}

    print(f"\n{'─' * 50}")
    print(f"  feedback_responses ({len(responses)} respostas)")
    print(f"{'─' * 50}")

    accepted = {}
    for key, val in responses.items():
        print(f"\n  [{key}]: \"{val}\"")
        resp = input("    (a)ceitar | (r)ejeitar | (e)ditar: ").strip()
        if resp in ("a", ""):
            accepted[key] = val
        elif resp.lower() == "e":
            new_val = input(f"    Novo texto: ").strip()
            accepted[key] = new_val if new_val else val
        # else: reject (skip)

    print(f"  → {len(accepted)}/{len(responses)} aceitos")
    return accepted


def interactive_review():
    """
    Interactive review of generated config — item by item.
    Allows accepting, rejecting, or editing each item individually.
    """
    data_dir = os.getenv("DATA_DIR", "./data")
    config_path = f"{data_dir}/{GENERATED_CONFIG_FILE}"

    if not os.path.exists(config_path):
        print("❌ Nenhuma configuração gerada encontrada. Execute sem argumentos primeiro.")
        return

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if config.get("approved"):
        print("✓ Configuração já aprovada. Use 'reset' para revisar novamente.")
        return

    print(f"\n{'=' * 60}")
    print(f"  REVIEW INTERATIVO — {config.get('domain', '?')}")
    print(f"  Gerado em: {config.get('generated_at', '?')}")
    print(f"{'=' * 60}")
    print("  Para cada item: (a)ceitar, (r)ejeitar, (e)ditar")
    print("  Atalhos: (A) aceitar todos restantes, (R) rejeitar todos restantes")

    # Review each category
    list_categories = [
        ("noise_terms", "Noise Terms"),
        ("spam_indicators", "Spam Indicators"),
        ("short_noise", "Short Noise"),
        ("feedback_positive", "Feedback Positivo"),
        ("feedback_negative", "Feedback Negativo"),
    ]

    for key, label in list_categories:
        items = config.get(key, [])
        if items:
            config[key] = _review_list_items(items, label)

    # Review feedback responses
    if config.get("feedback_responses"):
        config["feedback_responses"] = _review_feedback_responses(config["feedback_responses"])

    # Review KB entries
    if config.get("kb_entries"):
        config["kb_entries"] = _review_kb_entries(config["kb_entries"])

    # Summary
    print(f"\n{'=' * 60}")
    print("  RESUMO FINAL:")
    print(f"{'=' * 60}")
    print(f"  noise_terms:      {len(config.get('noise_terms', []))}")
    print(f"  spam_indicators:  {len(config.get('spam_indicators', []))}")
    print(f"  short_noise:      {len(config.get('short_noise', []))}")
    print(f"  feedback_positive: {len(config.get('feedback_positive', []))}")
    print(f"  feedback_negative: {len(config.get('feedback_negative', []))}")
    print(f"  feedback_responses: {len(config.get('feedback_responses', {}))}")
    print(f"  kb_entries:       {len(config.get('kb_entries', []))}")

    resp = input("\n  Salvar e aprovar resultado da revisão? (s/n): ").strip().lower()
    if resp in ("s", "sim", "y", "yes"):
        config["approved"] = True
        config["needs_human_review"] = False
        config["review_method"] = "interactive"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print("  ✓ Configuração revisada e APROVADA.")
    else:
        # Save reviewed state but not approved (can re-review)
        config["needs_human_review"] = True
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print("  ✗ Revisão salva mas NÃO aprovada. Execute 'review' novamente ou 'approve' para aprovar.")


# ==================== CLI ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else None

    if command == "review":
        # Interactive item-by-item review
        interactive_review()

    elif command == "approve":
        # Quick approve all (approve everything as-is)
        data_dir = os.getenv("DATA_DIR", "./data")
        config_path = f"{data_dir}/{GENERATED_CONFIG_FILE}"

        if not os.path.exists(config_path):
            print("❌ Nenhuma configuração gerada encontrada. Execute sem argumentos primeiro.")
            sys.exit(1)

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        if config.get("approved"):
            print("✓ Configuração já está aprovada.")
            sys.exit(0)

        print(f"=== Aprovar TUDO de uma vez: {config.get('domain', '?')} ===\n")
        print(f"  noise_terms:       {len(config.get('noise_terms', []))}")
        print(f"  spam_indicators:   {len(config.get('spam_indicators', []))}")
        print(f"  short_noise:       {len(config.get('short_noise', []))}")
        print(f"  feedback_positive: {len(config.get('feedback_positive', []))}")
        print(f"  feedback_negative: {len(config.get('feedback_negative', []))}")
        print(f"  kb_entries:        {len(config.get('kb_entries', []))}")

        print("\n" + "=" * 60)
        resp = input("Aprovar TUDO sem revisão? (s/n): ").strip().lower()

        if resp in ("s", "sim", "y", "yes"):
            config["approved"] = True
            config["needs_human_review"] = False
            config["review_method"] = "bulk_approve"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print("✓ Configuração APROVADA (bulk). O sistema usará esses termos.")
        else:
            print("✗ Não aprovada. Use 'review' para revisão item-a-item.")

    elif command == "reset":
        # Reset approval status (to re-review)
        data_dir = os.getenv("DATA_DIR", "./data")
        config_path = f"{data_dir}/{GENERATED_CONFIG_FILE}"

        if not os.path.exists(config_path):
            print("❌ Nenhuma configuração encontrada.")
            sys.exit(1)

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        config["approved"] = False
        config["needs_human_review"] = True
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print("✓ Status resetado. Execute 'review' para revisar novamente.")

    else:
        # Generate new config
        print("=== Gerando configuração do domínio via LLM ===\n")
        result = generate_domain_terms()

        if result:
            print(f"\n✓ Gerado com sucesso!")
            print(f"  noise_terms: {len(result.get('noise_terms', []))}")
            print(f"  spam_indicators: {len(result.get('spam_indicators', []))}")
            print(f"  short_noise: {len(result.get('short_noise', []))}")
            print(f"  feedback_positive: {len(result.get('feedback_positive', []))}")
            print(f"  feedback_negative: {len(result.get('feedback_negative', []))}")
            print(f"  kb_entries: {len(result.get('kb_entries', []))}")
            print(f"\n⚠️  Próximo passo:")
            print(f"   python -m kb.generate_domain_config review    → Revisão item-a-item")
            print(f"   python -m kb.generate_domain_config approve   → Aprovar tudo de uma vez")
        else:
            print("❌ Falha na geração.")
