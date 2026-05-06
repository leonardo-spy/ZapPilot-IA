"""
System prompts e templates para o agente.
Configurável por domínio via BOT_DOMAIN no .env.
Config de domínio carregada de config/domains/<domain>.yaml (padrão Quivr).
Prompts carregados de config/prompts.yaml.
"""
from config import load_domain_config, load_prompts


def get_domain_config() -> dict:
    """Retorna configuração do domínio atual (carregada do YAML)."""
    return load_domain_config()


# ==================== SYSTEM PROMPTS (dinâmicos por domínio) ====================

def _get_prompt_vars() -> dict:
    """Variáveis disponíveis para templates de prompt."""
    domain = get_domain_config()
    return {
        "domain_name": domain["name"],
        "domain_description": domain["description"],
        "products": ", ".join(domain["products"]),
    }


def _render_prompt(key: str, extra_vars: dict | None = None) -> str:
    """Renderiza um prompt template do YAML com as variáveis de domínio."""
    prompts = load_prompts()
    template = prompts.get(key, "")
    variables = _get_prompt_vars()
    if extra_vars:
        variables.update(extra_vars)
    return template.format(**variables)


def _build_base_system_prompt() -> str:
    """System prompt base carregado do YAML."""
    return _render_prompt("base_system_prompt")


def _get_sales_prompt() -> str:
    return _build_base_system_prompt() + "\n" + _render_prompt("sales_prompt")


def _get_support_prompt() -> str:
    return _build_base_system_prompt() + "\n" + _render_prompt("support_prompt")


def _get_general_prompt() -> str:
    return _build_base_system_prompt() + "\n" + _render_prompt("general_prompt")


# ==================== INTENT CLASSIFICATION ====================

def get_intent_classification_prompt() -> str:
    """Retorna prompt de classificação de intenção."""
    return load_prompts().get("intent_classification_prompt", "")


def get_feedback_detection_prompt() -> str:
    """Retorna prompt de detecção de feedback."""
    return load_prompts().get("feedback_detection_prompt", "")


# Manter variáveis de módulo para compatibilidade
INTENT_CLASSIFICATION_PROMPT = None  # Lazy — use get_intent_classification_prompt()
FEEDBACK_DETECTION_PROMPT = None  # Lazy — use get_feedback_detection_prompt()


def _init_prompt_constants():
    """Inicializa constantes de prompt (para compatibilidade com imports diretos)."""
    global INTENT_CLASSIFICATION_PROMPT, FEEDBACK_DETECTION_PROMPT
    prompts = load_prompts()
    INTENT_CLASSIFICATION_PROMPT = prompts.get("intent_classification_prompt", "")
    FEEDBACK_DETECTION_PROMPT = prompts.get("feedback_detection_prompt", "")


_init_prompt_constants()


# ==================== KNOWLEDGE GENERATION PROMPT ====================

def get_domain_kb_generation_prompt(quantity: int = 10) -> str:
    """Prompt para gerar KB complementar baseada no domínio."""
    return _render_prompt("kb_generation_prompt", {"quantity": quantity})


# ==================== HELPER FUNCTIONS ====================

def get_system_prompt(intent: str) -> str:
    """Retorna system prompt adequado para a intenção."""
    from config import get_setting
    sales_intents = get_setting("classification", "sales_intents", ["sales", "renewal", "billing"])
    if intent in sales_intents:
        return _get_sales_prompt()
    elif intent == "support":
        return _get_support_prompt()
    else:
        return _get_general_prompt()


def build_rag_prompt(system_prompt: str, context: str, memory_context: str = "") -> str:
    """Monta system prompt completo com contexto RAG e memória."""
    parts = [system_prompt]

    if memory_context:
        parts.append(f"\n--- Memória do cliente ---\n{memory_context}")

    if context:
        parts.append(f"\n--- Base de conhecimento ---\n{context}")

    return "\n".join(parts)
