"""
Loader centralizado de configuração: domínio, prompts e settings.
Padrão inspirado no Quivr (YAML para config de workflow/domínio).
"""
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).parent
_DOMAINS_DIR = _CONFIG_DIR / "domains"
_cache: dict[str, dict[str, Any]] = {}
_prompts_cache: dict[str, Any] | None = None
_settings_cache: dict[str, Any] | None = None
_locale_cache: dict[str, Any] | None = None


# ==================== LOCALE ====================


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override values take precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_locale(lang: str | None = None) -> dict[str, Any]:
    """
    Load locale with fallback: en_us (base) + regional override (e.g., pt_br).
    Keys not present in the regional file fallback to en_us.
    """
    global _locale_cache
    if _locale_cache is not None:
        return _locale_cache

    # Always load en_us as the base
    base_path = _CONFIG_DIR / "locale" / "en_us.yaml"
    base = {}
    if base_path.exists():
        with open(base_path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

    # Load regional override
    if lang is None:
        lang = os.getenv("BOT_LOCALE", "pt_br")

    if lang == "en_us":
        _locale_cache = base
        return _locale_cache

    override_path = _CONFIG_DIR / "locale" / f"{lang}.yaml"
    if override_path.exists():
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        _locale_cache = _deep_merge(base, override)
    else:
        _locale_cache = base

    return _locale_cache


# ==================== DOMAIN CONFIG ====================


def load_domain_config(domain: str | None = None) -> dict[str, Any]:
    """
    Carrega configuração do domínio a partir do YAML.

    Args:
        domain: Nome do domínio (sem extensão). Se None, usa BOT_DOMAIN do .env.

    Returns:
        Dict com a configuração do domínio.

    Raises:
        FileNotFoundError: Se o arquivo YAML do domínio não existir.
    """
    if domain is None:
        domain = os.getenv("BOT_DOMAIN", "custom")

    if domain in _cache:
        return _cache[domain]

    yaml_path = _DOMAINS_DIR / f"{domain}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Config de domínio não encontrada: {yaml_path}\n"
            f"Domínios disponíveis: {list_domains()}"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Normaliza para interface compatível com o antigo DOMAIN_CONFIG
    normalized = _normalize(config)
    _cache[domain] = normalized
    return normalized


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Normaliza YAML para interface compatível com código existente."""
    keywords = raw.get("keywords", {})
    feedback = raw.get("feedback", {})
    references = raw.get("references", {})

    return {
        "name": raw["name"],
        "description": raw["description"],
        "products": raw.get("products", []),
        "sale_keywords": keywords.get("sale", []),
        "support_keywords": keywords.get("support", []),
        "noise_terms": raw.get("noise_terms", []),
        "feedback_positive": feedback.get("positive", []),
        "feedback_negative": feedback.get("negative", []),
        "short_noise": raw.get("short_noise", []),
        "references": {
            "noise": references.get("noise", []),
            "valid_short": references.get("valid_short", []),
            "spam": references.get("spam", []),
            "feedback_positive": references.get("feedback_positive", []),
            "feedback_negative": references.get("feedback_negative", []),
            "neutral": references.get("neutral", []),
        },
    }


def list_domains() -> list[str]:
    """Lista domínios disponíveis (nomes dos arquivos YAML sem extensão)."""
    if not _DOMAINS_DIR.exists():
        return []
    return [p.stem for p in _DOMAINS_DIR.glob("*.yaml")]


# ==================== PROMPTS CONFIG ====================


def load_prompts() -> dict[str, Any]:
    """Carrega templates de prompts do config/prompts.yaml."""
    global _prompts_cache
    if _prompts_cache is not None:
        return _prompts_cache

    prompts_path = _CONFIG_DIR / "prompts.yaml"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts config não encontrado: {prompts_path}")

    with open(prompts_path, "r", encoding="utf-8") as f:
        _prompts_cache = yaml.safe_load(f)

    return _prompts_cache


# ==================== SETTINGS CONFIG ====================


def load_settings() -> dict[str, Any]:
    """Carrega settings/thresholds do config/settings.yaml."""
    global _settings_cache
    if _settings_cache is not None:
        return _settings_cache

    settings_path = _CONFIG_DIR / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings config não encontrado: {settings_path}")

    with open(settings_path, "r", encoding="utf-8") as f:
        _settings_cache = yaml.safe_load(f)

    return _settings_cache


def get_setting(section: str, key: str, default: Any = None) -> Any:
    """Acessa um setting específico: get_setting('extraction', 'noise_similarity_threshold')."""
    settings = load_settings()
    return settings.get(section, {}).get(key, default)


# ==================== PLAYBOOK CONFIG ====================

_playbook_cache: dict[str, dict[str, Any]] = {}


def load_playbook(domain: str | None = None) -> dict[str, Any]:
    """
    Carrega playbook do domínio (roteiros de conversa, mensagens, flows).

    Args:
        domain: Nome do domínio. Se None, usa BOT_DOMAIN do .env.

    Returns:
        Dict com: instructions, messages, flows
    """
    if domain is None:
        domain = os.getenv("BOT_DOMAIN", "custom")

    if domain in _playbook_cache:
        return _playbook_cache[domain]

    playbook_path = _CONFIG_DIR / "playbooks" / f"{domain}.yaml"
    if not playbook_path.exists():
        raise FileNotFoundError(
            f"Playbook não encontrado: {playbook_path}\n"
            f"Playbooks disponíveis: {list_playbooks()}"
        )

    with open(playbook_path, "r", encoding="utf-8") as f:
        playbook = yaml.safe_load(f)

    _playbook_cache[domain] = playbook
    return playbook


def get_playbook_instructions(domain: str | None = None) -> str:
    """Retorna as instruções gerais do playbook (injetadas no system prompt)."""
    playbook = load_playbook(domain)
    return playbook.get("instructions", "")


def get_playbook_messages(domain: str | None = None) -> dict[str, dict]:
    """Retorna dict de mensagens reutilizáveis do playbook."""
    playbook = load_playbook(domain)
    return playbook.get("messages", {})


def get_playbook_flows(domain: str | None = None) -> dict[str, dict]:
    """Retorna dict de flows do playbook."""
    playbook = load_playbook(domain)
    return playbook.get("flows", {})


def get_playbook_condition_hints(domain: str | None = None) -> dict[str, dict]:
    """Returns condition_hints from playbook (keywords + eval descriptions)."""
    playbook = load_playbook(domain)
    return playbook.get("condition_hints", {})


def get_flow_by_trigger(
    intent: str | None = None,
    conditions: dict[str, Any] | None = None,
    domain: str | None = None,
) -> dict | None:
    """
    Busca o flow mais adequado para um trigger (intent + condições).
    Retorna o flow de maior prioridade que casa com o trigger.

    Args:
        intent: Intenção classificada (venda, suporte, etc.)
        conditions: Dict de condições do contexto (client_is_new, etc.)
        domain: Domínio (default: BOT_DOMAIN)

    Returns:
        Dict do flow ou None se nenhum casar.
    """
    flows = get_playbook_flows(domain)
    conditions = conditions or {}

    matching = []
    for flow_name, flow in flows.items():
        trigger = flow.get("trigger", {})

        # Checar intent
        if "intent" in trigger and trigger["intent"] != intent:
            continue

        # Checar condition (simplificado — verifica se condição está nas conditions passadas)
        if "condition" in trigger:
            cond = trigger["condition"]
            if cond == "manual":
                continue  # flows manuais não são auto-selecionados
            if cond not in conditions or not conditions[cond]:
                continue

        matching.append({"name": flow_name, **flow})

    if not matching:
        return None

    # Retornar o de maior prioridade
    matching.sort(key=lambda f: f.get("priority", 0), reverse=True)
    return matching[0]


def list_playbooks() -> list[str]:
    """Lista playbooks disponíveis."""
    playbooks_dir = _CONFIG_DIR / "playbooks"
    if not playbooks_dir.exists():
        return []
    return [p.stem for p in playbooks_dir.glob("*.yaml")]


# ==================== CACHE MANAGEMENT ====================


def clear_cache() -> None:
    """Limpa todos os caches (útil para testes ou hot-reload)."""
    global _prompts_cache, _settings_cache
    _cache.clear()
    _playbook_cache.clear()
    _prompts_cache = None
    _settings_cache = None
