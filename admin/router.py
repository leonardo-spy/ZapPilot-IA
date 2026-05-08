"""
Admin API router — KB review, YAML editor, domain management.
"""
import json
import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import (
    clear_cache,
    get_active_domain,
    list_domains,
    list_playbooks,
)

router = APIRouter(prefix="/admin", tags=["admin"])

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))


# ==================== MODELS ====================


class DomainListResponse(BaseModel):
    domains: list[str]
    playbooks: list[str]
    active: str


class KBItemAction(BaseModel):
    source: str  # "generated_domain_config" | "extracted_patterns"
    category: str  # key within the JSON (e.g. "noise_terms", "kb_entries")
    index: int  # item index in the list
    action: str  # "approve" | "reject" | "delete"


class KBItemAdd(BaseModel):
    source: str
    category: str
    item: Any  # string or dict depending on category


class YAMLFileUpdate(BaseModel):
    content: str


# ==================== DOMAIN MANAGEMENT ====================


@router.get("/domains")
async def get_domains():
    """List all available domains and playbooks."""
    return DomainListResponse(
        domains=list_domains(),
        playbooks=list_playbooks(),
        active=get_active_domain(),
    )


# ==================== KNOWLEDGE GAPS ====================


@router.get("/knowledge-gaps")
async def knowledge_gaps(days: int = 30, top: int = 20, domain: str = None):
    """Returns knowledge gap analysis."""
    from scripts.knowledge_gaps_report import generate_report

    if domain is None:
        domain = get_active_domain()

    report = generate_report(days=days, top_n=top, domain=domain)
    return {"report": report, "domain": domain}


@router.get("/knowledge-gaps/json")
async def knowledge_gaps_json(days: int = 30, limit: int = 100, domain: str = None):
    """Returns raw knowledge gaps data as JSON."""
    from memory.sqlite_memory import SQLiteMemory

    if domain is None:
        domain = get_active_domain()

    memory = SQLiteMemory(os.getenv("DATA_DIR", "./data") + "/memory.db")
    gaps = memory.get_knowledge_gaps(limit=limit, since_days=days, domain=domain)
    summary = memory.get_knowledge_gaps_summary(since_days=days, domain=domain)

    return {"domain": domain, "summary": summary, "gaps": gaps}


# ==================== KB REVIEW ====================

def _get_kb_file_path(source: str) -> Path:
    """Maps source name to JSON file path."""
    mapping = {
        "generated_domain_config": _DATA_DIR / "generated_domain_config.json",
        "extracted_patterns": _DATA_DIR / "extracted_patterns.json",
    }
    path = mapping.get(source)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"Source '{source}' not found")
    return path


def _load_kb_json(source: str) -> dict:
    path = _get_kb_file_path(source)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_kb_json(source: str, data: dict):
    path = _get_kb_file_path(source)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@router.get("/kb/sources")
async def kb_sources():
    """List available KB sources for review."""
    sources = []
    for name, filename in [
        ("generated_domain_config", "generated_domain_config.json"),
        ("extracted_patterns", "extracted_patterns.json"),
    ]:
        path = _DATA_DIR / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sources.append({
                "name": name,
                "domain": data.get("domain", "unknown"),
                "approved": data.get("approved", False),
                "needs_human_review": data.get("needs_human_review", True),
                "categories": [
                    k for k, v in data.items()
                    if isinstance(v, list)
                ],
            })
    return {"sources": sources}


@router.get("/kb/{source}")
async def kb_get_source(source: str):
    """Get full content of a KB source for review."""
    data = _load_kb_json(source)
    return data


@router.get("/kb/{source}/{category}")
async def kb_get_category(source: str, category: str):
    """Get items of a specific category within a KB source."""
    data = _load_kb_json(source)
    items = data.get(category)
    if items is None:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found in '{source}'")
    return {"source": source, "category": category, "items": items, "count": len(items)}


@router.post("/kb/action")
async def kb_item_action(action: KBItemAction):
    """Approve, reject, or delete a specific KB item."""
    data = _load_kb_json(action.source)
    items = data.get(action.category)

    if items is None or not isinstance(items, list):
        raise HTTPException(status_code=404, detail=f"Category '{action.category}' not found")
    if action.index < 0 or action.index >= len(items):
        raise HTTPException(status_code=400, detail=f"Index {action.index} out of range (0-{len(items)-1})")

    if action.action == "delete":
        removed = items.pop(action.index)
        _save_kb_json(action.source, data)
        return {"status": "deleted", "item": removed}
    elif action.action == "approve":
        # Mark the whole file as reviewed if all items approved
        data["needs_human_review"] = False
        data["approved"] = True
        _save_kb_json(action.source, data)
        return {"status": "approved"}
    elif action.action == "reject":
        removed = items.pop(action.index)
        _save_kb_json(action.source, data)
        return {"status": "rejected", "item": removed}
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {action.action}")


@router.post("/kb/add")
async def kb_item_add(item: KBItemAdd):
    """Add a new item to a KB source category."""
    data = _load_kb_json(item.source)
    items = data.get(item.category)

    if items is None or not isinstance(items, list):
        raise HTTPException(status_code=404, detail=f"Category '{item.category}' not found")

    items.append(item.item)
    _save_kb_json(item.source, data)
    return {"status": "added", "new_count": len(items)}


@router.post("/kb/{source}/approve-all")
async def kb_approve_all(source: str):
    """Mark entire KB source as approved."""
    data = _load_kb_json(source)
    data["needs_human_review"] = False
    data["approved"] = True
    _save_kb_json(source, data)
    return {"status": "approved", "source": source}


# ==================== YAML EDITOR ====================

# Registry of editable YAML files and what they control
YAML_REGISTRY = {
    "settings": {
        "path": "config/settings.yaml",
        "description": "Global thresholds and tuning parameters (extraction, retrieval, classification)",
        "reload": "clear_cache",
    },
    "locale_en": {
        "path": "config/locale/en_us.yaml",
        "description": "English locale strings (base/fallback for all locales)",
        "reload": "clear_cache",
    },
    "locale_pt_br": {
        "path": "config/locale/pt_br.yaml",
        "description": "Portuguese locale strings (override for PT-BR clients)",
        "reload": "clear_cache",
    },
    "prompts_global": {
        "path": "config/prompts.yaml",
        "description": "Global prompt templates (fallback for domains without custom prompts)",
        "reload": "clear_cache",
    },
}


def _discover_domain_yamls() -> dict[str, dict]:
    """Dynamically discover per-domain YAML files."""
    extra = {}
    for domain in list_domains():
        extra[f"domain_{domain}"] = {
            "path": f"config/domains/{domain}.yaml",
            "description": f"Domain config for '{domain}' (keywords, products, noise)",
            "reload": "clear_cache",
        }
    for pb in list_playbooks():
        extra[f"playbook_{pb}"] = {
            "path": f"config/playbooks/{pb}.yaml",
            "description": f"Playbook for '{pb}' (flows, messages, condition_hints)",
            "reload": "clear_cache",
        }
    prompts_dir = _CONFIG_DIR / "prompts"
    if prompts_dir.exists():
        for p in prompts_dir.glob("*.yaml"):
            extra[f"prompts_{p.stem}"] = {
                "path": f"config/prompts/{p.name}",
                "description": f"Prompt templates for domain '{p.stem}'",
                "reload": "clear_cache",
            }
    return extra


@router.get("/yaml/files")
async def yaml_list_files():
    """List all editable YAML files with descriptions."""
    all_files = {**YAML_REGISTRY, **_discover_domain_yamls()}
    result = []
    for key, info in sorted(all_files.items()):
        full_path = _PROJECT_ROOT / info["path"]
        result.append({
            "key": key,
            "path": info["path"],
            "description": info["description"],
            "exists": full_path.exists(),
        })
    return {"files": result}


@router.get("/yaml/{file_key}")
async def yaml_get_file(file_key: str):
    """Read content of a YAML file."""
    all_files = {**YAML_REGISTRY, **_discover_domain_yamls()}
    info = all_files.get(file_key)
    if not info:
        raise HTTPException(status_code=404, detail=f"File key '{file_key}' not found")

    full_path = _PROJECT_ROOT / info["path"]
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {info['path']}")

    content = full_path.read_text(encoding="utf-8")
    return {
        "key": file_key,
        "path": info["path"],
        "description": info["description"],
        "content": content,
    }


@router.put("/yaml/{file_key}")
async def yaml_update_file(file_key: str, update: YAMLFileUpdate):
    """Update a YAML file and reload config."""
    all_files = {**YAML_REGISTRY, **_discover_domain_yamls()}
    info = all_files.get(file_key)
    if not info:
        raise HTTPException(status_code=404, detail=f"File key '{file_key}' not found")

    # Validate YAML syntax before saving
    try:
        yaml.safe_load(update.content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML syntax: {e}")

    full_path = _PROJECT_ROOT / info["path"]
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(update.content, encoding="utf-8")

    # Reload config
    if info.get("reload") == "clear_cache":
        clear_cache()

    return {"status": "saved", "path": info["path"], "reloaded": True}
