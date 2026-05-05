"""
Módulo de ingestão de dados do WhatsApp.
Lê JSON exportado e normaliza a estrutura.
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_whatsapp_json(filepath: str) -> list[dict]:
    """
    Carrega JSON exportado do WhatsApp e retorna lista normalizada de mensagens.

    Suporta dois formatos:
    - Formato bot-zdg: {status, conversations: [{id, name, messages: [{id, from, to, body, fromMe, timestamp, type, hasMedia}]}]}
    - Formato direto: [{id, jid, subject, messages: [{id, text, timestamp, from_me, sender, media}]}]
    """
    logger.info(f"Carregando JSON: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        raw = json.load(f)

    # Detectar formato
    if isinstance(raw, dict) and "conversations" in raw:
        return _parse_botzdg_format(raw)
    elif isinstance(raw, list):
        return _parse_direct_format(raw)
    else:
        raise ValueError(f"Formato de JSON não reconhecido: {type(raw)}")


def _parse_botzdg_format(raw: dict) -> list[dict]:
    """Formato bot-zdg: {status, conversations: [{id, name, messages: [...]}]}"""
    conversations = raw.get("conversations", [])
    all_messages = []
    skipped = 0

    for chat in conversations:
        chat_id = chat.get("id", "")
        subject = chat.get("name", "")

        for msg in chat.get("messages", []):
            text = (msg.get("body") or "").strip()
            msg_type = msg.get("type", "")

            # Filtrar mensagens sem texto útil
            if not text or msg_type in ("notification_template", "protocol"):
                skipped += 1
                continue

            # Filtrar mensagens que são apenas mídia sem texto
            if msg.get("hasMedia") and not text:
                skipped += 1
                continue

            all_messages.append({
                "chat_id": chat_id,
                "subject": subject,
                "message_id": msg.get("id", ""),
                "sender": msg.get("from", ""),
                "from_me": msg.get("fromMe", False),
                "timestamp": msg.get("timestamp", 0),
                "text": text,
            })

    # Ordenar por timestamp
    all_messages.sort(key=lambda x: x["timestamp"])

    logger.info(
        f"Formato bot-zdg: {len(conversations)} conversas, "
        f"{len(all_messages)} mensagens retidas, {skipped} filtradas"
    )
    return all_messages


def _parse_direct_format(raw: list) -> list[dict]:
    """Formato direto: [{id, jid, subject, messages: [{id, text, timestamp, from_me, sender, media}]}]"""
    all_messages = []
    skipped = 0

    for chat in raw:
        chat_id = str(chat.get("id", chat.get("jid", "")))
        subject = chat.get("subject", "")

        for msg in chat.get("messages", []):
            text = (msg.get("text") or msg.get("raw_text") or "").strip()

            # Filtrar mensagens sem texto
            if not text:
                skipped += 1
                continue

            # Filtrar mensagens que são apenas referência de mídia
            if text.startswith("<Media") and text.endswith(">"):
                skipped += 1
                continue

            # Timestamp pode estar em ms ou s
            ts = msg.get("timestamp", 0)
            if ts > 1e12:
                ts = ts / 1000  # converter ms para s

            all_messages.append({
                "chat_id": chat_id,
                "subject": subject,
                "message_id": str(msg.get("id", "")),
                "sender": msg.get("sender", msg.get("jid", "")),
                "from_me": msg.get("from_me", False),
                "timestamp": ts,
                "text": text,
            })

    # Ordenar por timestamp
    all_messages.sort(key=lambda x: x["timestamp"])

    logger.info(
        f"Formato direto: {len(raw)} chats, "
        f"{len(all_messages)} mensagens retidas, {skipped} filtradas"
    )
    return all_messages


def load_by_chat(filepath: str) -> dict[str, list[dict]]:
    """Carrega e agrupa mensagens por chat_id."""
    messages = load_whatsapp_json(filepath)
    by_chat: dict[str, list[dict]] = {}

    for msg in messages:
        cid = msg["chat_id"]
        if cid not in by_chat:
            by_chat[cid] = []
        by_chat[cid].append(msg)

    logger.info(f"{len(by_chat)} chats únicos encontrados")
    return by_chat


def load_whatsapp_auto(json_path: str = None, db_path: str = None) -> list[dict]:
    """
    Carrega mensagens do WhatsApp automaticamente.

    Prioridade:
    1. Se db_path for fornecido e o arquivo existir → usa msgstore.db
    2. Se json_path for fornecido e o arquivo existir → usa JSON
    3. Verifica variáveis de ambiente WHATSAPP_DB e WHATSAPP_JSON

    Retorna lista normalizada de mensagens (mesmo formato em ambos os casos).
    """
    import os

    # Resolver paths
    if not db_path:
        db_path = os.getenv("WHATSAPP_DB", "./input/msgstore.db")
    if not json_path:
        json_path = os.getenv("WHATSAPP_JSON", "./input/whatsapp_chats.json")

    # Verificar modo forçado via env
    source_mode = os.getenv("WHATSAPP_SOURCE", "auto").lower()

    if source_mode == "db" or (source_mode == "auto" and os.path.exists(db_path)):
        if os.path.exists(db_path):
            logger.info(f"Fonte: msgstore.db ({db_path})")
            from ingestion.msgstore_loader import load_whatsapp_db
            return load_whatsapp_db(db_path)
        elif source_mode == "db":
            raise FileNotFoundError(f"WHATSAPP_SOURCE=db mas arquivo não encontrado: {db_path}")

    if source_mode == "json" or source_mode == "auto":
        if os.path.exists(json_path):
            logger.info(f"Fonte: JSON ({json_path})")
            return load_whatsapp_json(json_path)

    raise FileNotFoundError(
        f"Nenhuma fonte de dados encontrada. "
        f"Verifique WHATSAPP_JSON ({json_path}) ou WHATSAPP_DB ({db_path})"
    )



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    messages = load_whatsapp_auto()
    print(f"\nTotal de mensagens carregadas: {len(messages)}")
    if messages:
        print(f"Primeira: {messages[0]['text'][:80]}...")
        print(f"Última: {messages[-1]['text'][:80]}...")
