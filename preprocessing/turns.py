"""
Módulo de construção de turns semânticos.
Agrupa mensagens em blocos de atendimento (problema do cliente + resposta do atendente).
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Tempo máximo entre mensagens para considerar mesmo atendimento (2 horas)
MAX_GAP_SECONDS = 7200


def build_turns(merged_blocks: list[dict]) -> list[dict]:
    """
    Constrói turns semânticos a partir de blocos merged.

    Um turn é um par: contexto do cliente (problema/pergunta) + contexto do assistente (resposta/solução).
    Não assume alternância estrita.

    Args:
        merged_blocks: Saída de merge_consecutive() — blocos com 'author', 'text', 'timestamp_start', etc.

    Returns:
        Lista de turns com client_context, assistant_context, full_context, etc.
    """
    if not merged_blocks:
        return []

    # Primeiro, segmentar em sessões de atendimento por gap temporal
    sessions = _segment_sessions(merged_blocks)

    # Depois, extrair turns de cada sessão
    turns = []
    for session in sessions:
        session_turns = _extract_turns_from_session(session)
        turns.extend(session_turns)

    logger.info(f"Turns construídos: {len(turns)} turns de {len(sessions)} sessões")
    return turns


def _segment_sessions(blocks: list[dict]) -> list[list[dict]]:
    """Segmenta blocos em sessões por gap temporal e por chat_id."""
    if not blocks:
        return []

    sessions = []
    current_session = [blocks[0]]

    for i in range(1, len(blocks)):
        prev = blocks[i - 1]
        curr = blocks[i]

        # Novo chat = nova sessão
        if curr.get("chat_id") != prev.get("chat_id"):
            if current_session:
                sessions.append(current_session)
            current_session = [curr]
            continue

        # Gap temporal grande = nova sessão
        gap = curr["timestamp_start"] - prev["timestamp_end"]
        if gap > MAX_GAP_SECONDS:
            if current_session:
                sessions.append(current_session)
            current_session = [curr]
            continue

        current_session.append(curr)

    if current_session:
        sessions.append(current_session)

    return sessions


def _extract_turns_from_session(session: list[dict]) -> list[dict]:
    """
    Extrai turns de uma sessão.

    Lógica:
    - Acumula blocos de cliente até encontrar bloco(s) do atendente
    - Forma um turn quando há resposta do atendente ao contexto do cliente
    - Se sessão começa com atendente (proativo), ignora até ter contexto do cliente
    """
    turns = []
    client_blocks = []
    i = 0

    while i < len(session):
        block = session[i]

        if block["author"] == "client":
            client_blocks.append(block)
            i += 1

        elif block["author"] == "me":
            if not client_blocks:
                # Atendente falou primeiro (mensagem proativa), pular
                i += 1
                continue

            # Acumular todos os blocos consecutivos do atendente
            assistant_blocks = []
            while i < len(session) and session[i]["author"] == "me":
                assistant_blocks.append(session[i])
                i += 1

            # Formar o turn
            client_text = "\n".join(b["text"] for b in client_blocks)
            assistant_text = "\n".join(b["text"] for b in assistant_blocks)

            # Coletar metadata
            all_msg_ids = []
            for b in client_blocks + assistant_blocks:
                all_msg_ids.extend(b.get("message_ids", []))

            turn = {
                "chat_id": session[0].get("chat_id", ""),
                "subject": session[0].get("subject", ""),
                "start_timestamp": client_blocks[0]["timestamp_start"],
                "end_timestamp": assistant_blocks[-1]["timestamp_end"],
                "client_context": client_text,
                "assistant_context": assistant_text,
                "full_context": f"Cliente: {client_text}\n\nAtendente: {assistant_text}",
                "message_ids": all_msg_ids,
                "participants": list(set(
                    b.get("author") for b in client_blocks + assistant_blocks
                )),
            }
            turns.append(turn)

            # Reset para próximo turn
            client_blocks = []

        else:
            i += 1

    return turns


def build_turns_from_messages(messages: list[dict]) -> list[dict]:
    """
    Pipeline completo: mensagens brutas → turns semânticos.
    Usa cleaner internamente.
    """
    from preprocessing.cleaner import clean_messages, merge_consecutive

    cleaned = clean_messages(messages)
    merged = merge_consecutive(cleaned)
    turns = build_turns(merged)
    return turns


if __name__ == "__main__":
    import os, json
    from dotenv import load_dotenv
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.whatsapp_loader import load_whatsapp_auto
    from preprocessing.cleaner import clean_messages, merge_consecutive

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    messages = load_whatsapp_auto()
    cleaned = clean_messages(messages)
    merged = merge_consecutive(cleaned)
    turns = build_turns(merged)

    print(f"\nTotal turns: {len(turns)}")
    if turns:
        print(f"\n--- Exemplo de turn ---")
        t = turns[0]
        print(f"Chat: {t['subject']}")
        print(f"Cliente: {t['client_context'][:200]}...")
        print(f"Atendente: {t['assistant_context'][:200]}...")
