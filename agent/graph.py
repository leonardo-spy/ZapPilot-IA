"""
Agentic RAG com LangGraph.
Fluxo: load_memory → classify_intent → retrieve → generate_response → feedback_handler → human_handoff
"""
import json
import logging
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from agent.prompts import (
    get_system_prompt,
    build_rag_prompt,
    INTENT_CLASSIFICATION_PROMPT,
    FEEDBACK_DETECTION_PROMPT,
)
from retrieval.hybrid_retriever import hybrid_search
from memory.sqlite_memory import SQLiteMemory
from llm.providers import get_default_provider, LLMProvider

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    customer_id: str
    user_message: str
    intent: str
    confidence: float
    needs_retrieval: bool
    retrieval_top_k: int
    retrieved_docs: list
    memory_context: str
    response: str
    route: str
    is_feedback: bool
    feedback_type: str
    resolved: bool
    active_flow: str  # nome do flow ativo do playbook
    flow_step: int  # step atual dentro do flow


# ==================== SINGLETON DEPENDENCIES ====================

_memory: SQLiteMemory = None
_llm: LLMProvider = None


def _get_memory() -> SQLiteMemory:
    global _memory
    if _memory is None:
        _memory = SQLiteMemory()
    return _memory


def _get_llm() -> LLMProvider:
    global _llm
    if _llm is None:
        _llm = get_default_provider()
    return _llm


def init_dependencies(memory: SQLiteMemory = None, llm: LLMProvider = None):
    """Inicializa dependências externamente (para testes/injeção)."""
    global _memory, _llm
    if memory:
        _memory = memory
    if llm:
        _llm = llm


# ==================== GRAPH NODES ====================

def load_memory(state: AgentState) -> dict:
    """Carrega contexto de memória do cliente + flow state."""
    memory = _get_memory()
    customer_id = state["customer_id"]

    memory_context = memory.get_memory_context(customer_id)

    # Carregar flow state persistido
    flow_state = memory.get_flow_state(customer_id)
    active_flow = ""
    flow_step = 0
    if flow_state:
        active_flow = flow_state.get("flow", "")
        flow_step = flow_state.get("step", 0)

    logger.info(f"[load_memory] customer={customer_id}, context_len={len(memory_context)}, flow={active_flow}@step{flow_step}")
    return {
        "memory_context": memory_context,
        "active_flow": active_flow,
        "flow_step": flow_step,
    }


def classify_intent(state: AgentState) -> dict:
    """Classifica intenção da mensagem usando LLM."""
    llm = _get_llm()

    messages = [
        {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
        {"role": "user", "content": state["user_message"]},
    ]

    # Se tiver contexto de memória, incluir
    if state.get("memory_context"):
        messages[0]["content"] += f"\n\nContexto do cliente:\n{state['memory_context']}"

    try:
        response = llm.chat(messages, temperature=0.1, max_tokens=100)
        # Extrair JSON da resposta
        result = _parse_json_response(response)

        intent = result.get("intent", "geral")
        confidence = float(result.get("confidence", 0.5))
        needs_retrieval = result.get("needs_retrieval", True)
        retrieval_top_k = int(result.get("retrieval_top_k", 3))

    except Exception as e:
        logger.warning(f"[classify_intent] Erro ao classificar: {e}")
        intent = "geral"
        confidence = 0.3
        needs_retrieval = True
        retrieval_top_k = 3

    logger.info(f"[classify_intent] intent={intent}, confidence={confidence:.2f}, needs_retrieval={needs_retrieval}")

    return {
        "intent": intent,
        "confidence": confidence,
        "needs_retrieval": needs_retrieval,
        "retrieval_top_k": retrieval_top_k,
    }


def check_feedback(state: AgentState) -> dict:
    """Verifica se a mensagem é feedback — usa detecção semântica + LLM."""
    # Primeiro tenta detecção semântica rápida via embeddings do domínio
    try:
        from preprocessing.cleaner import is_semantic_feedback
        semantic_result = is_semantic_feedback(state["user_message"], threshold=0.7)
        if semantic_result["is_feedback"] and semantic_result["score"] > 0.8:
            logger.info(f"[check_feedback] Detectado via embeddings: {semantic_result}")
            return {
                "is_feedback": True,
                "feedback_type": semantic_result["type"],
                "resolved": semantic_result["type"] == "positive",
            }
    except Exception as e:
        logger.debug(f"[check_feedback] Semantic fallback: {e}")

    # Fallback: usa LLM para detecção mais precisa
    llm = _get_llm()

    messages = [
        {"role": "system", "content": FEEDBACK_DETECTION_PROMPT},
        {"role": "user", "content": state["user_message"]},
    ]

    try:
        response = llm.chat(messages, temperature=0.1, max_tokens=80)
        result = _parse_json_response(response)

        is_feedback = result.get("is_feedback", False)
        feedback_type = result.get("type", "neutral")
        resolved = result.get("resolved", False)

    except Exception as e:
        logger.warning(f"[check_feedback] Erro: {e}")
        is_feedback = False
        feedback_type = "neutral"
        resolved = False

    return {
        "is_feedback": is_feedback,
        "feedback_type": feedback_type,
        "resolved": resolved,
    }


def retrieve(state: AgentState) -> dict:
    """Busca documentos relevantes com retriever híbrido."""
    if not state.get("needs_retrieval", False):
        logger.info("[retrieve] Skipping (needs_retrieval=False)")
        return {"retrieved_docs": [], "route": "direct"}

    query = state["user_message"]
    top_k = state.get("retrieval_top_k", 3)

    # Filtrar por categoria se intent for clara
    category_filter = None
    if state["intent"] == "venda":
        category_filter = "venda"
    elif state["intent"] == "suporte":
        category_filter = "suporte"

    try:
        docs = hybrid_search(
            query=query,
            top_k=top_k,
            category_filter=category_filter,
        )
        retrieved = [d.to_dict() for d in docs]
    except Exception as e:
        logger.warning(f"[retrieve] Erro na busca: {e}")
        retrieved = []

    route = "rag" if retrieved else "no_context"
    logger.info(f"[retrieve] {len(retrieved)} docs recuperados, route={route}")

    return {"retrieved_docs": retrieved, "route": route}


def generate_response(state: AgentState) -> dict:
    """Gera resposta usando LLM com contexto RAG + memória + playbook."""
    llm = _get_llm()
    intent = state.get("intent", "geral")
    confidence = state.get("confidence", 0.5)

    # Se é feedback, tratar diferente
    if state.get("is_feedback"):
        return _handle_feedback_response(state)

    # Se confiança muito baixa ou intent é humano, encaminhar
    if confidence < 0.3 or intent == "humano":
        return _human_handoff_response(state)

    # Montar contexto
    system_prompt = get_system_prompt(intent)

    # === PLAYBOOK: injetar instruções e flow no prompt ===
    playbook_context, detected_flow = _build_playbook_context(state)
    if playbook_context:
        system_prompt += "\n" + playbook_context

    # Contexto dos documentos recuperados
    docs_context = ""
    if state.get("retrieved_docs"):
        docs_context = "\n\n---\n\n".join(
            d["content"] for d in state["retrieved_docs"][:5]
        )

    full_system = build_rag_prompt(
        system_prompt,
        context=docs_context,
        memory_context=state.get("memory_context", ""),
    )

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": state["user_message"]},
    ]

    try:
        response = llm.chat(messages, temperature=0.3, max_tokens=512)
    except Exception as e:
        logger.error(f"[generate_response] Erro LLM: {e}")
        response = "Desculpe, estou com dificuldade técnica. Vou encaminhar para suporte humano."

    # Se route era no_context e não temos docs, pode ter baixa confiança
    if state.get("route") == "no_context" and confidence < 0.6:
        response += "\n\n_Se precisar de mais informações, posso encaminhar para um atendente._"

    logger.info(f"[generate_response] intent={intent}, flow={detected_flow or 'none'}, response_len={len(response)}")

    result = {"response": response, "route": state.get("route", "direct")}
    # Propagar flow detectado para state (usado por save_to_memory)
    if detected_flow and not state.get("active_flow"):
        result["active_flow"] = detected_flow
        result["flow_step"] = 0
    return result


def save_to_memory(state: AgentState) -> dict:
    """Salva interação na memória + atualiza flow state."""
    memory = _get_memory()
    customer_id = state["customer_id"]
    intent = state.get("intent", "")

    # Salvar mensagem do usuário
    memory.save_message(customer_id, "user", state["user_message"], intent=intent)

    # Salvar resposta do bot
    if state.get("response"):
        memory.save_message(customer_id, "assistant", state["response"], intent=intent)

    # Se é feedback, atualizar caso
    if state.get("is_feedback") and state.get("feedback_type") == "positive":
        cases = memory.get_open_cases(customer_id)
        for case in cases:
            memory.resolve_case(case["id"])

    # Se é suporte/venda, criar/atualizar caso
    if intent in ("suporte", "venda") and state.get("confidence", 0) > 0.5:
        memory.create_or_update_case(
            customer_id=customer_id,
            intent=intent,
            summary=state["user_message"][:200],
            solution_tried=state.get("response", "")[:200] if state.get("response") else None,
            resolved=state.get("resolved", False),
        )

    # === FLOW STATE: avançar ou limpar ===
    _update_flow_state(state, memory, customer_id)

    logger.info(f"[save_to_memory] Salvo para customer={customer_id}")
    return {}


def _update_flow_state(state: AgentState, memory, customer_id: str):
    """Atualiza o flow state após cada turno com lógica de pausa/retomada."""
    active_flow = state.get("active_flow", "")
    flow_step = state.get("flow_step", 0)
    intent = state.get("intent", "")

    # Se mudou de intent para feedback/humano/fora_da_base → ABANDONAR flow
    if intent in ("feedback_positivo", "feedback_negativo", "humano", "fora_da_base"):
        if active_flow:
            memory.clear_flow_state(customer_id)
            logger.info(f"[flow_state] Flow '{active_flow}' abandonado (intent={intent})")
        return

    # Se não tem flow ativo → tentar selecionar um novo
    if not active_flow:
        try:
            from config import get_flow_by_trigger
            conditions = _resolve_flow_conditions(state)
            flow = get_flow_by_trigger(intent=intent, conditions=conditions)
            if flow:
                active_flow = flow["name"]
                flow_step = 0
                memory.save_flow_state(customer_id, active_flow, flow_step)
                logger.info(f"[flow_state] Novo flow: '{active_flow}'")
        except Exception as e:
            logger.debug(f"[flow_state] Sem flow: {e}")
        return

    # Tem flow ativo → verificar se intent bate com o trigger do flow
    try:
        from config import get_playbook_flows
        flows = get_playbook_flows()
        flow_def = flows.get(active_flow, {})
        trigger_intent = flow_def.get("trigger", {}).get("intent", "")

        if trigger_intent and trigger_intent != intent:
            # Intent diferente → PAUSAR (manter state, não avançar)
            logger.info(f"[flow_state] Flow '{active_flow}' pausado (intent={intent} != trigger={trigger_intent})")
            return

        # Intent bate → AVANÇAR step
        steps = flow_def.get("steps", [])
        next_step = _find_next_wait_step(steps, flow_step)

        if next_step is None or next_step >= len(steps):
            # Flow concluído
            memory.clear_flow_state(customer_id)
            logger.info(f"[flow_state] Flow '{active_flow}' concluído")
        else:
            memory.save_flow_state(customer_id, active_flow, next_step)
            logger.info(f"[flow_state] Flow '{active_flow}' → step {next_step}")
    except Exception as e:
        logger.debug(f"[flow_state] Erro ao avançar: {e}")
        memory.save_flow_state(customer_id, active_flow, flow_step + 2)


def _find_next_wait_step(steps: list, current_step: int) -> int | None:
    """
    Encontra o próximo wait_response APÓS o step atual.
    Retorna o índice do step APÓS esse wait (onde o flow deve continuar).
    """
    # Percorrer a partir do step atual + 1
    for i in range(current_step + 1, len(steps)):
        if steps[i].get("action") == "wait_response":
            # O flow vai esperar aqui; na próxima msg, continuamos do step seguinte
            return i + 1

    # Se não encontrou mais wait_response, o flow está no final
    return len(steps)


# ==================== HELPERS ====================

def _handle_feedback_response(state: AgentState) -> dict:
    """Gera resposta para feedback usando respostas geradas (se aprovadas) ou defaults."""
    from kb.generate_domain_config import get_feedback_responses
    responses = get_feedback_responses()

    if state.get("feedback_type") == "positive":
        response = responses["positive"]
        route = "feedback_resolved"
    elif state.get("feedback_type") == "negative":
        # Buscar na KB se há segunda abordagem
        try:
            from retrieval.hybrid_retriever import hybrid_search
            docs = hybrid_search(
                state["user_message"],
                top_k=2,
            )
            if docs:
                response = responses["negative_with_docs"]
            else:
                response = responses["negative_no_docs"]
        except Exception:
            response = responses["negative_no_docs"]
        route = "feedback_unresolved"
    else:
        response = responses["neutral"]
        route = "feedback"

    return {"response": response, "route": route}


def _human_handoff_response(state: AgentState) -> dict:
    """Resposta de encaminhamento para humano."""
    response = "Vou encaminhar para suporte humano para verificar melhor seu caso. Um atendente entrará em contato em breve."
    return {"response": response, "route": "human_handoff"}


def _build_playbook_context(state: AgentState) -> tuple[str, str]:
    """
    Constrói contexto do playbook para injetar no system prompt.
    Inclui: instruções gerais + flow ativo (se intent compatível).
    Returns: (context_str, flow_name) — flow_name pode ser vazio.

    Lógica de pausa/retomada:
    - Se há flow ativo E intent bate com o trigger → mostra flow (retoma)
    - Se há flow ativo MAS intent diferente → NÃO mostra flow (pausa)
    - Se não há flow → tenta descobrir um novo
    """
    try:
        from config import get_playbook_instructions, get_flow_by_trigger, get_playbook_flows
    except Exception:
        return "", ""

    parts = []

    # 1. Instruções gerais do dono (sempre presentes)
    instructions = get_playbook_instructions()
    if instructions:
        parts.append(f"--- Instruções do atendimento ---\n{instructions.strip()}")

    # 2. Flow: verificar compatibilidade de intent
    active_flow = state.get("active_flow", "")
    flow_step = state.get("flow_step", 0)
    current_intent = state.get("intent", "")

    flow = None
    detected_flow_name = ""

    if active_flow:
        flows = get_playbook_flows()
        flow_def = flows.get(active_flow)
        if flow_def:
            trigger_intent = flow_def.get("trigger", {}).get("intent", "")

            if not trigger_intent or trigger_intent == current_intent:
                # Intent compatível → RETOMAR flow
                flow = {**flow_def, "name": active_flow}
                detected_flow_name = active_flow
            else:
                # Intent diferente → PAUSAR (não mostrar flow, LLM responde livre)
                # Adicionar nota ao LLM sobre o flow pausado
                parts.append(
                    f"--- Nota: há um roteiro de {trigger_intent} em andamento com este cliente. "
                    f"Se ele voltar ao assunto, retome de onde parou. ---"
                )
    else:
        # Sem flow ativo → tentar descobrir um novo
        conditions = _resolve_flow_conditions(state)
        flow = get_flow_by_trigger(
            intent=current_intent,
            conditions=conditions,
        )
        flow_step = 0
        if flow:
            detected_flow_name = flow.get("name", "")

    if flow:
        formatted = _format_flow_for_prompt(flow, state, from_step=flow_step)
        if formatted:
            parts.append(formatted)

    context = "\n\n".join(parts) if parts else ""
    return context, detected_flow_name


def _resolve_flow_conditions(state: AgentState) -> dict:
    """
    Resolve condições do cliente para seleção de flow.
    Baseado na memória e estado da conversa.
    """
    memory = _get_memory()
    customer_id = state.get("customer_id", "")
    conditions = {}

    try:
        # Cliente novo = sem histórico de mensagens
        history = memory.get_recent_history(customer_id, limit=1)
        conditions["client_is_new"] = len(history) == 0
        conditions["client_has_history"] = len(history) > 0

        # Cliente comprador = tem caso de venda resolvido
        cases = memory.get_open_cases(customer_id)
        # Se NÃO tem casos abertos de venda, pode já ter comprado antes
        # (heurística: se tem histórico mas não tem caso aberto = já comprou)
        conditions["client_is_buyer"] = (
            len(history) > 0 and len(cases) == 0
        )

    except Exception as e:
        logger.debug(f"[_resolve_flow_conditions] Fallback: {e}")
        conditions["client_is_new"] = True

    return conditions


def _format_flow_for_prompt(flow: dict, state: AgentState, from_step: int = 0) -> str:
    """
    Formata o flow selecionado como contexto para o LLM.
    Mostra apenas steps a partir de from_step (estado persistido).
    """
    try:
        from config import get_playbook_messages
        messages = get_playbook_messages()
    except Exception:
        messages = {}

    all_steps = flow.get("steps", [])
    remaining_steps = all_steps[from_step:]

    if not remaining_steps:
        return ""

    parts = [f"--- Roteiro ativo: {flow.get('description', flow['name'])} ---"]
    if from_step > 0:
        parts.append(f"(Continuação — etapa {from_step + 1} de {len(all_steps)})")
    parts.append("Siga este roteiro de conversa (adapte o tom, não copie literalmente):\n")

    for i, step in enumerate(remaining_steps, from_step + 1):
        action = step.get("action", "")

        if action == "send":
            msg_key = step.get("message", "")
            msg = messages.get(msg_key, {})
            content = msg.get("content", step.get("content", ""))
            if content:
                parts.append(f"  {i}. ENVIAR: \"{content.strip()[:200]}\"")

        elif action == "send_sequence":
            msg_keys = step.get("messages", [])
            contents = []
            for mk in msg_keys:
                m = messages.get(mk, {})
                if m.get("type") == "text":
                    contents.append(m.get("content", "")[:80].strip())
                elif m.get("type") == "image":
                    contents.append(f"[IMAGEM: {m.get('caption', mk)}]")
            parts.append(f"  {i}. ENVIAR SEQUÊNCIA: {', '.join(contents[:3])}...")

        elif action == "wait_response":
            parts.append(f"  {i}. AGUARDAR resposta do cliente")

        elif action == "condition":
            cond = step.get("if", "")
            parts.append(f"  {i}. SE {cond}:")
            if step.get("then"):
                for sub in step["then"][:2]:
                    if sub.get("action") == "send":
                        mk = sub.get("message", "")
                        m = messages.get(mk, {})
                        parts.append(f"       → enviar: \"{m.get('content', mk)[:100].strip()}\"")
                    elif sub.get("action") == "goto_flow":
                        parts.append(f"       → seguir roteiro: {sub.get('flow')}")
            if step.get("else"):
                parts.append(f"     SENÃO:")
                for sub in step["else"][:2]:
                    if sub.get("action") == "send":
                        mk = sub.get("message", "")
                        m = messages.get(mk, {})
                        parts.append(f"       → enviar: \"{m.get('content', mk)[:100].strip()}\"")

        elif action == "generate_response":
            parts.append(f"  {i}. RESPONDER usando base de conhecimento ({step.get('context', '')})")

        elif action == "goto_flow":
            parts.append(f"  {i}. SEGUIR roteiro: {step.get('flow', '')}")

        elif action == "escalate":
            parts.append(f"  {i}. ENCAMINHAR para humano: {step.get('reason', '')}")

    parts.append("\nIMPORTANTE: Use este roteiro como GUIA. Adapte a linguagem ao contexto.")
    parts.append("Se o cliente perguntar algo fora do roteiro, use a base de conhecimento.")

    return "\n".join(parts)


def _parse_json_response(text: str) -> dict:
    """Extrai JSON de resposta do LLM (pode ter texto antes/depois)."""
    text = text.strip()

    # Tentar parse direto
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tentar extrair JSON de dentro do texto
    import re
    match = re.search(r'\{[^{}]+\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


# ==================== GRAPH ROUTING ====================

def route_after_classify(state: AgentState) -> str:
    """Decide rota após classificação."""
    intent = state.get("intent", "")

    if intent in ("feedback_positivo", "feedback_negativo"):
        return "check_feedback"
    elif intent == "humano":
        return "generate_response"
    elif intent == "fora_da_base":
        return "generate_response"
    else:
        return "retrieve"


def route_after_feedback(state: AgentState) -> str:
    """Decide rota após verificar feedback."""
    if state.get("is_feedback"):
        return "generate_response"
    # Não era feedback, seguir fluxo normal
    return "retrieve"


# ==================== BUILD GRAPH ====================

def build_graph() -> StateGraph:
    """Constrói o grafo LangGraph."""
    graph = StateGraph(AgentState)

    # Adicionar nós
    graph.add_node("load_memory", load_memory)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("check_feedback", check_feedback)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_response", generate_response)
    graph.add_node("save_to_memory", save_to_memory)

    # Definir entry point
    graph.set_entry_point("load_memory")

    # Edges
    graph.add_edge("load_memory", "classify_intent")

    # Conditional edge após classificação
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "check_feedback": "check_feedback",
            "retrieve": "retrieve",
            "generate_response": "generate_response",
        }
    )

    # Conditional edge após feedback check
    graph.add_conditional_edges(
        "check_feedback",
        route_after_feedback,
        {
            "generate_response": "generate_response",
            "retrieve": "retrieve",
        }
    )

    # Edges lineares
    graph.add_edge("retrieve", "generate_response")
    graph.add_edge("generate_response", "save_to_memory")
    graph.add_edge("save_to_memory", END)

    return graph


# Compilar grafo
_compiled_graph = None


def get_graph():
    """Retorna grafo compilado (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


def run_agent(customer_id: str, message: str) -> dict:
    """
    Executa o agente para uma mensagem.

    Returns:
        Dict com response, intent, route, confidence, retrieved_docs
    """
    graph = get_graph()

    initial_state: AgentState = {
        "customer_id": customer_id,
        "user_message": message,
        "intent": "",
        "confidence": 0.0,
        "needs_retrieval": False,
        "retrieval_top_k": 3,
        "retrieved_docs": [],
        "memory_context": "",
        "response": "",
        "route": "",
        "is_feedback": False,
        "feedback_type": "",
        "resolved": False,
        "active_flow": "",
        "flow_step": 0,
    }

    result = graph.invoke(initial_state)

    return {
        "response": result.get("response", ""),
        "intent": result.get("intent", ""),
        "route": result.get("route", ""),
        "confidence": result.get("confidence", 0.0),
        "retrieved_docs": result.get("retrieved_docs", []),
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Teste
    test_messages = [
        ("5511999999999", "quanto custa a tirzec?"),
        ("5511999999999", "como aplico?"),
        ("5511999999999", "obrigado, resolveu!"),
    ]

    for customer_id, msg in test_messages:
        print(f"\n{'='*60}")
        print(f"User ({customer_id}): {msg}")
        result = run_agent(customer_id, msg)
        print(f"Intent: {result['intent']} (conf: {result['confidence']:.2f})")
        print(f"Route: {result['route']}")
        print(f"Bot: {result['response']}")
