"""Agentic RAG with LangGraph.
Flow: load_memory → classify_intent → retrieve → generate_response → feedback_handler → human_handoff
"""
import json
import logging
import re
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from config import get_locale
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
    domain: str  # active domain for this request
    active_flow: str  # active playbook flow name
    flow_step: int  # current step within the flow


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


def _get_domain(state: dict | None = None) -> str:
    """Returns the domain for the current request (from state, thread-local, or env var)."""
    if state and state.get("domain"):
        return state["domain"]
    from config import get_active_domain
    return get_active_domain()


def init_dependencies(memory: SQLiteMemory = None, llm: LLMProvider = None):
    """Initialize dependencies externally (for tests/injection)."""
    global _memory, _llm
    if memory:
        _memory = memory
    if llm:
        _llm = llm


# ==================== GRAPH NODES ====================

def load_memory(state: AgentState) -> dict:
    """Load customer memory context + flow state."""
    memory = _get_memory()
    customer_id = state["customer_id"]

    memory_context = memory.get_memory_context(customer_id, domain=_get_domain(state))

    # Load persisted flow state
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
    """Classify message intent using LLM."""
    llm = _get_llm()

    prompt = INTENT_CLASSIFICATION_PROMPT

    # Add domain products so LLM recognizes product terms
    try:
        from agent.prompts import get_domain_config
        domain = get_domain_config()
        products = domain.get("products", [])
        if products:
            locale = get_locale()
            ctx = locale.get("classify_intent", {})
            prompt += ctx.get("products_context", "").format(products=", ".join(products))
    except Exception:
        pass

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": state["user_message"]},
    ]

    # Include memory context if available
    if state.get("memory_context"):
        locale = get_locale()
        ctx = locale.get("classify_intent", {})
        messages[0]["content"] += ctx.get("memory_context_label", "").format(
            memory_context=state["memory_context"]
        )

    # Include active flow context — helps classify responses within a sales flow
    active_flow = state.get("active_flow", "")
    if active_flow:
        locale = get_locale()
        ctx = locale.get("classify_intent", {})
        messages[0]["content"] += ctx.get("active_flow_context", "").format(flow=active_flow)

    # Keyword pre-check: farewell detection (avoids LLM misclassifying "valeu, tchau" as general)
    msg_lower = state["user_message"].lower().strip()
    locale = get_locale()
    fw_conf = locale.get("farewell_precheck", {})
    farewell_single = set(fw_conf.get("single_keywords", "").split())
    farewell_multi = fw_conf.get("multi_keywords", [])
    has_farewell = bool(farewell_single.intersection(set(msg_lower.split())))
    if not has_farewell:
        has_farewell = any(kw in msg_lower for kw in farewell_multi)
    if has_farewell and len(state["user_message"].strip()) <= 50:
        logger.info(f"[classify_intent] Keyword pre-check → farewell")
        intent = "farewell"
        confidence = 0.9
        needs_retrieval = False
        return {
            **state,
            "intent": intent,
            "intent_confidence": confidence,
            "needs_retrieval": needs_retrieval,
            "route": "direct",
        }

    try:
        response = llm.chat(messages, temperature=0.1, max_tokens=100)
        # Extrair JSON da resposta
        result = _parse_json_response(response)

        intent = result.get("intent", "general")
        confidence = float(result.get("confidence", 0.5))
        needs_retrieval = result.get("needs_retrieval", True)
        retrieval_top_k = int(result.get("retrieval_top_k", 3))

        # Guard: after farewell, don't classify anything as greeting (avoid restart)
        if intent == "greeting" and _recent_farewell(state.get("memory_context", "")):
            logger.info("[classify_intent] Overriding greeting after farewell → general")
            intent = "general"
            confidence = 0.4

        # Guard: very short messages cannot be reliably classified as feedback
        user_msg_stripped = state["user_message"].strip()
        if len(user_msg_stripped) <= 5 and intent in ("feedback_positive", "feedback_negative"):
            logger.info(f"[classify_intent] Overriding feedback for short msg ({len(user_msg_stripped)} chars)")
            intent = "general"
            confidence = 0.3
            needs_retrieval = True

    except Exception as e:
        logger.warning(f"[classify_intent] Classification error: {e}")
        intent = "general"
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
    """Check if message is feedback — uses semantic detection + LLM."""
    # Guard: messages with ≤5 chars can't be reliable feedback (defense-in-depth)
    user_msg = state.get("user_message", "").strip()
    if len(user_msg) <= 5:
        logger.info(f"[check_feedback] Skipping — message too short ({len(user_msg)} chars)")
        return {"is_feedback": False, "feedback_type": "neutral", "resolved": False}

    # First try fast semantic detection via domain embeddings
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

    # Fallback: use LLM for more precise detection
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
        logger.warning(f"[check_feedback] Error: {e}")
        is_feedback = False
        feedback_type = "neutral"
        resolved = False

    return {
        "is_feedback": is_feedback,
        "feedback_type": feedback_type,
        "resolved": resolved,
    }


def retrieve(state: AgentState) -> dict:
    """Search relevant documents with hybrid retriever."""
    if not state.get("needs_retrieval", False):
        logger.info("[retrieve] Skipping (needs_retrieval=False)")
        return {"retrieved_docs": [], "route": "direct"}

    query = state["user_message"]
    top_k = state.get("retrieval_top_k", 3)

    # Filter by category if intent is clear
    category_filter = None
    if state["intent"] == "sales":
        category_filter = "sales"
    elif state["intent"] == "support":
        category_filter = "support"

    try:
        docs = hybrid_search(
            query=query,
            top_k=top_k,
            category_filter=category_filter,
            domain=_get_domain(state),
        )
        retrieved = [d.to_dict() for d in docs]
    except Exception as e:
        logger.warning(f"[retrieve] Search error: {e}")
        retrieved = []

    route = "rag" if retrieved else "no_context"
    logger.info(f"[retrieve] {len(retrieved)} docs retrieved, route={route}")

    return {"retrieved_docs": retrieved, "route": route}


def generate_response(state: AgentState) -> dict:
    """Generate response using LLM with RAG context + memory + playbook."""
    llm = _get_llm()
    intent = state.get("intent", "general")
    confidence = state.get("confidence", 0.5)

    # If feedback, handle differently
    if state.get("is_feedback"):
        return _handle_feedback_response(state)

    # If confidence too low or intent is human, hand off
    # BUT: if there's an active flow, try flow/LLM response first
    active_flow = state.get("active_flow", "")
    if intent == "human":
        return _human_handoff_response(state)
    if intent == "farewell":
        return _farewell_response(state)
    if confidence < 0.3 and not active_flow:
        return _human_handoff_response(state)

    # === PLAYBOOK: try direct execution (literal messages) ===
    direct = _try_direct_flow_response(state)
    if direct:
        return direct

    # Build context
    system_prompt = get_system_prompt(intent)

    # === PLAYBOOK: inject instructions and flow into prompt ===
    playbook_context, detected_flow = _build_playbook_context(state)
    if playbook_context:
        system_prompt += "\n" + playbook_context

    # Post-farewell note: keep responses brief, no handoff
    if _recent_farewell(state.get("memory_context", "")):
        locale = get_locale()
        post_farewell = locale.get("post_farewell_note", {}).get("prompt_suffix", "")
        system_prompt += post_farewell

    # Retrieved documents context
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
        logger.error(f"[generate_response] LLM error: {e}")
        locale = get_locale()
        response = locale.get("errors", {}).get("technical_difficulty", "Technical error.")

    # If route was no_context and we have no docs, might be low confidence
    if state.get("route") == "no_context" and confidence < 0.6:
        locale = get_locale()
        response += locale.get("errors", {}).get("low_confidence_suffix", "")

    # Strip patterns that the LLM may copy from context (configurable via locale)
    locale = get_locale()
    sanitization = locale.get("response_sanitization", {})
    for pattern in sanitization.get("strip_patterns", []):
        response = re.sub(pattern, '', response)
    for literal in sanitization.get("strip_literals", []):
        response = response.replace(literal, '')
    response = response.strip()

    logger.info(f"[generate_response] intent={intent}, flow={detected_flow or 'none'}, response_len={len(response)}")

    result = {"response": response, "route": state.get("route", "direct")}
    # Propagate detected flow to state (used by save_to_memory)
    if detected_flow and not state.get("active_flow"):
        result["active_flow"] = detected_flow
        result["flow_step"] = 0
    return result


def save_to_memory(state: AgentState) -> dict:
    """Save interaction to memory + update flow state."""
    memory = _get_memory()
    customer_id = state["customer_id"]
    intent = state.get("intent", "")
    domain = _get_domain(state)

    # Save user message
    memory.save_message(customer_id, "user", state["user_message"], intent=intent, domain=domain)

    # Save bot response
    if state.get("response"):
        memory.save_message(customer_id, "assistant", state["response"], intent=intent, domain=domain)

    # If feedback, update case
    if state.get("is_feedback") and state.get("feedback_type") == "positive":
        cases = memory.get_open_cases(customer_id)
        for case in cases:
            memory.resolve_case(case["id"])

    # If support/sales, create/update case
    if intent in ("support", "sales") and state.get("confidence", 0) > 0.5:
        memory.create_or_update_case(
            customer_id=customer_id,
            intent=intent,
            summary=state["user_message"][:200],
            solution_tried=state.get("response", "")[:200] if state.get("response") else None,
            resolved=state.get("resolved", False),
        )

    # === FLOW STATE: advance or clear ===
    _update_flow_state(state, memory, customer_id)

    # === KNOWLEDGE GAP DETECTION ===
    _detect_knowledge_gap(state, memory, customer_id)

    logger.info(f"[save_to_memory] Saved for customer={customer_id}")
    return {}


def _update_flow_state(state: AgentState, memory, customer_id: str):
    """Update flow state after each turn with pause/resume logic."""
    active_flow = state.get("active_flow", "")
    flow_step = state.get("flow_step", 0)
    intent = state.get("intent", "")

    # If intent changed to feedback/human/farewell → ABANDON flow
    # out_of_scope does NOT abandon (may be a question within the sales context)
    if intent in ("feedback_positive", "feedback_negative", "human", "farewell"):
        if active_flow:
            memory.clear_flow_state(customer_id)
            logger.info(f"[flow_state] Flow '{active_flow}' abandoned (intent={intent})")
        return

    # If no active flow → try to select a new one
    if not active_flow:
        # Guard: don't discover flow if conversation recently ended with farewell
        if not _recent_farewell(state.get("memory_context", "")):
            try:
                from config import get_flow_by_trigger
                conditions = _resolve_flow_conditions(state)
                flow = get_flow_by_trigger(intent=intent, conditions=conditions)
                if flow:
                    active_flow = flow["name"]
                    # Advance past opening messages (LLM already handled them)
                    steps = flow.get("steps", [])
                    flow_step = 0
                    for j in range(len(steps)):
                        if steps[j].get("action") == "wait_response":
                            flow_step = j
                            break
                    memory.save_flow_state(customer_id, active_flow, flow_step)
                    logger.info(f"[flow_state] New flow: '{active_flow}'")
            except Exception as e:
                logger.debug(f"[flow_state] No flow: {e}")
        return

    # Active flow → check if intent matches the flow trigger
    try:
        from config import get_playbook_flows
        flows = get_playbook_flows()
        flow_def = flows.get(active_flow, {})
        trigger_intent = flow_def.get("trigger", {}).get("intent", "")

        if trigger_intent and trigger_intent != intent and not (
            intent in ("greeting", "info", "billing") and trigger_intent == "sales"
        ):
            # Check if we're on a condition/wait_response step
            # (client responding to flow question → always continue)
            steps_list = flow_def.get("steps", [])
            current_step_action = ""
            if flow_step < len(steps_list):
                current_step_action = steps_list[flow_step].get("action", "")

            if current_step_action not in ("condition", "wait_response", "send", "send_sequence"):
                # Different intent and not responding → PAUSE
                logger.info(f"[flow_state] Flow '{active_flow}' paused (intent={intent} != trigger={trigger_intent})")
                return
            else:
                logger.info(f"[flow_state] Flow '{active_flow}' continues (responding to {current_step_action})")

        # Intent matches → ADVANCE step
        # Logic: if current step is wait_response, advance to next step (what bot must execute).
        # If current step was executed this turn (send/condition/etc), advance past next wait_response.
        steps = flow_def.get("steps", [])
        current_action = steps[flow_step].get("action", "") if flow_step < len(steps) else ""

        if current_action == "wait_response":
            # Client just responded to a wait → bot should execute NEXT step
            next_step = flow_step + 1
        else:
            # Bot just executed this step → find next wait_response and advance past it
            next_step = _find_next_wait_step(steps, flow_step)

        if next_step is None or next_step >= len(steps):
            # Flow completed
            memory.clear_flow_state(customer_id)
            logger.info(f"[flow_state] Flow '{active_flow}' completed")
        else:
            memory.save_flow_state(customer_id, active_flow, next_step)
            logger.info(f"[flow_state] Flow '{active_flow}' → step {next_step}")
    except Exception as e:
        logger.debug(f"[flow_state] Error advancing: {e}")
        memory.save_flow_state(customer_id, active_flow, flow_step + 1)


def _find_next_wait_step(steps: list, current_step: int) -> int | None:
    """
    Find the next wait_response AFTER the current step.
    Returns the index of that wait_response (where the flow pauses waiting for client).
    On next client message, _update_flow_state will advance from wait→next step.
    """
    for i in range(current_step + 1, len(steps)):
        if steps[i].get("action") == "wait_response":
            return i

    # If no more wait_response found, flow is at the end
    return len(steps)


def _detect_knowledge_gap(state: AgentState, memory, customer_id: str):
    """
    Detects when the agent lacked sufficient context to answer well.
    Records to knowledge_gaps table for later analysis/recommendations.

    Triggers:
    - route == "no_context" (retrieval returned nothing)
    - low confidence + needs_retrieval (tried to search but got poor results)
    - intent is info/support/sales but 0 docs retrieved
    """
    route = state.get("route", "")
    intent = state.get("intent", "")
    confidence = state.get("confidence", 0.0)
    retrieved_docs = state.get("retrieved_docs", [])
    needs_retrieval = state.get("needs_retrieval", False)

    # Skip non-informational intents (greeting, feedback, human, farewell, etc.)
    skip_intents = ("greeting", "farewell", "feedback_positive", "feedback_negative", "human", "out_of_scope")
    if intent in skip_intents:
        return

    # Skip playbook-handled messages (they have their own scripted responses)
    if route == "playbook":
        return

    is_gap = False

    # Case 1: Retrieval returned nothing
    if route == "no_context":
        is_gap = True

    # Case 2: Needed retrieval, got docs but confidence is very low
    elif needs_retrieval and len(retrieved_docs) == 0:
        is_gap = True

    # Case 3: Has retrieval but confidence still below threshold
    elif needs_retrieval and confidence < 0.4 and len(retrieved_docs) <= 1:
        is_gap = True

    if is_gap:
        try:
            memory.record_knowledge_gap(
                customer_id=customer_id,
                query=state.get("user_message", ""),
                intent=intent,
                route=route,
                confidence=confidence,
                retrieved_docs_count=len(retrieved_docs),
                domain=_get_domain(state),
            )
            logger.info(f"[knowledge_gap] Recorded: intent={intent}, route={route}, docs={len(retrieved_docs)}")
        except Exception as e:
            logger.debug(f"[knowledge_gap] Failed to record: {e}")


# ==================== HELPERS ====================

def _handle_feedback_response(state: AgentState) -> dict:
    """Generate response for feedback using pre-approved responses or defaults."""
    from kb.generate_domain_config import get_feedback_responses
    responses = get_feedback_responses()

    if state.get("feedback_type") == "positive":
        response = responses["positive"]
        route = "feedback_resolved"
    elif state.get("feedback_type") == "negative":
        # Search KB for second approach
        try:
            from retrieval.hybrid_retriever import hybrid_search
            docs = hybrid_search(
                state["user_message"],
                top_k=2,
                domain=_get_domain(state),
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


# Condition keywords loaded from config (cached)
_condition_hints_cache: dict | None = None


def _get_condition_hints() -> dict[str, dict]:
    """Load condition hints from playbook config (cached)."""
    global _condition_hints_cache
    if _condition_hints_cache is None:
        try:
            from config import get_playbook_condition_hints
            _condition_hints_cache = get_playbook_condition_hints()
        except Exception:
            _condition_hints_cache = {}
    return _condition_hints_cache


def _keyword_precheck(condition: str, user_msg: str, memory_context: str = "") -> bool | None:
    """Returns True/False if keywords match, None if LLM is needed."""
    hints = _get_condition_hints()
    hint = hints.get(condition)
    if not hint or not isinstance(hint, dict):
        return None
    keywords = hint.get("keywords_true", [])
    if not keywords:
        return None
    # For "client state" conditions, also check conversation history
    # For "asks_*" conditions, only check current message (avoid false positives from history)
    check_history = hint.get("check_history", False)
    combined = f" {user_msg.lower().strip()} "
    if check_history and memory_context:
        combined += f" {memory_context.lower()} "
    for kw in keywords:
        if kw in combined:
            return True
    return None  # No positive match → let LLM decide


def _evaluate_condition_branch(
    step: dict, state: AgentState, messages: dict
) -> tuple[list[str] | None, str]:
    """
    Evaluates a condition step using LLM to decide the branch (then/else).
    Returns (list of literal messages from chosen branch, goto_target).
    goto_target is set if a goto_flow was encountered inside the branch.
    """
    condition = step.get("if", "")
    then_actions = step.get("then", [])
    else_actions = step.get("else", [])
    user_msg = state.get("user_message", "")

    if not condition or (not then_actions and not else_actions):
        return None, ""

    # Pre-check via keywords before calling LLM (avoids rate limit + inconsistencies)
    memory_context = state.get("memory_context", "")
    is_true = _keyword_precheck(condition, user_msg, memory_context)
    if is_true is not None:
        logger.info(f"[condition_eval] '{condition}' → {'yes' if is_true else 'no'} (keyword_match)")
        branch = then_actions if is_true else else_actions
        if not branch:
            return [], ""
        return _collect_branch_messages(branch, state, messages)

    # Build eval prompt dynamically from config hints + locale
    hints = _get_condition_hints()
    locale = get_locale()
    cond_locale = locale.get("condition_eval", {})

    # Focus on the specific condition being evaluated
    specific_hint = hints.get(condition, {})
    hint_description = ""
    if isinstance(specific_hint, dict) and "description" in specific_hint:
        hint_description = specific_hint["description"]

    negative_examples = hints.get("negative_examples", "")

    eval_intro = cond_locale.get("eval_intro", "")
    cond_label = cond_locale.get("condition_label", "").format(condition=condition)
    msg_label = cond_locale.get("message_label", "").format(message=user_msg)

    eval_prompt = f"{eval_intro}\n{cond_label}\n{msg_label}\n\n"

    # Include conversation context for better evaluation
    memory_context = state.get("memory_context", "")
    if memory_context:
        eval_prompt += f"Conversation context (previous messages):\n{memory_context}\n\n"

    if hint_description:
        eval_prompt += cond_locale.get("hints_header", "") + "\n"
        eval_prompt += f"- {condition}: {hint_description}\n\n"
    if negative_examples:
        prefix = cond_locale.get("important_prefix", "")
        eval_prompt += f"{prefix}{negative_examples}\n"
    eval_prompt += cond_locale.get("response_instruction", "")

    positive_kw = cond_locale.get("positive_answer", "yes")

    # LLM call: just decides yes or no
    llm = _get_llm()
    try:
        system_content = cond_locale.get("system_prompt", "")
        eval_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": eval_prompt},
        ]
        result = llm.chat(eval_messages)
        answer = result.strip().lower()
        is_true = answer.startswith(positive_kw)
        logger.info(f"[condition_eval] '{condition}' → {answer} (is_true={is_true})")
    except Exception as e:
        logger.warning(f"[condition_eval] Failed to evaluate '{condition}': {e}")
        return None, ""

    # Choose branch
    branch = then_actions if is_true else else_actions
    if not branch:
        return [], ""

    # Collect literal messages from branch
    parts, goto_target = _collect_branch_messages(branch, state, messages)
    return parts, goto_target


def _collect_branch_messages(
    actions: list[dict], state: AgentState, messages: dict
) -> tuple[list[str], str]:
    """Collect literal messages from an action list (supports nested conditions).

    Returns (parts, goto_target) where goto_target is the target flow name
    if a goto_flow was encountered, empty string otherwise.
    """
    parts: list[str] = []
    goto_target = ""
    for sub in actions:
        sub_action = sub.get("action", "")
        if sub_action == "send":
            mk = sub.get("message", "")
            m = messages.get(mk, {})
            m_type = m.get("type", "text")
            content = m.get("content", "")
            if content:
                if m_type == "image":
                    parts.append(f"[IMAGEM: {m.get('caption', mk)}]\n{content}")
                else:
                    parts.append(content.strip())
        elif sub_action == "send_sequence":
            msg_keys = sub.get("messages", [])
            for smk in msg_keys:
                sm = messages.get(smk, {})
                sm_type = sm.get("type", "text")
                sm_content = sm.get("content", "")
                if sm_content:
                    if sm_type == "image":
                        parts.append(f"[IMAGEM: {sm.get('caption', smk)}]\n{sm_content}")
                    else:
                        parts.append(sm_content.strip())
        elif sub_action == "condition":
            # Nested condition — evaluate recursively
            nested_parts, nested_goto = _evaluate_condition_branch(sub, state, messages)
            if nested_parts is not None:
                parts.extend(nested_parts)
            if nested_goto:
                goto_target = nested_goto
                break
            if nested_parts is None and not nested_goto:
                break  # Evaluation failed — stop
        elif sub_action == "goto_flow":
            goto_target = sub.get("flow", "")
            break
        elif sub_action == "wait_response":
            break
        else:
            # generate_response, end etc → need LLM
            if not parts and not goto_target:
                return None, ""
            break
    return parts, goto_target


def _human_handoff_response(state: AgentState) -> dict:
    """Human handoff response."""
    locale = get_locale()
    response = locale.get("human_handoff", {}).get("response", "")
    return {"response": response, "route": "human_handoff"}


def _farewell_response(state: AgentState) -> dict:
    """Farewell response — client ending conversation."""
    locale = get_locale()
    response = locale.get("farewell", {}).get("response", "")
    return {"response": response, "route": "farewell"}


def _recent_farewell(memory_context: str) -> bool:
    """Check if the LAST assistant message was a farewell — prevents flow restart.

    Only checks the most recent assistant line in memory_context to avoid
    false positives when farewell keywords appeared earlier in the conversation.
    """
    if not memory_context:
        return False

    # Extract only the last assistant message from context
    last_assistant = ""
    for line in reversed(memory_context.splitlines()):
        stripped = line.strip()
        if stripped.startswith("assistant:"):
            last_assistant = stripped
            break

    if not last_assistant:
        return False

    last_lower = last_assistant.lower()
    locale = get_locale()
    farewell_keywords = locale.get("farewell_detection", {}).get("keywords", [])
    for kw in farewell_keywords:
        if kw in last_lower:
            return True
    return False


def _inline_collect_target_flow(
    target_flow: str,
    flows: dict,
    messages: dict,
    state: "AgentState",
    response_parts: list[str],
) -> tuple[str, int, list[dict]] | None:
    """Collect literal messages from a target flow's initial steps.

    Appends messages to response_parts in-place.
    Returns (flow_name, flow_step, steps_list) on success, None if target not found.
    """
    if not target_flow or target_flow not in flows:
        return None

    target_steps = flows[target_flow].get("steps", [])
    flow_step = 0

    for k in range(len(target_steps)):
        ts = target_steps[k]
        ta = ts.get("action", "")

        if ta == "send":
            mk = ts.get("message", "")
            m = messages.get(mk, {})
            mt = m.get("type", "text")
            mc = m.get("content", ts.get("content", ""))
            if mc:
                if mt == "image":
                    cap = m.get("caption", "")
                    response_parts.append(f"[IMAGEM: {cap}]\n{mc}")
                else:
                    response_parts.append(mc.strip())

        elif ta == "send_sequence":
            for smk in ts.get("messages", []):
                m = messages.get(smk, {})
                m_type = m.get("type", "text")
                m_c = m.get("content", "")
                if m_c:
                    if m_type == "image":
                        cap = m.get("caption", "")
                        response_parts.append(f"[IMAGEM: {cap}]\n{m_c}")
                    else:
                        response_parts.append(m_c.strip())

        elif ta == "wait_response":
            flow_step = k
            break

        elif ta == "condition":
            bm, bg = _evaluate_condition_branch(ts, state, messages)
            if bm is not None:
                response_parts.extend(bm)
            if bg:
                # Nested goto inside this target flow — recurse
                nested = _inline_collect_target_flow(bg, flows, messages, state, response_parts)
                if nested is not None:
                    return nested
                break

        elif ta == "generate_response":
            if not response_parts:
                return None
            flow_step = k
            break

        else:
            break
    else:
        flow_step = 0  # Exhausted all steps

    return target_flow, flow_step, target_steps


def _try_direct_flow_response(state: AgentState) -> dict | None:
    """
    Try to execute flow steps directly (without LLM).
    When the current step is send/send_sequence, returns LITERAL messages.
    Returns None if LLM is needed (condition, generate_response, etc).
    """
    # Guard: don't continue/restart a flow if the conversation just ended with farewell
    if _recent_farewell(state.get("memory_context", "")):
        logger.info("[_try_direct_flow_response] Recent farewell detected — skipping flow")
        return None

    active_flow = state.get("active_flow", "")
    flow_step = state.get("flow_step", 0)

    # If no active flow, try to discover one (for greeting → sales)
    if not active_flow:

        try:
            from config import get_flow_by_trigger
            current_intent = state.get("intent", "")
            conditions = _resolve_flow_conditions(state)

            flow = get_flow_by_trigger(intent=current_intent, conditions=conditions)

            # If greeting didn't find flow, try with sales
            if not flow and current_intent == "greeting":
                flow = get_flow_by_trigger(intent="sales", conditions=conditions)

            if flow:
                active_flow = flow["name"]
                flow_step = 0
            else:
                return None
        except Exception:
            return None

    try:
        from config import get_playbook_flows, get_playbook_messages
        flows = get_playbook_flows()
        messages = get_playbook_messages()
    except Exception:
        return None

    flow_def = flows.get(active_flow)
    if not flow_def:
        return None

    steps = flow_def.get("steps", [])
    if flow_step >= len(steps):
        return None

    current = steps[flow_step]
    action = current.get("action", "")

    # If current step is wait_response, user just responded — advance to execute NEXT step
    # This ensures send/send_sequence fires on the SAME turn as the user's response
    if action == "wait_response":
        flow_step += 1
        if flow_step >= len(steps):
            return None
        current = steps[flow_step]
        action = current.get("action", "")

    # Executes send/send_sequence/condition/goto_flow directly
    if action == "goto_flow":
        # Transition to the target flow
        target_flow = current.get("flow", "")
        if target_flow and target_flow in flows:
            target_def = flows[target_flow]
            target_steps = target_def.get("steps", [])
            # Execute the target flow's initial steps
            active_flow = target_flow
            flow_step = 0
            steps = target_steps
            if not target_steps:
                return None
            current = target_steps[0]
            action = current.get("action", "")
            if action not in ("send", "send_sequence", "condition"):
                return None
        else:
            return None
    elif action not in ("send", "send_sequence", "condition"):
        return None

    # Collect all consecutive messages until wait_response
    response_parts = []
    for i in range(flow_step, len(steps)):
        step = steps[i]
        act = step.get("action", "")

        if act == "send":
            msg_key = step.get("message", "")
            msg = messages.get(msg_key, {})
            msg_type = msg.get("type", "text")
            content = msg.get("content", step.get("content", ""))
            if content:
                if msg_type == "image":
                    caption = msg.get("caption", "")
                    response_parts.append(f"[IMAGEM: {caption}]\n{content}")
                else:
                    response_parts.append(content.strip())

        elif act == "send_sequence":
            msg_keys = step.get("messages", [])
            for mk in msg_keys:
                m = messages.get(mk, {})
                m_type = m.get("type", "text")
                m_content = m.get("content", "")
                if m_content:
                    if m_type == "image":
                        caption = m.get("caption", "")
                        response_parts.append(f"[IMAGEM: {caption}]\n{m_content}")
                    else:
                        response_parts.append(m_content.strip())

        elif act == "condition":
            # Evaluate condition with LLM (yes/no) and pick correct branch
            branch_msgs, branch_goto = _evaluate_condition_branch(step, state, messages)
            if branch_msgs is not None:
                response_parts.extend(branch_msgs)
            if branch_goto:
                # goto_flow was triggered inside the branch — transition now
                result = _inline_collect_target_flow(
                    branch_goto, flows, messages, state, response_parts
                )
                if result is not None:
                    active_flow, flow_step, steps = result
                    logger.info(f"[direct_flow] goto_flow(cond) → '{active_flow}' (from branch)")
                    break
                else:
                    logger.warning(f"[direct_flow] goto_flow(cond) target '{branch_goto}' not found")
            if branch_msgs is not None:
                # Continue collecting steps after the condition
                pass
            else:
                # Could not evaluate — stop and let full LLM handle
                break

        elif act == "wait_response":
            break  # Stop here — bot waits for client response

        elif act == "goto_flow":
            # Transition mid-collection: switch to target flow and continue collecting
            target_flow = step.get("flow", "")
            result = _inline_collect_target_flow(
                target_flow, flows, messages, state, response_parts
            )
            if result is not None:
                active_flow, flow_step, steps = result
            break

        elif act == "generate_response":
            break  # Needs LLM — stop and let next node handle

        else:
            break

    if not response_parts:
        return None

    # Join with separator for individual messages
    response = "\n---MSG---\n".join(response_parts)

    logger.info(
        f"[direct_flow] Flow '{active_flow}' step {flow_step}: "
        f"{len(response_parts)} literal messages sent"
    )

    return {
        "response": response,
        "route": "playbook",
        "active_flow": active_flow,
        "flow_step": flow_step,
    }


def _build_playbook_context(state: AgentState) -> tuple[str, str]:
    """
    Build playbook context to inject into system prompt.
    Includes: general instructions + active flow (if intent is compatible).
    Returns: (context_str, flow_name) — flow_name may be empty.

    Pause/resume logic:
    - If there's an active flow AND intent matches trigger → show flow (resume)
    - If there's an active flow BUT different intent → DON'T show flow (pause)
    - If no flow → try to discover a new one
    """
    try:
        from config import get_playbook_instructions, get_flow_by_trigger, get_playbook_flows
    except Exception:
        return "", ""

    parts = []

    # 1. Owner instructions (always present)
    instructions = get_playbook_instructions()
    if instructions:
        locale = get_locale()
        header = locale.get("playbook_context", {}).get(
            "instructions_header", ""
        )
        parts.append(f"{header}\n{instructions.strip()}")

    # 2. Flow: check intent compatibility
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
            steps = flow_def.get("steps", [])

            # If we're on a condition/wait step, client is RESPONDING
            # to the flow → always continue, regardless of classified intent
            current_step_action = ""
            if flow_step < len(steps):
                current_step_action = steps[flow_step].get("action", "")

            is_responding_to_flow = current_step_action in (
                "condition", "wait_response", "send", "send_sequence"
            )

            if is_responding_to_flow or not trigger_intent or trigger_intent == current_intent or (
                current_intent in ("greeting", "info", "billing") and trigger_intent == "sales"
            ):
                # Compatible intent or responding to flow → RESUME
                flow = {**flow_def, "name": active_flow}
                detected_flow_name = active_flow
            elif current_intent == "human":
                # Asked for human → abandon flow
                pass
            else:
                # Different intent → PAUSE (don't show flow, LLM responds freely)
                locale = get_locale()
                note = locale.get("playbook_context", {}).get(
                    "flow_paused_note", ""
                ).format(intent=trigger_intent)
                parts.append(note)
    else:
        # No active flow → try to discover a new one
        # Guard: don't discover flow if conversation recently ended with farewell
        if not _recent_farewell(state.get("memory_context", "")):
            conditions = _resolve_flow_conditions(state)
            flow = get_flow_by_trigger(
                intent=current_intent,
                conditions=conditions,
            )
            # If greeting didn't find a flow, try with sales (in sales bots, greeting = sales start)
            if not flow and current_intent == "greeting":
                flow = get_flow_by_trigger(intent="sales", conditions=conditions)
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
    Resolve client conditions for flow selection.
    Based on memory and conversation state.
    """
    memory = _get_memory()
    customer_id = state.get("customer_id", "")
    conditions = {}

    try:
        # New client = no message history
        history = memory.get_recent_history(customer_id, limit=1)
        conditions["client_is_new"] = len(history) == 0
        conditions["client_has_history"] = len(history) > 0

        # Buyer client = has resolved sales case
        cases = memory.get_open_cases(customer_id)
        # If NO open sales cases, may have already bought
        # (heuristic: has history but no open case = already bought)
        conditions["client_is_buyer"] = (
            len(history) > 0 and len(cases) == 0
        )

    except Exception as e:
        logger.debug(f"[_resolve_flow_conditions] Fallback: {e}")
        conditions["client_is_new"] = True

    return conditions


def _format_flow_for_prompt(flow: dict, state: AgentState, from_step: int = 0) -> str:
    """
    Format the selected flow as context for the LLM.
    Shows only steps from from_step onwards (persisted state).
    Includes FULL messages for LLM to reproduce LITERALLY.
    """
    try:
        from config import get_playbook_messages
        messages = get_playbook_messages()
    except Exception:
        messages = {}

    locale = get_locale()
    fmt = locale.get("flow_format", {})

    all_steps = flow.get("steps", [])
    remaining_steps = all_steps[from_step:]

    if not remaining_steps:
        return ""

    header = fmt.get("active_flow_header", "")
    parts = [header.format(description=flow.get("description", flow["name"]))]
    if from_step > 0:
        cont = fmt.get("continuation", "")
        parts.append(cont.format(step=from_step + 1, total=len(all_steps)))
    parts.append(fmt.get("literal_instruction", "") + "\n")

    for i, step in enumerate(remaining_steps, from_step + 1):
        action = step.get("action", "")

        if action == "send":
            msg_key = step.get("message", "")
            msg = messages.get(msg_key, {})
            content = msg.get("content", step.get("content", ""))
            if content:
                label = fmt.get("send_literal", "").format(i=i)
                parts.append(f"{label}\n\"\"\"\n{content.strip()}\n\"\"\"")

        elif action == "send_sequence":
            msg_keys = step.get("messages", [])
            label = fmt.get("send_sequence", "").format(i=i)
            parts.append(label)
            for mk in msg_keys:
                m = messages.get(mk, {})
                if m.get("type") == "text":
                    parts.append(f"     → \"\"\"\n{m.get('content', '').strip()}\n\"\"\"")
                elif m.get("type") == "image":
                    parts.append(f"     → [IMAGEM: {m.get('caption', mk)}]")

        elif action == "wait_response":
            label = fmt.get("wait_response", "").format(i=i)
            parts.append(label)
            break  # Don't show steps after the next wait

        elif action == "condition":
            cond = step.get("if", "")
            label = fmt.get("evaluate_condition", "").format(i=i, condition=cond)
            parts.append(label)
            if step.get("then"):
                parts.append(fmt.get("then_label", ""))
                for sub in step["then"]:
                    if sub.get("action") == "send":
                        mk = sub.get("message", "")
                        m = messages.get(mk, {})
                        content = m.get("content", mk)
                        send_label = fmt.get("send_in_branch", "")
                        parts.append(f"{send_label}\n\"\"\"\n{content.strip()}\n\"\"\"")
                    elif sub.get("action") == "goto_flow":
                        goto = fmt.get("goto_flow_branch", "").format(flow=sub.get("flow"))
                        parts.append(goto)
            if step.get("else"):
                parts.append(fmt.get("else_label", ""))
                for sub in step["else"]:
                    if sub.get("action") == "send":
                        mk = sub.get("message", "")
                        m = messages.get(mk, {})
                        content = m.get("content", mk)
                        send_label = fmt.get("send_in_branch", "")
                        parts.append(f"{send_label}\n\"\"\"\n{content.strip()}\n\"\"\"")
                    elif sub.get("action") == "goto_flow":
                        goto = fmt.get("goto_flow_branch", "").format(flow=sub.get("flow"))
                        parts.append(goto)

        elif action == "generate_response":
            label = fmt.get("generate_response", "").format(
                i=i, context=step.get("context", "")
            )
            parts.append(label)

        elif action == "goto_flow":
            label = fmt.get("follow_flow", "").format(i=i, flow=step.get("flow", ""))
            parts.append(label)

        elif action == "escalate":
            label = fmt.get("escalate", "").format(
                i=i, reason=step.get("reason", "")
            )
            parts.append(label)

    parts.append(fmt.get("rules_header", ""))
    parts.append(fmt.get("rule_literal", ""))
    parts.append(fmt.get("rule_separate", ""))
    parts.append(fmt.get("rule_generate", ""))

    return "\n".join(parts)


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response (may have text before/after)."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block (supports nested braces)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


# ==================== GRAPH ROUTING ====================

def route_after_classify(state: AgentState) -> str:
    """Decide route after classification."""
    intent = state.get("intent", "")

    if intent in ("feedback_positive", "feedback_negative"):
        return "check_feedback"
    elif intent in ("human", "farewell"):
        return "generate_response"
    elif intent == "out_of_scope":
        return "generate_response"
    else:
        return "retrieve"


def route_after_feedback(state: AgentState) -> str:
    """Decide route after feedback check."""
    if state.get("is_feedback"):
        return "generate_response"
    # Not feedback, continue normal flow
    return "retrieve"


# ==================== BUILD GRAPH ====================

def build_graph() -> StateGraph:
    """Build the LangGraph graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("load_memory", load_memory)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("check_feedback", check_feedback)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_response", generate_response)
    graph.add_node("save_to_memory", save_to_memory)

    # Set entry point
    graph.set_entry_point("load_memory")

    # Edges
    graph.add_edge("load_memory", "classify_intent")

    # Conditional edge after classification
    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "check_feedback": "check_feedback",
            "retrieve": "retrieve",
            "generate_response": "generate_response",
        }
    )

    # Conditional edge after feedback check
    graph.add_conditional_edges(
        "check_feedback",
        route_after_feedback,
        {
            "generate_response": "generate_response",
            "retrieve": "retrieve",
        }
    )

    # Linear edges
    graph.add_edge("retrieve", "generate_response")
    graph.add_edge("generate_response", "save_to_memory")
    graph.add_edge("save_to_memory", END)

    return graph


# Compile graph
_compiled_graph = None


def get_graph():
    """Return compiled graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph().compile()
    return _compiled_graph


def run_agent(customer_id: str, message: str, domain: str | None = None) -> dict:
    """
    Executa o agente para uma mensagem.

    Args:
        customer_id: Customer identifier.
        message: User message.
        domain: Domain to use. If None, falls back to BOT_DOMAIN env var.

    Returns:
        Dict com response, intent, route, confidence, retrieved_docs
    """
    from config import get_active_domain, domain_context

    if domain is None:
        domain = get_active_domain()

    with domain_context(domain):
        graph = get_graph()

        initial_state: AgentState = {
            "customer_id": customer_id,
            "user_message": message,
            "domain": domain,
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
