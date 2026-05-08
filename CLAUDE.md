# ZapPilot IA — Copilot Instructions

## Project Overview
WhatsApp sales/support chatbot with agentic RAG architecture. Handles multi-step sales flows with pause/resume logic, hybrid retrieval, and playbook-driven conversations.

**License:** GPL-3.0  
**Stack:** Python 3.12, FastAPI, LangGraph, SQLite, ChromaDB (broken — only BM25 works), Groq API (llama-3.1-8b-instant)

## Architecture

```
app.py (FastAPI) → agent/graph.py (LangGraph) → llm/providers.py (Groq/Local fallback)
                                                → retrieval/hybrid_retriever.py (BM25 + Chroma)
                                                → memory/sqlite_memory.py (SQLite WAL)
                                                → config/ (playbooks, locale, prompts)
```

### Core Flow (LangGraph)
`load_memory → classify_intent → retrieve → generate_response → save_to_memory`

- **Playbook execution:** `_try_direct_flow_response()` sends literal messages without LLM
- **Flow state:** Persisted per customer in SQLite. Pause/resume on intent mismatch
- **Condition evaluation:** `_keyword_precheck()` first (fast), LLM fallback (slow, yes/no)
- **Response sanitization:** Configurable regex patterns from `config/locale/en_us.yaml` → `response_sanitization`

### Key Behaviors
- `send_sequence` at step 4 sends 5 literal messages (photos + prices) — tests must account for this extra step
- `---MSG---` separator joins multiple messages internally; `app.py` splits into `response_parts`
- `[IMAGEM: caption]` tags are injected by `_format_flow_for_prompt()` for image-type messages in playbook context
- `_keyword_precheck` does substring matching: `if kw in combined` — case insensitive
- ChromaDB expects dim=384 but embeddings produce dim=768 → only BM25/keyword search works currently

## Coding Conventions

### Language Rules
- **Code:** ALL English (variable names, function names, comments, docstrings, log messages)
- **User-facing text:** NEVER hardcoded in Python. Goes in YAML configs:
  - `config/locale/en_us.yaml` — base locale (ALL keys must exist here)
  - `config/locale/pt_br.yaml` — override only user-facing strings
  - `config/prompts.yaml` — LLM prompt templates (English, locale-agnostic)
  - `config/prompts/tizerdral.yaml` — domain-specific prompt overrides
  - `config/playbooks/tizerdral.yaml` — sales flow definitions + messages

### Configuration Pattern
```python
# CORRECT — load from locale
locale = get_locale()
ctx = locale.get("classify_intent", {})
messages[0]["content"] += ctx.get("active_flow_context", "").format(flow=active_flow)

# WRONG — hardcoded Portuguese in Python
messages[0]["content"] += f"IMPORTANTE: Há um fluxo de VENDA ativo..."
```

### When Adding New Features
1. Define text templates in `config/locale/en_us.yaml` (English base)
2. Override user-facing text in `config/locale/pt_br.yaml` (Portuguese)
3. Reference from Python via `get_locale()` dict access
4. Regex patterns, strip rules → `response_sanitization` section in locale YAML

## Environment

- **Python venv:** `.venv/` — use `.venv/bin/python`
- **CUDA:** Always set `CUDA_VISIBLE_DEVICES=""` for all Python commands (no GPU needed)
- **Server:** `uvicorn app:app --host 0.0.0.0 --port 8001`
- **Domain:** `BOT_DOMAIN=tizerdral`, `BOT_LOCALE=pt_br`
- **Groq:** 6 API keys (comma-separated in GROQ_API_KEY). Rate limiting is aggressive — retries 1-50s
- **Embeddings:** Google Gemini Embedding 2, 5 keys, dimension=768
- **Port conflicts:** Use `fuser -k 8001/tcp` before restarting server

## Testing

- **Test script:** `scripts/test_flow_conversations.py` — 24 scenarios
- **Run specific:** `CUDA_VISIBLE_DEVICES="" TEST_DELAY=8 .venv/bin/python scripts/test_flow_conversations.py 7 9`
- **TEST_DELAY:** seconds between scenarios (rate limit mitigation)
- **Scenario 9** is slow (~60-90s) due to multiple LLM condition evaluations + Groq rate limits

### Test Patterns
- After `send_sequence` (photos at step 4), send another `chat()` before checking payment/closing content
- `route == "playbook"` means literal messages — `[IMAGEM:]` tags are legitimate there
- Asset testing: use specific file path (e.g., `/assets/tizerdral/foto_caixa_tizerdral.jpg`), not directory listing

## Playbook Structure (tizerdral.yaml)

```yaml
flows:
  venda_cliente_novo:
    trigger: { intent: sales }
    steps:
      - step 0: send greeting
      - step 1: wait_response
      - step 2: condition (client_already_uses_similar_product)
      - step 3: wait_response
      - step 4: send_sequence (5 messages with photos)  # ← important for tests
      - step 5: wait_response
      - step 6-8: conditions (asks_payment_method, asks_about_tg, asks_about_tirzec)
      - step 9: goto_flow: fechamento_venda
```

### condition_hints
Defined in playbook YAML under `condition_hints:`. Each has:
- `description`: Natural language for LLM evaluation
- `keywords_true`: Fast substring matches (skips LLM call)
- `check_history`: Whether to include memory_context in matching

## Production Hardening (Implemented)

- **SQLite WAL:** `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`
- **Per-user async lock:** `asyncio.Lock()` per `customer_id` in `app.py` — prevents race conditions
- **`asyncio.to_thread`:** `run_agent` executes in thread pool, lock serializes same-user requests
- **Response sanitization:** Strips `[IMAGEM:]`, `[IMAGE:]`, `[FOTO:]`, `assets/...`, `---MSG---` from LLM output
- **Key rotation:** 6 Groq keys with automatic rotation on 429
- **BM25 normalization:** Already normalized 0-1 in `bm25_index.py` before hybrid merge

## Internal Patterns & Pitfalls

### Keyword Matching — Two Different Mechanisms
There are **two separate keyword matching systems**. Confusing them causes subtle bugs:

1. **`_keyword_precheck()`** — used for condition_hints evaluation
   - Substring matching: `if kw in combined` (case insensitive)
   - Pads user message with spaces: `f" {user_msg.lower().strip()} "` → so `" tg "` matches end-of-message
   - Supports multi-word keywords naturally ("quero tg" works as substring)

2. **Farewell keyword pre-check** — used in `classify_intent()` for farewell detection
   - Uses **two separate sets**: `farewell_single` (word-level via `set.split()`) and `farewell_multi` (substring matching)
   - Single-word: `farewell_single & set(msg_lower.split())` — fast intersection
   - Multi-word: `any(kw in msg_lower for kw in farewell_multi)` — substring scan
   - **PITFALL:** If you put a multi-word keyword (e.g., "até logo") in the single-word set, it will NEVER match because `set.split()` breaks it into individual words

### `_recent_farewell()` — Last Assistant Message Only
- Checks only the **last assistant line** in `memory_context`, not the entire history
- Parses `memory_context` line-by-line in reverse, looking for `assistant:` prefix
- **PITFALL:** If this ever scans the full memory_context, a farewell keyword from early in a long conversation would permanently block all future flows for that customer

### `_inline_collect_target_flow()` — DRY Helper for Flow Step Collection
- Extracted helper that collects literal messages from a target flow's steps
- Used by `_try_direct_flow_response()` for: main loop, `branch_goto`, and `goto_flow` handlers
- Handles step types: `send`, `send_sequence`, `wait_response`, `condition` (recursive), `generate_response`, nested `goto_flow`
- Returns `(flow_name, flow_step, steps_list) | None`
- **Any changes to flow step handling must go through this single helper** — never duplicate the logic

### Feedback Guards — Consolidated Threshold
- Very short messages (≤5 chars like "?", "ok", "sim") are classified as `general` with `needs_retrieval=True` in `classify_intent()`
- Defense-in-depth: `check_feedback()` also skips feedback logic for messages ≤5 chars
- **PITFALL:** If thresholds diverge across functions, edge cases like "?" may get misclassified as feedback/out_of_scope instead of continuing the active flow

### Post-Farewell Behavior
- After farewell, if user sends another message (e.g., "ah ok"), `_recent_farewell()` detects it
- `classify_intent()` overrides greeting → `general` to prevent flow restart
- `_try_direct_flow_response()` skips flow execution
- `generate_response()` appends a locale-based suffix (`post_farewell_note.prompt_suffix`) instructing the LLM to respond briefly without restarting sales
- The suffix text lives in `config/locale/en_us.yaml` (English) and `config/locale/pt_br.yaml` (Portuguese override)

## Known Issues

- **ChromaDB dimension mismatch:** Expects 384, gets 768. Only keyword/BM25 search works
- **`client_already_uses_similar_product`:** "nunca usei" matches keyword "usei" → false positive (by design, not critical)
- **Groq rate limits:** Scenarios with multiple LLM calls (conditions) can take 60-90s due to 429 retries
- **`_user_locks` dict:** Grows unbounded — needs cleanup for long-running production (TTL eviction)
