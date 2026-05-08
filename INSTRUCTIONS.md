# ZapPilot IA — General AI Instructions

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

## File Map (what lives where)

| File | Purpose |
| --- | --- |
| `app.py` | FastAPI — `/chat`, `/feedback`, `/health`, `/admin/*` endpoints. Per-user async locks. Splits `---MSG---` into `response_parts` |
| `agent/graph.py` | LangGraph agentic flow. ALL core logic: classify, retrieve, generate, flow execution, condition eval |
| `agent/prompts.py` | Builds system prompt from YAML templates + domain config |
| `config/__init__.py` | Centralized loader: domain, prompts, settings, locale, playbooks. `get_locale()` deep-merges `en_us` + override |
| `config/prompts.yaml` | LLM prompt templates (English, locale-agnostic) |
| `config/prompts/tizerdral.yaml` | Domain-specific prompt overrides |
| `config/settings.yaml` | Thresholds, tuning params (similarity, clustering, retrieval weights) |
| `config/domains/<domain>.yaml` | Products, keywords, noise terms, references per domain |
| `config/locale/en_us.yaml` | Base locale — ALL keys must exist here. LLM instructions in English |
| `config/locale/pt_br.yaml` | Override — only user-facing strings (errors, handoff) in Portuguese |
| `config/playbooks/<domain>.yaml` | Sales flows, instructions, messages, condition_hints |
| `llm/providers.py` | LLM interface — Groq primary, llama.cpp local fallback. Multi-key rotation on 429 |
| `llm/embeddings.py` | Google Gemini Embedding 2 (768d) primary, SentenceTransformers (384d) fallback |
| `retrieval/hybrid_retriever.py` | 65% semantic (Chroma) + 35% keyword (BM25) with weighted reranking |
| `retrieval/chroma_index.py` | ChromaDB indexing/search — domain-aware metadata filter |
| `retrieval/bm25_index.py` | BM25 indexing/search — domain-aware. Already normalized 0-1 |
| `memory/sqlite_memory.py` | Persistent memory: history, customer facts, flow state, knowledge gaps. WAL mode |
| `ingestion/whatsapp_loader.py` | Loads exported WhatsApp JSON |
| `ingestion/msgstore_loader.py` | Loads decrypted msgstore.db (WhatsApp Android SQLite) |
| `preprocessing/cleaner.py` | Cleans messages + semantic noise/spam/feedback detection |
| `preprocessing/turns.py` | Groups messages into semantic Q&A turns |
| `kb/build_knowledge_base.py` | Builds KB via DBSCAN clustering on embeddings |
| `kb/generate_domain_kb.py` | LLM-generated complementary KB entries |
| `kb/generate_domain_config.py` | LLM-generated noise/spam/feedback terms |
| `kb/extract_patterns.py` | Extracts real noise/spam patterns from WhatsApp data |
| `scripts/build_all.py` | Full pipeline: ingest → clean → turns → KB → index |
| `scripts/test_flow_conversations.py` | 24 end-to-end flow test scenarios |
| `scripts/test_chat.py` | Interactive CLI for manual testing |

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

### Flow Actions
`text`, `image`, `wait_response`, `condition`, `goto_flow`, `generate_response`, `send_sequence`, `escalate`, `end`

### Flow Pause/Resume Logic
| Scenario | Behavior |
| --- | --- |
| Customer mid-**sales** asks **info** | Flow **pauses** — LLM answers freely, state preserved |
| Customer returns to **sales** | Flow **resumes** from saved step |
| Customer asks for **human** or gives **feedback** | Flow **abandoned** (state cleared) |
| Flow reaches last step | **Completed** automatically |

## Production Hardening (Implemented)

- **SQLite WAL:** `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`
- **Per-user async lock:** `asyncio.Lock()` per `customer_id` in `app.py` — prevents race conditions
- **`asyncio.to_thread`:** `run_agent` executes in thread pool, lock serializes same-user requests
- **Response sanitization:** Strips `[IMAGEM:]`, `[IMAGE:]`, `[FOTO:]`, `assets/...`, `---MSG---` from LLM output
- **Key rotation:** 6 Groq keys with automatic rotation on 429
- **BM25 normalization:** Already normalized 0-1 in `bm25_index.py` before hybrid merge

## Known Issues

- **ChromaDB dimension mismatch:** Expects 384, gets 768. Only keyword/BM25 search works
- **`client_already_uses_similar_product`:** "nunca usei" matches keyword "usei" → false positive (by design, not critical)
- **Groq rate limits:** Scenarios with multiple LLM calls (conditions) can take 60-90s due to 429 retries
- **`_user_locks` dict:** Grows unbounded — needs cleanup for long-running production (TTL eviction)
