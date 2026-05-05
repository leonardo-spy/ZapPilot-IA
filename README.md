# ZapPilot IA

Chatbot agentic de vendas e suporte técnico com RAG (Retrieval-Augmented Generation), construído a partir de conversas reais do WhatsApp.

O sistema ingere exportações de chats do WhatsApp, extrai padrões de atendimento, constrói uma base de conhecimento via clustering semântico e responde perguntas de clientes usando busca híbrida (semântica + keywords) com LLM.

---

## Arquitetura

```text
                    ┌─ WhatsApp JSON (whatsapp_chats.json)
Fonte de dados ───┼                                       ───→ Ingestão → Limpeza → Turns → KB (DBSCAN) → Chroma + BM25
                    └─ msgstore.db (SQLite decriptado)                                                         │
                                                                                                              ▼
                    Cliente ←─── LLM (Groq) ←─── Retriever Híbrido (65% semântico + 35% keyword)
                       ↕              ↑
                   Memória       LangGraph (Agentic Flow)
                   (SQLite)     load_memory → classify → retrieve → generate
                                                                       ↑
                                                        ┌──────────────┴──────────────┐
                                                        │     System Prompt Final     │
                                                        ├─────────────────────────────┤
                                                        │ 1. Persona (prompts.yaml)   │
                                                        │ 2. Playbook instructions    │
                                                        │ 3. Flow ativo (roteiro)     │
                                                        │ 4. RAG (docs recuperados)   │
                                                        │ 5. Memória do cliente       │
                                                        └─────────────────────────────┘

Config (YAML):
  config/domains/<domain>.yaml     → produtos, keywords, references, noise
  config/prompts.yaml              → templates de persona/regras
  config/settings.yaml             → thresholds e tuning
  config/playbooks/<domain>.yaml   → instruções do dono + mensagens + flows
```

### Stack

| Componente | Tecnologia |
| --- | --- |
| LLM | Groq (llama-3.1-8b-instant) / llama.cpp local |
| Embeddings (primário) | Google Gemini Embedding 2 (768 dims) |
| Embeddings (fallback) | SentenceTransformers all-MiniLM-L6-v2 (384 dims) |
| Busca semântica | ChromaDB |
| Busca keyword | BM25 (rank-bm25) |
| Orquestração | LangGraph |
| Memória | SQLite |
| API | FastAPI |
| Clustering | DBSCAN (scikit-learn) |

---

## Estrutura do Projeto

```text
ZapPilot IA/
├── app.py                      # FastAPI — endpoints /chat, /feedback, /health
├── agent/
│   ├── graph.py                # LangGraph agentic flow (classify → retrieve → generate)
│   └── prompts.py              # System prompts (carrega domain config do YAML)
├── config/
│   ├── __init__.py             # Loader centralizado (domínio, prompts, settings, playbooks)
│   ├── prompts.yaml            # Templates de system prompts (editável sem código)
│   ├── settings.yaml           # Thresholds e parâmetros de tuning
│   ├── domains/
│   │   ├── android_box.yaml    # Config domínio: produtos, keywords, references, short_noise
│   │   └── tirzepatida.yaml    # Config domínio Tirzepatida/Tirzec
│   └── playbooks/
│       ├── android_box.yaml    # Roteiros de conversa (flows, mensagens, instruções)
│       └── tirzepatida.yaml    # Playbook Tirzepatida
├── ingestion/
│   ├── whatsapp_loader.py      # Ingestão de JSON exportado + auto-detect fonte
│   └── msgstore_loader.py      # Ingestão direta do msgstore.db decriptado (SQLite)
├── preprocessing/
│   ├── cleaner.py              # Limpeza + detecção semântica de noise/spam/feedback
│   └── turns.py                # Construção de turns semânticos (pergunta + resposta)
├── kb/
│   ├── build_knowledge_base.py # KB via embeddings + DBSCAN clustering
│   ├── generate_domain_config.py # Geração de termos de domínio via LLM
│   ├── generate_domain_kb.py   # KB complementar gerada por LLM
│   └── extract_patterns.py     # Extração de padrões noise/spam do WhatsApp real
├── llm/
│   ├── providers.py            # Interface LLM (Groq, llama.cpp local, fallback)
│   └── embeddings.py           # Embedding providers (Google Gemini + SentenceTransformers)
├── retrieval/
│   ├── chroma_index.py         # Indexação e busca ChromaDB
│   ├── bm25_index.py           # Indexação e busca BM25
│   └── hybrid_retriever.py     # Retriever híbrido com reranking ponderado
├── memory/
│   └── sqlite_memory.py        # Memória persistente (histórico, fatos, casos, flow state)
├── scripts/
│   ├── build_all.py            # Pipeline completo de build (ingestão → indexação)
│   ├── review_kb.py            # Revisão manual da KB gerada
│   └── test_chat.py            # CLI interativo para testar o chatbot
├── web/
│   └── index.html              # Interface web de chat
├── input/
│   ├── whatsapp_chats.json     # Dados brutos do WhatsApp em JSON (não versionado)
│   └── msgstore.db             # Banco SQLite decriptado do WhatsApp (não versionado)
├── data/                        # Artefatos gerados (KB, Chroma, BM25, memória)
├── requirements.txt
├── .env.example
└── .env                         # Variáveis de ambiente (não versionado)
```

---

## Instalação

```bash
# Clone o repositório
git clone <repo-url> && cd "ZapPilot IA"

# Crie o venv
python3 -m venv .venv
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Configure o .env
cp .env.example .env
# Edite com suas chaves de API
```

---

## Configuração (.env)

```env
# LLM
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
LOCAL_LLM_URL=http://127.0.0.1:8081/v1      # Opcional: llama.cpp local

# Embeddings
GOOGLE_API_KEY=your_google_api_key
GOOGLE_EMBEDDING_MODEL=gemini-embedding-2
GOOGLE_EMBEDDING_DIM=768
EMBEDDING_MODEL=all-MiniLM-L6-v2             # Fallback local

# Rate limit (Google Embedding API)
GOOGLE_EMBEDDING_WAIT_ON_LIMIT=true          # true=aguarda e retenta, false=fallback imediato
GOOGLE_EMBEDDING_MAX_RETRIES=3               # Retries nos scripts (build, extract)
GOOGLE_EMBEDDING_SERVER_MAX_RETRIES=10       # Retries no servidor (mais tolerante)
GOOGLE_EMBEDDING_RPM=100                     # Free tier: 100 req/min

# Domínio
BOT_DOMAIN=android_box                       # android_box | tirzepatida | custom
COLLECTION_NAME=android_box_suporte

# Dados
DATA_DIR=./data
WHATSAPP_JSON=./input/whatsapp_chats.json
WHATSAPP_DB=./input/msgstore.db              # Banco SQLite decriptado do WhatsApp
WHATSAPP_SOURCE=auto                         # auto | json | db
CUDA_VISIBLE_DEVICES=                        # Forçar CPU para embeddings locais
```

---

## Uso

### 1. Build da Knowledge Base

Coloque a fonte de dados na pasta `input/`:

- **JSON:** `input/whatsapp_chats.json` (exportado via bot-zdg, whatsapp-forensic-tool, etc.)
- **SQLite:** `input/msgstore.db` (banco decriptado do WhatsApp Android)

O sistema auto-detecta a fonte disponível (`WHATSAPP_SOURCE=auto`), priorizando o `.db` se existir.

<details>
<summary><strong>💡 Como obter o msgstore.db (método WSA + Root)</strong></summary>

Se você não tem acesso a um dispositivo Android com root, pode usar o **WSA (Windows Subsystem for Android)** com root:

1. Instale o WSA com root via [WSABuilds](https://github.com/MustardChef/WSABuilds/)
2. Instale o WhatsApp (ou WA Business) dentro do WSA
3. Vincule o WhatsApp como **dispositivo adicional** (não precisa ser o principal — funciona como "extensão" do seu celular)
4. Com um explorador de arquivos com root (ex: MT Manager, Root Explorer), navegue até:

   ```text
   /data/data/com.whatsapp/databases/msgstore.db
   ```

   ou, em algumas versões:

   ```text
   /data/data/com.whatsapp.w4b/databases/msgstore.db
   ```

5. Copie o `msgstore.db` — ele já estará **decriptografado** (sem necessidade de key file)
6. Coloque em `input/msgstore.db`

> **Nota:** Como o WSA roda com root, o banco está acessível diretamente sem criptografia.
> Este método é mais simples que decriptar backups `crypt14/15`.

</details>

```bash
python scripts/build_all.py
```

O pipeline executa:

1. **Ingestão** — Carrega do JSON ou msgstore.db (auto-detect)
2. **Limpeza** — Remove spam, URLs, mensagens curtas, detecta noise via embeddings
3. **Turns** — Agrupa mensagens em pares pergunta/resposta semânticos
4. **Knowledge Base** — Clusteriza problemas similares (DBSCAN) e extrai respostas canônicas
5. **KB Complementar** — Gera entradas adicionais via LLM (opcional)
6. **Indexação Chroma** — Indexa a KB para busca semântica
7. **Indexação BM25** — Indexa para busca por keywords

### 2. Usando msgstore.db diretamente (opcional)

Se você tem o banco decriptado do WhatsApp (`msgstore.db`), pode usá-lo diretamente sem exportar para JSON:

```bash
# Ver estatísticas do banco
python -m ingestion.msgstore_loader ./input/msgstore.db --stats

# Exportar banco para JSON (compatível com pipeline)
python -m ingestion.msgstore_loader ./input/msgstore.db --export ./input/whatsapp_chats.json

# Ou simplesmente coloque o .db em input/ e rode build_all (auto-detect)
python scripts/build_all.py
```

> **Nota:** O banco precisa estar decriptado. Para decriptar backups crypt12/14/15, use o
> [whatsapp-forensic-tool](https://github.com/cedroid/whatsapp-forensic-tool) ou
> [wa-crypt-tools](https://github.com/ElDavoo/wa-crypt-tools).

### 3. Geração de Config de Domínio (opcional)

Gera termos de noise, spam e feedback via LLM para o domínio configurado:

```bash
python -m kb.generate_domain_config           # Gera config
python -m kb.generate_domain_config approve   # Aprova para uso
```

### 4. Extração de Padrões do WhatsApp (opcional)

Extrai padrões reais de noise/spam/feedback diretamente dos dados exportados:

```bash
python -m kb.extract_patterns                 # Extrai padrões
python -m kb.extract_patterns approve         # Aprova para uso
```

### 5. Iniciar o Servidor

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 6. Testar via CLI

```bash
python scripts/test_chat.py
```

### 7. Interface Web

Acesse `http://localhost:8000/web` após iniciar o servidor.

---

## API

### POST /chat

```json
{
  "customer_id": "5511999999999",
  "message": "meu box está travando"
}
```

Resposta:

```json
{
  "response": "Vou te ajudar! Primeiro, tente reiniciar o box...",
  "intent": "suporte",
  "route": "rag",
  "confidence": 0.92,
  "retrieved_docs": [...]
}
```

### POST /feedback

```json
{
  "customer_id": "5511999999999",
  "feedback": "resolved"
}
```

### GET /health

Retorna status do serviço.

---

## Fluxo Agentic (LangGraph)

```text
load_memory → classify_intent → check_feedback ─┬─→ feedback_response
                                                 │
                                                 └─→ retrieve → generate_response → save_memory
                                                                      ↑
                                                              playbook context
                                                     (instructions + flow + messages)
```

1. **load_memory** — Carrega histórico, fatos do cliente e **flow state** persistido (SQLite)
2. **classify_intent** — LLM classifica intenção (venda, suporte, geral) + confiança
3. **check_feedback** — Detecta feedback via embeddings semânticos + LLM
4. **retrieve** — Busca híbrida Chroma (65%) + BM25 (35%) com reranking
5. **generate_response** — LLM gera resposta com: system prompt + playbook (instruções + roteiro ativo a partir do step atual) + RAG + memória
6. **save_memory** — Persiste interação, atualiza contexto e **avança/pausa/limpa flow state**

O `generate_response` monta o prompt final combinando:

- System prompt (persona/regras do `config/prompts.yaml`)
- Playbook instructions + flow ativo (selecionado por intent + estado do cliente)
- Documentos recuperados (RAG)
- Memória do cliente (histórico + fatos)

---

## Detecção Semântica de Noise/Feedback

O sistema usa 3 níveis de termos para filtragem, em ordem de prioridade:

1. **Extraídos do WhatsApp** (`kb.extract_patterns`) — Padrões reais dos dados
2. **Gerados por LLM** (`kb.generate_domain_config`) — Expansão inteligente do domínio
3. **Base do domínio** (`config/domains/<domain>.yaml`) — Configurado via YAML

A detecção compara embeddings da mensagem contra embeddings dos termos de referência (definidos no YAML do domínio em `references`) via similaridade coseno. Thresholds configuráveis em `config/settings.yaml`.

---

## Embeddings

O provider de embeddings usa **Google Gemini Embedding 2** como primário com fallback automático para **SentenceTransformers** local:

- **Task prefixes** no texto (padrão do gemini-embedding-2):
  - Documentos: `"title: {title} | text: {content}"`
  - Queries: `"task: question answering | query: {text}"`
  - Classificação: `"task: classification | query: {text}"`
  - Clustering: `"task: clustering | query: {text}"`

- **Rate limit handling**: Retry automático com backoff ao receber 429, throttle preventivo baseado em RPM

---

## Domínios Suportados

O bot é multi-domínio, configurável via `BOT_DOMAIN` no `.env`.  
Cada domínio é um arquivo YAML em `config/domains/` (padrão inspirado no [Quivr](https://github.com/QuivrHQ/quivr)):

| Domínio | Arquivo | Descrição |
| --- | --- | --- |
| `android_box` | `config/domains/android_box.yaml` | Vendas e suporte de Android Box / IPTV |
| `tirzepatida` | `config/domains/tirzepatida.yaml` | Vendas de Tirzepatida / Tirzec |

Para criar um novo domínio, copie um YAML existente e ajuste os valores:

```bash
cp config/domains/android_box.yaml config/domains/meu_dominio.yaml
# edite meu_dominio.yaml
# no .env: BOT_DOMAIN=meu_dominio
```

Cada YAML define: `name`, `products`, `keywords` (sale/support), `noise_terms`, `feedback` (positive/negative), `references` (embeddings de referência para classificação) e `short_noise`.

---

## Playbooks (Roteiros de Conversa)

Além da configuração de domínio, cada domínio tem um **playbook** em `config/playbooks/<domain>.yaml` com roteiros modulares:

```yaml
# Seções de um playbook:
instructions: |          # Texto livre do dono ("Eu vendo X, minhas regras são...")
  ...
messages:               # Templates de mensagens reutilizáveis
  boas_vindas_novo:
    type: text
    content: "Olá! 👋 ..."
  tabela_precos:
    type: text
    content: "📋 Valores..."
  tutorial_img:
    type: image
    content: "assets/tutorial.jpg"
    caption: "Como usar"
flows:                  # Cenários modulares com steps
  venda_cliente_novo:
    trigger:
      intent: venda
      condition: client_is_new
    steps:
      - action: send
        message: boas_vindas_novo
      - action: wait_response
      - action: condition
        if: "asks_price"
        then: ...
```

**Actions disponíveis:** `send`, `send_sequence`, `wait_response`, `condition`, `goto_flow`, `generate_response`, `escalate`, `end`

O playbook é injetado no system prompt como contexto — a LLM usa como guia mas adapta ao contexto real.

### Flow State Persistente

O estado do flow é salvo entre turnos na memória do cliente (`customer_facts`), permitindo conversas multi-turno:

| Cenário | Comportamento |
| --- | --- |
| Cliente no meio de **venda** e pergunta **info** | Flow **pausa** — LLM responde info livremente, state mantido |
| Cliente volta a falar de **venda** | Flow **retoma** do step onde parou |
| Cliente pede **suporte humano** ou dá **feedback** | Flow **abandonado** (state limpo) |
| Flow chega ao último step | **Concluído** automaticamente |

O prompt mostra apenas os steps restantes (a partir da posição atual), não o flow inteiro.

---

## Requisitos

- Python 3.12+
- Chave de API Groq (gratuita em groq.com)
- Chave de API Google (para Gemini Embedding 2, gratuita)
- Dados do WhatsApp em uma das formas:
  - JSON exportado (via bot-zdg, whatsapp-forensic-tool, etc.)
  - `msgstore.db` decriptado (extraido do Android + chave de backup)

---

## Créditos

- A lógica de leitura do `msgstore.db` em [ingestion/msgstore_loader.py](ingestion/msgstore_loader.py) foi baseada no
  [whatsapp-forensic-tool](https://github.com/cedroid/whatsapp-forensic-tool) por Cedroid, licenciado sob MIT.

---

## Licença

Este projeto é licenciado sob a [GPL-3.0](LICENSE) — veja o arquivo LICENSE para detalhes.

O módulo `ingestion/msgstore_loader.py` é derivado de código MIT ([whatsapp-forensic-tool](https://github.com/cedroid/whatsapp-forensic-tool)), compatível com GPL-3.0.
