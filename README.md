# ZapPilot IA

Chatbot agentic de vendas e suporte técnico com RAG (Retrieval-Augmented Generation), construído a partir de conversas reais do WhatsApp.

O sistema ingere exportações de chats do WhatsApp, extrai padrões de atendimento, constrói uma base de conhecimento via clustering semântico e responde perguntas de clientes usando busca híbrida (semântica + keywords) com LLM. Suporta múltiplos domínios isolados, internacionalização (i18n) e detecção automática de lacunas no conhecimento.

---

## Arquitetura

```text
                    ┌─ WhatsApp JSON (whatsapp_chats.json)
Fonte de dados ───┼                                       ───→ Ingestão → Limpeza → Turns → KB (DBSCAN) → Chroma + BM25
                    └─ msgstore.db (SQLite decriptado)                                                         │
                                                                                                              ▼
                    Cliente ←─── LLM (Groq) ←─── Retriever Híbrido (65% semântico + 35% keyword)
                       ↕              ↑                         ↑
                   Memória       LangGraph (Agentic Flow)   domain filter
                   (SQLite)     load_memory → classify → retrieve → generate
                     ↑                                                 ↑
                  domain tag                            ┌──────────────┴──────────────┐
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
  config/locale/en_us.yaml         → locale base (LLM prompts, templates)
  config/locale/pt_br.yaml         → override user-facing (erros, handoff)
```

### Stack

| Componente | Tecnologia |
| --- | --- |
| LLM | Groq (llama-3.1-8b-instant) / llama.cpp local (fallback) |
| Embeddings (primário) | Google Gemini Embedding 2 (768 dims) |
| Embeddings (fallback) | SentenceTransformers all-MiniLM-L6-v2 (384 dims) |
| Busca semântica | ChromaDB (filtro por domain + category) |
| Busca keyword | BM25 (rank-bm25, filtro por domain) |
| Orquestração | LangGraph |
| Memória | SQLite (com isolamento por domain) |
| API | FastAPI |
| Clustering | DBSCAN (scikit-learn) |
| i18n | YAML locale com deep-merge fallback |

---

## Estrutura do Projeto

```text
ZapPilot IA/
├── app.py                      # FastAPI — endpoints /chat, /feedback, /health, /admin/*
├── agent/
│   ├── graph.py                # LangGraph agentic flow (classify → retrieve → generate)
│   └── prompts.py              # System prompts (carrega domain config do YAML)
├── config/
│   ├── __init__.py             # Loader centralizado (domínio, prompts, settings, locale, playbooks)
│   ├── prompts.yaml            # Templates de system prompts (editável sem código)
│   ├── settings.yaml           # Thresholds e parâmetros de tuning
│   ├── domains/
│   │   ├── android_box.yaml    # Config domínio: produtos, keywords, references
│   │   ├── tirzepatida.yaml    # Config domínio Tirzepatida/Tirzec
│   │   └── tizerdral.yaml      # Config domínio Tizerdral
│   ├── locale/
│   │   ├── en_us.yaml          # Locale base (TODAS as chaves — LLM prompts, templates)
│   │   └── pt_br.yaml          # Override — apenas strings user-facing em português
│   └── playbooks/
│       ├── android_box.yaml    # Roteiros de conversa Android Box
│       ├── tirzepatida.yaml    # Playbook Tirzepatida
│       └── tizerdral.yaml      # Playbook Tizerdral
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
│   ├── chroma_index.py         # Indexação e busca ChromaDB (domain-aware)
│   ├── bm25_index.py           # Indexação e busca BM25 (domain-aware)
│   └── hybrid_retriever.py     # Retriever híbrido com reranking ponderado
├── memory/
│   └── sqlite_memory.py        # Memória persistente (histórico, fatos, casos, flow state, knowledge gaps)
├── scripts/
│   ├── build_all.py            # Pipeline completo de build (ingestão → indexação)
│   ├── knowledge_gaps_report.py # Relatório de lacunas no conhecimento
│   ├── test_classification.py  # Testes de classificação de intent
│   ├── test_flow_conversations.py # Testes end-to-end de flows
│   ├── review_kb.py            # Revisão manual da KB gerada
│   └── test_chat.py            # CLI interativo para testar o chatbot
├── web/
│   └── index.html              # Interface web de chat
├── assets/                     # Imagens e mídia dos playbooks
├── input/
│   ├── whatsapp_chats.json     # Dados brutos do WhatsApp em JSON (não versionado)
│   └── msgstore.db             # Banco SQLite decriptado do WhatsApp (não versionado)
├── data/                       # Artefatos gerados (KB, Chroma, BM25, memória)
├── requirements.txt
├── .env.example
└── .env                        # Variáveis de ambiente (não versionado)
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
LOCAL_LLM_MODEL=prism-ml/Bonsai-8B-gguf:Q1_0

# Embeddings
GOOGLE_API_KEY=your_google_api_key
GOOGLE_EMBEDDING_MODEL=gemini-embedding-2
GOOGLE_EMBEDDING_DIM=768
EMBEDDING_MODEL=all-MiniLM-L6-v2             # Fallback local

# Rate limit (Google Embedding API)
GOOGLE_EMBEDDING_WAIT_ON_LIMIT=true          # true=aguarda e retenta, false=fallback imediato
GOOGLE_EMBEDDING_MAX_RETRIES=3               # Retries nos scripts (build, extract)
GOOGLE_EMBEDDING_RPM=100                     # Free tier: 100 req/min

# Domínio
BOT_DOMAIN=tizerdral                         # android_box | tirzepatida | tizerdral

# Locale (override de strings user-facing)
BOT_LOCALE=pt_br                             # pt_br | en_us (default: pt_br)

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
6. **Indexação Chroma** — Indexa a KB para busca semântica (com tag `domain`)
7. **Indexação BM25** — Indexa para busca por keywords (com tag `domain`)

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
uvicorn app:app --host 0.0.0.0 --port 8001
```

### 6. Testar via CLI

```bash
python scripts/test_chat.py
```

### 7. Interface Web

Acesse `http://localhost:8001/web` após iniciar o servidor.

---

## API

### POST /chat

```json
{
  "customer_id": "5511999999999",
  "message": "quanto custa a tirzepatida?"
}
```

Resposta:

```json
{
  "response": "A tirzepatida está por R$ 1.490,00...",
  "response_parts": ["A tirzepatida está por R$ 1.490,00..."],
  "intent": "sales",
  "route": "playbook",
  "confidence": 0.92,
  "retrieved_docs": [...]
}
```

> **`response_parts`**: Quando o playbook envia mensagens múltiplas (separadas por `\n---MSG---\n`), cada item é uma mensagem individual para enviar ao WhatsApp sequencialmente.

### POST /feedback

```json
{
  "customer_id": "5511999999999",
  "feedback": "resolved"
}
```

### GET /health

Retorna status do serviço.

### GET /admin/knowledge-gaps?days=30&top=20&domain=tizerdral

Retorna relatório de gaps de conhecimento (tópicos que a KB não cobre bem). Filtrado por domain.

### GET /admin/knowledge-gaps/json?days=30&limit=100&domain=tizerdral

Retorna dados brutos de knowledge gaps para uso programático.

---

## Fluxo Agentic (LangGraph)

```text
load_memory → classify_intent → check_feedback ─┬─→ feedback_response
                                                 │
                                                 └─→ retrieve → generate_response → save_memory
                                                         ↑               ↑
                                                   domain filter   playbook context
                                                                (instructions + flow + messages)
```

1. **load_memory** — Carrega histórico (filtrado por domain ativo), fatos do cliente e **flow state** persistido
2. **classify_intent** — LLM classifica intenção (`greeting`, `sales`, `info`, `support`, `billing`, `renewal`, `feedback_positive`, `feedback_negative`, `human`, `out_of_scope`) + confiança
3. **check_feedback** — Detecta feedback via embeddings semânticos + LLM
4. **retrieve** — Busca híbrida Chroma (65%) + BM25 (35%) com reranking, **filtrada por domain ativo**
5. **generate_response** — LLM gera resposta com: system prompt + playbook (instruções + roteiro ativo a partir do step atual) + RAG + memória
6. **save_memory** — Persiste interação (com tag domain), atualiza contexto, **avança/pausa/limpa flow state**, e detecta knowledge gaps

### Keyword Precheck (Condition Evaluation)

Quando o playbook tem uma `condition` (ex: "já usou o produto?"), o sistema primeiro tenta responder via keywords sem chamar a LLM:

- Palavras positivas (já, sim, usei, conheço, etc.) → `true`
- Palavras negativas (não, nunca, nenhum, etc.) → `false`
- Sem match → LLM avalia e responde `yes`/`no`

Isso reduz chamadas à API e melhora a latência.

### Knowledge Gap Detection

Após cada interação, o sistema detecta quando faltou contexto para responder bem:

| Condição | Gap detectado |
| --- | --- |
| Route = `no_context` (nenhum doc recuperado) | ✅ |
| `needs_retrieval` mas 0 docs retornados | ✅ |
| Confiança < 0.4 e ≤ 1 doc retornado | ✅ |
| Route = `playbook` | ❌ (tem roteiro) |

Gaps são gravados no SQLite com `domain` e analisáveis via `/admin/knowledge-gaps`.

---

## Isolamento por Domain (Multi-tenant)

Todos os dados são isolados por `BOT_DOMAIN`:

| Componente | Isolamento |
| --- | --- |
| **ChromaDB** | Metadata `domain` em cada doc — busca filtra por domain |
| **BM25** | Index taggeado com domain — rejeita resultados cross-domain |
| **Conversations** | Coluna `domain` — histórico filtrado por domain ativo |
| **Knowledge Gaps** | Coluna `domain` — relatórios filtrados |
| **Playbooks** | Arquivo `config/playbooks/<domain>.yaml` |
| **Domain Config** | Arquivo `config/domains/<domain>.yaml` |

Uma única instância do bot serve um domain por vez (`BOT_DOMAIN`). Para operar múltiplos domínios simultaneamente, rode instâncias separadas com `BOT_DOMAIN` diferente.

A collection ChromaDB é compartilhada (nome padrão: `knowledge_base`), com separação feita via metadata filter. Isso permite que um mesmo Chroma DB sirva múltiplos domains sem conflito.

---

## Internacionalização (i18n)

O sistema usa locale YAML com deep-merge:

```text
config/locale/
├── en_us.yaml   ← BASE (todas as chaves definidas aqui)
└── pt_br.yaml   ← OVERRIDE (apenas strings user-facing)
```

**Como funciona:**

1. `en_us.yaml` é carregado como base (contém templates de LLM, instruções de flow, condition eval, etc.)
2. O override regional (ex: `pt_br.yaml`) é deep-merged por cima
3. Chaves não presentes no override caem automaticamente para en_us

**Seções do locale:**

| Seção | Propósito | Override em pt_br? |
| --- | --- | --- |
| `flow_format.*` | Templates de formatação de flows para o LLM | ❌ (inglês) |
| `playbook_context.*` | Headers de contexto do playbook | ❌ (inglês) |
| `condition_eval.*` | Instruções para avaliar condições (yes/no) | ❌ (inglês) |
| `human_handoff.*` | Mensagem de encaminhamento p/ humano | ✅ (português) |
| `errors.*` | Mensagens de erro (dificuldade técnica, etc.) | ✅ (português) |

> **Por que inglês no LLM?** A LLM (llama-3.1-8b-instant) funciona melhor com instruções em inglês, mesmo processando conteúdo em português. As mensagens finais ao cliente saem em português via playbook.

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
| `tizerdral` | `config/domains/tizerdral.yaml` | Vendas de Tizerdral (tirzepatida) |

Para criar um novo domínio:

```bash
# 1. Criar configs
cp config/domains/tizerdral.yaml config/domains/meu_dominio.yaml
cp config/playbooks/tizerdral.yaml config/playbooks/meu_dominio.yaml

# 2. Editar com dados do novo domínio
# 3. Atualizar .env
# BOT_DOMAIN=meu_dominio

# 4. Rebuild da KB
python scripts/build_all.py
```

Cada domínio YAML define: `name`, `description`, `products`, `keywords` (sale/support), `noise_terms`, `feedback` (positive/negative), `references` (embeddings de referência).

---

## Playbooks (Roteiros de Conversa)

Cada domínio tem um **playbook** em `config/playbooks/<domain>.yaml` com roteiros modulares:

```yaml
instructions: |          # Texto livre do dono ("Eu vendo X, minhas regras são...")
  ...
messages:               # Templates de mensagens reutilizáveis
  abertura:
    type: text
    content: "Olá! 👋 ..."
  tabela_precos:
    type: image
    content: "assets/tizerdral/foto_preco.jpeg"
    caption: "Valores atuais"
flows:
  venda_cliente_novo:
    name: "Venda - Cliente Novo"
    trigger:
      intent: sales
      condition: client_is_new
    priority: 10
    steps:
      - action: text
        content: "Mensagem literal enviada ao cliente"
      - action: wait_response
      - action: condition
        if: "already_used_similar_product"
        keywords:
          positive: [ja, sim, usei, conheço, ozempic]
          negative: [não, nunca, nao, nenhum, primeiro]
        then:
          - action: text
            content: "Mensagem se SIM"
        else:
          - action: text
            content: "Mensagem se NÃO"
      - action: generate_response
        context: "preços e condições"
```

**Actions disponíveis:** `text`, `image`, `wait_response`, `condition`, `goto_flow`, `generate_response`, `escalate`, `end`

**Triggers por intent:** `sales`, `info`, `support`, `greeting`

### Flow State Persistente

O estado do flow é salvo entre turnos na memória do cliente (`customer_facts`), permitindo conversas multi-turno:

| Cenário | Comportamento |
| --- | --- |
| Cliente no meio de **sales** e pergunta **info** | Flow **pausa** — LLM responde info livremente, state mantido |
| Cliente volta a falar de **sales** | Flow **retoma** do step onde parou |
| Cliente pede **suporte humano** ou dá **feedback** | Flow **abandonado** (state limpo) |
| Flow chega ao último step | **Concluído** automaticamente |

O prompt mostra apenas os steps restantes (a partir da posição atual), não o flow inteiro.

---

## Scripts de Teste

```bash
# Teste interativo via CLI
python scripts/test_chat.py

# Teste de classificação de intents
python scripts/test_classification.py

# Teste end-to-end de flows (cenários completos)
python scripts/test_flow_conversations.py

# Relatório de knowledge gaps
python scripts/knowledge_gaps_report.py --days 30 --top 20 --domain tizerdral
```

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
