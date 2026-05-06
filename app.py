"""
FastAPI app — endpoints /chat, /feedback, /health
"""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa dependências no startup."""
    from agent.graph import init_dependencies
    from memory.sqlite_memory import SQLiteMemory
    from llm.providers import get_default_provider
    from llm.embeddings import get_embedding_provider

    logger.info("Inicializando dependências...")
    memory = SQLiteMemory(os.getenv("DATA_DIR", "./data") + "/memory.db")
    llm = get_default_provider()
    init_dependencies(memory=memory, llm=llm)
    logger.info(f"LLM Provider: {llm.name()}")

    # Pré-carregar embeddings (evita delay na primeira mensagem)
    embedding = get_embedding_provider()
    logger.info(f"Embedding Provider: {embedding.name()} (dim={embedding.dimension})")

    logger.info("Pronto!")
    yield


app = FastAPI(
    title="ZapPilot IA",
    description="Chatbot de vendas e suporte com RAG agentic",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir arquivos estáticos (web chat)
if os.path.isdir("web"):
    app.mount("/web", StaticFiles(directory="web", html=True), name="web")


# ==================== MODELS ====================

class ChatRequest(BaseModel):
    customer_id: str
    message: str


class ChatResponse(BaseModel):
    response: str
    response_parts: list[str] = []
    intent: str
    route: str
    confidence: float
    retrieved_docs: list = []


class FeedbackRequest(BaseModel):
    customer_id: str
    message_id: str = ""
    feedback: str  # "resolved" | "not_resolved"


class FeedbackResponse(BaseModel):
    status: str
    message: str


# ==================== ENDPOINTS ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint principal de chat."""
    from agent.graph import run_agent

    logger.info(f"[/chat] customer={request.customer_id}, msg='{request.message[:80]}'")

    result = run_agent(request.customer_id, request.message)

    # Separar mensagens múltiplas (playbook envia com separador)
    raw_response = result["response"]
    if "\n---MSG---\n" in raw_response:
        parts = [p.strip() for p in raw_response.split("\n---MSG---\n") if p.strip()]
    else:
        parts = [raw_response]

    return ChatResponse(
        response=raw_response.replace("\n---MSG---\n", "\n\n"),
        response_parts=parts,
        intent=result["intent"],
        route=result["route"],
        confidence=result["confidence"],
        retrieved_docs=result.get("retrieved_docs", []),
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    """Endpoint de feedback explícito."""
    from memory.sqlite_memory import SQLiteMemory

    memory = SQLiteMemory(os.getenv("DATA_DIR", "./data") + "/memory.db")

    if request.feedback == "resolved":
        cases = memory.get_open_cases(request.customer_id)
        for case in cases:
            memory.resolve_case(case["id"])
        return FeedbackResponse(status="ok", message="Caso marcado como resolvido")
    else:
        return FeedbackResponse(status="ok", message="Feedback registrado")


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "service": "ZapPilot IA",
        "version": "1.0.0",
    }


# ==================== ADMIN ENDPOINTS ====================

@app.get("/admin/knowledge-gaps")
async def knowledge_gaps(days: int = 30, top: int = 20, domain: str = None):
    """Returns knowledge gap analysis — what info the KB is missing. Filtered by domain."""
    import os
    from scripts.knowledge_gaps_report import generate_report

    if domain is None:
        domain = os.getenv("BOT_DOMAIN", "android_box")

    report = generate_report(days=days, top_n=top, domain=domain)
    return {"report": report, "domain": domain}


@app.get("/admin/knowledge-gaps/json")
async def knowledge_gaps_json(days: int = 30, limit: int = 100, domain: str = None):
    """Returns raw knowledge gaps data as JSON, filtered by domain."""
    import os
    from memory.sqlite_memory import SQLiteMemory

    if domain is None:
        domain = os.getenv("BOT_DOMAIN", "android_box")

    memory = SQLiteMemory(os.getenv("DATA_DIR", "./data") + "/memory.db")
    gaps = memory.get_knowledge_gaps(limit=limit, since_days=days, domain=domain)
    summary = memory.get_knowledge_gaps_summary(since_days=days, domain=domain)

    return {
        "domain": domain,
        "summary": summary,
        "gaps": gaps,
    }
