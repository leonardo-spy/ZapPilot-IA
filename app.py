"""
FastAPI app — endpoints /chat, /feedback, /health
"""
import asyncio
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

# Admin router (KB review, YAML editor, domain management)
from admin.router import router as admin_router
app.include_router(admin_router)

# Servir arquivos estáticos (web chat)
if os.path.isdir("web"):
    app.mount("/web", StaticFiles(directory="web", html=True), name="web")

# Admin UI
if os.path.isdir("admin/static"):
    app.mount("/admin-ui", StaticFiles(directory="admin/static", html=True), name="admin-ui")

# Servir assets (imagens de playbooks)
if os.path.isdir("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")


# ==================== MODELS ====================

class ChatRequest(BaseModel):
    customer_id: str
    message: str
    domain: str | None = None


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


# ==================== PER-USER LOCK ====================

_user_locks: dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()


async def _get_user_lock(user_id: str) -> asyncio.Lock:
    """Get or create an asyncio.Lock for a specific user (prevents race conditions)."""
    async with _locks_lock:
        if user_id not in _user_locks:
            _user_locks[user_id] = asyncio.Lock()
        return _user_locks[user_id]


# ==================== ENDPOINTS ====================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint principal de chat."""
    from agent.graph import run_agent

    logger.info(f"[/chat] customer={request.customer_id}, domain={request.domain}, msg='{request.message[:80]}'")

    lock = await _get_user_lock(request.customer_id)
    async with lock:
        result = await asyncio.to_thread(
            run_agent, request.customer_id, request.message, domain=request.domain
        )

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
