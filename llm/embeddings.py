"""
Embedding Providers: Google Gemini Embedding 2 (primário) + SentenceTransformers (fallback).

Gemini Embedding 2:
- SDK: google-genai (from google import genai)
- Modelo: gemini-embedding-2
- Task prefixes no texto (não usa task_type param):
  - RAG Document: "title: {title} | text: {content}"
  - RAG Query: "task: question answering | query: {content}"
  - Classification: "task: classification | query: {content}"
  - Clustering: "task: clustering | query: {content}"
  - Similarity: "task: sentence similarity | query: {content}"
- Dimensão: 3072 padrão, truncável para 768/1536 via output_dimensionality
- Batch: Cada texto em types.Content separado para embeddings individuais
- Rate Limit: Retry com backoff configurável via GOOGLE_EMBEDDING_RETRY_*
"""
import os
import re
import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Literal

logger = logging.getLogger(__name__)

# Task types suportados
TaskType = Literal[
    "retrieval_document",  # Para indexar documentos no RAG
    "retrieval_query",     # Para queries de busca no RAG
    "classification",      # Para classificação / detecção de spam/feedback
    "clustering",          # Para clustering (DBSCAN, etc)
    "similarity",          # Para similaridade semântica
]


class EmbeddingProvider(ABC):
    """Interface base para provedores de embeddings."""

    @abstractmethod
    def encode(self, texts: Union[str, list[str]], task_type: TaskType = "retrieval_document", **kwargs) -> np.ndarray:
        """Gera embeddings para texto(s). Retorna array numpy [n_texts, dim]."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class GoogleEmbeddingProvider(EmbeddingProvider):
    """
    Google Gemini Embedding 2 via google-genai SDK.
    
    Usa task prefixes no conteúdo (não parâmetro task_type).
    Dimensão configurável via output_dimensionality (recomendado: 768).
    
    Rate Limit Handling:
    - Detecta erro 429 RESOURCE_EXHAUSTED
    - Extrai retry_delay da resposta de erro
    - Aguarda e retenta (configurável via .env)
    
    Configuração via .env:
    - GOOGLE_EMBEDDING_WAIT_ON_LIMIT=true  → aguarda e retenta (default: true)
    - GOOGLE_EMBEDDING_MAX_RETRIES=3       → máximo de retries (default: 3)
    - GOOGLE_EMBEDDING_RPM=100             → requests/min para throttle preventivo
    """

    # Mapeamento de task_type para prefixo no texto
    _TASK_PREFIXES = {
        "retrieval_document": lambda text, title=None: f"title: {title or 'none'} | text: {text}",
        "retrieval_query": lambda text, **_: f"task: question answering | query: {text}",
        "classification": lambda text, **_: f"task: classification | query: {text}",
        "clustering": lambda text, **_: f"task: clustering | query: {text}",
        "similarity": lambda text, **_: f"task: sentence similarity | query: {text}",
    }

    def __init__(self, api_key: str = None, model: str = "gemini-embedding-2", output_dimensionality: int = 768):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        self._output_dim = output_dimensionality

        # Rate limit config
        self._wait_on_limit = os.getenv("GOOGLE_EMBEDDING_WAIT_ON_LIMIT", "true").lower() == "true"
        self._max_retries = int(os.getenv("GOOGLE_EMBEDDING_MAX_RETRIES", "3"))
        self._rpm_limit = int(os.getenv("GOOGLE_EMBEDDING_RPM", "100"))

        # Server mode: permite override com mais retries (app.py seta antes de inicializar)
        server_retries = os.getenv("GOOGLE_EMBEDDING_SERVER_MAX_RETRIES")
        if server_retries:
            self._max_retries = int(server_retries)

        # Throttle: track request timestamps para prevenção
        self._request_times: list[float] = []

        from google import genai
        from google.genai import types
        self._client = genai.Client(api_key=self.api_key)
        self._types = types
        logger.info(
            f"Google Gemini Embedding inicializado: {model} (dim={output_dimensionality}, "
            f"wait_on_limit={self._wait_on_limit}, max_retries={self._max_retries}, rpm={self._rpm_limit})"
        )

    def name(self) -> str:
        return f"google/{self.model}"

    @property
    def dimension(self) -> int:
        return self._output_dim

    def _prepare_text(self, text: str, task_type: TaskType, **kwargs) -> str:
        """Aplica task prefix ao texto conforme gemini-embedding-2 requer."""
        prefix_fn = self._TASK_PREFIXES.get(task_type)
        if prefix_fn:
            return prefix_fn(text, **kwargs)
        return text

    def _throttle(self):
        """Throttle preventivo: aguarda se estiver perto do RPM limit."""
        now = time.time()
        # Remove timestamps com mais de 60s
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self._rpm_limit - 5:  # margem de 5
            # Calcular quanto falta para o mais antigo sair da janela
            oldest = self._request_times[0]
            wait_time = 60 - (now - oldest) + 0.5
            if wait_time > 0:
                logger.info(f"[rate-limit] Throttle preventivo: aguardando {wait_time:.1f}s (RPM: {len(self._request_times)}/{self._rpm_limit})")
                time.sleep(wait_time)

    def _extract_retry_delay(self, error_msg: str) -> float:
        """Extrai o tempo de retry da mensagem de erro 429."""
        # Procurar "retryDelay": "33s" ou "Please retry in 33.235994438s"
        match = re.search(r'retry\s*(?:Delay|in)\D*(\d+(?:\.\d+)?)\s*s', str(error_msg), re.IGNORECASE)
        if match:
            return float(match.group(1))
        return 60.0  # Default: 1 minuto

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Verifica se o erro é rate limit (429)."""
        error_str = str(error)
        return "429" in error_str or "RESOURCE_EXHAUSTED" in error_str

    def _identify_exhausted_quota(self, error_msg: str) -> str:
        """Identifica qual quota estourou a partir da mensagem de erro."""
        error_str = str(error_msg)
        # Procurar quotaMetric na resposta
        metric_match = re.search(r'quotaMetric["\s:]+([^"]+)', error_str)
        if metric_match:
            metric = metric_match.group(1)
            limit_match = re.search(r'quotaValue["\s:]+["\s]*(\d+)', error_str)
            limit = limit_match.group(1) if limit_match else "?"
            if "requests" in metric.lower():
                return f"RPM (Requests/min) - limit: {limit}/min"
            elif "tokens" in metric.lower():
                return f"TPM (Tokens/min) - limit: {limit}/min"
            return f"{metric} - limit: {limit}"

        # Fallback por keywords
        if "requests" in error_str.lower() or "per_minute" in error_str.lower():
            return "RPM (Requests/min)"
        elif "tokens" in error_str.lower():
            return "TPM (Tokens/min)"
        elif "day" in error_str.lower():
            return "RPD (Requests/day)"
        return "desconhecida (verifique o log completo)"

    def _call_with_retry(self, contents: list, config) -> object:
        """Executa chamada à API com retry em caso de rate limit."""
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                self._throttle()
                self._request_times.append(time.time())

                result = self._client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                return result

            except Exception as e:
                last_error = e

                if not self._is_rate_limit_error(e):
                    raise  # Erro não-recuperável

                retry_delay = self._extract_retry_delay(str(e))
                quota = self._identify_exhausted_quota(str(e))

                if not self._wait_on_limit:
                    logger.warning(
                        f"[rate-limit] 429 - Quota estourada: {quota}. "
                        f"GOOGLE_EMBEDDING_WAIT_ON_LIMIT=false → propagando erro para fallback."
                    )
                    raise

                if attempt >= self._max_retries:
                    logger.error(
                        f"[rate-limit] 429 - Max retries ({self._max_retries}) atingido. "
                        f"Quota: {quota}. Propagando erro."
                    )
                    raise

                logger.warning(
                    f"[rate-limit] 429 - Quota estourada: {quota}. "
                    f"Aguardando {retry_delay:.1f}s antes de retry "
                    f"(tentativa {attempt + 1}/{self._max_retries})..."
                )
                time.sleep(retry_delay)

        raise last_error

    def encode(self, texts: Union[str, list[str]], task_type: TaskType = "retrieval_document", **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        # Aplica task prefix a cada texto
        title = kwargs.get("title", None)
        prepared = [self._prepare_text(t, task_type, title=title) for t in texts]

        # Para embeddings separados, cada texto vai em um types.Content
        batch_size = kwargs.get("batch_size", 100)
        all_embeddings = []

        config = self._types.EmbedContentConfig(
            output_dimensionality=self._output_dim
        )

        for i in range(0, len(prepared), batch_size):
            batch = prepared[i:i + batch_size]

            # Wrap cada texto em Content para obter embeddings individuais
            contents = [
                self._types.Content(parts=[self._types.Part.from_text(text=t)])
                for t in batch
            ]

            result = self._call_with_retry(contents, config)

            for emb in result.embeddings:
                all_embeddings.append(emb.values)

        return np.array(all_embeddings, dtype=np.float32)


class SentenceTransformerProvider(EmbeddingProvider):
    """SentenceTransformers local (all-MiniLM-L6-v2 etc). Lazy loading — só carrega na primeira chamada."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # default para all-MiniLM-L6-v2

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Carregando SentenceTransformer: {self.model_name}...")
            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._dimension = self._model.get_embedding_dimension()

    def name(self) -> str:
        return f"local/{self.model_name}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Union[str, list[str]], task_type: TaskType = "retrieval_document", **kwargs) -> np.ndarray:
        self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        show_progress = kwargs.get("show_progress_bar", len(texts) > 10)
        batch_size = kwargs.get("batch_size", 64)
        return self._model.encode(texts, show_progress_bar=show_progress, batch_size=batch_size)


class FallbackEmbeddingProvider(EmbeddingProvider):
    """Provider com fallback: tenta o primário, se falhar usa secundário."""

    def __init__(self, providers: list[EmbeddingProvider]):
        if not providers:
            raise ValueError("Pelo menos um provider é necessário")
        self.providers = providers
        self._active = None

    def name(self) -> str:
        if self._active:
            return f"fallback(active={self._active.name()})"
        return f"fallback({', '.join(p.name() for p in self.providers)})"

    @property
    def dimension(self) -> int:
        if self._active:
            return self._active.dimension
        return self.providers[0].dimension

    def encode(self, texts: Union[str, list[str]], task_type: TaskType = "retrieval_document", **kwargs) -> np.ndarray:
        last_error = None

        for provider in self.providers:
            try:
                result = provider.encode(texts, task_type=task_type, **kwargs)
                self._active = provider
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Embedding provider {provider.name()} falhou: {e}")
                continue

        raise RuntimeError(f"Todos os embedding providers falharam. Último erro: {last_error}")


# ==================== SINGLETON ====================

_embedding_provider: EmbeddingProvider = None


def get_embedding_provider() -> EmbeddingProvider:
    """Retorna provider de embeddings configurado via .env."""
    global _embedding_provider
    if _embedding_provider is not None:
        return _embedding_provider

    from dotenv import load_dotenv
    load_dotenv()

    providers = []

    # Google Gemini Embedding 2 como primário (se tiver key)
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        try:
            google_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-2")
            output_dim = int(os.getenv("GOOGLE_EMBEDDING_DIM", "768"))
            providers.append(GoogleEmbeddingProvider(
                api_key=google_key,
                model=google_model,
                output_dimensionality=output_dim,
            ))
            logger.info(f"Google embeddings configurado: {google_model} (dim={output_dim})")
        except Exception as e:
            logger.warning(f"Falha ao inicializar Google embeddings: {e}")

    # SentenceTransformers como fallback
    local_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    providers.append(SentenceTransformerProvider(model_name=local_model))
    logger.info(f"Local embeddings configurado: {local_model}")

    if len(providers) == 1:
        _embedding_provider = providers[0]
    else:
        _embedding_provider = FallbackEmbeddingProvider(providers)

    logger.info(f"Embedding provider: {_embedding_provider.name()}")
    return _embedding_provider


def reset_provider():
    """Reset singleton (para testes)."""
    global _embedding_provider
    _embedding_provider = None


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    provider = get_embedding_provider()
    print(f"Provider: {provider.name()}")
    print(f"Dimensão: {provider.dimension}")

    # Teste com task_type para RAG
    docs = provider.encode(
        ["Como configurar o XCIPTV", "Minha box não conecta no wifi"],
        task_type="retrieval_document",
    )
    print(f"Docs shape: {docs.shape}")

    query = provider.encode(
        "box não conecta wifi",
        task_type="retrieval_query",
    )
    print(f"Query shape: {query.shape}")

    # Teste classificação (usado para detecção de spam/feedback)
    clf = provider.encode("obrigado, resolveu!", task_type="classification")
    print(f"Classification shape: {clf.shape}")
    print(f"Primeiros valores: {docs[0][:5]}")
