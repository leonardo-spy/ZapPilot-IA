"""
LLM Providers: interface unificada para Groq, llama.cpp local e fallbacks.
"""
import os
import logging
from abc import ABC, abstractmethod
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Interface base para provedores de LLM."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        """Envia mensagens e retorna resposta."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class GroqProvider(LLMProvider):
    """Provider Groq via API compatível com OpenAI."""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def name(self) -> str:
        return f"groq/{self.model}"

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        if stream:
            content = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                content += delta
            return content
        else:
            return response.choices[0].message.content or ""


class LocalLlamaProvider(LLMProvider):
    """Provider llama.cpp local via endpoint compatível com OpenAI."""

    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:8081/v1")
        self.model = model or os.getenv("LOCAL_LLM_MODEL", "local-model")
        self.client = OpenAI(
            api_key="not-needed",
            base_url=self.base_url,
        )

    def name(self) -> str:
        return f"local/{self.model}"

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        if stream:
            content = ""
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                content += delta
            return content
        else:
            return response.choices[0].message.content or ""


class FallbackProvider(LLMProvider):
    """Provider com fallback: tenta o primeiro, se falhar tenta o próximo."""

    def __init__(self, providers: list[LLMProvider]):
        if not providers:
            raise ValueError("Pelo menos um provider é necessário")
        self.providers = providers

    def name(self) -> str:
        return f"fallback({', '.join(p.name() for p in self.providers)})"

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        last_error = None

        for provider in self.providers:
            try:
                logger.debug(f"Tentando provider: {provider.name()}")
                result = provider.chat(messages, temperature, max_tokens, stream=False)
                logger.debug(f"Sucesso com: {provider.name()}")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider.name()} falhou: {e}")
                continue

        raise RuntimeError(f"Todos os providers falharam. Último erro: {last_error}")


def get_default_provider() -> LLMProvider:
    """Retorna provider padrão com fallback configurado via .env."""
    from dotenv import load_dotenv
    load_dotenv()

    providers = []

    # Groq como primário (se tiver key)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        providers.append(GroqProvider(api_key=groq_key))

    # Local llama como fallback
    local_url = os.getenv("LOCAL_LLM_URL")
    if local_url:
        providers.append(LocalLlamaProvider(base_url=local_url))

    if not providers:
        raise RuntimeError(
            "Nenhum LLM provider configurado. Defina GROQ_API_KEY ou LOCAL_LLM_URL no .env"
        )

    if len(providers) == 1:
        return providers[0]

    return FallbackProvider(providers)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    provider = get_default_provider()
    print(f"Provider: {provider.name()}")

    response = provider.chat([
        {"role": "system", "content": "Responda em português, brevemente."},
        {"role": "user", "content": "Olá, tudo bem?"},
    ])
    print(f"Resposta: {response}")
