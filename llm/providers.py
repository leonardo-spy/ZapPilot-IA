"""
LLM Providers: interface unificada para NVIDIA Build, Groq, llama.cpp local e fallbacks.
Suporta múltiplas API keys com rotação automática (comma-separated no .env).
Hierarquia padrão: NVIDIA → Groq → Local.
"""
import os
import re
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
    """Provider Groq via API compatível com OpenAI. Suporta múltiplas keys com rotação."""

    def __init__(self, api_key: str = None, model: str = None):
        raw_keys = api_key or os.getenv("GROQ_API_KEY", "")
        self._api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if not self._api_keys:
            raise ValueError("Nenhuma GROQ_API_KEY configurada")
        self._current_key_idx = 0
        self._exhausted_keys: set[int] = set()

        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = OpenAI(
            api_key=self._api_keys[0],
            base_url="https://api.groq.com/openai/v1",
        )

        key_count = len(self._api_keys)
        logger.info(f"Groq inicializado: {self.model} ({key_count} key{'s' if key_count > 1 else ''})")

    def _is_rate_limit_error(self, error: Exception) -> bool:
        error_str = str(error)
        return "429" in error_str or "rate_limit" in error_str.lower() or "quota" in error_str.lower()

    def _is_daily_quota(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return "day" in error_str or "daily" in error_str

    def _extract_retry_delay(self, error: Exception) -> float:
        match = re.search(r'try again in (\d+(?:\.\d+)?)\s*s', str(error), re.IGNORECASE)
        if match:
            return float(match.group(1))
        # Groq format: "Please try again in 1m20.345s"
        match = re.search(r'try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s', str(error), re.IGNORECASE)
        if match:
            minutes = int(match.group(1) or 0)
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        return 30.0

    def _switch_to_next_key(self) -> bool:
        self._exhausted_keys.add(self._current_key_idx)
        for i in range(len(self._api_keys)):
            candidate = (self._current_key_idx + 1 + i) % len(self._api_keys)
            if candidate not in self._exhausted_keys:
                self._current_key_idx = candidate
                self.client = OpenAI(
                    api_key=self._api_keys[candidate],
                    base_url="https://api.groq.com/openai/v1",
                )
                key_masked = self._api_keys[candidate][:8] + "..."
                logger.info(
                    f"[key-rotation] Groq: rotacionando para key "
                    f"{candidate + 1}/{len(self._api_keys)} ({key_masked})"
                )
                return True
        logger.error(f"[key-rotation] Groq: todas as {len(self._api_keys)} keys esgotadas.")
        return False

    def name(self) -> str:
        return f"groq/{self.model}"

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        max_retries = 3
        wait_retries = 0
        max_attempts = max_retries + len(self._api_keys) * 2
        last_error = None

        for _ in range(max_attempts):
            try:
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

            except Exception as e:
                last_error = e

                if not self._is_rate_limit_error(e):
                    raise

                if self._is_daily_quota(e):
                    logger.warning(f"[rate-limit] Groq: quota diária esgotada na key {self._current_key_idx + 1}.")
                    if self._switch_to_next_key():
                        continue
                    raise

                # RPM: tentar rotacionar para outra key
                if len(self._api_keys) > 1:
                    available = [
                        i for i in range(len(self._api_keys))
                        if i != self._current_key_idx and i not in self._exhausted_keys
                    ]
                    if available:
                        next_key = available[0]
                        self._current_key_idx = next_key
                        self.client = OpenAI(
                            api_key=self._api_keys[next_key],
                            base_url="https://api.groq.com/openai/v1",
                        )
                        key_masked = self._api_keys[next_key][:8] + "..."
                        logger.info(f"[key-rotation] Groq: RPM hit, rotacionando para key {next_key + 1}/{len(self._api_keys)} ({key_masked})")
                        continue

                wait_retries += 1
                if wait_retries > max_retries:
                    raise

                retry_delay = self._extract_retry_delay(e)
                logger.warning(
                    f"[rate-limit] Groq 429: aguardando {retry_delay:.1f}s "
                    f"(tentativa {wait_retries}/{max_retries}, "
                    f"key {self._current_key_idx + 1}/{len(self._api_keys)})..."
                )
                import time
                time.sleep(retry_delay + 0.5)

        raise last_error


class NvidiaProvider(LLMProvider):
    """Provider NVIDIA Build (NIM) via API compatível com OpenAI. Suporta múltiplas keys com rotação."""

    NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(self, api_key: str = None, model: str = None):
        raw_keys = api_key or os.getenv("NVIDIA_API_KEY", "")
        self._api_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        if not self._api_keys:
            raise ValueError("Nenhuma NVIDIA_API_KEY configurada")
        self._current_key_idx = 0
        self._exhausted_keys: set[int] = set()

        self.model = model or os.getenv("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
        self.client = OpenAI(
            api_key=self._api_keys[0],
            base_url=self.NVIDIA_BASE_URL,
        )

        key_count = len(self._api_keys)
        logger.info(f"NVIDIA inicializado: {self.model} ({key_count} key{'s' if key_count > 1 else ''})")

    def _is_rate_limit_error(self, error: Exception) -> bool:
        error_str = str(error)
        return "429" in error_str or "rate_limit" in error_str.lower() or "quota" in error_str.lower()

    def _is_daily_quota(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return "day" in error_str or "daily" in error_str or "credit" in error_str

    def _extract_retry_delay(self, error: Exception) -> float:
        match = re.search(r'try again in (\d+(?:\.\d+)?)\s*s', str(error), re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r'try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s', str(error), re.IGNORECASE)
        if match:
            minutes = int(match.group(1) or 0)
            seconds = float(match.group(2))
            return minutes * 60 + seconds
        return 30.0

    def _switch_to_next_key(self) -> bool:
        self._exhausted_keys.add(self._current_key_idx)
        for i in range(len(self._api_keys)):
            candidate = (self._current_key_idx + 1 + i) % len(self._api_keys)
            if candidate not in self._exhausted_keys:
                self._current_key_idx = candidate
                self.client = OpenAI(
                    api_key=self._api_keys[candidate],
                    base_url=self.NVIDIA_BASE_URL,
                )
                key_masked = self._api_keys[candidate][:12] + "..."
                logger.info(
                    f"[key-rotation] NVIDIA: rotacionando para key "
                    f"{candidate + 1}/{len(self._api_keys)} ({key_masked})"
                )
                return True
        logger.error(f"[key-rotation] NVIDIA: todas as {len(self._api_keys)} keys esgotadas.")
        return False

    def name(self) -> str:
        return f"nvidia/{self.model}"

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        max_retries = 3
        wait_retries = 0
        max_attempts = max_retries + len(self._api_keys) * 2
        last_error = None

        for _ in range(max_attempts):
            try:
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

            except Exception as e:
                last_error = e

                if not self._is_rate_limit_error(e):
                    raise

                if self._is_daily_quota(e):
                    logger.warning(f"[rate-limit] NVIDIA: quota esgotada na key {self._current_key_idx + 1}.")
                    if self._switch_to_next_key():
                        continue
                    raise

                # RPM: tentar rotacionar para outra key
                if len(self._api_keys) > 1:
                    available = [
                        i for i in range(len(self._api_keys))
                        if i != self._current_key_idx and i not in self._exhausted_keys
                    ]
                    if available:
                        next_key = available[0]
                        self._current_key_idx = next_key
                        self.client = OpenAI(
                            api_key=self._api_keys[next_key],
                            base_url=self.NVIDIA_BASE_URL,
                        )
                        key_masked = self._api_keys[next_key][:12] + "..."
                        logger.info(f"[key-rotation] NVIDIA: RPM hit, rotacionando para key {next_key + 1}/{len(self._api_keys)} ({key_masked})")
                        continue

                wait_retries += 1
                if wait_retries > max_retries:
                    raise

                retry_delay = self._extract_retry_delay(e)
                logger.warning(
                    f"[rate-limit] NVIDIA 429: aguardando {retry_delay:.1f}s "
                    f"(tentativa {wait_retries}/{max_retries}, "
                    f"key {self._current_key_idx + 1}/{len(self._api_keys)})..."
                )
                import time
                time.sleep(retry_delay + 0.5)

        raise last_error


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
    """Retorna provider padrão com fallback configurado via .env.

    Hierarquia: NVIDIA Build → Groq → Local llama.cpp
    """
    from dotenv import load_dotenv
    load_dotenv()

    providers = []

    # NVIDIA Build como primário (se tiver key)
    nvidia_keys = os.getenv("NVIDIA_API_KEY", "")
    if nvidia_keys.strip():
        providers.append(NvidiaProvider(api_key=nvidia_keys))

    # Groq como secundário (se tiver key)
    groq_keys = os.getenv("GROQ_API_KEY", "")
    if groq_keys.strip():
        providers.append(GroqProvider(api_key=groq_keys))

    # Local llama como último fallback
    local_url = os.getenv("LOCAL_LLM_URL")
    if local_url:
        providers.append(LocalLlamaProvider(base_url=local_url))

    if not providers:
        raise RuntimeError(
            "Nenhum LLM provider configurado. "
            "Defina NVIDIA_API_KEY, GROQ_API_KEY ou LOCAL_LLM_URL no .env"
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
