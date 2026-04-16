"""
Cliente LLM unificado com rastreamento de tokens e latência.

Estratégia de rate limiting inteligente:
  - Dispara requests sem delay artificial
  - Quando recebe 429, lê o Retry-After do header
  - Espera apenas o tempo necessário e retenta
  - Salva checkpoints para retomar em caso de interrupção
"""

import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Resposta do LLM com metadados operacionais."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = ""


class LLMClient:
    """Cliente unificado para chamadas LLM com retry inteligente."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: Optional[int] = 42,
        max_retries: int = 10,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_retries = max_retries

        # Contadores globais
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.total_retries = 0
        self.total_wait_seconds = 0.0

        self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            from openai import OpenAI

            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            import anthropic

            self.client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"Provider não suportado: {self.provider}")

    def _extract_retry_after(self, error) -> float:
        """
        Extrai o tempo de espera de um erro 429.

        Tenta (em ordem):
          1. Header Retry-After da resposta HTTP
          2. Campo 'retry_after' no corpo do erro
          3. Parse do texto "Please try again in Xs"
          4. Fallback: 20 segundos
        """
        error_str = str(error)

        # Tentar extrair do response header (openai lib expõe via .response)
        try:
            if hasattr(error, "response") and error.response is not None:
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    return float(retry_after) + 1.0
        except Exception:
            pass

        # Tentar parsear "Please try again in Xs" ou "try again in X.XXs"
        import re
        match = re.search(r"try again in (\d+\.?\d*)s", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 1.0

        # Tentar "Please retry after X seconds"
        match = re.search(r"retry after (\d+\.?\d*)\s*second", error_str, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 1.0

        # Fallback conservador
        return 20.0

    def _is_rate_limit_error(self, error) -> bool:
        """Verifica se o erro é um rate limit (429)."""
        error_str = str(error).lower()

        # Verificar código HTTP diretamente
        if hasattr(error, "status_code") and error.status_code == 429:
            return True

        return any(term in error_str for term in [
            "rate_limit", "429", "rate limit", "quota",
            "too many requests", "resource_exhausted",
        ])

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
    ) -> LLMResponse:
        """Faz uma chamada ao LLM com retry inteligente em 429."""
        for attempt in range(self.max_retries):
            start = time.time()
            try:
                if self.provider == "openai":
                    return self._complete_openai(prompt, system, start)
                elif self.provider == "anthropic":
                    return self._complete_anthropic(prompt, system, start)

            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < self.max_retries - 1:
                    wait = self._extract_retry_after(e)
                    self.total_retries += 1
                    self.total_wait_seconds += wait
                    print(
                        f"    ⏳ 429 rate limit (tentativa {attempt + 1}): "
                        f"aguardando {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    continue
                else:
                    raise

        raise RuntimeError(f"Falha após {self.max_retries} tentativas de rate limit")

    def _complete_openai(
        self, prompt: str, system: Optional[str], start: float
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed

        response = self.client.chat.completions.create(**kwargs)
        latency = time.time() - start

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            model=self.model,
        )

    def _complete_anthropic(
        self, prompt: str, system: Optional[str], start: float
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        latency = time.time() - start

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

        return LLMResponse(
            text=response.content[0].text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_seconds=latency,
            model=self.model,
        )

    def reset_counters(self):
        """Reseta contadores de tokens."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.total_retries = 0
        self.total_wait_seconds = 0.0

    def status(self) -> str:
        """Retorna status de uso atual."""
        return (
            f"Chamadas: {self.total_calls} | "
            f"Tokens: {self.total_input_tokens + self.total_output_tokens:,} | "
            f"Retries: {self.total_retries} | "
            f"Tempo esperando: {self.total_wait_seconds:.0f}s"
        )
