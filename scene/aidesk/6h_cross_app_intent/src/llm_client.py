import json
import threading
import time

import requests

from src.utils import env, extract_json


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible Chat Completions client with usage recording.

    Per-instance `model`, so the generator and the judge agents can use different model families
    (a blind annotation made by the model that wrote the events is not an independent signal).
    """

    def __init__(self, base_url=None, api_key=None, model=None, timeout=None, max_retries=None,
                 use_response_format=None, thinking=None):
        self.base_url = (base_url or env("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or env("LLM_API_KEY", "")
        self.model = model or env("LLM_MODEL", "gpt-4.1-mini")
        self.timeout = int(timeout or env("LLM_TIMEOUT", "120"))
        self.max_retries = int(max_retries or env("LLM_MAX_RETRIES", "3"))
        self._use_response_format = (env("LLM_RESPONSE_FORMAT", "json_object") == "json_object") \
            if use_response_format is None else use_response_format
        # Optional reasoning budget for thinking models: enabled | disabled | low | high | max (None = provider default).
        self.thinking = thinking or env("LLM_THINKING")
        self._thinking_ok = self.thinking is not None
        self._lock = threading.Lock()
        # Connect to the gateway directly unless LLM_TRUST_ENV=1. On macOS `requests` otherwise inherits the
        # system proxy (e.g. 127.0.0.1:7890), which in testing dropped connections mid-batch (ProxyError /
        # NameResolutionError) while the gateway itself was reachable.
        self.session = requests.Session()
        self.session.trust_env = (env("LLM_TRUST_ENV", "0") == "1")
        if not self.api_key:
            raise RuntimeError("Missing LLM_API_KEY. Copy .env.example to .env and configure it.")

    @classmethod
    def from_env(cls, prefix="LLM"):
        """Build a client from <prefix>_BASE_URL / _API_KEY / _MODEL / _TIMEOUT / _MAX_RETRIES / _THINKING.

        Variables that are missing fall back to the plain LLM_* ones, so `LLM_JUDGE_MODEL` alone is enough
        to give the judge agents a different model on the same endpoint.
        """
        return cls(
            base_url=env(f"{prefix}_BASE_URL") or env("LLM_BASE_URL"),
            api_key=env(f"{prefix}_API_KEY") or env("LLM_API_KEY"),
            model=env(f"{prefix}_MODEL") or env("LLM_MODEL"),
            timeout=env(f"{prefix}_TIMEOUT") or env("LLM_TIMEOUT"),
            max_retries=env(f"{prefix}_MAX_RETRIES") or env("LLM_MAX_RETRIES"),
            thinking=env(f"{prefix}_THINKING") or env("LLM_THINKING"),
        )

    @property
    def host(self):
        return self.base_url.split("//", 1)[-1].split("/", 1)[0]

    def generate_json(self, system_prompt, user_input, schema=None, temperature=0.2, recorder=None,
                      max_tokens=None, thinking=None):
        url = self.base_url + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)}
            ],
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        with self._lock:
            use_rf = self._use_response_format
            use_thinking = self._thinking_ok
        if use_rf:
            payload["response_format"] = {"type": "json_object"}
        thinking = self.thinking if thinking is None else thinking
        if use_thinking and thinking in ("enabled", "disabled", "low", "high", "max"):
            payload["thinking"] = {"type": thinking}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 400 and "thinking" in payload:
                    # Provider rejects this reasoning budget (400, whatever the wording) → provider default.
                    payload.pop("thinking", None)
                    with self._lock:
                        self._thinking_ok = False
                    continue
                if response.status_code == 400 and "response_format" in payload:
                    # Endpoint does not support JSON mode → drop it once and continue.
                    payload.pop("response_format", None)
                    with self._lock:
                        self._use_response_format = False
                    continue
                if response.status_code == 400 and "thinking" in payload and "thinking" in response.text:
                    # Provider rejects this reasoning budget → leave the provider default.
                    payload.pop("thinking", None)
                    with self._lock:
                        self._thinking_ok = False
                    continue
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                content = message.get("content") or ""
                if not content.strip() and message.get("reasoning_content"):
                    content = message["reasoning_content"]
                result = extract_json(content)
                if not isinstance(result, dict):
                    raise ValueError("Model response must be a JSON object")
                usage = data.get("usage") or {}
                if recorder is not None:
                    recorder.append({
                        "model": data.get("model") or self.model,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "latency_s": round(time.time() - t0, 2),
                        "attempt": attempt + 1,
                    })
                return result
            except Exception as exc:  # noqa: BLE001 - retry any transport/parse failure
                last_error = exc
                if attempt + 1 < self.max_retries:
                    transport = isinstance(exc, (requests.ConnectionError, requests.Timeout)) or \
                        (isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code >= 500)
                    # DNS blips and gateway 5xx need seconds, bad JSON does not: 4/10/20/30 s vs 1/2/4 s.
                    time.sleep(min(4 * 2 ** (attempt - 1), 30) if transport and attempt >= 1 else min(2 ** attempt, 8))
        raise RuntimeError(f"LLM request failed after retries: {last_error}")
