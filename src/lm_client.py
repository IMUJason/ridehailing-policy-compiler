from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"


class LMStudioClient:
    """Minimal client for an OpenAI-compatible local serving endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        temperature: float | None = 0.0,
    ) -> None:
        if not model:
            raise ValueError("model identifier is required")
        if timeout_s <= 0 or max_retries < 0:
            raise ValueError("invalid timeout or retry configuration")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.temperature = temperature

    def _payload(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        seed: int | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if response_format is not None:
            payload["response_format"] = response_format
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if seed is not None:
            payload["seed"] = seed
        return payload

    @staticmethod
    def _retryable(exc: BaseException) -> bool:
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code in {408, 409, 429} or 500 <= exc.code < 600
        return isinstance(
            exc,
            (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException),
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        payload = self._payload(messages, response_format, seed)
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        endpoint = f"{self.base_url}/chat/completions"
        request_summary = {
            "endpoint": endpoint,
            "model": self.model,
            "message_count": len(messages),
            "response_format_type": (response_format or {}).get("type"),
            "seed": seed,
            "timeout_s": self.timeout_s,
            "max_attempts": self.max_retries + 1,
        }

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                parsed = json.loads(raw)
                latency = time.perf_counter() - started
                choice = (parsed.get("choices") or [{}])[0] if isinstance(parsed, dict) else {}
                message = choice.get("message") or {}
                return {
                    "model": parsed.get("model", self.model),
                    "requested_model": self.model,
                    "content": message.get("content"),
                    "usage": parsed.get("usage"),
                    "latency_s": latency,
                    "finish_reason": choice.get("finish_reason"),
                    "response": parsed,
                    "raw_response": raw,
                    "request_summary": request_summary,
                    "error": None,
                    "retries": attempt,
                }
            except Exception as exc:  # noqa: BLE001 - transport errors are recorded, not raised
                last_error = f"{type(exc).__name__}: {exc}"
                if not self._retryable(exc) or attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff_s * (attempt + 1))

        return {
            "model": self.model,
            "requested_model": self.model,
            "content": None,
            "usage": None,
            "latency_s": None,
            "finish_reason": None,
            "response": None,
            "raw_response": None,
            "request_summary": request_summary,
            "error": last_error,
            "retries": self.max_retries,
        }
