from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable

import numpy as np
import requests


class OllamaHTTPError(RuntimeError):
    def __init__(self, endpoint: str, status_code: int, detail: str, model: str | None = None) -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        self.detail = detail
        self.model = model
        model_part = f" for model '{model}'" if model else ""
        super().__init__(f"Ollama {endpoint} failed{model_part}: HTTP {status_code}. {detail}")


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embed_model: str,
        timeout_sec: int = 120,
        num_ctx: int | None = None,
        num_gpu: int | None = None,
        num_batch: int | None = None,
        keep_alive: str | None = None,
        manage_vram: bool = True,
        dry_run: bool = False,
        retry_attempts: int = 1,
        retry_backoff_sec: float = 2.0,
        fallback_num_ctx: int | None = 3072,
        fallback_num_gpu: int | None = 35,
        fallback_num_batch: int | None = 64,
        fallback_max_tokens: int | None = 1200,
        fallback_keep_alive: str | None = "10s",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout_sec = timeout_sec
        self.num_ctx = num_ctx
        self.num_gpu = num_gpu
        self.num_batch = num_batch
        self.keep_alive = keep_alive
        self.manage_vram = manage_vram
        self.dry_run = dry_run
        self.retry_attempts = max(0, retry_attempts)
        self.retry_backoff_sec = max(0.0, retry_backoff_sec)
        self.fallback_num_ctx = fallback_num_ctx
        self.fallback_num_gpu = fallback_num_gpu
        self.fallback_num_batch = fallback_num_batch
        self.fallback_max_tokens = fallback_max_tokens
        self.fallback_keep_alive = fallback_keep_alive

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=min(self.timeout_sec, 5))
            self._raise_for_status(response, "/api/tags")
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach Ollama /api/tags at {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Ollama /api/tags returned invalid JSON.") from exc
        names = [item.get("name") for item in payload.get("models", []) if item.get("name")]
        return sorted(names)

    def running_models(self) -> list[dict[str, Any]]:
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=min(self.timeout_sec, 5))
            self._raise_for_status(response, "/api/ps")
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach Ollama /api/ps at {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Ollama /api/ps returned invalid JSON.") from exc
        return payload.get("models", [])

    def unload_model(self, model: str) -> bool:
        if self.dry_run or not model:
            return False
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=min(self.timeout_sec, 30),
            )
            self._raise_for_status(response, "/api/generate", model)
            return True
        except Exception:
            return False

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        json_mode: bool = False,
        stream_callback: Callable[[str], None] | None = None,
    ) -> str:
        if self.dry_run:
            text = self._dry_chat(prompt)
            if stream_callback is not None:
                for char in text:
                    stream_callback(char)
            return text
        if self.manage_vram and self.embed_model and self.embed_model != self.chat_model:
            self.unload_model(self.embed_model)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        last_error: Exception | None = None
        total_attempts = 1 + self.retry_attempts
        for attempt in range(total_attempts):
            recovery_mode = attempt > 0
            body = {
                "model": self.chat_model,
                "messages": messages,
                "stream": stream_callback is not None,
                "think": False,
                "options": self._chat_options(temperature, max_tokens, recovery_mode=recovery_mode),
            }
            keep_alive = self._effective_keep_alive(recovery_mode=recovery_mode)
            if keep_alive:
                body["keep_alive"] = keep_alive
            if json_mode:
                body["format"] = "json"

            delivered = {"any": False}
            callback = stream_callback
            if stream_callback is not None:

                def tracking_callback(chunk: str) -> None:
                    delivered["any"] = True
                    stream_callback(chunk)

                callback = tracking_callback

            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=body,
                    timeout=self.timeout_sec,
                    stream=callback is not None,
                )
                self._raise_for_status(response, "/api/chat", self.chat_model)
                if callback is not None:
                    content, payload = self._read_streaming_chat(response, callback)
                else:
                    payload = response.json()
                    content = payload.get("message", {}).get("content", "")
                break
            except OllamaHTTPError as exc:
                last_error = exc
                if not self._should_retry_chat_error(exc, attempt, total_attempts, delivered["any"]):
                    if attempt > 0 or exc.status_code >= 500:
                        raise self._with_recovery_hint(exc) from exc
                    raise exc
                self._prepare_chat_retry(attempt)
            except requests.RequestException as exc:
                last_error = exc
                if delivered["any"] or attempt >= total_attempts - 1:
                    raise RuntimeError(f"Could not reach Ollama /api/chat at {self.base_url}: {exc}") from exc
                self._prepare_chat_retry(attempt)
            except ValueError as exc:
                raise RuntimeError("Ollama /api/chat returned invalid JSON.") from exc
        else:
            raise RuntimeError(f"Ollama /api/chat failed after retry. Last error: {last_error}")

        message = payload.get("message", {})
        if not content.strip():
            done_reason = payload.get("done_reason", "unknown")
            thinking = message.get("thinking", "")
            detail = f" done_reason={done_reason}."
            if thinking:
                detail += " The model returned thinking text but no final content."
            raise RuntimeError(
                f"Ollama /api/chat returned empty content for model '{self.chat_model}'.{detail} "
                "Try a larger max token value or a non-thinking chat model."
            )
        return content

    def _read_streaming_chat(
        self,
        response: requests.Response,
        stream_callback: Callable[[str], None],
    ) -> tuple[str, dict[str, Any]]:
        chunks: list[str] = []
        last_payload: dict[str, Any] = {}
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            payload = json.loads(line)
            last_payload = payload
            chunk = payload.get("message", {}).get("content", "")
            if chunk:
                chunks.append(chunk)
                stream_callback(chunk)
            if payload.get("done"):
                break
        return "".join(chunks), last_payload

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.dry_run:
            return np.vstack([self._stable_embedding(text) for text in texts]).astype("float32")
        if self.manage_vram and self.chat_model and self.chat_model != self.embed_model:
            self.unload_model(self.chat_model)
        body: dict[str, Any] = {"model": self.embed_model, "input": texts}
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json=body,
                timeout=self.timeout_sec,
            )
            self._raise_for_status(response, "/api/embed", self.embed_model)
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach Ollama /api/embed at {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Ollama /api/embed returned invalid JSON.") from exc
        vectors = payload.get("embeddings")
        if not vectors:
            raise RuntimeError("Ollama /api/embed returned no embeddings.")
        return np.asarray(vectors, dtype="float32")

    def _chat_options(self, temperature: float, max_tokens: int, recovery_mode: bool = False) -> dict[str, Any]:
        num_predict = max_tokens
        num_ctx = self.num_ctx
        num_gpu = self.num_gpu
        num_batch = self.num_batch
        if recovery_mode:
            if self.fallback_max_tokens:
                num_predict = min(num_predict, self.fallback_max_tokens)
            if self.fallback_num_ctx and num_ctx:
                num_ctx = min(num_ctx, self.fallback_num_ctx)
            elif self.fallback_num_ctx:
                num_ctx = self.fallback_num_ctx
            if self.fallback_num_gpu is not None and num_gpu is not None:
                num_gpu = min(num_gpu, self.fallback_num_gpu)
            elif self.fallback_num_gpu is not None:
                num_gpu = self.fallback_num_gpu
            if self.fallback_num_batch and num_batch:
                num_batch = min(num_batch, self.fallback_num_batch)
            elif self.fallback_num_batch:
                num_batch = self.fallback_num_batch
        options: dict[str, Any] = {"temperature": temperature, "num_predict": num_predict}
        if num_ctx:
            options["num_ctx"] = num_ctx
        if num_gpu is not None:
            options["num_gpu"] = num_gpu
        if num_batch:
            options["num_batch"] = num_batch
        return options

    def _effective_keep_alive(self, recovery_mode: bool = False) -> str | None:
        if recovery_mode and self.fallback_keep_alive is not None:
            return self.fallback_keep_alive
        return self.keep_alive

    def _should_retry_chat_error(
        self,
        exc: OllamaHTTPError,
        attempt: int,
        total_attempts: int,
        stream_delivered: bool,
    ) -> bool:
        if stream_delivered or attempt >= total_attempts - 1:
            return False
        detail = exc.detail.lower()
        if exc.status_code in {500, 502, 503, 504}:
            return True
        return any(
            marker in detail
            for marker in [
                "model runner",
                "resource",
                "memory",
                "cuda",
                "gpu",
                "unexpectedly stopped",
            ]
        )

    def _prepare_chat_retry(self, attempt: int) -> None:
        if self.manage_vram:
            self.unload_model(self.chat_model)
            if self.embed_model and self.embed_model != self.chat_model:
                self.unload_model(self.embed_model)
        if self.retry_backoff_sec:
            time.sleep(self.retry_backoff_sec * (attempt + 1))

    def _with_recovery_hint(self, exc: OllamaHTTPError) -> RuntimeError:
        hint = (
            "The app retried after unloading Ollama models with conservative fallback options. "
            "If this keeps happening, lower Ollama context length, GPU layers, batch size, or generation max tokens."
        )
        return RuntimeError(f"{exc} {hint}")

    def _raise_for_status(self, response: requests.Response, endpoint: str, model: str | None = None) -> None:
        if response.ok:
            return
        detail = response.text.strip()
        try:
            payload = response.json()
            detail = payload.get("error") or detail
        except ValueError:
            pass
        raise OllamaHTTPError(endpoint, response.status_code, detail, model)

    def _dry_chat(self, prompt: str) -> str:
        if "JSON" in prompt.upper() or "json" in prompt:
            from src.llm.prompts import diversity_plan

            idx_match = re.search(r"sample\s*#?(\d+)", prompt, flags=re.IGNORECASE)
            idx = int(idx_match.group(1)) if idx_match else 1
            plan = diversity_plan(idx)
            prompt_keys = {
                "세부 장르": "subgenre",
                "장면 기능": "plot_function",
                "감정 변화": "emotion_arc",
                "핵심 갈등": "conflict",
                "모티프": "motif",
                "관계 긴장": "relationship",
                "장면 목표": "scene_goal",
                "다음 훅": "next_hook",
            }
            for label, key in prompt_keys.items():
                match = re.search(rf"-\s*{label}:\s*(.+)", prompt)
                if match:
                    plan[key] = match.group(1).strip()
            names = [
                ("서윤", "기억 복원가"),
                ("이린", "궁중 기록관"),
                ("도하", "오컬트 팟캐스터"),
                ("하린", "법의학 사진가"),
                ("무겸", "몰락 문파의 제자"),
                ("연우", "배급국 견습 관리"),
            ]
            name, role = names[(idx - 1) % len(names)]
            sample = {
                "world": {
                    "genre": plan["subgenre"],
                    "premise": f"{plan['motif']}가 사건의 방향을 바꾸는 한국형 장편 서사",
                    "rules": [plan["conflict"], "선택의 대가는 다음 장면의 관계와 상태에 남는다"],
                },
                "characters": [
                    {
                        "name": name,
                        "role": role,
                        "goal": f"{plan['plot_function']}을 통해 숨겨진 진실에 접근한다",
                        "fear": "잘못된 선택으로 소중한 사람을 잃는 것",
                        "relationship": plan["relationship"],
                    },
                    {
                        "name": "민재",
                        "role": "비밀을 아는 협력자",
                        "goal": "자신이 숨긴 과거를 덮으면서도 더 큰 위기를 막는다",
                        "fear": "자신의 죄가 드러나는 것",
                        "relationship": "주인공에게 협력하지만 핵심 사실을 숨긴다",
                    },
                ],
                "scene_t": {
                    "summary": f"{name}은 {plan['motif']}와 연결된 첫 단서를 발견하지만, 그 단서가 협력자의 거짓말과 맞물린다는 사실을 눈치챈다 {idx}.",
                    "emotion": plan["emotion_arc"].split("에서")[0],
                    "conflict": plan["conflict"],
                    "state": [plan["motif"], "숨겨진 목표", "관계 균열"],
                    "plot_function": plan["plot_function"],
                },
                "scene_t_plus_1": {
                    "summary": f"다음 장면에서 {name}은 {plan['conflict']}는 압박 속에서 새 증거를 얻고, 협력자와의 관계를 재정의하는 선택을 한다 {idx}.",
                    "emotion": plan["emotion_arc"],
                    "conflict": f"{plan['conflict']}는 선택이 더 큰 대가를 부른다.",
                    "state": ["새 증거 확보", "선택 압박", plan["relationship"]],
                    "plot_function": "상태 변화",
                },
            }
            return json.dumps(sample, ensure_ascii=False)
        return (
            "서윤은 차가운 형광등 아래에서 숨을 골랐다. 기록 장치가 토해낸 잔향은 동생의 목소리였지만, "
            "그 안에는 도망치는 사람의 공포보다 무언가를 선택한 사람의 단단함이 남아 있었다.\n\n"
            "민재는 심층 구역 좌표를 보는 순간 얼굴빛을 잃었다. 서윤은 그 침묵이 대답이라는 걸 알았다. "
            "이제 그녀가 찾아야 할 것은 동생의 행방만이 아니었다. 왜 모두가 그 선택을 숨기려 했는지, "
            "그리고 자신이 잃어버린 첫 번째 기억이 무엇인지 확인해야 했다."
        )

    def _stable_embedding(self, text: str, dim: int = 384) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=dim)
        norm = np.linalg.norm(vector)
        return (vector / norm).astype("float32") if norm else vector.astype("float32")
