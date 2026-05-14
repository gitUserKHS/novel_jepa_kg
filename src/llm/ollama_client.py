from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np
import requests


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embed_model: str,
        timeout_sec: int = 120,
        dry_run: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.timeout_sec = timeout_sec
        self.dry_run = dry_run

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

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1200,
        json_mode: bool = False,
    ) -> str:
        if self.dry_run:
            return self._dry_chat(prompt)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=body,
                timeout=self.timeout_sec,
            )
            self._raise_for_status(response, "/api/chat", self.chat_model)
            payload = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach Ollama /api/chat at {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Ollama /api/chat returned invalid JSON.") from exc

        message = payload.get("message", {})
        content = message.get("content", "")
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

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.dry_run:
            return np.vstack([self._stable_embedding(text) for text in texts]).astype("float32")
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embed_model, "input": texts},
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

    def _raise_for_status(self, response: requests.Response, endpoint: str, model: str | None = None) -> None:
        if response.ok:
            return
        detail = response.text.strip()
        try:
            payload = response.json()
            detail = payload.get("error") or detail
        except ValueError:
            pass
        model_part = f" for model '{model}'" if model else ""
        raise RuntimeError(f"Ollama {endpoint} failed{model_part}: HTTP {response.status_code}. {detail}")

    def _dry_chat(self, prompt: str) -> str:
        if "JSON" in prompt.upper() or "json" in prompt:
            from src.llm.prompts import diversity_plan

            idx_match = re.search(r"sample\s*#?(\d+)", prompt, flags=re.IGNORECASE)
            idx = int(idx_match.group(1)) if idx_match else 1
            plan = diversity_plan(idx)
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
