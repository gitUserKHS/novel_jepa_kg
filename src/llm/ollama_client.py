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

    def chat(self, prompt: str, system: str | None = None, temperature: float = 0.7, max_tokens: int = 1200) -> str:
        if self.dry_run:
            return self._dry_chat(prompt)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.chat_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("message", {}).get("content", "")

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.dry_run:
            return np.vstack([self._stable_embedding(text) for text in texts]).astype("float32")
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embed_model, "input": texts},
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = payload.get("embeddings")
        if not vectors:
            raise RuntimeError("Ollama /api/embed returned no embeddings.")
        return np.asarray(vectors, dtype="float32")

    def _dry_chat(self, prompt: str) -> str:
        if "JSON" in prompt.upper() or "json" in prompt:
            idx_match = re.search(r"sample\s*#?(\d+)", prompt, flags=re.IGNORECASE)
            idx = int(idx_match.group(1)) if idx_match else 1
            sample = {
                "world": {
                    "genre": "한국형 SF 미스터리",
                    "premise": "기억의 잔향이 도시의 물건에 남는 근미래 서울",
                    "rules": ["강한 감정은 장소에 흔적으로 남는다", "흔적을 읽으면 대가로 자신의 기억이 흐려진다"],
                },
                "characters": [
                    {
                        "name": "서윤",
                        "role": "기억 복원가",
                        "goal": "사라진 동생의 마지막 동선을 찾는다",
                        "fear": "동생을 잊어버리는 것",
                        "relationship": "민재를 믿고 싶지만 연구소와의 관계를 의심한다",
                    },
                    {
                        "name": "민재",
                        "role": "전직 연구원",
                        "goal": "연구소가 숨긴 사고 기록을 공개한다",
                        "fear": "자신의 죄가 드러나는 것",
                        "relationship": "서윤에게 협력하지만 핵심 사실을 숨긴다",
                    },
                ],
                "scene_t": {
                    "summary": f"서윤은 폐쇄된 역무실에서 동생의 이름이 적힌 파손된 기록 장치를 발견한다 {idx}.",
                    "emotion": "불안과 기대",
                    "conflict": "장치를 읽으면 자신의 기억 일부를 잃을 수 있다.",
                    "state": ["폐쇄 역무실", "파손된 장치", "동생의 흔적"],
                    "plot_function": "단서 발견",
                },
                "scene_t_plus_1": {
                    "summary": f"장치 속 잔향은 동생이 자발적으로 연구소 심층 구역에 들어갔다는 장면을 보여준다 {idx}.",
                    "emotion": "충격에서 결심으로",
                    "conflict": "동생이 피해자인지 공모자인지 판단할 수 없다.",
                    "state": ["심층 구역 좌표", "흐려진 어린 시절 기억", "새 목표"],
                    "plot_function": "목표 강화",
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
