"""생성 품질용 로짓 프로세서 — 장 내부 구절 반복(루프) 억제.

DRY (Don't Repeat Yourself, llama.cpp 의 샘플러를 HF LogitsProcessor 로 이식):
지금까지 *생성한* 토큰열의 끝부분이 앞서 생성한 어딘가와 같은 순서로 이어지고 있으면,
그 다음에 왔던 토큰에 multiplier * base**(match_len - allowed_length) 만큼 페널티를 준다.
- 프롬프트(문맥)는 보지 않으므로 직전 장 꼬리·요약을 베끼는 것은 다른 장치(게이트)가 맡는다.
- 문장 부호·줄바꿈 같은 sequence breaker 에서 매칭을 끊어, 흔한 어미·조사의 자연스러운
  반복까지 벌하지 않는다.
- repetition_penalty 와 달리 "이미 쓴 단어" 전체가 아니라 "이미 쓴 구절의 연속" 만 막으므로
  인물 이름·핵심어 사용을 해치지 않는다 (2026-08-23 설계 근거: 반복 페널티 상향은
  한자/영어 이탈, no_repeat_ngram 은 이름 훼손을 일으켰음).
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

DEFAULT_BREAKERS = ["\n", ".", ",", "!", "?", "…", '"', "'", "“", "”", "‘", "’", "(", ")", "[", "]", ":", ";", "·"]


class DRYLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        multiplier: float = 0.8,
        base: float = 1.75,
        allowed_length: int = 2,
        breakers: list[str] | None = None,
        max_penalty: float = 20.0,
        window: int = 2048,
    ) -> None:
        self.prompt_length = int(prompt_length)
        self.multiplier = float(multiplier)
        self.base = float(base)
        self.allowed_length = max(1, int(allowed_length))
        self.max_penalty = float(max_penalty)
        self.window = int(window)
        breaker_ids: set[int] = set()
        for text in breakers or DEFAULT_BREAKERS:
            for variant in (text, " " + text):
                for token_id in tokenizer.encode(variant, add_special_tokens=False):
                    decoded = tokenizer.decode([token_id])
                    if decoded.strip() == text.strip() or decoded == text:
                        breaker_ids.add(int(token_id))
        self.breaker_ids = breaker_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.multiplier <= 0:
            return scores
        generated = input_ids[0, self.prompt_length:].tolist()
        if len(generated) < self.allowed_length + 1:
            return scores
        if len(generated) > self.window:
            generated = generated[-self.window:]
        last = generated[-1]
        if last in self.breaker_ids:
            return scores
        n = len(generated)
        # 끝에서부터의 접미사가 앞 어딘가와 일치하는 최장 길이를 후보 토큰별로 모은다.
        penalties: dict[int, int] = {}
        for i in range(n - 1):
            if generated[i] != last:
                continue
            # generated[..i] 가 generated[..n-1] 과 얼마나 길게 일치하는가 (breaker 에서 중단)
            length = 0
            while i - length >= 0 and generated[i - length] == generated[n - 1 - length]:
                if generated[i - length] in self.breaker_ids and length > 0:
                    break
                length += 1
                if length >= 64:
                    break
            if length >= self.allowed_length:
                nxt = generated[i + 1]
                if nxt in self.breaker_ids:
                    continue
                penalties[nxt] = max(penalties.get(nxt, 0), length)
        if not penalties:
            return scores
        ids = torch.tensor(list(penalties.keys()), device=scores.device)
        lengths = torch.tensor([penalties[k] for k in penalties], device=scores.device, dtype=torch.float32)
        amount = (self.multiplier * self.base ** (lengths - self.allowed_length)).clamp(max=self.max_penalty)
        scores[0, ids] = scores[0, ids] - amount.to(scores.dtype)
        return scores
