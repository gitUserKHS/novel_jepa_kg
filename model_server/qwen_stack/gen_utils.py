"""생성 공용: 청크 프리필 + generate (GQA-split 어텐션 전제) + 한국어 전용 토큰 필터.
(출처: C:/연구_프로젝트/ai_아키텍처/applied/gen_utils.py, 2026-08-23 복사)

프리필 순간 텐서(선형어텐션 층 스캔, 토큰당 ~170KB)를 청크 크기로 묶고,
full-attention 층은 _patches.gqa_split 이 마스크·repeat_kv 없이 처리한다.

foreign_token_ids(tok): 한자·가나·키릴 등 외국 문자를 담은 어휘 토큰 id 목록
(generate(suppress_tokens=...) 에 넣어 누출 차단). 08-22 실측: 기본 모델도
한국어 답변 10개 중 1개꼴로 한자 토큰이 섞임 (예: "바쁜日子里엔").
"""

import json
import re
import pathlib
import time

import torch
from transformers import LogitsProcessor, LogitsProcessorList

HERE = pathlib.Path(__file__).resolve().parent
_BAN_CACHE = HERE / "data" / "ban_foreign_ids.json"
# 한글·라틴·그리스(π 등 수식)·기호·이모지는 허용. 아래 범위만 차단.
FOREIGN_RANGES = [
    (0x3040, 0x30FF),    # 히라가나·가타카나
    (0x31F0, 0x31FF),    # 가타카나 확장
    (0x3400, 0x4DBF),    # CJK 확장 A
    (0x4E00, 0x9FFF),    # CJK 통합 한자
    (0xF900, 0xFAFF),    # CJK 호환 한자
    (0x20000, 0x2FA1F),  # CJK 확장 B~
    (0x0400, 0x04FF),    # 키릴
    (0x0590, 0x05FF),    # 히브리
    (0x0600, 0x06FF),    # 아랍
    (0x0900, 0x097F),    # 데바나가리
    (0x0E00, 0x0E7F),    # 타이
]


def is_foreign_char(c):
    o = ord(c)
    return any(lo <= o <= hi for lo, hi in FOREIGN_RANGES)


def foreign_token_ids(tok):
    """외국 문자를 온전히 담은 토큰 id 목록 (바이트 조각 토큰은 제외). 캐시됨."""
    n = len(tok)
    if _BAN_CACHE.exists():
        d = json.loads(_BAN_CACHE.read_text(encoding="utf-8"))
        if d.get("vocab") == n:
            return d["ids"]
    ids = []
    for i in range(n):
        s = tok.decode([i])
        if "�" in s:
            continue
        if any(is_foreign_char(c) for c in s):
            ids.append(i)
    _BAN_CACHE.parent.mkdir(exist_ok=True)
    _BAN_CACHE.write_text(json.dumps(dict(vocab=n, ids=ids)), encoding="utf-8")
    return ids


class ForeignTokenFilter(LogitsProcessor):
    """외국 문자 토큰을 -inf 로 막는 프로세서 (디바이스별 bool 마스크 1회 계산)."""

    def __init__(self, tok):
        self.ids = torch.tensor(foreign_token_ids(tok), dtype=torch.long)
        self._mask = {}

    def __call__(self, input_ids, scores):
        key = (str(scores.device), scores.shape[-1])
        m = self._mask.get(key)
        if m is None:
            m = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
            m[self.ids.to(scores.device)] = True
            self._mask[key] = m
        return scores.masked_fill(m, float("-inf"))


def korean_filter(tok):
    """generate(logits_processor=korean_filter(tok)) 로 사용."""
    return LogitsProcessorList([ForeignTokenFilter(tok)])


_LATIN_CACHE = HERE / "data" / "ban_latin_ids.json"
_LATIN_RE = re.compile(r"^\s?[A-Za-z]{2,}$")


def latin_token_ids(tok):
    """라틴 문자만으로 된 토큰 id (2자 이상) — 소설 본문의 영어 단어 누출 차단용. 캐시됨."""
    n = len(tok)
    if _LATIN_CACHE.exists():
        d = json.loads(_LATIN_CACHE.read_text(encoding="utf-8"))
        if d.get("vocab") == n:
            return d["ids"]
    ids = [i for i in range(n) if _LATIN_RE.match(tok.decode([i]) or "")]
    _LATIN_CACHE.parent.mkdir(exist_ok=True)
    _LATIN_CACHE.write_text(json.dumps(dict(vocab=n, ids=ids)), encoding="utf-8")
    return ids


class LatinTokenFilter(ForeignTokenFilter):
    def __init__(self, tok):
        self.ids = torch.tensor(foreign_token_ids(tok) + latin_token_ids(tok), dtype=torch.long)
        self._mask = {}


def korean_strict_filter(tok):
    """한자·가나·키릴 + 라틴 단어 토큰까지 차단 (숫자·기호·단일 문자는 허용)."""
    return LogitsProcessorList([LatinTokenFilter(tok)])


@torch.no_grad()
def prefill_chunked(model, ids, chunk=2048):
    """ids [1, n] 의 앞 n-1 토큰을 청크로 프리필, 캐시 반환."""
    n = ids.shape[1]
    pkv = None
    for s in range(0, n - 1, chunk):
        out = model(input_ids=ids[:, s:min(s + chunk, n - 1)],
                    past_key_values=pkv, use_cache=True, logits_to_keep=1)
        pkv = out.past_key_values
    return pkv


def generate(model, tok, ids, chunk=2048, **gen):
    """청크 프리필 후 generate. 반환: (new_ids[1,m], info dict)."""
    n = ids.shape[1]
    torch.cuda.synchronize()
    t0 = time.time()
    pkv = prefill_chunked(model, ids, chunk) if n > chunk else None
    torch.cuda.synchronize()
    t1 = time.time()
    kw = dict(input_ids=ids, max_new_tokens=gen.pop("max_new_tokens", 512), **gen)
    if pkv is not None:
        kw["past_key_values"] = pkv
    out = model.generate(**kw)
    torch.cuda.synchronize()
    t2 = time.time()
    new = out[:, n:]
    m = new.shape[1]
    return new, dict(n_in=n, n_out=m, prefill_s=round(t1 - t0, 1),
                     decode_s=round(t2 - t1, 1),
                     tok_s=round(m / max(t2 - t1, 1e-9), 1),
                     peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))


def chat_ids(tok, messages, device):
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, return_tensors="pt")["input_ids"].to(device)


def set_adapter_scale(model, name, factor):
    """PEFT LoRA 어댑터 `name` 의 출력 배율을 factor 배로 (1.0 = 학습 그대로)."""
    n = 0
    for m in model.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict) and name in sc:
            sc[name] = sc[name] * factor
            n += 1
    return n
