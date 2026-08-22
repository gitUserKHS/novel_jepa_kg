"""CUDA 그래프 디코드 — Qwen3.5-4B 4bit 의 1토큰 forward 를 캡처해 재생한다.

실측 근거 (2026-08-23, RTX 4060): eager 디코드는 토큰당 CPU 111 ms > GPU 92 ms, 커널 런치
~1만 개 (선형어텐션 24층의 순수 PyTorch 구현). 그래프 재생은 20 ms/토큰 (5배), 그리디 토큰
완전 일치, bitsandbytes 4bit 커널·양쪽 할당자 모두 캡처 가능.

설계
- GraphCache: full-attention 8층의 KV 는 정적 버퍼(버킷 길이로 슬라이스), 선형어텐션 24층의
  conv/recurrent 상태는 in-place 갱신. 모델 코드가 Cache 에 기대하는 표면만 구현한다.
- 프리필은 eager(2K 청크, gqa_split 경로)로 같은 캐시에 기록하고, 디코드는 그래프 재생.
- 그래프 키 = (버킷 길이, 어댑터 이름, 배율). LoRA 배율은 캡처 시 상수로 굳으므로 키에 포함.
- 디코드 어텐션은 버킷 전체 버퍼에 대해 `arange(L) <= pos` 마스크(정적 pos 텐서)로 계산.
- 샘플링(온도·top-k·top-p·min-p·반복 페널티·한국어 필터·DRY)은 자체 루프에서 수행.
- 버킷 상한을 넘는 문맥은 호출자가 eager 경로로 폴백한다.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers.modeling_utils import AttentionInterface

import _patches
from samplers import DRYLogitsProcessor

# 버킷은 max_bucket 크기 정적 버퍼의 슬라이스 — 큰 버킷의 비용은 최대 버퍼 VRAM 뿐이다.
# full-attn 8층 KV 32 KiB/토큰: 16K=512 MiB · 32K=1 GiB · 64K=2 GiB.
BUCKET_STEPS = (2048, 4096, 8192, 16384, 32768, 65536)
DEFAULT_MAX_BUCKET = 16384
PREFILL_CHUNK = 2048


class _LinearLayerState:
    __slots__ = ("record_past", "conv_states", "recurrent_states", "has_previous")

    def __init__(self) -> None:
        self.record_past = False
        self.conv_states: list[torch.Tensor | None] = [None]
        self.recurrent_states: list[torch.Tensor | None] = [None]
        self.has_previous = False


class GraphCache:
    """정적 KV 버퍼(최대 버킷) + 선형층 상태. 한 요청 동안만 산다 (버퍼는 디코더가 재사용)."""

    def __init__(self, decoder: "GraphDecoder", bucket: int) -> None:
        self.d = decoder
        self.bucket = bucket
        self.k = {i: decoder.k_buf[i][:, :, :bucket] for i in decoder.attn_layers}
        self.v = {i: decoder.v_buf[i][:, :, :bucket] for i in decoder.attn_layers}
        for i in decoder.attn_layers:
            self.k[i].zero_()
            self.v[i].zero_()
        self.layers = [_LinearLayerState() for _ in decoder.layer_types]
        self.filled = 0
        self.pos = decoder.pos
        self.pos.zero_()
        self.graph_mode = False

    # --- attention layers (Qwen3_5Attention 이 호출) ---
    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int, cache_kwargs: Any = None):
        L = key.shape[2]
        if self.graph_mode and L == 1:
            self.k[layer_idx].index_copy_(2, self.pos, key)
            self.v[layer_idx].index_copy_(2, self.pos, value)
            return self.k[layer_idx], self.v[layer_idx]
        start = self.filled
        self.k[layer_idx][:, :, start:start + L] = key
        self.v[layer_idx][:, :, start:start + L] = value
        return self.k[layer_idx][:, :, :start + L], self.v[layer_idx][:, :, :start + L]

    # --- linear attention layers (Qwen3_5GatedDeltaNet 이 호출) ---
    def has_previous_state(self, layer_idx: int) -> bool:
        return self.layers[layer_idx].has_previous

    def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, conv_kernel_size: int | None = None, **_: Any):
        st = self.layers[layer_idx]
        k = conv_kernel_size or self.d.conv_kernel
        if st.conv_states[0] is None:
            st.conv_states[0] = self.d.conv_buf[layer_idx]
            st.conv_states[0].zero_()
        if not st.has_previous:
            full = conv_states
            st.has_previous = True
            if full.shape[-1] < k:
                full = F.pad(full, (k - full.shape[-1], 0), value=0)
        else:
            full = torch.cat([st.conv_states[0], conv_states], dim=-1)
        st.conv_states[0].copy_(full[..., -k:])
        return full

    def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **_: Any) -> torch.Tensor:
        st = self.layers[layer_idx]
        if st.recurrent_states[0] is None:
            st.recurrent_states[0] = self.d.rec_buf[layer_idx]
        st.recurrent_states[0].copy_(recurrent_states)
        return st.recurrent_states[0]

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        return self.filled

    def advance(self, n: int) -> None:
        self.filled += n
        self.pos.fill_(self.filled)


class GraphDecoder:
    def __init__(self, model: torch.nn.Module, tokenizer: Any, *, max_bucket: int = DEFAULT_MAX_BUCKET) -> None:
        self.model = model
        self.tok = tokenizer
        # PEFT 는 속성을 base_model 로 포워딩하므로 hasattr 로는 구분이 안 된다 — 명시적으로 푼다.
        core = model.base_model.model if hasattr(model, "peft_config") else model
        # core: Qwen3_5ForCausalLM. 텍스트 모델은 core.model (멀티모달 클래스면 .language_model)
        self.core = core
        self.text_model = getattr(core.model, "language_model", core.model)
        self.lm_head = core.lm_head
        if not hasattr(self.text_model, "layers"):
            raise RuntimeError(f"text model resolution failed: {type(self.text_model).__name__}")
        cfg = getattr(core.config, "text_config", core.config)
        self.cfg = cfg
        self.device = next(core.parameters()).device
        self.layer_types = list(cfg.layer_types)
        self.attn_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        self.linear_layers = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
        self.conv_kernel = int(cfg.linear_conv_kernel_dim)
        self.max_bucket = int(max_bucket)
        self.buckets = tuple(b for b in BUCKET_STEPS if b < self.max_bucket) + (self.max_bucket,)
        hkv, hd = cfg.num_key_value_heads, cfg.head_dim
        self.k_buf = {i: torch.zeros(1, hkv, self.max_bucket, hd, dtype=torch.bfloat16, device=self.device) for i in self.attn_layers}
        self.v_buf = {i: torch.zeros(1, hkv, self.max_bucket, hd, dtype=torch.bfloat16, device=self.device) for i in self.attn_layers}
        conv_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2 + cfg.linear_value_head_dim * cfg.linear_num_value_heads
        self.conv_buf = {i: torch.zeros(1, conv_dim, self.conv_kernel, dtype=torch.bfloat16, device=self.device) for i in self.linear_layers}
        self.rec_buf = {
            i: torch.zeros(1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim,
                           dtype=torch.float32, device=self.device)
            for i in self.linear_layers
        }
        self.pos = torch.zeros(1, dtype=torch.long, device=self.device)
        self.static_tok = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.static_pos = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.graphs: dict[tuple[int, str | None, float], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.cache: GraphCache | None = None
        self.mask = {"full_attention": None, "linear_attention": None}
        self.eos_ids = set()
        for value in (getattr(cfg, "eos_token_id", None), tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")):
            if isinstance(value, int) and value >= 0:
                self.eos_ids.add(value)
            elif isinstance(value, (list, tuple)):
                self.eos_ids.update(int(v) for v in value)
        self._orig_attention = _patches.gqa_split_attention_forward
        AttentionInterface.register("sdpa", self._attention)

    # ---- attention dispatch ---------------------------------------------------------------
    def _attention(self, module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        cache = self.cache
        if cache is None or not cache.graph_mode or query.shape[2] != 1:
            return self._orig_attention(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                        is_causal=is_causal, **kwargs)
        B, Hq, _, D = query.shape
        Hkv, Lk = key.shape[1], key.shape[2]
        n_rep = Hq // Hkv
        scale = scaling if scaling is not None else D ** -0.5
        qg = query.reshape(B, Hkv, n_rep, D)
        scores = torch.matmul(qg, key.transpose(-1, -2)).float() * scale
        allowed = torch.arange(Lk, device=query.device) <= cache.pos
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = scores.softmax(-1).to(value.dtype)
        out = torch.matmul(probs, value).reshape(B, Hq, 1, D)
        return out.transpose(1, 2).contiguous(), None

    # ---- forward helpers ------------------------------------------------------------------
    def _step(self, tok_ids: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        out = self.text_model(input_ids=tok_ids, position_ids=pos_ids, attention_mask=self.mask,
                              past_key_values=self.cache, use_cache=True)
        return self.lm_head(out.last_hidden_state[:, -1]).float()

    def bucket_for(self, total: int) -> int | None:
        for b in self.buckets:
            if total <= b:
                return b
        return None

    def _graph(self, bucket: int, adapter: str | None, scale: float) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        key = (bucket, adapter, round(float(scale), 4))
        if key in self.graphs:
            return self.graphs[key]
        assert self.cache is not None and self.cache.graph_mode
        # 캡처 전 워밍업은 별도 스트림에서 (torch 권장). 상태는 되돌린다.
        snap = self._snapshot()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._restore(snap)
                self._step(self.static_tok, self.static_pos)
        torch.cuda.current_stream().wait_stream(s)
        self._restore(snap)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            logits = self._step(self.static_tok, self.static_pos)
        self._restore(snap)
        self.graphs[key] = (graph, logits)
        return self.graphs[key]

    def _snapshot(self) -> dict[str, Any]:
        # 워밍업·캡처가 KV 에 쓰는 곳은 index_copy_(pos=filled) 한 칸뿐 — 그 칸만 저장한다
        # (전체 clone 은 64K 버킷에서 2 GiB). conv/rec 상태는 전체가 갱신되므로 통째로.
        c = self.cache
        assert c is not None
        f = c.filled
        return dict(
            k={i: t[:, :, f:f + 1].clone() for i, t in c.k.items()},
            v={i: t[:, :, f:f + 1].clone() for i, t in c.v.items()},
            conv=[None if s.conv_states[0] is None else s.conv_states[0].clone() for s in c.layers],
            rec=[None if s.recurrent_states[0] is None else s.recurrent_states[0].clone() for s in c.layers],
            filled=f,
        )

    def _restore(self, snap: dict[str, Any]) -> None:
        c = self.cache
        assert c is not None
        f = snap["filled"]
        for i, t in snap["k"].items():
            c.k[i][:, :, f:f + 1].copy_(t)
        for i, t in snap["v"].items():
            c.v[i][:, :, f:f + 1].copy_(t)
        for st, cv, rc in zip(c.layers, snap["conv"], snap["rec"]):
            if cv is not None and st.conv_states[0] is not None:
                st.conv_states[0].copy_(cv)
            if rc is not None and st.recurrent_states[0] is not None:
                st.recurrent_states[0].copy_(rc)
        c.filled = snap["filled"]
        c.pos.fill_(c.filled)

    # ---- sampling ---------------------------------------------------------------------------
    @staticmethod
    def _sample(logits: torch.Tensor, *, temperature: float, top_k: int, top_p: float, min_p: float) -> int:
        if temperature <= 0:
            return int(logits.argmax(-1))
        scores = logits / float(temperature)
        if top_k and top_k > 0 and top_k < scores.shape[-1]:
            kth = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
            scores = scores.masked_fill(scores < kth, float("-inf"))
        if min_p and min_p > 0:
            probs = scores.softmax(-1)
            scores = scores.masked_fill(probs < min_p * probs.max(-1, keepdim=True).values, float("-inf"))
        if top_p and 0 < top_p < 1:
            sorted_scores, order = torch.sort(scores, descending=True, dim=-1)
            cum = sorted_scores.softmax(-1).cumsum(-1)
            remove = cum - sorted_scores.softmax(-1) > top_p
            sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
            scores = torch.full_like(scores, float("-inf")).scatter(-1, order, sorted_scores)
        probs = scores.softmax(-1)
        return int(torch.multinomial(probs, 1))

    @staticmethod
    def _repetition_penalty(logits: torch.Tensor, ids: torch.Tensor, penalty: float) -> torch.Tensor:
        if penalty == 1.0 or ids.numel() == 0:
            return logits
        unique = torch.unique(ids)
        vals = logits[0, unique]
        vals = torch.where(vals > 0, vals / penalty, vals * penalty)
        logits[0, unique] = vals
        return logits

    # ---- public ------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        filter_mask: torch.Tensor | None,
        dry: DRYLogitsProcessor | None,
        adapter: str | None,
        adapter_scale: float,
        stop: list[str],
        cancel: threading.Event,
        on_piece: Callable[[str | None], None] | None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        n_in = int(ids.shape[1])
        bucket = self.bucket_for(n_in + max_new_tokens)
        if bucket is None:
            raise ValueError("context exceeds graph buckets")
        if seed is not None:
            torch.manual_seed(int(seed))
        self.cache = GraphCache(self, bucket)
        t0 = time.time()
        pos_all = torch.arange(n_in, device=self.device)[None]
        logits = None
        for s in range(0, n_in, PREFILL_CHUNK):
            e = min(s + PREFILL_CHUNK, n_in)
            logits = self._step(ids[:, s:e], pos_all[:, s:e])
            self.cache.advance(e - s)
        torch.cuda.synchronize()
        t1 = time.time()
        self.cache.graph_mode = True
        self.static_pos.fill_(self.cache.filled)
        graph, static_logits = self._graph(bucket, adapter, adapter_scale)
        assert logits is not None
        vocab = int(logits.shape[-1])
        if filter_mask is not None and filter_mask.numel() != vocab:
            # lm_head 는 config.vocab_size(248320) 폭, 토크나이저는 248077 — 패딩 id 는 모두 차단한다.
            padded = torch.ones(vocab, dtype=torch.bool, device=logits.device)
            padded[: filter_mask.numel()] = filter_mask
            filter_mask = padded

        generated: list[int] = []
        emitted = ""
        text = ""
        finish = "length"
        all_ids = ids[0]
        gen_tensor = torch.zeros(max_new_tokens, dtype=torch.long, device=self.device)
        try:
            for step_index in range(max_new_tokens):
                if cancel.is_set():
                    finish = "cancelled"
                    break
                scores = logits.clone()
                if repetition_penalty != 1.0:
                    scores = self._repetition_penalty(scores, all_ids, repetition_penalty)
                if filter_mask is not None:
                    scores = scores.masked_fill(filter_mask, float("-inf"))
                if dry is not None and generated:
                    scores = dry(gen_tensor[:len(generated)][None], scores)
                nxt = self._sample(scores, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
                if nxt in self.eos_ids:
                    finish = "stop"
                    break
                generated.append(nxt)
                gen_tensor[len(generated) - 1] = nxt
                all_ids = torch.cat([all_ids, gen_tensor[len(generated) - 1:len(generated)]])
                text = self.tok.decode(generated, skip_special_tokens=True)
                if text and not text.endswith("�") and len(text) > len(emitted):
                    piece = text[len(emitted):]
                    stop_at = min((text.find(s_) for s_ in stop if s_ and s_ in text), default=-1)
                    if stop_at >= 0:
                        text = text[:stop_at]
                        if on_piece is not None:
                            on_piece(None)
                        finish = "stop"
                        break
                    emitted = text
                    if on_piece is not None:
                        on_piece(piece)
                # 다음 토큰 forward (그래프 재생)
                self.static_tok.fill_(nxt)
                self.static_pos.fill_(self.cache.filled)
                graph.replay()
                logits = static_logits
                self.cache.advance(1)
                if self.cache.filled >= bucket - 1:
                    finish = "length"
                    break
        finally:
            self.cache.graph_mode = False
            self.cache = None
        torch.cuda.synchronize()
        t2 = time.time()
        n_out = len(generated)
        return dict(
            text=text, finish_reason=finish, n_in=n_in, n_out=n_out,
            prefill_s=round(t1 - t0, 2), decode_s=round(t2 - t1, 2),
            tok_s=round(n_out / max(t2 - t1, 1e-9), 1), bucket=bucket, engine="graph",
        )
