"""플랫폼 패치. (출처: C:/연구_프로젝트/ai_아키텍처/applied/_patches.py, 2026-08-23 복사)

1) SDPA GQA 우회 (import 만 하면 적용, 2026-08-21 실측): 이 Windows torch 2.13
   빌드의 fused 커널은 dense GQA(enable_gqa=True)를 지원하지 않아 math 백엔드
   (이차 메모리)로 추락 → 4K 프리필부터 OOM. transformers 의 use_gqa_in_sdpa
   게이트를 끄면 repeat_kv 경로 → memory-efficient 커널.

2) GQA-split 어텐션 (enable_gqa_split() 로 옵트인, 2026-08-22 실측): 1)의
   repeat_kv 경로는 K/V 를 16헤드로 실체화하고(64K 문맥에서 1 GiB) 청크
   프리필의 bool 마스크를 float 바이어스로 실체화(0.5 GiB)해 4K×64K 어텐션
   한 번에 1.6 GiB 스파이크. 대신
     - KV 헤드 그룹(4개)별로 expand 뷰(복사 없음)를 memory-efficient 커널에
       직접 넘기고,
     - 마스크 없이 과거 구간(전부 가시)과 현재 청크(is_causal)를 따로 계산해
       logsumexp 로 병합(flash-decoding 식)
   하면 같은 결과(합성 텐서 maxdiff 2e-4, bf16)에 스파이크 64 MiB.
   디코드(q_len=1)는 행렬곱 GQA 경로 — 120K 문맥에서 층당 36→3.9 ms (08-23).
   배치 1·패딩 없음·인과 어텐션 전제 — 그 외(비인과 비전 타워, dropout,
   position_bias, 배치>1)는 원래 sdpa 경로로 폴백.
"""

import torch
import torch.nn.functional as F
import transformers.integrations.sdpa_attention as _S
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

_S.use_gqa_in_sdpa = lambda *a, **k: False

_ORIG_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]
_ORIG_SDPA_MASK = ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
_EFF = torch.ops.aten._scaled_dot_product_efficient_attention


def gqa_split_sdpa(q, k, v, scale):
    """q [B,Hq,Lq,D] (연속 비필수), k/v [B,Hkv,Lk,D]. 하단-우측 정렬 인과.
    반환 [B,Hq,Lq,D]."""
    B, Hq, Lq, D = q.shape
    Hkv, Lk = k.shape[1], k.shape[2]
    n_rep = Hq // Hkv
    P = Lk - Lq
    if Lq == 1:
        # 디코드: kv 헤드별 [n_rep, D] 쿼리 묶음을 K^T 와 한 번에 행렬곱 → K/V 를 한 번만
        # 읽는 대역폭 최적 경로 (120K 실측: per-group SDPA 36 ms → 3.9 ms / 층).
        qg = q.reshape(B, Hkv, n_rep, D)
        scores = torch.matmul(qg, k.transpose(-1, -2)).float() * scale
        probs = scores.softmax(-1).to(v.dtype)
        return torch.matmul(probs, v).reshape(B, Hq, 1, D)
    outs = []
    for g in range(Hkv):
        qg = q[:, g * n_rep:(g + 1) * n_rep]
        kg, vg = k[:, g:g + 1], v[:, g:g + 1]
        kc = kg[:, :, P:].expand(B, n_rep, Lq, D)
        vc = vg[:, :, P:].expand(B, n_rep, Lq, D)
        o_c, lse_c, _, _ = _EFF(qg, kc, vc, None, P > 0, 0.0, True, scale=scale)
        if P > 0:
            kp = kg[:, :, :P].expand(B, n_rep, P, D)
            vp = vg[:, :, :P].expand(B, n_rep, P, D)
            o_p, lse_p, _, _ = _EFF(qg, kp, vp, None, True, 0.0, False, scale=scale)
            lse_p, lse_c = lse_p[..., :Lq], lse_c[..., :Lq]
            lse = torch.logaddexp(lse_p, lse_c)
            o = (o_p * (lse_p - lse).exp()[..., None].to(o_p.dtype)
                 + o_c * (lse_c - lse).exp()[..., None].to(o_c.dtype))
        else:
            o = o_c
        outs.append(o)
    return torch.cat(outs, 1)


def gqa_split_attention_forward(module, query, key, value, attention_mask,
                                dropout=0.0, scaling=None, is_causal=None,
                                **kwargs):
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    fallback = (not is_causal or dropout or query.shape[0] != 1
                or kwargs.get("position_bias") is not None
                or not query.is_cuda
                or (attention_mask is not None and attention_mask.dtype != torch.bool))
    if fallback:
        return _ORIG_SDPA(module, query, key, value, attention_mask, dropout=dropout,
                          scaling=scaling, is_causal=is_causal, **kwargs)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    out = gqa_split_sdpa(query, key, value, scale)
    return out.transpose(1, 2).contiguous(), None


def _no_mask(*args, **kwargs):
    """full_attention 층 마스크를 만들지 않는다 (인과성은 gqa_split 이 담당)."""
    return None


def enable_gqa_split():
    AttentionInterface.register("sdpa", gqa_split_attention_forward)
    AttentionMaskInterface.register("sdpa", _no_mask)


def disable_gqa_split():
    AttentionInterface.register("sdpa", _ORIG_SDPA)
    AttentionMaskInterface.register("sdpa", _ORIG_SDPA_MASK)


def _selftest():
    torch.manual_seed(0)
    dev = "cuda"
    B, Hq, Hkv, D = 1, 16, 4, 256
    for Lq, P in [(4096, 60000), (1, 65535), (1, 120000), (3000, 0), (512, 100), (7, 3)]:
        Lk = P + Lq
        # 모델과 같은 레이아웃: [B,L,H,D] → transpose(1,2) (비연속 q)
        q = torch.randn(B, Lq, Hq, D, device=dev, dtype=torch.bfloat16).transpose(1, 2)
        k = torch.randn(B, Hkv, Lk, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        iq = torch.arange(Lq, device=dev)[:, None] + P
        ik = torch.arange(Lk, device=dev)[None, :]
        mask = (ik <= iq)[None, None]
        ref = F.scaled_dot_product_attention(q, _S.repeat_kv(k, 4), _S.repeat_kv(v, 4),
                                             attn_mask=mask, scale=D ** -0.5)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = gqa_split_sdpa(q, k, v, D ** -0.5)
        torch.cuda.synchronize()
        spike = (torch.cuda.max_memory_allocated() - base) / 2**20
        diff = (out.float() - ref.float()).abs().max().item()
        print(f"Lq={Lq:5d} P={P:5d}  maxdiff {diff:.4f}  spike {spike:6.0f} MiB")
        assert diff < 0.02, "GQA-split 수치 불일치"
        del q, k, v, ref, out, mask
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
