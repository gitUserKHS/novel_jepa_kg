"""공용 로더: Qwen3.5-4B 4bit (함정 3조 반영: lm_head 제외·오프라인·클래스 폴백).

출처: C:/연구_프로젝트/ai_아키텍처/applied/loader.py (2026-08-23 복사). 모델 경로는
NOVEL_QWEN_MODEL_DIR 환경변수로 지정 (기본: C:/llm_files/qwen35-4b).
"""

import os
import pathlib

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import _patches  # noqa: F401  SDPA GQA 우회
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = "C:/llm_files/qwen35-4b"
MODEL_ID = os.environ.get("NOVEL_QWEN_MODEL_DIR") or DEFAULT_MODEL_DIR
BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         llm_int8_skip_modules=["lm_head"])


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


def load_model():
    torch.cuda.set_per_process_memory_fraction(0.90)
    kw = dict(device_map={"": 0}, dtype=torch.bfloat16, quantization_config=BNB)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kw)
    model.eval()
    return model
