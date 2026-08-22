"""Qwen3.5-4B 로컬 모델 서버 — OpenAI 호환 /v1/chat/completions (SSE 스트리밍).

RTX 4060 8GB 에서 실측으로 확정한 경로를 그대로 쓴다 (2026-08-23):
- 4bit nf4 로드(lm_head 제외), GQA-split 어텐션(_patches), 2K 청크 프리필 → 120K 문맥
- 한국어 전용 토큰 필터(한자·가나·키릴 등 93,504 토큰 차단) — 요청별 on/off
- LoRA 어댑터(cute 등) 요청별 선택 + 출력 배율(기본 0.6: 1.0 은 사실 오류·반복 증가)
- 단일 GPU 이므로 생성은 직렬화(락). 스트리밍 중 클라이언트가 끊으면 생성을 멈춘다.

환경변수:
  NOVEL_QWEN_MODEL_DIR   모델 디렉터리 (기본: 랩의 applied/models/qwen35-4b)
  NOVEL_QWEN_ADAPTERS    "이름=경로;이름=경로" (기본: cute=랩의 applied/adapters/cute)
  NOVEL_QWEN_HOST / NOVEL_QWEN_PORT  (기본 127.0.0.1 / 8765)
실행: run_model_server.bat  (transformers 5.15·bitsandbytes·peft 가 있는 venv 필요)
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

STACK_DIR = Path(__file__).resolve().parent / "qwen_stack"
if str(STACK_DIR) not in sys.path:
    sys.path.insert(0, str(STACK_DIR))

from loader import MODEL_ID, load_model, load_tokenizer  # noqa: E402  (torch 보다 먼저)
import _patches  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from gen_utils import foreign_token_ids, korean_filter, korean_strict_filter, prefill_chunked  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from graph_decode import GraphDecoder  # noqa: E402
from samplers import DRYLogitsProcessor  # noqa: E402
from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer  # noqa: E402

DEFAULT_ADAPTERS = "cute=C:/연구_프로젝트/ai_아키텍처/applied/adapters/cute"
MAX_CONTEXT = 110_000  # 실측 한계 120K 에 여유 (디코드 재할당 스톨 회피)
# 그래프 디코드 상한. KV 버퍼 32 KiB/tok: 32768=1 GiB, 65536=2 GiB. 초과 문맥은 eager 폴백.
MAX_BUCKET = int(os.environ.get("NOVEL_QWEN_MAX_BUCKET", "32768"))
PREFILL_CHUNK = 2048
MODEL_NAME = "qwen3.5-4b-4bit"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 20
    max_tokens: int = Field(default=1024, ge=1, le=16384)
    stream: bool = False
    adapter: str | None = None
    adapter_scale: float = Field(default=0.6, ge=0.0, le=1.5)
    korean_filter: bool | str = True  # True | False | "strict" (라틴 단어까지 차단)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=1.5)
    no_repeat_ngram_size: int = Field(default=0, ge=0, le=64)
    seed: int | None = None
    stop: list[str] = Field(default_factory=list)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    # DRY (구절 반복 억제): multiplier 0 이면 꺼짐. 생성 토큰만 본다.
    dry_multiplier: float = Field(default=0.0, ge=0.0, le=5.0)
    dry_base: float = Field(default=1.75, ge=1.0, le=4.0)
    dry_allowed_length: int = Field(default=2, ge=1, le=16)
    engine: str = "auto"  # auto | graph | eager  (graph: 문맥+생성 <= 최대 버킷이면 CUDA 그래프 디코드)


class _CancelCriteria(StoppingCriteria):
    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # type: ignore[override]
        return self.event.is_set()


def _parse_adapters(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in [part.strip() for part in spec.split(";") if part.strip()]:
        if "=" not in item:
            continue
        name, path = item.split("=", 1)
        if Path(path.strip()).exists():
            out[name.strip()] = path.strip()
    return out


class Engine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        # 단일 GPU 스레드: CUDA 그래프는 캡처와 재생이 같은 스레드여야 한다 (08-23 크래시 원인).
        # 모든 생성 요청을 이 스레드로 마셜한다.
        self.gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-gpu")
        self.loaded_at = time.time()
        self.tok = load_tokenizer()
        _patches.enable_gqa_split()
        self.model = load_model()
        self.filter = korean_filter(self.tok)
        self.strict_filter = korean_strict_filter(self.tok)
        self.filter_size = len(foreign_token_ids(self.tok))
        self.adapters: dict[str, str] = _parse_adapters(os.environ.get("NOVEL_QWEN_ADAPTERS", DEFAULT_ADAPTERS))
        self._base_scaling: dict[str, list[tuple[Any, float]]] = {}
        self.peft = False
        if self.adapters:
            from peft import PeftModel

            first = True
            for name, path in self.adapters.items():
                if first:
                    self.model = PeftModel.from_pretrained(self.model, path, adapter_name=name)
                    first = False
                else:
                    self.model.load_adapter(path, adapter_name=name)
                self._base_scaling[name] = [
                    (module, float(module.scaling[name]))
                    for module in self.model.modules()
                    if isinstance(getattr(module, "scaling", None), dict) and name in module.scaling
                ]
            self.peft = True
            self.model.base_model.disable_adapter_layers()
        self.model.eval()
        self.active_adapter: str | None = None
        self.active_scale: float = 1.0
        self.generations = 0
        self.graph_disabled = False  # 그래프 캡처/재생이 CUDA 오류를 내면 eager 로 강등
        # CUDA 그래프 디코더 (08-23 실측 5배). 어휘 필터는 bool 마스크로도 보관.
        vocab = len(self.tok)
        self.filter_mask = torch.zeros(vocab, dtype=torch.bool, device=self.model.device)
        self.filter_mask[torch.tensor(self.filter[0].ids, device=self.model.device)] = True
        self.strict_mask = torch.zeros(vocab, dtype=torch.bool, device=self.model.device)
        self.strict_mask[torch.tensor(self.strict_filter[0].ids, device=self.model.device)] = True
        self.graph = GraphDecoder(self.model, self.tok, max_bucket=MAX_BUCKET)
        print(f"[model-server] graph decoder ready (buckets {self.graph.buckets})", flush=True)

    # ---- adapters -------------------------------------------------------------------------
    def _apply_adapter(self, name: str | None, scale: float) -> None:
        if not self.peft:
            return
        if name is None:
            if self.active_adapter is not None:
                self.model.base_model.disable_adapter_layers()
                self.active_adapter = None
            return
        if name not in self.adapters:
            raise HTTPException(status_code=400, detail=f"unknown adapter '{name}'. available: {sorted(self.adapters)}")
        for module, base in self._base_scaling[name]:
            module.scaling[name] = base * scale
        if self.active_adapter is None:
            self.model.base_model.enable_adapter_layers()
        if self.active_adapter != name:
            self.model.set_adapter(name)
        self.active_adapter, self.active_scale = name, scale

    # ---- prompt ---------------------------------------------------------------------------
    def encode(self, messages: list[ChatMessage]) -> torch.Tensor:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        text = self.tok.apply_chat_template(
            payload, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tok(text, return_tensors="pt")["input_ids"].to(self.model.device)

    # ---- generation -----------------------------------------------------------------------
    def run(self, req: ChatRequest, on_piece=None, cancel: threading.Event | None = None) -> dict[str, Any]:
        cancel = cancel or threading.Event()
        # 모든 CUDA 작업(프리필·그래프 캡처·재생·샘플링)을 단일 GPU 스레드에서 실행한다.
        return self.gpu.submit(self._run_impl, req, on_piece, cancel).result()

    def _run_impl(self, req: ChatRequest, on_piece, cancel: threading.Event) -> dict[str, Any]:
        with self.lock:
            self._apply_adapter(req.adapter, req.adapter_scale)
            ids = self.encode(req.messages)
            n_in = int(ids.shape[1])
            if n_in + req.max_tokens > MAX_CONTEXT:
                raise HTTPException(
                    status_code=413,
                    detail=f"prompt {n_in} tokens + max_tokens {req.max_tokens} exceeds {MAX_CONTEXT}",
                )
            if req.seed is not None:
                torch.manual_seed(int(req.seed))
            torch.cuda.reset_peak_memory_stats()
            use_graph = (not self.graph_disabled) and (
                req.engine == "graph"
                or (req.engine == "auto" and self.graph.bucket_for(n_in + req.max_tokens) is not None)
            )
            if use_graph:
                try:
                    return self._run_graph(req, ids, n_in, on_piece, cancel)
                except torch.cuda.CUDAError as exc:
                    self.graph_disabled = True
                    print(f"[model-server] graph decode disabled after CUDA error: {exc}", flush=True)
                    raise HTTPException(status_code=503, detail="graph decode failed; retry (eager fallback engaged)")
            t0 = time.time()
            pkv = prefill_chunked(self.model, ids, PREFILL_CHUNK) if n_in > PREFILL_CHUNK else None
            t1 = time.time()
            streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
            kwargs: dict[str, Any] = dict(
                input_ids=ids,
                max_new_tokens=int(req.max_tokens),
                streamer=streamer,
                stopping_criteria=StoppingCriteriaList([_CancelCriteria(cancel)]),
                repetition_penalty=float(req.repetition_penalty),
            )
            if pkv is not None:
                kwargs["past_key_values"] = pkv
            if req.temperature <= 0:
                kwargs["do_sample"] = False
            else:
                kwargs.update(do_sample=True, temperature=float(req.temperature), top_p=float(req.top_p), top_k=int(req.top_k))
            if req.no_repeat_ngram_size:
                kwargs["no_repeat_ngram_size"] = int(req.no_repeat_ngram_size)
            processors = LogitsProcessorList()
            if req.korean_filter == "strict":
                processors.extend(self.strict_filter)
            elif req.korean_filter:
                processors.extend(self.filter)
            if req.dry_multiplier > 0:
                processors.append(DRYLogitsProcessor(
                    self.tok, n_in, multiplier=req.dry_multiplier, base=req.dry_base,
                    allowed_length=req.dry_allowed_length,
                ))
            if processors:
                kwargs["logits_processor"] = processors
            if req.min_p > 0 and req.temperature > 0:
                kwargs["min_p"] = float(req.min_p)

            errors: list[BaseException] = []

            def _worker() -> None:
                try:
                    with torch.no_grad():
                        self.model.generate(**kwargs)
                except BaseException as exc:  # noqa: BLE001 - surfaced to the client below
                    errors.append(exc)
                    streamer.end()

            thread = threading.Thread(target=_worker, name="qwen-generate", daemon=True)
            thread.start()
            pieces: list[str] = []
            text = ""
            finish = "stop"
            try:
                for piece in streamer:
                    if not piece:
                        continue
                    pieces.append(piece)
                    text = "".join(pieces)
                    stop_at = min((text.find(s) for s in req.stop if s and s in text), default=-1)
                    if stop_at >= 0:
                        text = text[:stop_at]
                        cancel.set()
                        if on_piece is not None:
                            on_piece(None)  # 신호: 이후 조각 무시, 최종 text 로 대체
                        break
                    if on_piece is not None:
                        on_piece(piece)
            finally:
                cancel.set()
                thread.join()
            if errors:
                raise errors[0]
            n_out = len(self.tok.encode(text)) if text else 0
            if n_out >= req.max_tokens:
                finish = "length"
            if cancel.is_set() and not req.stop:
                pass
            t2 = time.time()
            self.generations += 1
            return dict(
                text=text,
                finish_reason=finish,
                usage=dict(prompt_tokens=n_in, completion_tokens=n_out, total_tokens=n_in + n_out),
                timing=dict(prefill_s=round(t1 - t0, 2), decode_s=round(t2 - t1, 2),
                            tok_s=round(n_out / max(t2 - t1, 1e-9), 1)),
                peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                adapter=req.adapter, adapter_scale=req.adapter_scale if req.adapter else None,
            )

    def _run_graph(self, req: ChatRequest, ids: torch.Tensor, n_in: int, on_piece, cancel: threading.Event) -> dict[str, Any]:
        mask = None
        if req.korean_filter == "strict":
            mask = self.strict_mask
        elif req.korean_filter:
            mask = self.filter_mask
        dry = None
        if req.dry_multiplier > 0:
            dry = DRYLogitsProcessor(self.tok, 0, multiplier=req.dry_multiplier, base=req.dry_base,
                                     allowed_length=req.dry_allowed_length)
        result = self.graph.generate(
            ids, max_new_tokens=int(req.max_tokens), temperature=float(req.temperature), top_p=float(req.top_p),
            top_k=int(req.top_k), min_p=float(req.min_p), repetition_penalty=float(req.repetition_penalty),
            filter_mask=mask, dry=dry, adapter=req.adapter if self.peft else None, adapter_scale=float(req.adapter_scale),
            stop=list(req.stop), cancel=cancel, on_piece=on_piece, seed=req.seed,
        )
        self.generations += 1
        n_out = int(result["n_out"])
        return dict(
            text=result["text"],
            finish_reason="stop" if result["finish_reason"] == "stop" else "length",
            usage=dict(prompt_tokens=n_in, completion_tokens=n_out, total_tokens=n_in + n_out),
            timing=dict(prefill_s=result["prefill_s"], decode_s=result["decode_s"], tok_s=result["tok_s"],
                        engine="graph", bucket=result["bucket"]),
            peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
            adapter=req.adapter, adapter_scale=req.adapter_scale if req.adapter else None,
        )

    def status(self) -> dict[str, Any]:
        return dict(
            status="ok",
            model=MODEL_NAME,
            model_dir=MODEL_ID,
            adapters=sorted(self.adapters),
            korean_filter_tokens=self.filter_size,
            busy=self.lock.locked(),
            vram_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
            max_context=MAX_CONTEXT,
            graph_buckets=list(self.graph.buckets),
            graph_disabled=self.graph_disabled,
            graphs_captured=len(self.graph.graphs),
            generations=self.generations,
            uptime_s=int(time.time() - self.loaded_at),
        )


app = FastAPI(title="Novel Qwen model server", version="1.0")
engine: Engine | None = None


@app.on_event("startup")
def _startup() -> None:
    global engine
    t0 = time.time()
    engine = Engine()
    print(f"[model-server] loaded {MODEL_NAME} in {time.time()-t0:.0f}s, adapters={sorted(engine.adapters)}, "
          f"VRAM {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)


def _engine() -> Engine:
    if engine is None:
        raise HTTPException(status_code=503, detail="model is still loading")
    return engine


@app.get("/")
@app.get("/health")
def health() -> dict[str, Any]:
    if engine is None:
        return {"status": "loading", "model": MODEL_NAME}
    return engine.status()


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}]}


@app.post("/v1/count_tokens")
def count_tokens(payload: dict[str, Any]) -> dict[str, int]:
    eng = _engine()
    if "messages" in payload:
        messages = [ChatMessage(**m) for m in payload["messages"]]
        return {"tokens": int(eng.encode(messages).shape[1])}
    return {"tokens": len(eng.tok.encode(str(payload.get("text", ""))))}


def _completion_payload(result: dict[str, Any], request_id: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]},
                     "finish_reason": result["finish_reason"]}],
        "usage": result["usage"],
        "timing": result["timing"],
        "peak_gib": result["peak_gib"],
        "adapter": result["adapter"],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    eng = _engine()
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    if not req.stream:
        result = await run_in_threadpool(eng.run, req)
        return JSONResponse(_completion_payload(result, request_id))

    q: queue.Queue = queue.Queue()
    cancel = threading.Event()
    box: dict[str, Any] = {}

    def _producer() -> None:
        try:
            box["result"] = eng.run(req, on_piece=q.put, cancel=cancel)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            q.put(_END)

    threading.Thread(target=_producer, name="qwen-producer", daemon=True).start()

    async def _events():
        def chunk(delta: str | None, finish: str | None = None, extra: dict[str, Any] | None = None) -> str:
            body: dict[str, Any] = {
                "id": request_id, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": ({"content": delta} if delta else {}), "finish_reason": finish}],
            }
            if extra:
                body.update(extra)
            return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

        ignore_rest = False
        try:
            while True:
                try:
                    item = await run_in_threadpool(q.get, True, 0.5)
                except queue.Empty:
                    if await request.is_disconnected():
                        cancel.set()
                        return
                    continue
                if item is _END:
                    break
                if item is None:
                    ignore_rest = True
                    continue
                if not ignore_rest:
                    yield chunk(item)
            if "error" in box:
                err = box["error"]
                detail = err.detail if isinstance(err, HTTPException) else f"{type(err).__name__}: {err}"
                yield f"data: {json.dumps({'error': detail}, ensure_ascii=False)}\n\n"
            else:
                result = box["result"]
                yield chunk(None, result["finish_reason"],
                            {"usage": result["usage"], "timing": result["timing"], "peak_gib": result["peak_gib"],
                             "text": result["text"] if ignore_rest else None})
            yield "data: [DONE]\n\n"
        finally:
            cancel.set()

    return StreamingResponse(
        _events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


_END = object()


def main() -> None:
    import uvicorn

    host = os.environ.get("NOVEL_QWEN_HOST", "127.0.0.1")
    port = int(os.environ.get("NOVEL_QWEN_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="info", workers=1)


if __name__ == "__main__":
    main()
