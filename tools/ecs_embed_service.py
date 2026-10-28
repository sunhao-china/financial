# -*- coding: utf-8 -*-
"""ECS 在线查询阶段 embedding 服务（P3 起用，与 AutoDL 离线管线一致）。

- 同一模型文件（tools/model_lock.json sha256 锁定）+ sentence-transformers + truncate_dim=512 + normalize
- FastAPI /v1/embeddings（OpenAI 兼容，供检索链路调用）
- CPU fp32 推理（一致性实测：与 GPU fp16 余弦 ≥0.99986，见 model_lock.json）

部署（ECS 上）：
    pip install fastapi uvicorn sentence-transformers pyarrow
    python tools/ecs_embed_service.py --model-dir /data/models/bge-m3 --port 8010
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

app = FastAPI(title="embedding-service", version="1.0")
MODEL = None
ARGS = None
LOAD_TS = 0.0


class EmbedRequest(BaseModel):
    model: str = "bge-m3"
    input: list[str] = Field(min_length=1, max_length=256)
    dimensions: int | None = None  # 兼容 OpenAI 参数；实际由服务端 --dim 决定，忽略


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbedResponse(BaseModel):
    object: str = "list"
    model: str
    data: list[EmbeddingData]
    usage: dict


def verify_model_lock(model_dir: Path) -> dict:
    """校验模型文件 sha256 与 tools/model_lock.json 一致（一致性锁）。"""
    lock_path = Path(__file__).parent / "model_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for entry in lock["models"].values():
        if model_dir.name == Path(entry["local_dir"]).name:
            bad = []
            for rel, want in entry["files"].items():
                f = model_dir / rel
                got = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
                if got != want:
                    bad.append((rel, got))
            return {"model": entry["local_dir"], "missing_or_mismatch": bad,
                    "ok": not bad}
    return {"model": str(model_dir), "missing_or_mismatch": ["未在锁文件中"], "ok": False}


@app.get("/health")
def health():
    return {"status": "ok", "model": ARGS.model_dir, "dim": ARGS.dim,
            "load_ts": LOAD_TS, "lock": verify_model_lock(Path(ARGS.model_dir))}


@app.post("/v1/embeddings", response_model=EmbedResponse)
def embeddings(req: EmbedRequest):
    t0 = time.time()
    vecs = MODEL.encode(req.input, truncate_dim=ARGS.dim, normalize_embeddings=True,
                        batch_size=32, convert_to_numpy=True)
    vecs = vecs.astype(np.float32).reshape(len(req.input), -1)
    data = [EmbeddingData(index=i, embedding=[float(x) for x in v])
            for i, v in enumerate(vecs)]
    return EmbedResponse(model=req.model, data=data,
                         usage={"prompt_tokens": sum(len(t) for t in req.input),
                                "total_tokens": sum(len(t) for t in req.input),
                                "latency_ms": int((time.time() - t0) * 1000)})


def main():
    global MODEL, ARGS, LOAD_TS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/data/models/bge-m3")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--port", type=int, default=8010)
    ARGS = ap.parse_args()

    lock = verify_model_lock(Path(ARGS.model_dir))
    if not lock["ok"]:
        print("⚠️ 模型文件与 model_lock.json 不一致:", lock["missing_or_mismatch"])
        print("  为保证离线/在线一致性，拒绝启动。请重新传输模型文件。")
        raise SystemExit(1)
    print("模型锁校验通过 ✓")

    t0 = time.time()
    MODEL = SentenceTransformer(ARGS.model_dir, device="cpu", trust_remote_code=True)
    LOAD_TS = round(time.time() - t0, 1)
    print(f"模型加载完成（CPU fp32，{ARGS.dim} 维 MRL），耗时 {LOAD_TS}s")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port, log_level="info")


if __name__ == "__main__":
    main()
