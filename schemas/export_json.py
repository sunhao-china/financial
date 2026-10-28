# -*- coding: utf-8 -*-
"""把 S1~S9 pydantic 模型导出为 JSON Schema（大纲 §9.1：集中定义、随代码入库）。

pydantic 模型是运行时校验的唯一事实源，JSON Schema 用于存储/文档/外部工具。
用法: python schemas/export_json.py
输出: schemas/json/*.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import schemas.models as m

OUT = Path(__file__).parent / "json"

MODELS = {
    "s1": m.S1Record,
    "s2": m.S2IntentExpand,
    "s3": m.S3SelfCheck,
    "s4": m.S4Verify,
    "s5": m.S5Hit,
    "s6": m.S6Context,
    "s7": m.S7Snapshot,
    "s8": m.S8Event,
    "s9_chat_request": m.S9ChatRequest,
    "s9_event": m.S9Event,
    "s9_sources_response": m.S9SourcesResponse,
}


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cls in MODELS.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(cls.model_json_schema(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"导出 {path.name}")
    print(f"共 {len(MODELS)} 个结构 → {OUT}")


if __name__ == "__main__":
    export()
