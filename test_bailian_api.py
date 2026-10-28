# -*- coding: utf-8 -*-
"""百炼平台接口连通性测试：对话 / Embedding / Rerank。"""
import json
import os
import sys

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
BASE = "https://dashscope.aliyuncs.com"
CHAT_URL = f"{BASE}/compatible-mode/v1/chat/completions"
EMBED_URL = f"{BASE}/api/v1/services/embeddings/text-embedding/text-embedding"
RERANK_URL = f"{BASE}/api/v1/services/rerank/text-rerank/text-rerank"

HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def test_chat():
    """qwen-turbo 对话测试（走 OpenAI 兼容模式）。"""
    payload = {
        "model": "qwen-turbo",
        "messages": [{"role": "user", "content": "用一句话回答：什么是综合保税区？"}],
        "response_format": {"type": "json_object"},
    }
    r = requests.post(CHAT_URL, json=payload, headers=HEADERS, timeout=30)
    data = r.json()
    if r.status_code == 200:
        choice = data["choices"][0]
        usage = data.get("usage", {})
        print(f"[OK] qwen-turbo 对话: {choice['message']['content'][:60]!r}")
        print(f"     usage: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}")
    else:
        print(f"[FAIL] qwen-turbo 对话: HTTP {r.status_code} {json.dumps(data, ensure_ascii=False)[:200]}")


def test_embedding():
    """text-embedding-v3 向量化测试。"""
    payload = {
        "model": "text-embedding-v3",
        "input": {"texts": ["2021年全国综合保税区进出口值5.9万亿元", "测试文本"]},
    }
    r = requests.post(EMBED_URL, json=payload, headers=HEADERS, timeout=30)
    data = r.json()
    if r.status_code == 200 and data.get("output"):
        embs = data["output"]["embeddings"]
        print(f"[OK] text-embedding-v3: {len(embs)} 条向量, 维度={len(embs[0]['embedding'])}, "
              f"text_index={embs[0]['text_index']}")
    else:
        print(f"[FAIL] text-embedding-v3: HTTP {r.status_code} {json.dumps(data, ensure_ascii=False)[:200]}")


def test_rerank():
    """gte-rerank 精排测试。"""
    payload = {
        "model": "gte-rerank",
        "input": {
            "query": "综合保税区进出口值",
            "documents": [
                "全国综合保税区实现进出口值5.9万亿元，同比增长23.8%",
                "今天天气晴朗，适合出门散步",
                "海关总署公布综合保税区最新统计数据",
            ],
        },
    }
    r = requests.post(RERANK_URL, json=payload, headers=HEADERS, timeout=30)
    data = r.json()
    if r.status_code == 200 and data.get("output"):
        results = data["output"].get("results", [])
        print(f"[OK] gte-rerank: {len(results)} 条打分, 前2条 = "
              f"{[{'index': x['index'], 'score': round(x['relevance_score'], 3)} for x in results[:2]]}")
    else:
        print(f"[FAIL] gte-rerank: HTTP {r.status_code} {json.dumps(data, ensure_ascii=False)[:200]}")


if __name__ == "__main__":
    if not API_KEY:
        print("未找到 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)
    test_chat()
    test_embedding()
    test_rerank()
