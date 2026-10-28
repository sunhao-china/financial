# -*- coding: utf-8 -*-
"""对比 deepseek-flash 与 deepseek-v4-pro 在 RAG 合成任务上的质量/延迟/成本。"""
import json
import os
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_KEY = os.environ["DEEPSEEK_API_KEY"]
URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

PROMPT = """你是严谨的金融信息助手。只依据下面资料回答，每个结论后标注引用编号 [1][2]。
资料中未提及的内容明确说"现有资料中未提及"，禁止编造数字。

【资料】
[1] 从海关总署获悉:2021年,全国综合保税区实现进出口值5.9万亿元,同比增长23.8%,较同期全国外贸进出口21.4%的增幅高2.4个百分点,占同期全国外贸进出口总值的15.1%。
[2] 2022年前三季度,全国综合保税区进出口4.39万亿元,同比增长12.5%,占我国外贸进出口总值的比重为17.1%,较2021年提升2个百分点。
[3] 综合保税区是设立在内陆地区、具有保税港区功能的海关特殊监管区域,实行封闭监管,集保税加工、保税物流、保税服务等功能于一体。

【问题】2021年综合保税区进出口值是多少？占外贸比重如何？"""


def run(model: str) -> dict:
    t0 = time.time()
    r = requests.post(URL, json={"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 1500}, headers=HEADERS, timeout=180)
    d = r.json()
    elapsed = time.time() - t0
    if r.status_code != 200:
        return {"model": model, "error": d, "elapsed": elapsed}
    msg = d["choices"][0]["message"]
    usage = d["usage"]
    return {
        "model": model,
        "elapsed": elapsed,
        "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
        "completion_tokens": usage["completion_tokens"],
        "answer": msg["content"],
    }


if __name__ == "__main__":
    for model in ["deepseek-flash", "deepseek-v4-pro"]:
        res = run(model)
        print("=" * 70)
        if "error" in res:
            print(f"[FAIL] {model}: {json.dumps(res['error'], ensure_ascii=False)[:150]}")
            continue
        print(f"[{model}] 耗时 {res['elapsed']:.1f}s | 推理token {res['reasoning_tokens']} | 总completion {res['completion_tokens']}")
        print(res["answer"][:600])
        print()
