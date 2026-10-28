# -*- coding: utf-8 -*-
"""doc_id → 文献名映射构建（2026-10-19 加，§14.5 引用来源可读性）。

语料为匿名化财经新闻正文（无标题/URL 字段），文献名由正文首句规则提炼：
剥离导语前缀（「XX网X月X日消息:」「据XX报道」「从XX获悉:」等）→ 取首个句末
标点前主干 → 截 44 字符。纯规则零 LLM 成本，46 万篇约 1-2 分钟。

用法: python tools/build_title_map.py [--input data/cleaned.parquet] [--output data/title_map.parquet]
产物: parquet 两列 doc_id / title（title 为空的行不写入，前端回退「文章 #id」）
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

# 导语前缀（顺序重要：长的先匹配）。全部剥离后剩下的才是正文主干。
LEAD_PATTERNS = [
    # 「(原标题:…)」包裹：剥掉包裹符（收尾括号在 extract_title 里单独处理）
    re.compile(r"^[（(]?\s*原标题[:：]\s*"),
    # 正文前的栏目标签：「导 读」「导读:」「摘 要」「(经济观察)」
    re.compile(r"^[导读摘]\s*[读要]\s*[:：]?\s*"),
    re.compile(r"^[（(]?(?:经济观察|财经观察)[)）]\s*"),
    # 「XX网X月X日电(记者 XX)」通讯社导语（含括号记者名）
    re.compile(r"^[^，,。；;:：]{1,30}?[电讯][（(][^)）]{1,24}[)）][,，]?\s*"),
    # 「XX网X月X日消息:」
    re.compile(r"^[^\s，,。；;:：]{1,14}?(?:网|报|社|台)\s*\d{1,4}年?\d{0,2}月?\d{0,2}日?"
               r"(?:消息|电|讯|快讯)[:：]?\s*"),
    re.compile(r"^据[^，,。；;]{1,24}?(?:报道|消息|介绍|透露|获悉)[:：]?\s*"),
    re.compile(r"^从[^，,。；;]{1,24}?获悉[:：]?\s*"),
    re.compile(r"^(?:新华社|中新社|路透社|彭博)[^，,。；;]{0,20}?电\s*"),
]
# 句首时间短语（剥离后文献名更紧凑）
TIME_PREFIX = re.compile(r"^(?:今日|今天|昨日|昨天|近日|日前|本周|本月|今年|去年)"
                         r"[,，、\s]*")
TITLE_MAX = 44  # 超长在词边界截断


def extract_title(text: str) -> str:
    """正文首句 → 文献名。提炼失败返回空串（调用方回退 doc_id 展示）。"""
    if not text:
        return ""
    seg = text.split("\n")[0].strip()
    wrapped = bool(re.match(r"^[（(]?\s*原标题[:：]", seg))
    for pat in LEAD_PATTERNS:
        m = pat.match(seg)
        if m:
            seg = seg[m.end():]
    if wrapped:  # 「(原标题:X)」的收尾括号
        seg = re.sub(r"[)）]\s*$", "", seg)
    # 双空格分隔的记者署名（通讯社格式「……电 题:」）截断于署名前
    seg = re.split(r"\s{2,}", seg, 1)[0]
    seg = TIME_PREFIX.sub("", seg).strip()
    # 取首个句末标点前的主干
    m = re.match(r"^[^。！？!?]+", seg)
    if m:
        seg = m.group(0).strip()
    seg = re.sub(r'["\'「」『』]', "", seg).strip(" ，,、;；:：")
    if len(seg) < 6:  # 过短无信息量（如「本报讯」）
        return ""
    if len(seg) > TITLE_MAX:
        seg = seg[:TITLE_MAX].rstrip(" ，,、;；:：") + "…"
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(config.CLEANED_PARQUET))
    ap.add_argument("--output", default=str(config.DATA_DIR / "title_map.parquet"))
    args = ap.parse_args()

    t0 = time.time()
    t = pq.read_table(args.input, columns=["_id", "text"], memory_map=True)
    ids = t.column("_id").to_pylist()
    texts = t.column("text").to_pylist()
    n = len(ids)
    print(f"语料 {n:,} 篇，开始提炼…", flush=True)

    titles, kept = [], 0
    for i, (did, txt) in enumerate(zip(ids, texts)):
        title = extract_title(txt)
        if title:
            titles.append((did, title))
            kept += 1
        if (i + 1) % 100_000 == 0:
            print(f"  {i + 1:,}/{n:,}（有效 {kept:,}）{time.time() - t0:.0f}s", flush=True)

    out = Path(args.output)
    # doc_id 转字符串：chunks.parquet 的 doc_id = str(_id)（corpus/chunk_corpus.py），必须同型对齐
    pq.write_table(pa.table({"doc_id": pa.array([str(d) for d, _ in titles]),
                             "title": pa.array([s for _, s in titles])}), out)
    print(f"✓ 有效文献名 {kept:,}/{n:,}（{(kept / n * 100):.1f}%）→ {out} "
          f"({out.stat().st_size / 1e6:.1f}MB，{time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
