# -*- coding: utf-8 -*-
"""P3 检索评估（大纲 §12）：qa_set.json × 5 检索配置消融 → recall@k / MRR@10。

正例口径：源文档全部 chunk（同文档相邻 chunk 是 P2 的口径；P3 正例 = 问题来源文档）。
拒答类问题不参与检索打分（P4 评估拒绝策略），报告单独注明。

用法（ECS）: .venv/bin/python corpus/eval_retrieval.py [--limit N]
产物: data/reports/p3_eval.json + p3_eval.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.retrieval import HybridRetriever

CONFIGS = {
    "vector":        dict(expand=False, use_bm25=False, rerank=False),
    "vector_expand": dict(expand=True, use_bm25=False, rerank=False),
    "bm25":          dict(expand=False, use_vector=False, rerank=False),
    "hybrid":        dict(expand=True, rerank=False),
    # 旧生产配置（深度 50/30、等权 sum、向量路无原问句）：A/B 基线
    "hybrid_rerank": dict(expand=True, rerank=True, bm25_top=50, rerank_top=30,
                          rrf_w=(1.0, 1.0), rrf_mode="sum", vec_raw=False),
    # 新生产配置（BM25 深度 300、精排 100、max 融合等权、原问句入向量路）
    "hybrid_rerank_max": dict(expand=True, rerank=True),
    # 扩展消融：无 LLM 扩展（queries/keywords 全空），验证变体/关键词零贡献后砍掉
    "hybrid_rerank_max_noexpand": dict(expand=False, rerank=True),
    # comparison 分解检索已证伪（2026-10-19 全量：MRR 0.325→0.147 + 意图误分类泄漏），
    # 配置移除；retrieval.py 中 decomp 分支保留但默认关闭，仅作记录
}
KS = [5, 10, 20]
REPORT = config.DATA_DIR / "reports"


def score(hits, positives):
    ranks = [h.rank for h in hits if h.label in positives]
    out = {f"recall@{k}": sum(1 for h in hits[:k] if h.label in positives) / max(len(positives), 1)
           for k in KS}
    out["mrr@10"] = 1 / min(ranks) if ranks and min(ranks) <= 10 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    qa = json.loads((config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))
    scored = [q for q in qa if q["category"] != "refuse" and q.get("source_doc_ids")
              and q.get("question")]
    refuse = [q for q in qa if q["category"] == "refuse"]
    if args.limit:
        scored = scored[:args.limit]
    print(f"评估 {len(scored)} 个文档类问题（拒答类 {len(refuse)} 个留待 P4）", flush=True)

    ret = HybridRetriever()
    rows = []          # [{question_id, category, config, **metrics}]
    skipped = 0
    t0 = time.time()
    for i, q in enumerate(scored, 1):
        positives = ret.labels_for_docs(q["source_doc_ids"])
        if not positives:
            skipped += 1
            print(f"  {i}/{len(scored)} {q['id']} 正例为空，跳过", flush=True)
            continue
        for name, cfg in CONFIGS.items():
            t1 = time.time()
            try:
                hits = ret.retrieve(q["question"], n=20, **cfg)
            except Exception as e:  # noqa: BLE001
                print(f"  {q['id']} {name} 失败: {e}", flush=True)
                continue
            m = score(hits, positives)
            m["question_id"], m["category"], m["config"] = q["id"], q["category"], name
            rows.append(m)
        print(f"  {i}/{len(scored)} {q['id']} {q['category']} {time.time()-t0:.0f}s", flush=True)

    # 汇总
    summary = {}
    lines = ["# P3 检索评估报告", "",
             f"- 评估集：{len(scored)} 个文档类问题（{len(refuse)} 个拒答类留待 P4），"
             f"跳过 {skipped} 个正例为空", f"- 正例：问题来源文档的全部 chunk", ""]
    for name in CONFIGS:
        by_cat = {}
        for r in rows:
            if r["config"] != name:
                continue
            c = by_cat.setdefault(r["category"], [])
            c.append(r)
        if not by_cat:
            continue
        all_r = [r for c in by_cat.values() for r in c]
        summary[name] = {f"recall@{k}": float(np.mean([r[f"recall@{k}"] for r in all_r]))
                         for k in KS}
        summary[name]["mrr@10"] = float(np.mean([r["mrr@10"] for r in all_r]))
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| 类别 | 问题数 | recall@5 | recall@10 | recall@20 | MRR@10 |")
        lines.append("|---|---|---|---|---|---|")
        for c, rs in sorted(by_cat.items()):
            lines.append(f"| {c} | {len(rs)} | "
                         + " | ".join(f"{np.mean([r[f'recall@{k}'] for r in rs]):.3f}" for k in KS)
                         + f" | {np.mean([r['mrr@10'] for r in rs]):.3f} |")
        lines.append(f"| **合计** | {len(all_r)} | "
                     + " | ".join(f"**{summary[name][f'recall@{k}']:.3f}**" for k in KS)
                     + f" | **{summary[name]['mrr@10']:.3f}** |")
        lines.append("")

    lines.append("## 对比汇总（recall@10 / MRR@10）")
    lines.append("")
    lines.append("| 配置 | " + " | ".join(f"recall@{k}" for k in KS) + " | MRR@10 |")
    lines.append("|---|---|---|---|---|")
    for name in CONFIGS:
        if name in summary:
            s = summary[name]
            lines.append(f"| {name} | " + " | ".join(f"{s[f'recall@{k}']:.3f}" for k in KS)
                         + f" | {s['mrr@10']:.3f} |")

    md = "\n".join(lines) + "\n"
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "p3_eval.md").write_text(md, encoding="utf-8")
    (REPORT / "p3_eval.json").write_text(json.dumps(
        {"summary": summary, "rows": rows, "total_seconds": time.time() - t0},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 汇总（recall@10 / MRR@10）===")
    for name, s in summary.items():
        print(f"  {name:14s} {s['recall@10']:.3f}  {s['mrr@10']:.3f}")
    print(f"报告 → {REPORT/'p3_eval.md'}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
