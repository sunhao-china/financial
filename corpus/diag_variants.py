# -*- coding: utf-8 -*-
"""诊断「同事实异表述」规模（回应检索 recall@10≈0.13 的根因分析）。

对每道题：定位 gold doc 中的证据 chunk（含 gold_answer 关键数字的段落）→
编码其向量 → HNSW 查 top-200 → 统计余弦 ≥0.85/0.80/0.75 的 distractor
chunk 数与来源文档数。这直接量化「正例在向量空间里有多少几乎重合的变体」。

对照设计：22 个 retrieval 失败题（recall@10=0）+ 10 个命中题，
验证「变体规模大 → 检索失败」的因果假设。

用法: python corpus/diag_variants.py [--topk 200]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import numpy as np
from corpus.retrieval import HybridRetriever

FAIL_IDS = ["q004", "q005", "q013", "q015", "q021", "q022", "q045", "q025",
            "q030", "q031", "q049", "q037", "q050", "q053", "q046", "q056",
            "q057", "q064", "q065", "q060"]
OK_IDS = ["q001", "q002", "q003", "q006", "q008", "q009", "q010", "q011",
          "q012", "q014"]

NUM_PAT = re.compile(r"\d+(?:\.\d+)?%?")


def find_evidence_chunk(ret, question, gold_answer, gold_docs):
    """在 gold doc 的 chunks 中定位证据 chunk。

    1) 优先：gold_answer 中的数字串命中某 chunk 文本（出题证据锚定在原文数字上）
    2) 兜底：question 向量与 gold doc 各 chunk 余弦最大者
    """
    keys = [m for m in NUM_PAT.findall(gold_answer) if len(m) >= 2]
    cand_labels = []
    for did in gold_docs:
        cand_labels.extend(ret.doc_to_labels.get(did, []))
    for lab in cand_labels:
        if any(k in ret.texts[lab] for k in keys):
            return ret.ids[lab], ret.doc_ids[lab], ret.texts[lab]
    # 兜底：question 向量与 gold doc 各 chunk 余弦最大者
    qv = ret.model.encode([question], normalize_embeddings=True)[0]
    best, best_cid, best_did, best_text = -1.0, None, None, ""
    for lab in cand_labels:
        cv = ret.model.encode([ret.texts[lab]], normalize_embeddings=True)[0]
        s = float(np.dot(qv, cv))
        if s > best:
            best, best_cid, best_did, best_text = s, ret.ids[lab], ret.doc_ids[lab], ret.texts[lab]
    return best_cid, best_did, best_text


def variants_of(ret, text, topk):
    """证据 chunk 文本 → 向量 → HNSW top-k → 各阈值 distractor 统计。"""
    v = ret.model.encode([text], normalize_embeddings=True)[0]
    labels, dists = ret.idx.knn_query(np.array([v]), k=topk)
    labels, dists = labels[0], dists[0]
    sims = 1.0 - dists   # cosine 距离 → 相似度
    stats = {t: {"chunks": 0, "docs": set()} for t in (0.85, 0.80, 0.75)}
    samples, gold_own = [], 0
    for lab, sim in zip(labels, sims):
        did = ret.doc_ids[lab]
        for t in stats:
            if sim >= t:
                stats[t]["chunks"] += 1
                stats[t]["docs"].add(did)
        if sim >= 0.85 and len(samples) < 3:
            samples.append((round(float(sim), 3), did, ret.texts[lab][:60].replace("\n", " ")))
    out = {}
    for t, d in stats.items():
        out[t] = {"chunks": d["chunks"], "docs": len(d["docs"])}
    return out, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=200)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    qa = json.loads((config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))
    by_id = {it["id"]: it for it in qa}
    ret = HybridRetriever()
    rows = []
    for grp, ids in (("fail", FAIL_IDS), ("ok", OK_IDS)):
        for iid in ids:
            it = by_id.get(iid)
            if not it:
                continue
            raw = it["source_doc_ids"]
            gold_docs = set(raw) if isinstance(raw, list) else set(eval(raw))
            t0 = time.time()
            cid, did, text = find_evidence_chunk(ret, it["question"], it["gold_answer"], gold_docs)
            if cid is None:
                rows.append({"id": iid, "grp": grp, "error": "no evidence chunk in gold doc"})
                print(f"{iid} [{grp}] 证据 chunk 未定位", flush=True)
                continue
            vstats, samples = variants_of(ret, text, args.topk)
            row = {"id": iid, "grp": grp, "chunk_id": cid, "doc_id": did,
                   "variants": vstats, "samples": samples,
                   "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"{iid} [{grp}] 证据={cid} | >=0.85: {vstats[0.85]['chunks']}chunks/"
                  f"{vstats[0.85]['docs']}docs | >=0.80: {vstats[0.80]['chunks']}/"
                  f"{vstats[0.80]['docs']} | >=0.75: {vstats[0.75]['chunks']}/"
                  f"{vstats[0.75]['docs']} | {row['seconds']}s", flush=True)
            for s in samples[:2]:
                print(f"     样本 [{s[0]}] doc={s[1]} {s[2]}", flush=True)

    out = config.REPORT_DIR / "diag_variants.json"
    out.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    # 汇总：fail vs ok 组的变体规模均值
    print("\n汇总（>=0.85 的 distractor 规模）：", flush=True)
    for grp in ("fail", "ok"):
        rs = [r for r in rows if r.get("grp") == grp and "variants" in r]
        if not rs:
            continue
        avg_ch = sum(r["variants"][0.85]["chunks"] for r in rs) / len(rs)
        avg_dc = sum(r["variants"][0.85]["docs"] for r in rs) / len(rs)
        print(f"  {grp}: n={len(rs)} 平均变体 chunks={avg_ch:.1f} 平均来源文档={avg_dc:.1f}", flush=True)


if __name__ == "__main__":
    main()
