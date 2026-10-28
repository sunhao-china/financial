# -*- coding: utf-8 -*-
"""q065/q060 全量正例 chunk + gold_answer 检查：判断证据是否存在于源文档（评估集质量 vs 语料问题）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.retrieval import HybridRetriever

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    qa = {q["id"]: q for q in json.loads(
        (config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))}
    ret = HybridRetriever()
    for qid in ["q065", "q060"]:
        q = qa[qid]
        pos = sorted(ret.labels_for_docs(q["source_doc_ids"]))
        print(f"\n[{qid}] {q['question'][:60]}", flush=True)
        print(f"  gold_answer: {q['gold_answer'][:120]}", flush=True)
        for lab in pos:
            t = ret.texts[lab].replace("\n", " ")
            print(f"    chunk[{lab}] ({len(t)}字): {t[:110]}", flush=True)


if __name__ == "__main__":
    main()
