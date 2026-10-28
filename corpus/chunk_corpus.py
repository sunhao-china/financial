# -*- coding: utf-8 -*-
"""P2 文档切分（大纲 §7.4）：段落 → 超长按句边界，chunk 300~500 字，重叠 10%~15%。

- 流式分批（iter_batches），与 P0 一致，不整表入内存
- 边界优先级：句边界（。！？；）> 硬切；短段落跨段合并到目标长度
- 重叠实现：新 chunk 开头补上一 chunk 末尾 OVERLAP_CHARS 字（从句边界起，不截断句子）
- 输出对齐 S1 metadata（schemas/models.py: S1Metadata）：
  id = {source_file}_{doc_id}_c{index}，char_start/char_end 为原文偏移（可重建原文验证）
- 参数组合写入 chunking_method（paragraph_sentence_{max}_{overlap}），
  chunking_version 记录版本（换参数重切时新旧数据可区分对比，§8）

小批试验: python corpus/chunk_corpus.py --trial --limit 3000   # 输出 chunks_trial.parquet
全量:     python corpus/chunk_corpus.py                        # 输出 chunks.parquet
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# 切分参数（§7.4；全量前先在服务器小批试验对比检索 recall，再回填确认）
CHUNK_MIN = 300          # 目标最短 chunk（字符）
CHUNK_MAX = 500          # 目标最长 chunk（字符）
OVERLAP_CHARS = 50       # 重叠字符数（≈ CHUNK_MAX 的 10%~15%）
CHUNKING_VERSION = "v1.0"

# 句边界：句号/全角句点/叹号/问号/分号后，以及半角 . 后跟非数字非空白
# （金融语料大量用半角 . 代句号；后跟数字的是 8.11 式小数点，排除。
#  实测：纳入后 3000 篇 chunks 24,796 → 19,147，中位长度 350 → 466，偏移不变量保持）
SENT_SPLIT = re.compile(r"(?<=[。．！？；])|(?<=\.)(?=[^\d\s])")
PARA_SPLIT = re.compile(r"\n+")             # 段落边界：连续换行

CHUNKS_STATS = config.DATA_DIR / "reports" / "chunk_stats.md"


def _method_name() -> str:
    return f"paragraph_sentence_{CHUNK_MAX}_{OVERLAP_CHARS}"


def _atoms(text: str):
    """文本 → 原子区间列表 [(start, end)]：段落内容按句边界切，偏移严格对应原文。

    位置法：不拼接字符串，段间换行保留在原文区间里（P0 实测教训：丢 \n 会导致
    chunk 文本与 char_start/char_end 切片不一致，偏移漂移）。
    """
    atoms = []
    pos = 0
    for m in PARA_SPLIT.finditer(text):
        seg = text[pos:m.start()]
        if seg:
            s2 = 0
            for b in SENT_SPLIT.finditer(seg):
                atoms.append((pos + s2, pos + b.end()))
                s2 = b.end()
            if s2 < len(seg):
                atoms.append((pos + s2, pos + len(seg)))
        pos = m.end()
    seg = text[pos:]
    if seg:
        s2 = 0
        for b in SENT_SPLIT.finditer(seg):
            atoms.append((pos + s2, pos + b.end()))
            s2 = b.end()
        if s2 < len(seg):
            atoms.append((pos + s2, pos + len(seg)))
    return atoms


def split_doc(text: str):
    """单篇切分 → [(chunk_text, char_start, char_end), ...]，偏移为闭开区间。

    不变量：chunk_text == text[char_start:char_end]（tests 里全量验证）。
    """
    atoms = _atoms(text)
    chunks = []
    buf_s = buf_e = 0        # 当前正文区间（不含 carry 重叠）
    carry_s = carry_e = 0    # 上一 chunk 尾部，用作下一 chunk 重叠开头

    for a_s, a_e in atoms:
        if buf_e == buf_s:
            # 正文起点 = 重叠起点（首 chunk 为 0）
            buf_s = carry_e if carry_e > carry_s else a_s
        buf_e = a_e
        # while：超长单句/无句边界文本也要切到目标长度（实测单次 if 会漏切 1.8 万字尾部）
        while buf_e - buf_s >= CHUNK_MAX:
            window = text[buf_s:buf_s + CHUNK_MAX]
            bounds = [m.end() for m in SENT_SPLIT.finditer(window)]
            cut = min(max(bounds[-1] if bounds else CHUNK_MIN, CHUNK_MIN), CHUNK_MAX)
            body_e = buf_s + cut
            c_s = carry_s if carry_e > carry_s else buf_s
            chunks.append((c_s, body_e))
            buf_s = body_e
            ov = max(0, cut - OVERLAP_CHARS)
            carry_s, carry_e = buf_s - cut + ov, buf_s  # 上一 body 的尾部 OVERLAP 字
    if buf_e > buf_s:  # 尾部不足 CHUNK_MAX 的正文
        c_s = carry_s if carry_e > carry_s else buf_s
        chunks.append((c_s, buf_e))
    return [(text[s:e], s, e) for s, e in chunks]


def chunk_table(t: pa.Table) -> pa.Table:
    """一批文章 → 展开为 chunk 表（对齐 S1 metadata 字段）。"""
    rows = t.to_pydict()
    out = {"id": [], "text": [], "doc_id": [], "source_file": [], "industry_type": [],
           "quality_score": [], "perplexity": [], "num_words": [],
           "chunk_index": [], "chunk_count": [], "char_start": [], "char_end": [],
           "chunking_method": [], "chunking_version": [], "data_version": [],
           "doc_char_len": []}
    method = _method_name()
    for i, text in enumerate(rows["text"]):
        chunks = split_doc(text or "")
        for ci, (ctext, cs, ce) in enumerate(chunks):
            out["id"].append(f"{rows['source_file'][i]}_{rows['_id'][i]}_c{ci}")
            out["text"].append(ctext)
            out["doc_id"].append(str(rows["_id"][i]))
            out["source_file"].append(rows["source_file"][i])
            out["industry_type"].append(rows["industry_type"][i])
            out["quality_score"].append(float(rows["quality_score"][i]))
            out["perplexity"].append(float(rows["perplexity"][i]))
            out["num_words"].append(int(rows["num_words"][i]))
            out["chunk_index"].append(ci)
            out["chunk_count"].append(len(chunks))
            out["char_start"].append(int(cs))
            out["char_end"].append(int(ce))
            out["chunking_method"].append(method)
            out["chunking_version"].append(CHUNKING_VERSION)
            out["data_version"].append("2026-10-16")
            out["doc_char_len"].append(len(text or ""))
    return pa.table({k: pa.array(v) for k, v in out.items()})


def run_chunking(limit: int = 0, out_path=None):
    """流式切分（limit>0 时只处理前 limit 篇，用于试验）。"""
    out_path = out_path or (config.DATA_DIR / "chunks.parquet")
    pf = pq.ParquetFile(config.CLEANED_PARQUET)
    n_total = pf.metadata.num_rows
    writer = None
    n_docs = n_chunks = 0
    lens = []  # 每 chunk 长度（全量 ~185 万 int ≈ 15MB，可接受；分位数在批间不可合并）
    start = time.time()
    n_batches = n_total // 50_000 + 1 if not limit else min(limit, n_total) // 50_000 + 1
    for batch in tqdm(pf.iter_batches(batch_size=50_000), total=n_batches, desc="切分"):
        t = pa.Table.from_batches([batch])
        if limit and n_docs + t.num_rows > limit:
            t = t.slice(0, limit - n_docs)
        out = chunk_table(t)
        n_docs += t.num_rows
        n_chunks += out.num_rows
        lens.extend(pc.utf8_length(out.column("text")).to_pylist())
        if writer is None:
            writer = pq.ParquetWriter(out_path, out.schema)
        writer.write_table(out)
        if limit and n_docs >= limit:
            break
    if writer:
        writer.close()
    ls = sorted(lens)
    pct = lambda p: ls[int(len(ls) * p)] if ls else 0
    stats = (
        f"# chunk 统计（{CHUNKING_VERSION}，{_method_name()}）\n\n"
        f"- 处理文档: {n_docs:,} → chunks: {n_chunks:,}（平均 {n_chunks / max(n_docs, 1):.1f} chunk/篇）\n"
        f"- 长度分布: min={ls[0]}, p25={pct(0.25)}, 中位={pct(0.5)}, p75={pct(0.75)}, "
        f"p95={pct(0.95)}, max={ls[-1]}\n"
        f"- 耗时 {time.time()-start:.0f}s\n"
    )
    print(stats)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_STATS.write_text(stats, encoding="utf-8")
    print(f"已写入 {out_path} / {config.DATA_DIR / 'reports' / 'chunk_stats.md'}")
    return n_docs, n_chunks


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", action="store_true", help="小批试验模式（前 limit 篇）")
    ap.add_argument("--limit", type=int, default=3000)
    args = ap.parse_args()
    if args.trial:
        run_chunking(limit=args.limit, out_path=config.DATA_DIR / "chunks_trial.parquet")
    else:
        run_chunking()


if __name__ == "__main__":
    main()
