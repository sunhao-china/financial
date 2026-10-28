# -*- coding: utf-8 -*-
"""切分验证关卡（大纲 §7.4：任何切分参数改动先跑此脚本再全量）。

不变量（对 chunks parquet 全量验证）：
1. 每 chunk 文本 == 原文[char_start:char_end]（偏移精确性）
2. 首 chunk 从 0 起、末 chunk 到 len(原文) 止（覆盖完整）
3. 相邻 chunk 重叠恰好 OVERLAP_CHARS（除首尾/极短文档）
4. chunk 长度分布落在目标区间（报告，不硬失败）

用法: python corpus/validate_chunking.py [chunks 路径] [limit]
"""
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.chunk_corpus import CHUNK_MAX, CHUNK_MIN, OVERLAP_CHARS

BAD_LEN = 0.02  # 允许的超长 chunk 比例上限（分句窗口极值）


def validate(chunks_path=None, limit=0):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    chunks_path = Path(chunks_path) if chunks_path else config.DATA_DIR / "chunks.parquet"
    t = pq.read_table(chunks_path)
    docs = t.to_pydict()
    by_doc = defaultdict(list)
    for i in range(t.num_rows):
        by_doc[docs["doc_id"][i]].append(i)

    n_bad_offset = n_bad_cover = n_bad_overlap = 0
    n_chunks = n_docs_checked = 0
    n_over_ok = n_over_bad = 0
    pf = pq.ParquetFile(config.CLEANED_PARQUET)
    for b in pf.iter_batches(batch_size=50_000):
        tb = pa.Table.from_batches([b])
        ids = tb.column("_id").cast("string").to_pylist()
        for j, did in enumerate(ids):
            if limit and n_docs_checked >= limit:
                break
            if did not in by_doc:
                continue
            n_docs_checked += 1
            orig = tb.column("text")[j].as_py()
            idxs = sorted(by_doc[did], key=lambda i: docs["chunk_index"][i])
            n_chunks += len(idxs)
            # 1) 偏移精确
            for i in idxs:
                cs, ce = docs["char_start"][i], docs["char_end"][i]
                if docs["text"][i] != orig[cs:ce]:
                    n_bad_offset += 1
            # 2) 覆盖完整
            if docs["char_start"][idxs[0]] != 0 or docs["char_end"][idxs[-1]] != len(orig):
                n_bad_cover += 1
            # 3) 相邻重叠 = OVERLAP_CHARS（末 chunk 与全文边界除外）
            for a, b in zip(idxs, idxs[1:]):
                gap = docs["char_start"][b] - docs["char_end"][a]
                # 前 chunk 是 carry 尾部重叠，后 chunk start 应比前 end 早 OVERLAP_CHARS；
                # 极短文档（全文 < CHUNK_MAX 无切分）例外
                if gap == -OVERLAP_CHARS:
                    n_over_ok += 1
                elif gap >= 0 and docs["char_end"][a] >= len(orig) - 1:
                    pass  # 尾部 chunk 无后继，正常
                else:
                    n_over_bad += 1
                    if n_over_bad <= 5:
                        print(f"重叠异常 doc={did} chunk={docs['chunk_index'][a]}→{docs['chunk_index'][b]} gap={gap}")
            if limit and n_docs_checked >= limit:
                break
        if limit and n_docs_checked >= limit:
            break

    # 全量 text 列 >2GB 时 combine_chunks 会触发 int32 offset 溢出，
    # pc.utf8_length 对 ChunkedArray 逐 chunk 计算，不 concat
    import pyarrow.compute as pc
    arr = pc.utf8_length(t.column("text")).to_numpy()
    n_overlong = int((arr > CHUNK_MAX + OVERLAP_CHARS + 10).sum())

    print(f"验证 {n_docs_checked:,} 篇 / {n_chunks:,} chunks（{chunks_path.name}）")
    print(f"  偏移精确: 违例 {n_bad_offset}")
    print(f"  覆盖完整: 违例 {n_bad_cover}")
    print(f"  重叠校验: ok={n_over_ok:,} bad={n_over_bad}")
    print(f"  长度: min={arr.min()} 中位={int(__import__('numpy').median(arr))} max={arr.max()}"
          f"（> {CHUNK_MAX + OVERLAP_CHARS + 10} 的 {n_overlong} 个，{n_overlong / max(len(arr), 1):.2%}）")
    passed = (n_bad_offset == 0 and n_bad_cover == 0 and n_over_bad == 0
              and n_overlong / max(len(arr), 1) <= BAD_LEN)
    print("结论:", "通过 ✓" if passed else "不通过 ✗")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    sys.exit(validate(Path(sys.argv[1]) if len(sys.argv) > 1 else None,
                      int(sys.argv[2]) if len(sys.argv) > 2 else 0))
