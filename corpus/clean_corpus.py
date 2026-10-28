# -*- coding: utf-8 -*-
"""P0-4 清洗（大纲 §7.3，保守策略：只做安全项，不动正文内容）。

流式分批处理（5 万行/批），不用 pandas（全表往返实测 OOM）。
- 换行规范化（\\r\\n → \\n），连续 3+ 换行折叠为 2
- 去除文首「来源：xxx」头（仅文首一次）
- 去首尾空白；清洗后为空的剔除（null 文本一并剔除）
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

BATCH = 50_000


def clean_batch(t: pa.Table) -> pa.Table:
    s = t.column("text")
    s = pc.replace_substring(s, "\r\n", "\n")
    s = pc.replace_substring_regex(s, r"\n{3,}", "\n\n")
    s = pc.replace_substring_regex(s, r"^来源[:：][^\n]{0,50}\n", "")
    s = pc.utf8_trim_whitespace(s)
    t = t.set_column(t.column_names.index("text"), "text", s)
    t = t.filter(pc.greater(pc.utf8_length(s), 0))  # 空/None 剔除
    w = pc.count_substring_regex(t.column("text"), r"\S+")  # 非空白 token 数
    return t.set_column(t.column_names.index("num_words"), "num_words", w.cast(pa.int64()))


def clean():
    pf = pq.ParquetFile(config.DEDUPED_PARQUET)
    n0 = pf.metadata.num_rows
    writer = None
    n1 = 0
    len_before = len_after = 0
    for batch in tqdm(pf.iter_batches(batch_size=BATCH), desc="清洗"):
        t = pa.Table.from_batches([batch])
        len_before += pc.sum(pc.utf8_length(t.column("text"))).as_py()
        out = clean_batch(t)
        len_after += pc.sum(pc.utf8_length(out.column("text"))).as_py()
        if out.num_rows == 0:
            continue
        if writer is None:
            writer = pq.ParquetWriter(config.CLEANED_PARQUET, out.schema)
        writer.write_table(out)
        n1 += out.num_rows
    if writer:
        writer.close()
    print(f"清洗: {n0:,} → {n1:,}（剔除空文本 {n0 - n1:,}）")
    print(f"平均文本长度: {len_before/max(n0,1):.0f} → {len_after/max(n1,1):.0f} 字符")
    print(f"已写入 {config.CLEANED_PARQUET}")


if __name__ == "__main__":
    clean()
