# -*- coding: utf-8 -*-
"""P0-2 质量过滤（大纲 §7.2）。

按 config.FILTER_RULES 过滤三个文件，输出 data/filtered.parquet（含 source_file 列）。
逐文件读取，逐条规则统计保留数。
"""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def build_mask(table: pa.Table) -> pc.Expression:
    """组装过滤 mask（null 一律剔除）。"""
    mask = None
    col = lambda n: table.column(n)
    checks = [
        ("quality_score", pc.greater_equal(col("quality_score"), config.FILTER_RULES["quality_score_min"])),
        ("perplexity", pc.less_equal(col("perplexity"), config.FILTER_RULES["perplexity_max"])),
        ("char_rep_ratio", pc.less_equal(col("char_rep_ratio"), config.FILTER_RULES["char_rep_ratio_max"])),
        ("flagged_words_ratio", pc.less_equal(col("flagged_words_ratio"), config.FILTER_RULES["flagged_words_ratio_max"])),
    ]
    for _, check in checks:
        mask = check if mask is None else pc.and_(mask, check)
    return pc.and_(mask, pc.is_valid(col("quality_score")))


def filter_files():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    total_in = total_out = 0
    parts = []

    for f in config.SOURCE_FILES:
        table = pq.read_table(config.ROOT / f)
        table = table.append_column("source_file", pa.array([f] * table.num_rows))
        mask = build_mask(table)
        passed = table.filter(mask)
        n_in, n_out = table.num_rows, passed.num_rows
        total_in += n_in
        total_out += n_out
        # 各规则单独统计
        detail = {
            "quality_score": int(pc.sum(pc.cast(pc.greater_equal(table.column("quality_score"),
                          config.FILTER_RULES["quality_score_min"]), pa.int64())).as_py()),
            "perplexity": int(pc.sum(pc.cast(pc.less_equal(table.column("perplexity"),
                          config.FILTER_RULES["perplexity_max"]), pa.int64())).as_py()),
            "char_rep_ratio": int(pc.sum(pc.cast(pc.less_equal(table.column("char_rep_ratio"),
                          config.FILTER_RULES["char_rep_ratio_max"]), pa.int64())).as_py()),
            "flagged_words_ratio": int(pc.sum(pc.cast(pc.less_equal(table.column("flagged_words_ratio"),
                          config.FILTER_RULES["flagged_words_ratio_max"]), pa.int64())).as_py()),
        }
        print(f"{f}: 输入 {n_in:,} → 保留 {n_out:,} ({n_out/n_in*100:.1f}%) | 各规则: {detail}")
        parts.append(passed)

    combined = pa.concat_tables(parts)
    pq.write_table(combined, config.FILTERED_PARQUET)
    print(f"\n合计: {total_in:,} → {total_out:,} ({total_out/total_in*100:.1f}%)")
    print(f"已写入 {config.FILTERED_PARQUET}")


if __name__ == "__main__":
    filter_files()
