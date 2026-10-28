# -*- coding: utf-8 -*-
"""测试读取三个 rank parquet 文件，各取前几行数据。"""
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

# Windows 控制台默认 GBK，强制 UTF-8 输出避免中文乱码
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FILES = ["rank_00045.parquet", "rank_00046.parquet", "rank_00047.parquet"]
N_ROWS = 5  # 每个文件取的行数
TEXT_PREVIEW = 80  # text 列只显示前 80 个字符


def read_sample(path: Path, n: int) -> None:
    t0 = time.time()
    pf = pq.ParquetFile(path)

    # 只读第一个 row group，避免加载整个文件
    table = pf.read_row_group(0)

    print("=" * 70)
    print(f"文件: {path.name}")
    print(f"  总行数     : {pf.metadata.num_rows:,}")
    print(f"  row groups : {pf.metadata.num_row_groups}")
    print(f"  列名       : {table.column_names}")
    print(f"  耗时(读1个row group): {time.time() - t0:.2f}s")

    # 转 pandas 取前 n 行
    df = table.to_pandas().head(n)
    df["text"] = df["text"].astype(str).str[:TEXT_PREVIEW]
    print(f"  前 {n} 行 (text 列截断为前 {TEXT_PREVIEW} 字符):")
    print(df.to_string())


if __name__ == "__main__":
    for f in FILES:
        read_sample(Path(__file__).parent / f, N_ROWS)
