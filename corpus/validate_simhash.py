# -*- coding: utf-8 -*-
"""simhash（idf 加权版）验证脚本。

在 filtered.parquet 上随机抽 3000 篇，暴力比对全部 4.5M 对：
- 随机 5 万对的海明距离分布（期望：均值 ≈ 32，与内容无关）
- hamming ≤ 3 的配对抽查文本，确认都是真近重复
全部通过后，dedup_corpus.py 才允许重跑。
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from simhash import char_df, idf_table, compute_center, simhash, hamming

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t = pq.read_table(config.FILTERED_PARQUET, columns=["text"])
    texts = t.column("text").to_pylist()
    n = len(texts)

    # df 与 bit 中心用 ~10 万篇均匀抽样估计
    sample = texts[:: max(1, n // 100_000)]
    df = char_df(sample)
    idf = idf_table(df, len(sample))
    center = compute_center(sample[::4], idf)  # 2.5 万篇足够估计中心
    print(f"df 抽样 {len(sample):,} 篇，字符表 {len(df):,}，示例 idf：的={idf.get('的'):.2f} "
          f"股={idf.get('股'):.2f} 鑫={idf.get('鑫'):.2f}")

    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=3000, replace=False)
    hs = [simhash(texts[i], idf, center) for i in idx]

    # 1) 随机 5 万对基线
    d = [hamming(hs[a], hs[b]) for a, b in rng.integers(0, 3000, size=(50_000, 2))]
    arr = np.array(d)
    print(f"\n随机对 hamming: min={arr.min()} mean={arr.mean():.2f} "
          f"median={int(np.median(arr))} max={arr.max()}")

    # 2) 全样本暴力 hamming ≤ 3
    close = []
    for i in range(3000):
        for j in range(i + 1, 3000):
            d = hamming(hs[i], hs[j])
            if d <= 3:
                close.append((d, i, j))
    print(f"样本内 hamming ≤ 3 的对数: {len(close)}")
    dist = {}
    for d, i, j in close:
        dist[d] = dist.get(d, 0) + 1
    print("距离分布:", dict(sorted(dist.items())))
    d2 = [x for x in close if x[0] == 2]  # 边界抽查：只看 d=2（d=3 已知混入同主题异文）
    for d, i, j in d2[:8]:
        a = texts[idx[i]][:150].replace("\n", " ")
        b = texts[idx[j]][:150].replace("\n", " ")
        print(f"\n--- hamming={d} ---")
        print(" A:", a)
        print(" B:", b)


if __name__ == "__main__":
    main()
