# -*- coding: utf-8 -*-
"""Jaccard 近重复方案验证脚本（对 filtered.parquet 抽 3000 篇）。

验证三件事：
1. 随机文档对 Jaccard 分布——期望最大值远低于 0.8 阈值
2. 暴力全量配对 J ≥ 0.8 的配对是否全部为真转载（抽查最低 J 的边界对）
3. 排序窗口 ±100 扫描对暴力结果的召回率（决定全量跑的窗口参数）
全部通过后，dedup_corpus.py 才允许用本方案重跑。
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from jaccard import normalize, shingle_set, minhash_params, minhash_sketch, jaccard

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

N_SAMPLE = 3000
N_PERM = 64
SORT_KEYS = 12     # 排序用前 12 个 minhash 值
WINDOW = 100
J_MIN = 0.8


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t = pq.read_table(config.FILTERED_PARQUET, columns=["text"])
    texts = t.column("text").to_pylist()
    n = len(texts)

    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=N_SAMPLE, replace=False)
    subs = [texts[i] for i in idx]
    sets = [shingle_set(normalize(s)) for s in subs]
    a, b = minhash_params(N_PERM)
    sk = np.stack([minhash_sketch(s, a, b) for s in sets])  # (3000, 64)

    # 1) 随机对 Jaccard
    rand = rng.integers(0, N_SAMPLE, size=(10_000, 2))
    js = [jaccard(sets[i], sets[j]) for i, j in rand]
    arr = np.array(js)
    print(f"随机对 Jaccard: max={arr.max():.3f} mean={arr.mean():.4f} "
          f"p99={np.quantile(arr, 0.99):.3f}")

    # 2) 暴力全量配对 J ≥ J_MIN
    brute = []
    for i in range(N_SAMPLE):
        for j in range(i + 1, N_SAMPLE):
            jj = jaccard(sets[i], sets[j])
            if jj >= J_MIN:
                brute.append((jj, i, j))
    print(f"\n暴力 J ≥ {J_MIN} 的配对: {len(brute)}")
    brute.sort()
    for jj, i, j in brute[:6]:  # 最低 J 的边界对最危险
        print(f"\n--- J={jj:.3f} ---")
        print(" A:", subs[i][:150].replace("\n", " "))
        print(" B:", subs[j][:150].replace("\n", " "))

    # 3) 排序窗口扫描召回率
    order = np.lexsort([sk[:, k] for k in range(SORT_KEYS - 1, -1, -1)])
    pos = np.empty(N_SAMPLE, dtype=np.int64)
    pos[order] = np.arange(N_SAMPLE)
    found = set()
    for i in range(N_SAMPLE):
        for j in range(max(0, i - WINDOW), min(N_SAMPLE, i + WINDOW + 1)):
            if i == j:
                continue
            oi, oj = order[i], order[j]
            if oi > oj:
                oi, oj = oj, oi
            jj = jaccard(sets[oi], sets[oj])
            if jj >= J_MIN:
                found.add((oi, oj))
    missed = [(jj, i, j) for jj, i, j in brute if (i, j) not in found]
    print(f"\n窗口 ±{WINDOW} 扫描: 找到 {len(found)} 对，暴力结果漏掉 {len(missed)} 对")
    for jj, i, j in missed:
        print(f"  漏: J={jj:.3f} 排序距离={abs(int(pos[i]) - int(pos[j]))}")


if __name__ == "__main__":
    main()
