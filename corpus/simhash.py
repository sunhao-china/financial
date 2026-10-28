# -*- coding: utf-8 -*-
"""simhash 近重复检测（大纲 §7.3）。

字符 unigram × log-idf 加权。不用裸词频的原因：中文高频功能字
（的/了/是/一/不……）在每个文档都出现几十次，其 crc32 符号对每个
bit 是固定贡献，会把这些 bit 锁死成全体文档共享的偏置（实测随机
文档对海明距离均值只有 17.9，理论应为 32），导致 hamming ≤ 3
误合并无关文档。idf 加权把功能字权重压到 ~0，bit 由内容字符驱动。

64 位哈希 = crc32(正序) | crc32(倒序) << 32，跨进程稳定。
"""
import zlib
from collections import Counter

import numpy as np

N_BITS = 64


def char_df(texts) -> Counter:
    """字符文档频率：df[c] = 出现过 c 的文档数（每文档去重后计数）。"""
    df = Counter()
    for t in texts:
        df.update(set(t))
    return df


def idf_table(df: Counter, n_docs: int) -> dict:
    """纯 log-idf（无 +1 平滑）：log((N+1)/(df+1))，df=N 的功能字权重精确为 0。

    +1 平滑会让「的」等全语料功能字仍带 1.0×词频的权重，继续锁死 bit，
    实测随机对海明均值只有 22.7（理论 32）。
    """
    return {c: np.log((n_docs + 1.0) / (d + 1.0)) for c, d in df.items()}


def _weighted_bits(text: str, idf: dict) -> np.ndarray:
    """字符 unigram 词频 × log-idf 加权后逐位累加，返回 (64,) 符号和向量。"""
    counts = Counter(text)
    toks = list(counts.keys())
    weights = np.array([counts[t] * idf.get(t, 0.0) for t in toks], dtype=np.float32)
    if not toks:
        return np.zeros(N_BITS, dtype=np.float32)
    h64 = np.empty(len(toks), dtype=np.uint64)
    for i, tok in enumerate(toks):
        b = tok.encode("utf-8")
        h64[i] = np.uint64(zlib.crc32(b)) | (np.uint64(zlib.crc32(b[::-1])) << np.uint64(32))
    # 向量化：bits[m,64] ∈ {-1,+1}，v = bits^T @ weights
    bits = (((h64[:, None] >> np.arange(N_BITS, dtype=np.uint64)) & np.uint64(1))
            .astype(np.int8) * 2 - 1)
    return bits.astype(np.float32).T @ weights


def compute_center(texts, idf: dict) -> np.ndarray:
    """比特中心向量：抽样文档的逐位符号和均值。

    减去后消除「全体文档共享的 bit 偏置」（同字同符号贡献在每篇都相同），
    让无关文档对的海明距离回到 32 附近。
    """
    acc = np.zeros(N_BITS, dtype=np.float64)
    for t in texts:
        acc += _weighted_bits(t, idf)
    return acc / len(texts)


def simhash(text: str, idf: dict, center: np.ndarray = None) -> int:
    """64 位 simhash：加权符号和 − 语料中心，逐位取符号。"""
    v = _weighted_bits(text, idf)
    if center is not None:
        v = v - center
    h = 0
    for i in range(N_BITS):
        if v[i] > 0:
            h |= 1 << i
    return h


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
