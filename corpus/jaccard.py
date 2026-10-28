# -*- coding: utf-8 -*-
"""MinHash + 精确 Jaccard 近重复检测（大纲 §7.3）。

为什么不用字符 unigram simhash：金融语料主题高度聚集，同主题不同文章
共享主题词（税/股/审计……），simhash 海明距离 1~3 区间里「真转载」和
「同主题异文」混杂（实测 d=2/3 约半数误合并，d=1 也有风险），阈值分不开。
改用字符 4-gram 的 Jaccard：转载文共享长短语，J ≥ 0.8；同主题异文只有
零散共享 4-gram，J ≤ 0.3。且配对判定用精确 Jaccard（非 minhash 估计）。

minhash 仅用于排序近似（让相似文档排在一起），阈值判定永远用精确 Jaccard。
"""
import re
import zlib

import numpy as np

WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """折叠全部空白（转载文最常见的差异就是空白/换行）。"""
    return WS.sub("", text)


def shingle_set(text: str) -> np.ndarray:
    """字符 4-gram 集合 → 排序去重的 uint32 哈希数组（精确 Jaccard 用）。"""
    grams = {text[i:i + 4] for i in range(len(text) - 3)}
    h = np.array([zlib.crc32(g.encode("utf-8")) for g in grams], dtype=np.uint32)
    return np.unique(h)


def minhash_params(n_perm: int, seed: int = 20261016) -> tuple:
    """minhash 置换参数：h_i = (a_i * h + b_i) mod 2^32，a_i 取奇数保证双射。"""
    rng = np.random.default_rng(seed)
    a = (rng.integers(0, 2 ** 31, size=n_perm, dtype=np.uint64) * 2 + 1)
    b = rng.integers(0, 2 ** 32, size=n_perm, dtype=np.uint64)
    return a, b


def minhash_sketch(hset: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """n 个 minhash 值：对每个置换取集合内最小哈希。"""
    m = hset[None, :].astype(np.uint64)
    hi = (a[:, None] * m + b[:, None]) & np.uint64(0xFFFFFFFF)
    return hi.min(axis=1).astype(np.uint32)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """精确 Jaccard（两数组均需排序去重）。"""
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union else 0.0
