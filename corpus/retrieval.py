# -*- coding: utf-8 -*-
"""P3 混合检索（大纲 §5）：多查询扩展 + 向量(Qdrant+bge-m3) + BM25 → RRF → 元数据过滤 → rerank。

用法:
    ret = HybridRetriever()          # 加载 Qdrant/模型/BM25（~40s）
    hits = ret.retrieve("2021年综合保税区进出口值是多少", rerank=True)
    # -> [SearchHit(label, id, doc_id, score, text, source), ...]

P6 迁移（2026-10-21）：向量路从进程内 hnswlib 迁至独立 Qdrant 向量库——chunk 向量 +
8 列元数据（含 text）全量入 collection，point id = chunk 行号（= BM25 文档号 =
原 HNSW label），融合对齐契约不变；进程内只留 bge-m3 模型（查询向量编码）、BM25、
doc_titles 与查询侧缓存。Qdrant 断连/超时 → vector_fail_streak 降级纯 BM25（§13.3）。

评估消融（eval_retrieval.py 用）通过开关控制：
    retrieve(q, expand=False, use_bm25=False, ...) = 纯向量
    retrieve(q, expand=False, use_vector=False)   = 纯 BM25
    retrieve(q, rerank=True)                       = 混合 + 精排
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))   # CLI 直跑（corpus/retrieval.py）时可见项目根
import config
from corpus.bm25 import BM25Index

RRF_K = 60          # RRF 常数（§5.2）
VEC_TOP = 50        # 向量每路候选数
BM25_TOP = 300      # BM25 每路候选数（检索深度：正例排名中位 ~466，50 够不着）
RERANK_TOP = 100    # 精排输入数（与 BM25_TOP 配套，否则深度白捞）
RRF_W_VEC = 1.0     # RRF 加权（§5.2）：向量路权重
RRF_W_BM25 = 1.0    # RRF 加权：BM25 路权重（试点：max 模式下 1:3 反伤，等权最优）
QUALITY_MIN = config.FILTER_RULES["quality_score_min"]

# 事件簇配额（2026-10-18 变体诊断结论：fail 题正例周围平均 21.4 篇同事实变体文档
# vs 命中题 5.4 篇 → 检索后按余弦相似簇限制名额，强制分散 top-k）
DIV_THR = 0.85      # 余弦 ≥ 阈值视为同事实变体（诊断中 ≥0.85 的样本均为同事件异表述）
DIV_QUOTA = 2       # 每簇最多保留 chunk 数
DIV_CAND = 100      # 配额前候选池大小（配额后取 RERANK_TOP 送精排）


def _with_retry(fn):
    """Qdrant 调用快速重试（§13.2）：吸收瞬时抖动，耗尽后原样上抛走降级链。"""
    for attempt in range(1 + config.QDRANT_RETRIES):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if attempt >= config.QDRANT_RETRIES:
                raise
            time.sleep(0.1)


@dataclass
class SearchHit:
    """检索命中（对应 S5 结构，§9.3）。"""
    label: int           # Qdrant point id / BM25 统一文档号（= chunks 行号）
    id: str              # chunk id
    doc_id: str
    score: float
    text: str = ""
    source: str = ""     # vector / bm25 / rrf / rrf+rerank
    rank: int = 0
    # P6 迁移（2026-10-21）：元数据随 Qdrant payload 携带，不再查进程内数组
    quality: float = 4.0
    chunk_index: int = 0
    chunk_count: int = 1


@dataclass
class S2:
    """意图识别+查询扩展结果（schemas.S2IntentExpand 的轻量映射）。"""
    intent: str = "fact_qa"
    confidence: float = 0.0
    entities: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class HybridRetriever:
    # payload 字段（P6：与灌库脚本 corpus/build_qdrant.py 保持一致）
    PAYLOAD_FIELDS = ["id", "doc_id", "text", "quality_score", "industry_type",
                      "data_version", "chunk_index", "chunk_count"]

    def __init__(self):
        t0 = time.time()
        # P6（2026-10-21）：连通性与灌库完整性检查放最前——探针失败成本只有几秒
        # （模型加载最贵，放最后）；失败抛异常走既有 maintenance + 60s 探针重试
        # 整个 HybridRetriever() 的路径（server/state.py 零改动）。
        print(f"连接 Qdrant（{config.QDRANT_URL}）...", flush=True)
        self.client = QdrantClient(url=config.QDRANT_URL, timeout=config.QDRANT_TIMEOUT)
        last_err: Optional[Exception] = None
        for _ in range(3):   # 2 次 × 3s 快速重试，吸收容器冷启动竞态（§13.3）
            try:
                self.client.get_collection(config.QDRANT_COLLECTION)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(3)
        else:
            raise RuntimeError(f"Qdrant 不可用（{config.QDRANT_URL}）：{last_err}")
        self.n_chunks = self.client.count(config.QDRANT_COLLECTION).count
        # 灌库完整性守卫：防止「灌了一半的 collection」被服务当全量用（毫秒级元数据读）
        n_parquet = pq.ParquetFile(config.DATA_DIR / "chunks.parquet").metadata.num_rows
        if self.n_chunks != n_parquet:
            raise RuntimeError(f"Qdrant 灌库不完整：collection {self.n_chunks:,} 点 vs "
                               f"chunks.parquet {n_parquet:,} 行")
        print("加载 bge-m3（CPU fp32）...", flush=True)
        self.model = SentenceTransformer(str(config.DATA_DIR / "models" / "bge-m3"),
                                         device="cpu", trust_remote_code=True)
        print("加载 BM25...", flush=True)
        self.bm25 = BM25Index()
        # 8 列元数据不再全量进内存（文本等随 Qdrant payload 按需取回）；
        # 以下 ids/texts/... 为懒加载兼容属性，诊断脚本（diag_*）直接读仍可用
        self._cols: dict = {}
        # 正例映射（评估用）：惰性构建（labels_for_docs 首次调用），服务进程不持有
        self._doc_to_labels: Optional[dict] = None
        # 文献名映射（§14.5）：doc_id → 规则提炼标题；缺文件空 dict（S5Hit.doc_title=None）
        self.doc_titles: dict = {}
        if config.TITLE_MAP_PARQUET.is_file():
            _tt = pq.read_table(config.TITLE_MAP_PARQUET, columns=["doc_id", "title"])
            self.doc_titles = dict(zip(_tt.column("doc_id").to_pylist(),
                                       _tt.column("title").to_pylist()))
        self._cache = {}          # 热点缓存（§5.5，V1 进程内 LRU，P5 换 Redis）
        self._cache_order = []
        self._vec_cache = {}      # 查询文本 → 向量（评估多配置复用，省重复编码）
        # P5 服务化（§13.3）：并发安全 + 向量路健康观测
        self._cache_lock = threading.Lock()       # 4 并发下 LRU 三步序列（remove/append/pop）竞态保护
        self.last_vector_error: Optional[str] = None   # 最近一次向量检索异常（探针/状态机观测）
        self.vector_fail_streak = 0                    # 连续失败次数（≥config.VECTOR_FAIL_STREAK 转维护态）
        print(f"✓ 就绪 {self.n_chunks:,} chunks（向量路=Qdrant）{time.time()-t0:.0f}s", flush=True)

    def _load_col(self, name: str) -> list:
        """按需读 chunks.parquet 单列并缓存（懒加载兼容，服务运行期不触碰）。"""
        if name not in self._cols:
            t = pq.read_table(config.DATA_DIR / "chunks.parquet", columns=[name])
            self._cols[name] = t.column(name).to_pylist()
        return self._cols[name]

    @property
    def ids(self): return self._load_col("id")

    @property
    def doc_ids(self): return self._load_col("doc_id")

    @property
    def texts(self): return self._load_col("text")

    @property
    def quality(self): return self._load_col("quality_score")

    @property
    def industry(self): return self._load_col("industry_type")

    @property
    def data_version(self): return self._load_col("data_version")

    @property
    def chunk_index(self): return self._load_col("chunk_index")

    @property
    def chunk_counts(self): return self._load_col("chunk_count")

    @property
    def doc_to_labels(self) -> dict:
        """doc_id → 该文档全部 chunk label（评估正例集）；惰性流式构建（诊断脚本兼容）。"""
        if self._doc_to_labels is None:
            d: dict = {}
            pf = pq.ParquetFile(config.DATA_DIR / "chunks.parquet")
            offset = 0
            for batch in pf.iter_batches(batch_size=100_000, columns=["doc_id"]):
                for did in batch.column("doc_id").to_pylist():
                    d.setdefault(did, []).append(offset)
                    offset += 1
            self._doc_to_labels = d
        return self._doc_to_labels

    # ---------- 查询扩展 ----------

    def expand(self, question: str) -> S2:
        """LLM 多查询扩展（§5.1，prompts/intent_expand）。失败降级为原问题（§13.2）。"""
        try:
            from llm_client import client
            from prompts import remote
            from prompts.intent_expand import build_intent_messages
            r = client.chat(build_intent_messages(question), route="intent",
                            json_mode=True, max_tokens=1000,
                            prompt_client=remote.get_prompt("intent_expand"))
            d = client.parse_json_content(r.content)
            return S2(intent=d.get("intent", "fact_qa"), confidence=d.get("confidence", 0),
                      entities=d.get("entities", []), queries=d.get("queries", []),
                      keywords=d.get("keywords", []))
        except Exception:
            return S2()

    # ---------- 单路检索 ----------

    def vector_search(self, queries: List[str], k: int = VEC_TOP) -> List[tuple]:
        """批量编码多查询 → 每个查询 Qdrant top-k，返回 [(label, score, qidx)]。
        失败抛异常（P5 §13.3：由 retrieve 捕获降级纯 BM25）。"""
        with self._cache_lock:
            missing = [q for q in dict.fromkeys(queries) if q not in self._vec_cache]
        if missing:
            embs = self.model.encode(missing, truncate_dim=512, normalize_embeddings=True,
                                     batch_size=len(missing), convert_to_numpy=True)
            with self._cache_lock:
                for q, v in zip(missing, embs):
                    self._vec_cache[q] = v.astype(np.float32)
        with self._cache_lock:
            vecs = {q: self._vec_cache[q] for q in dict.fromkeys(queries)}
        return _with_retry(lambda: self._query_vectors(
            [vecs[q] for q in queries], k))

    def _query_vectors(self, vecs: List[np.ndarray], k: int) -> List[tuple]:
        """多查询向量检索（一批一 RTT）；返回 [(label, score, qidx)] 平铺列表。

        分数契约：bge-m3 向量已归一化，Qdrant cosine 分数 == hnswlib 的 1-dist
        （verify 脚本有 top-1 分数差断言）；qidx = 输入序下标（批量返回按输入序分组）。

        P6 坑位（2026-10-21）：query_points 的 query 传「向量列表」会被服务端当作
        单查询的多向量输入（multivector），普通 collection 直接 400 Conversion
        between multi and regular vectors failed；批量必须走 query_batch_points
        （requests=[QueryRequest...]，返回序 == 输入序，verify ④ 有对照断言锁定）。
        """
        if not vecs:
            return []
        qfilter = None
        if config.QDRANT_SERVER_FILTER:   # 质量过滤下推（服务端 filter，可环境开关）
            qfilter = models.Filter(must=[models.FieldCondition(
                key="quality_score", range=models.Range(gte=QUALITY_MIN))])
        resps = self.client.query_batch_points(
            collection_name=config.QDRANT_COLLECTION,
            requests=[models.QueryRequest(
                query=v.tolist(), filter=qfilter, with_payload=False,
                params=models.SearchParams(hnsw_ef=config.QDRANT_EF, exact=False),
                limit=k) for v in vecs],
        )
        out = []
        for qi, resp in enumerate(resps):
            out.extend((int(p.id), float(p.score), qi) for p in resp.points)
        return out

    def bm25_search(self, queries: List[str], k: int = BM25_TOP) -> List[tuple]:
        """每个查询 BM25 top-k，返回 [(label, score, qidx)]。"""
        out = []
        for i, q in enumerate(queries):
            for lab, s in self.bm25.search(q, k):
                out.append((lab, s, i))
        return out

    # ---------- 融合与过滤 ----------

    @staticmethod
    def fuse_rrf(lists: List[List[tuple]], k: int = RRF_K,
                 weights: Optional[List[float]] = None, mode: str = "sum") -> List[tuple]:
        """RRF 合并（§5.2）：按 label 去重，返回降序 [(label, score)]。

        mode="sum"：score = Σ w_i/(k+rank)（多查询共识累加）。
        mode="max"：score = max w_i/(k+rank)（保留任一查询的最强命中；
        诊断：sum 下单个强命中（向量#7）被 7 路查询共识噪声淹没）。
        weights 每路一个（None = 等权）。
        """
        if weights is None:
            weights = [1.0] * len(lists)
        acc: dict = {}
        for lst, w in zip(lists, weights):
            for rank, (lab, _, _) in enumerate(lst, 1):
                s = w / (k + rank)
                if mode == "max":
                    if s > acc.get(lab, 0.0):
                        acc[lab] = s
                else:
                    acc[lab] = acc.get(lab, 0.0) + s
        return sorted(acc.items(), key=lambda x: -x[1])

    def filter_meta(self, hits: List[tuple], industry: Optional[str] = None,
                    meta: Optional[dict] = None) -> List[tuple]:
        """元数据过滤（§5.4）：quality 下限 + 可选行业。

        meta = {label: payload}（P6：融合后一次取回的 payload 快照，filter_meta 与
        _make_hits 共用，免二次取数）；None 时回退懒加载数组（诊断脚本路径）。
        """
        out = []
        for lab, s in hits:
            if meta is not None:
                m = meta.get(lab)
                if not m:                      # payload 缺失视为不可用（正常不会发生）
                    continue
                if float(m.get("quality_score", 0)) < QUALITY_MIN:
                    continue
                if industry and m.get("industry_type") != industry:
                    continue
            else:
                if self.quality[lab] < QUALITY_MIN:
                    continue
                if industry and self.industry[lab] != industry:
                    continue
            out.append((lab, s))
        return out

    def _fetch_payloads(self, labels: List[int]) -> dict:
        """Qdrant 按 point id 取 payload（≤1 个本地 RTT）。返回 {label: payload}。"""
        if not labels:
            return {}
        pts = self.client.retrieve(
            collection_name=config.QDRANT_COLLECTION,
            ids=[int(l) for l in labels],
            with_payload=self.PAYLOAD_FIELDS,
            with_vectors=False,
        )
        return {int(p.id): (p.payload or {}) for p in pts}

    def _make_hits(self, ranked: List[tuple], n: int, source: str,
                   payloads: Optional[dict] = None) -> List[SearchHit]:
        """label 列表 → SearchHit（P6：元数据/文本来自 Qdrant payload；payloads
        缺省时自取；Qdrant 不可用回退懒加载列——degraded 纯 BM25 仍能出答案。

        故障演练实证（2026-10-21）：宕机后 payload 取回硬抛会直接 500，降级链断裂；
        懒加载列兜底（首请求触发全列读取，时间+内存换可用性，仅降级期触发）。"""
        if payloads is None:
            try:
                payloads = self._fetch_payloads([lab for lab, _ in ranked])
            except Exception:  # noqa: BLE001 — 宕机兜底
                payloads = None
        hits = []
        for rank, (lab, s) in enumerate(ranked[:n], 1):
            pl = payloads.get(lab) if payloads is not None else None
            if pl:
                hits.append(SearchHit(label=lab, id=pl.get("id", ""),
                                      doc_id=pl.get("doc_id", ""), score=s,
                                      text=pl.get("text", ""), source=source, rank=rank,
                                      quality=float(pl.get("quality_score", 4.0)),
                                      chunk_index=int(pl.get("chunk_index", 0)),
                                      chunk_count=int(pl.get("chunk_count", 1))))
            else:   # 懒加载列兜底（Qdrant 宕机）
                hits.append(SearchHit(label=lab, id=self.ids[lab], doc_id=self.doc_ids[lab],
                                      text=self.texts[lab], score=s, source=source,
                                      rank=rank, quality=self.quality[lab],
                                      chunk_index=self.chunk_index[lab],
                                      chunk_count=self.chunk_counts[lab]))
        return hits

    # ---------- 进程内缓存（§5.5，P5 加锁） ----------

    def _cache_get(self, key: str) -> Optional[List[SearchHit]]:
        with self._cache_lock:
            if key in self._cache:
                self._cache_order.remove(key)
                self._cache_order.append(key)
                return self._cache[key]
        return None

    def _cache_set(self, key: str, hits: List[SearchHit]) -> None:
        with self._cache_lock:
            self._cache[key] = hits
            self._cache_order.append(key)
            if len(self._cache) > 1024:   # LRU 上限（§5.5）
                self._cache.pop(self._cache_order.pop(0), None)

    # ---------- 入口 ----------

    def retrieve(self, question: str, *, expand: bool = True, s2: Optional[S2] = None,
                 use_vector: bool = True, use_bm25: bool = True, rerank: bool = False,
                 n: int = 10, industry: Optional[str] = None,
                 diversify: bool = False, bm25_top: int = BM25_TOP,
                 rerank_top: int = RERANK_TOP,
                 rrf_w: tuple = (RRF_W_VEC, RRF_W_BM25), rrf_mode: str = "max",
                 vec_raw: bool = True, decomp: bool = False) -> List[SearchHit]:
        """混合检索主入口（§5 流程）。开关组合供评估消融。

        s2 由调用方（Agent 循环）传入时跳过内部扩展调用，避免重复 LLM 开销。
        bm25_top/rerank_top/rrf_w/rrf_mode 为检索深度与融合方式（消融 A/B 用，默认=生产配置）。
        vec_raw=True 时原问句也参与向量检索（诊断：原问句向量 #7 命中曾被变体列表丢在门外）。
        diversify=True 时 rerank 候选先过事件簇配额（§5.7 诊断结论）。
        decomp：comparison 意图分解检索——**已证伪，勿启用**（2026-10-19 全量 64 题：
        comparison MRR 0.325→0.147、q063 正例从第 1 被挤出前 10；意图误分类泄漏到
        multi_doc 伤 q051；试点假阳性源于共用 s2 掩盖了变体/分类依赖。保留仅作记录）。
        """
        cache_key = json.dumps([question, expand, use_vector, use_bm25, rerank, n, industry,
                                diversify, bm25_top, rerank_top, rrf_w, rrf_mode, vec_raw,
                                decomp],
                               ensure_ascii=False)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        s2 = s2 if s2 is not None else (self.expand(question) if expand else S2())
        if decomp and s2.intent == "comparison":
            sides = [q for q in (s2.queries or []) if q.strip()]
            if sides and rerank:
                hits = self._decomp_retrieve(question, sides, n)
                self._cache_set(cache_key, hits)
                return hits
        variants = [q for q in (s2.queries or []) if q.strip()] or [question]
        lists = []
        weights = []
        if use_vector:
            vec_queries = ([question] if vec_raw else []) + variants
            try:
                lists.append(self.vector_search(vec_queries))
                weights.append(rrf_w[0])
                with self._cache_lock:
                    self.vector_fail_streak = 0
            except Exception as e:  # noqa: BLE001 — §13.3 向量路失败 → 纯 BM25 兜底（P5 收口）
                with self._cache_lock:
                    self.vector_fail_streak += 1
                    self.last_vector_error = f"{type(e).__name__}: {e}"
                print(f"[retrieval] 向量检索失败，降级纯 BM25（连续 {self.vector_fail_streak} 次）：{e}",
                      flush=True)
        if use_bm25:
            bm_queries = list(dict.fromkeys([question] + variants + [" ".join(s2.keywords)]))
            lists.append(self.bm25_search(bm_queries, k=bm25_top))
            weights.append(rrf_w[1])
        if not lists:
            return []
        fused_raw = self.fuse_rrf(lists, weights=weights, mode=rrf_mode)
        # P6：融合后一次取 payload（≤1 RTT），filter_meta 校验与 SearchHit 组装共用。
        # Qdrant 不可用时回退懒加载列（纯 BM25 降级路径仍能出答案——故障演练实证，
        # 若此处硬抛，/sources 直接 500，降级链断裂）
        try:
            payloads = self._fetch_payloads([lab for lab, _ in fused_raw])
            fused = self.filter_meta(fused_raw, industry, payloads)
        except Exception as e:  # noqa: BLE001
            print(f"[retrieval] payload 取回失败，回退进程内懒加载列：{e}", flush=True)
            payloads = None
            fused = self.filter_meta(fused_raw, industry)
        source = "rrf"
        if rerank and fused:
            cand_n = DIV_CAND if diversify else rerank_top
            cand = self._make_hits(fused, cand_n, source, payloads)
            if diversify:
                cand = self._diversify(cand)[:rerank_top]
            from llm_client import client
            try:
                scored = client.rerank(question, [h.text for h in cand])
                top = [cand[r["index"]] for r in scored[:n]]
                for r, h in zip(scored[:n], top):
                    h.score, h.rank, h.source = r["score"], 1 + scored.index(r), "rrf+rerank"
                hits = top
            except Exception:
                hits = self._make_hits(fused, n, source, payloads)   # rerank 失败降级 RRF 结果
        else:
            hits = self._make_hits(fused, n, source, payloads)

        self._cache_set(cache_key, hits)
        return hits

    def _decomp_retrieve(self, question: str, sides: List[str], n: int) -> List[SearchHit]:
        """comparison 意图分解检索（§5.8，diag_comp 三轮诊断结论）。
        两路并联：每侧独立检索+分侧 rerank 各取 top-10，全池原问句 rerank 取 top-10，
        跨路去重按 rerank 分合并成 n。
        **已证伪，勿启用**（2026-10-19 全量 64 题：comparison MRR 0.325→0.147，
        q063 正例从第 1 被挤出前 10；试点假阳性源于共用 s2 掩盖了变体/分类依赖）。"""
        from llm_client import client
        out = []
        for s in sides[:4]:
            fused = self.filter_meta(self.fuse_rrf(
                [self.vector_search([s]), self.bm25_search([s], k=BM25_TOP)],
                mode="max"))[:RERANK_TOP]
            cand = self._make_hits(fused, RERANK_TOP, "rrf")
            try:
                scored = client.rerank(s, [h.text for h in cand])
                for r in scored[:10]:
                    h = cand[r["index"]]
                    h.score, h.rank = r["score"], 1 + scored.index(r)
                    out.append(h)
            except Exception:
                out.extend(cand[:10])
        lists = [self.vector_search([question]), self.bm25_search([question], k=BM25_TOP)]
        for s in sides[:4]:
            lists.append(self.vector_search([s]))
            lists.append(self.bm25_search([s], k=BM25_TOP))
        gfused = self.filter_meta(self.fuse_rrf(lists, mode="max"))[:RERANK_TOP]
        gcand = self._make_hits(gfused, RERANK_TOP, "rrf")
        try:
            scored = client.rerank(question, [h.text for h in gcand])
            for r in scored[:10]:
                h = gcand[r["index"]]
                h.score, h.rank = r["score"], 1 + scored.index(r)
                out.append(h)
        except Exception:
            out.extend(gcand[:10])
        best = {}                      # label → SearchHit（跨路去重保留最高 rerank 分）
        for h in out:
            if h.label not in best or h.score > best[h.label].score:
                best[h.label] = h
        hits = sorted(best.values(), key=lambda x: -x.score)[:n]
        for i, h in enumerate(hits, 1):
            h.rank = i
        return hits

    def _diversify(self, hits: List[SearchHit]) -> List[SearchHit]:
        """事件簇配额（§5.7 诊断）：余弦 ≥DIV_THR 视为同一事实变体，每簇保留分数最高的
        DIV_QUOTA 个 chunk，其余跳过——强制 top-k 分散到不同事实，防止同事实异表述
        的多篇报道霸占候选（诊断：fail 题正例周围平均 21.4 篇变体 vs 命中题 5.4 篇）。"""
        if len(hits) <= DIV_QUOTA:
            return hits
        vecs = self.model.encode([h.text for h in hits], normalize_embeddings=True)
        kept, kept_vecs = [], []
        for h, v in zip(hits, vecs):
            same = sum(1 for kv in kept_vecs if float(np.dot(v, kv)) >= DIV_THR)
            if same >= DIV_QUOTA:
                continue
            kept.append(h)
            kept_vecs.append(v)
        return kept

    def labels_for_docs(self, doc_ids: List[str]) -> set:
        """评估用：源文档 → 全部 chunk label（正例集）。首次调用惰性构建映射（P6）。"""
        out = set()
        for d in doc_ids:
            out.update(self.doc_to_labels.get(d, []))
        return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ret = HybridRetriever()
    for q in sys.argv[1:] or ["2021年全国综合保税区进出口值是多少"]:
        t0 = time.time()
        hits = ret.retrieve(q, rerank=True, n=5)
        print(f"\nQ: {q}  {time.time()-t0:.1f}s")
        for h in hits:
            print(f"  [{h.source}] {h.score:.3f} {h.id} {h.text[:50].replace(chr(10),' ')}")
