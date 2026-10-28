# -*- coding: utf-8 -*-
"""P6 向量库灌库（大纲 §8）：emb.parquet + chunks.parquet → Qdrant collection。

- point id = chunk 行号（= BM25 文档号 = 原 HNSW label），融合对齐契约不变
- 向量 fp16→fp32（与 build_index.py 同款处理）；payload 8 列（含 text）on_disk
- HNSW 参数与原 hnswlib 完全一致：m=16 / ef_construct=200，查询 ef=100（不量化——
  迁移验收标准是与 hnswlib 基线召回对齐，量化损耗留作后续单独 A/B）
- 锁步读取双 parquet 并逐批断言 id 一致：行序错位 = 检索语义全毁，必须运行时校验
- 断点续传（checkpoint json 原子写，upsert 幂等，重复批次无害）
- 完成判定 = 服务端 count == parquet 行数 且 status green（不信任「已发送」）
- 用法:
    python corpus/build_qdrant.py                 # 灌库（collection 已存在则报错提示）
    python corpus/build_qdrant.py --resume        # 从 checkpoint 续传
    python corpus/build_qdrant.py --drop          # 删旧 collection 重建
    python corpus/build_qdrant.py --verify        # 只跑校验（灌库后）
    python corpus/build_qdrant.py --limit 10000   # 小批试验
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from qdrant_client import QdrantClient, models

M = 16                  # HNSW 连接数（与 build_index.py 对齐）
EF_CONSTRUCTION = 200   # 建索引搜索宽度
EF_SEARCH = 100         # 查询 ef（与 retrieval 原 set_ef(100) 对齐）

PAYLOAD_FIELDS = ["id", "doc_id", "text", "quality_score", "industry_type",
                  "data_version", "chunk_index", "chunk_count"]
CHECKPOINT = config.DATA_DIR / "index" / "qdrant_upload_checkpoint.json"


# ---------- 基础 ----------

def get_client(timeout: int = 300) -> QdrantClient:
    """灌库/校验客户端（大超时：批量写入与等待 green 都慢）。"""
    return QdrantClient(url=config.QDRANT_URL, timeout=timeout)


def load_checkpoint() -> int:
    if CHECKPOINT.is_file():
        return int(json.loads(CHECKPOINT.read_text(encoding="utf-8")).get("done", 0))
    return 0


def save_checkpoint(done: int, total: int) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    tmp = CHECKPOINT.with_suffix(".tmp")
    tmp.write_text(json.dumps({"done": done, "total": total,
                               "ts": time.time()}, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(CHECKPOINT)   # 原子写：中断时 checkpoint 不残缺


# ---------- collection ----------

def create_collection(client: QdrantClient) -> None:
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20_000),
        on_disk_payload=True,      # text 等 payload 落盘（RocksDB），不进容器堆内存
        shard_number=1,
    )


def build_payload_indexes(client: QdrantClient) -> None:
    """预建 payload 索引（wait=True），避免首个带 filter 查询触发全量构建尖峰。"""
    client.create_payload_index(collection_name=config.QDRANT_COLLECTION,
                                field_name="quality_score",
                                field_schema=models.PayloadSchemaType.FLOAT, wait=True)
    client.create_payload_index(collection_name=config.QDRANT_COLLECTION,
                                field_name="industry_type",
                                field_schema=models.PayloadSchemaType.KEYWORD, wait=True)


# ---------- 灌库 ----------

def upload(client: QdrantClient, emb_path: Path, chunks_path: Path,
           batch_size: int, limit: int, resume: bool) -> None:
    ef = pq.ParquetFile(emb_path)
    cf = pq.ParquetFile(chunks_path)
    n_total = ef.metadata.num_rows
    assert n_total == cf.metadata.num_rows, \
        f"emb 与 chunks 行数不一致：{n_total} vs {cf.metadata.num_rows}"
    if limit:
        n_total = min(n_total, limit)
    done = load_checkpoint() if resume else 0
    print(f"上传 {n_total:,} 点（起点 {done:,}，批 {batch_size}）...", flush=True)
    t0 = time.time()
    row = 0
    for eb, cb in zip(ef.iter_batches(batch_size=batch_size),
                      cf.iter_batches(batch_size=batch_size, columns=PAYLOAD_FIELDS)):
        if row + eb.num_rows <= done:          # 已完成的整批跳过（续传）
            row += eb.num_rows
            continue
        emb_ids = eb.column("id").to_pylist()
        chunk_ids = cb.column("id").to_pylist()
        assert emb_ids == chunk_ids, \
            f"第 {row:,} 行起 emb.id != chunks.id —— 行序错位，中止（灌错=检索语义全毁）"
        emb = np.asarray(eb.column("embedding").to_pylist(), dtype=np.float32)
        if emb.ndim == 1:                      # FixedSizeListArray 偶发展平
            emb = emb.reshape(eb.num_rows, -1)
        pts = []
        for i in range(eb.num_rows):
            pts.append(models.PointStruct(
                id=row + i,
                vector=emb[i].tolist(),
                payload={f: (emb_ids[i] if f == "id" else cb.column(f)[i].as_py())
                         for f in PAYLOAD_FIELDS},
            ))
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=pts, wait=False)
        row += eb.num_rows
        save_checkpoint(row, n_total)
        print(f"\r{row:,}/{n_total:,}  {(time.time()-t0):.0f}s", end="", flush=True)
        if limit and row >= n_total:
            break
    print(f"\n发送完成（{(time.time()-t0)/60:.1f} 分钟），等待服务端优化落库...")


def wait_green(client: QdrantClient, expected: int, timeout_s: int = 7200) -> None:
    t0 = time.time()
    while True:
        cnt = client.count(collection_name=config.QDRANT_COLLECTION).count
        status = str(client.get_collection(config.QDRANT_COLLECTION).status).lower()
        if cnt >= expected and status == "green":
            print(f"\n灌库完成：{cnt:,} 点，status=green，{(time.time()-t0)/60:.1f} 分钟")
            return
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"等待 green 超时：count={cnt:,} status={status}")
        print(f"\r等待优化完成：count={cnt:,} status={status}  {time.time()-t0:.0f}s",
              end="", flush=True)
        time.sleep(20)


# ---------- 校验（--verify，也是灌库后的验收闸门） ----------

def pick_rows(path: Path, labels: set, columns=None):
    """流式扫描 parquet 挑出指定行号的行 → {row: value}。

    value 为 dict（columns 给多列）或单值（columns 给单列名）。全量扫描 ~1-2 分钟。
    """
    pf = pq.ParquetFile(path)
    out = {}
    row = 0
    for b in pf.iter_batches(batch_size=100_000, columns=columns):
        if columns is None or (isinstance(columns, list) and len(columns) > 1):
            for i in range(b.num_rows):
                if row + i in labels:
                    out[row + i] = {c: b.column(c)[i].as_py() for c in columns}
        elif len(columns) == 1:
            c = columns[0]
            for i in range(b.num_rows):
                if row + i in labels:
                    out[row + i] = b.column(c)[i].as_py()
        row += b.num_rows
        if len(out) >= len(labels):
            break
    return out


def verify(client: QdrantClient, emb_path: Path, chunks_path: Path) -> None:
    import hnswlib
    n_parquet = pq.ParquetFile(emb_path).metadata.num_rows
    rng = random.Random(20261021)

    # ① 计数
    cnt = client.count(collection_name=config.QDRANT_COLLECTION).count
    assert cnt == n_parquet, f"count 不一致：Qdrant {cnt:,} vs parquet {n_parquet:,}"
    print(f"① 计数一致：{cnt:,}")

    # ② payload 抽样比对（200 个随机 label，含 meta.parquet 交叉验证）
    sample = rng.sample(range(n_parquet), 200)
    pts = client.retrieve(collection_name=config.QDRANT_COLLECTION,
                          ids=sample, with_payload=PAYLOAD_FIELDS, with_vectors=False)
    got = {int(p.id): (p.payload or {}) for p in pts}
    want = pick_rows(chunks_path, set(sample), PAYLOAD_FIELDS)
    bad = 0
    for lab in sample:
        for f in PAYLOAD_FIELDS:
            if got.get(lab, {}).get(f) != want[lab].get(f):
                print(f"  ✗ label={lab} 字段 {f} 不一致："
                      f"Qdrant={got.get(lab, {}).get(f)!r} parquet={want[lab].get(f)!r}")
                bad += 1
    assert bad == 0, f"payload 比对 {bad} 处不一致"
    meta_path = config.DATA_DIR / "index" / "meta.parquet"
    if meta_path.is_file():
        meta = pq.read_table(meta_path).to_pandas().set_index("label")["id"].to_dict()
        mbad = sum(1 for lab in sample if got.get(lab, {}).get("id") != meta.get(lab))
        assert mbad == 0, f"与 meta.parquet 交叉验证 {mbad} 处不一致"
        print(f"② payload 抽样 200/200 一致（含 meta.parquet 交叉验证）")
    else:
        print("② payload 抽样 200/200 一致（无 meta.parquet，跳过交叉验证）")

    # ③ 召回对齐：随机 50 向量 Qdrant top-50 vs hnswlib top-50。
    # 实测结论（2026-10-21 诊断）：两实现数值等价（同 label top-1 分数差 ~4e-7），
    # 但 HNSW 图结构不同 → 低 ef 近邻集有近似噪声（ef=100 overlap 0.956 → ef=400 0.985）。
    # 三重闸门：① 同 top-1 分数差 ≤1e-4（数值等价，硬门槛）② ef=100 overlap ≥0.95
    # （生产 ef 容差）③ ef=400 overlap ≥0.98（高 ef 收敛，排除系统性偏差）。
    # 最终生产对齐由 64 题 eval 闸门裁定（计划既定双重兜底）。
    vec_labels = rng.sample(range(n_parquet), 50)
    vec_rows = pick_rows(emb_path, set(vec_labels), ["embedding"])
    vecs = {}
    for lab in vec_labels:
        v = np.asarray(vec_rows[lab], dtype=np.float32)
        vecs[lab] = v.reshape(-1) if v.ndim == 1 else v
    idx = hnswlib.Index(space="cosine", dim=512)
    idx.load_index(str(config.DATA_DIR / "index" / "hnsw.bin"))
    idx.set_ef(EF_SEARCH)     # ef 不持久化，加载后必须重设（P2 验证结论）

    def compare(ef: int):
        """双侧 ef=ef 的 top-50 对比 → (overlap 列表, 同 top-1 的分数差列表)。"""
        idx.set_ef(ef)
        overlaps, score_diffs = [], []
        for lab, v in vecs.items():
            # P6 坑位：批量要走 query_batch_points；此处单查询直接传裸向量（传 [向量] 列表
            # 会被服务端当成 multivector 输入 → 400）
            resp = client.query_points(
                collection_name=config.QDRANT_COLLECTION, query=v.tolist(), limit=50,
                with_payload=False,
                search_params=models.SearchParams(hnsw_ef=ef, exact=False))
            q_labels = [int(p.id) for p in resp.points]
            q_scores = [p.score for p in resp.points]
            hl, hd = idx.knn_query(v, k=50)
            h_labels = [int(x) for x in hl[0]]
            h_scores = [float(1 - x) for x in hd[0]]
            overlaps.append(len(set(q_labels) & set(h_labels)) / 50)
            if q_labels[0] == h_labels[0]:   # 同 label 才比分数（label 不同分数差无意义）
                score_diffs.append(abs(q_scores[0] - h_scores[0]))
        return overlaps, score_diffs

    ov100, sd100 = compare(EF_SEARCH)
    ov400, _ = compare(400)
    mean_sd = sum(sd100) / len(sd100) if sd100 else 1.0   # 全翻转视为数值不等价
    print(f"③ 召回对齐：ef=100 overlap@50 {sum(ov100)/len(ov100):.4f}（阈值 0.95）、"
          f"ef=400 overlap@50 {sum(ov400)/len(ov400):.4f}（阈值 0.98）、"
          f"同 top-1 分数差 {mean_sd:.2e}（阈值 1e-4）")
    assert mean_sd <= 1e-4, f"同 top-1 分数差 {mean_sd:.2e} > 1e-4 —— 数值不等价，勿切换"
    assert sum(ov100) / len(ov100) >= 0.95, \
        f"ef=100 overlap 低于 0.95 —— 生产 ef 对齐不足，勿切换"
    assert sum(ov400) / len(ov400) >= 0.98, \
        f"ef=400 overlap 低于 0.98 —— 高 ef 不收敛，疑似系统性偏差，勿切换"

    # ④ 延迟 + 批量返回序 == 逐查询序（锁定客户端批量语义）
    qs = [v.tolist() for v in list(vecs.values())[:6]]
    resps = client.query_batch_points(
        collection_name=config.QDRANT_COLLECTION,
        requests=[models.QueryRequest(
            query=q, with_payload=False,
            params=models.SearchParams(hnsw_ef=EF_SEARCH, exact=False),
            limit=10) for q in qs])
    for qi, q in enumerate(qs):
        one = client.query_points(
            collection_name=config.QDRANT_COLLECTION, query=q, limit=10,
            with_payload=False,
            search_params=models.SearchParams(hnsw_ef=EF_SEARCH, exact=False))
        assert [p.id for p in one.points] == [p.id for p in resps[qi].points], \
            f"批量返回序/内容与逐查询不一致（查询 {qi}）"
    ts = []
    for v in list(vecs.values())[:20]:
        t0 = time.time()
        client.query_points(collection_name=config.QDRANT_COLLECTION,
                            query=v.tolist(), limit=10, with_payload=False,
                            search_params=models.SearchParams(hnsw_ef=EF_SEARCH, exact=False))
        ts.append((time.time() - t0) * 1000)
    ts.sort()
    print(f"④ 批量返回序一致；单查询延迟 p50={ts[len(ts)//2]:.0f}ms "
          f"p95={ts[int(len(ts)*0.95)]:.0f}ms（n={len(ts)}）")
    print("✅ 校验全部通过")


# ---------- 入口 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(config.DATA_DIR / "emb.parquet"))
    ap.add_argument("--chunks", default=str(config.DATA_DIR / "chunks.parquet"))
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="只灌前 N 行（试验用）")
    ap.add_argument("--resume", action="store_true", help="从 checkpoint 续传")
    ap.add_argument("--drop", action="store_true", help="删旧 collection 重建")
    ap.add_argument("--verify", action="store_true", help="只跑校验")
    ap.add_argument("--skip-verify", action="store_true", help="灌库后跳过校验")
    args = ap.parse_args()

    client = get_client()
    if args.verify:
        verify(client, Path(args.emb), Path(args.chunks))
        return

    if client.collection_exists(config.QDRANT_COLLECTION):
        if args.drop:
            client.delete_collection(config.QDRANT_COLLECTION)
            create_collection(client)
            print(f"已删除并重建 collection「{config.QDRANT_COLLECTION}」")
        elif not args.resume:
            print(f"collection「{config.QDRANT_COLLECTION}」已存在："
                  f"--resume 续传 / --drop 重建 / --verify 校验", file=sys.stderr)
            sys.exit(1)
    else:
        create_collection(client)
        print(f"已创建 collection「{config.QDRANT_COLLECTION}」")

    upload(client, Path(args.emb), Path(args.chunks), args.batch_size, args.limit,
           args.resume)
    wait_green(client, args.limit or pq.ParquetFile(args.emb).metadata.num_rows)
    if not args.limit:      # 试验灌库不建 payload 索引（正式库才需要 filter 下推）
        build_payload_indexes(client)
    if not args.skip_verify:
        verify(client, Path(args.emb), Path(args.chunks))


if __name__ == "__main__":
    main()
