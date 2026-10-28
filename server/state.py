# -*- coding: utf-8 -*-
"""维护态状态机（§13.3 降级矩阵收口）：ok / degraded_bm25 / maintenance。

- ok              ：正常（向量 + BM25 双路）
- degraded_bm25  ：向量路运行中失败，retrieve 内部已降级纯 BM25，继续服务
- maintenance    ：向量路连续失败 ≥ VECTOR_FAIL_STREAK，或启动加载失败；/chat 503 维护话术
- 探针线程（60s）：maintenance 下自愈探测（向量路自检 / 重试全量加载）；运行中监视 streak 迁移状态
"""
from __future__ import annotations

import threading
import time

import config


class HealthState:
    OK = "ok"
    DEGRADED_BM25 = "degraded_bm25"
    MAINTENANCE = "maintenance"

    def __init__(self, retriever=None, metrics=None, streak_threshold: int = None,
                 probe_interval: int = None):
        self.ret = retriever
        self.metrics = metrics
        self.threshold = streak_threshold or config.VECTOR_FAIL_STREAK
        self._lock = threading.Lock()
        self._status = self.OK
        self.reason = ""
        self.started_at = time.time()
        self._t = threading.Thread(target=self._loop,
                                   args=(probe_interval or config.PROBE_INTERVAL_SEC,),
                                   daemon=True, name="health-probe")
        self._t.start()

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def enter_maintenance(self, reason: str) -> None:
        with self._lock:
            self._status = self.MAINTENANCE
            self.reason = reason
        if self.metrics:
            self.metrics.inc("maintenance_entered")

    def _set(self, status: str) -> None:
        with self._lock:
            self._status = status

    # ---------- 探针 ----------

    def _loop(self, interval: int) -> None:
        while True:
            time.sleep(interval)
            try:
                self._tick()
            except Exception:  # noqa: BLE001 — 探针异常不致命，下一轮重试
                pass

    def _tick(self) -> None:
        if self.status == self.MAINTENANCE:
            self._recover()
        elif self.ret is not None:
            with self.ret._cache_lock:
                streak = self.ret.vector_fail_streak
            if streak >= self.threshold:
                self.enter_maintenance(f"向量检索连续失败 {streak} 次")
            elif streak > 0 and self.status == self.OK:
                self._set(self.DEGRADED_BM25)
            elif self.status == self.DEGRADED_BM25:
                # P6 故障演练发现（2026-10-21）：degraded 态恢复不能只等 streak 归零——
                # 同题请求命中查询缓存不触向量路，streak 无从复位，状态会永久滞后。
                # 主动向量自检（maintenance 同款探针），通了立即归零回 ok，不依赖流量。
                try:
                    self.ret.vector_search(["金融系统自检探针"])
                    with self.ret._cache_lock:
                        self.ret.vector_fail_streak = 0
                    self._set(self.OK)
                    self.reason = ""
                    if self.metrics:
                        self.metrics.inc("recovered")
                except Exception:  # noqa: BLE001 — 向量路仍不可用：保持 degraded，下轮再试
                    pass

    def _recover(self) -> None:
        """maintenance 自愈：启动加载失败 → 周期重试全量加载；运行中故障 → 向量路自检。"""
        if self.ret is None:
            ret = self._try_load()
            if ret is not None:
                self.ret = ret
                self._set(self.OK)
                self.reason = ""
                if self.metrics:
                    self.metrics.inc("recovered")
            return
        try:
            self.ret.vector_search(["金融系统自检探针"])
            with self.ret._cache_lock:
                self.ret.vector_fail_streak = 0
                self.ret.last_vector_error = None
            self._set(self.OK)
            self.reason = ""
            if self.metrics:
                self.metrics.inc("recovered")
        except Exception as e:  # noqa: BLE001
            with self.ret._cache_lock:
                self.ret.last_vector_error = f"探针: {e}"

    def _try_load(self):
        try:
            from corpus.retrieval import HybridRetriever
            return HybridRetriever()
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self.reason = f"{type(e).__name__}: {e}"[:200]
            return None
