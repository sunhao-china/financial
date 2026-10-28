# -*- coding: utf-8 -*-
"""进程内 Prometheus 指标（§13.5）：计数器 + 有界采样直方图，全部加锁（workers=1 多线程）。

计数器：chat_total / chat_error / chat_refuse / no_result / verify_pass / verify_fail /
        sse_drop / vector_degraded / maintenance_entered / recovered /
        retry_total / retry_{label} / token_total_{step}
直方图：step_latency_ms_{step}（2048 样本有界、10 分钟窗口，渲染时输出 p50/p95/avg）
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

HIST_MAX_SAMPLES = 2048     # 有界采样上限（防内存膨胀）
HIST_WINDOW_SEC = 600       # 采样窗口 10 分钟


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(int)
        self._hist = {}                 # name -> deque[(monotonic_ts, value)]
        self._started = time.time()

    # ---------- 采集 ----------

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] += n

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            dq = self._hist.setdefault(name, deque(maxlen=HIST_MAX_SAMPLES))
            dq.append((time.monotonic(), value))

    def on_retry(self, retry_state, label: str) -> None:
        """llm_client 重试回调（重试率指标唯一来源）。label 为路由/工序名。"""
        clean = "".join(c if (c.isalnum() or c == "_") else "_" for c in label)
        self.inc("retry_total")
        self.inc(f"retry_{clean}")

    # ---------- 渲染 ----------

    def render_prometheus(self) -> str:
        now = time.monotonic()
        with self._lock:
            counters = dict(self._counters)
            hists = {}
            for name, dq in self._hist.items():
                samples = sorted(v for t, v in dq if now - t <= HIST_WINDOW_SEC)
                if samples:
                    hists[name] = samples
            uptime = time.time() - self._started
        lines = []
        for name, v in sorted(counters.items()):
            lines.append(f"fin_{name} {v}")
        for name, samples in sorted(hists.items()):
            n = len(samples)
            avg = sum(samples) / n
            p50 = samples[int(n * 0.50)] if n > 1 else samples[0]
            p95 = samples[min(int(n * 0.95), n - 1)]
            lines.append(f"fin_{name}_avg {avg:.1f}")
            lines.append(f"fin_{name}_p50 {p50:.1f}")
            lines.append(f"fin_{name}_p95 {p95:.1f}")
        lines.append(f"fin_uptime_seconds {uptime:.0f}")
        return "\n".join(lines) + "\n"
