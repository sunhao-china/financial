# -*- coding: utf-8 -*-
"""告警规则检查（§13.5）：独立进程运行（tools/alert_check.py cron 每 5 分钟），
读 /metrics + S8 日志 + df，落盘 alerts.jsonl + 可选 webhook（config.ALERT_WEBHOOK_URL）。

设计（P5 核心决策）：告警走 ECS cron 而非进程内——服务重启不丢告警历史；
窗口指标用计数器差值（state 文件存上次快照），不受进程重启影响。
规则（config.ALERT_RULES 阈值）：
    api_unreachable    /metrics 不可达（服务挂了本身即告警）
    error_rate         chat_error / chat_total > 5%（窗口内差值）
    verify_fail_rate   验证拦截率 > 30%
    latency_p95        生成 p95 相对基线 > 2x（基线文件首次运行时建立）
    disk_usage         数据盘占用 > 85%
    daily_cost         日 token 成本达预算 80% 预警 / 100% 告警（预算 0 = 禁用）
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

import config


class AlertChecker:
    def __init__(self, rules: Optional[dict] = None, alerts_path: Optional[Path] = None,
                 webhook: Optional[str] = None):
        self.rules = rules or config.ALERT_RULES
        self.alerts_path = Path(alerts_path or config.ALERTS_PATH)
        self.webhook = webhook if webhook is not None else config.ALERT_WEBHOOK_URL
        self._state_path = config.LOG_DIR / "alert_state.json"
        self._baseline_path = config.LOG_DIR / "latency_baseline.json"
        self._metrics_url = f"http://127.0.0.1:{config.API_PORT}/metrics"

    # ---------- 主入口 ----------

    def check(self, daily_only: bool = False) -> List[dict]:
        alerts: List[dict] = []
        if daily_only:
            alerts += self._check_daily_cost()
            return alerts
        metrics = self._fetch_metrics()
        if metrics is None:
            return [self._alert("api_unreachable", "error",
                                f"/metrics 不可达（{self._metrics_url}），服务可能已宕")]
        alerts += self._check_windowed(metrics)
        alerts += self._check_latency(metrics)
        alerts += self._check_disk()
        return alerts

    # ---------- 规则 ----------

    def _fetch_metrics(self) -> Optional[Dict[str, float]]:
        try:
            r = requests.get(self._metrics_url, timeout=5)
            if r.status_code != 200:
                return None
            out = {}
            for line in r.text.splitlines():
                if line.startswith("fin_") and " " in line:
                    name, val = line.split()
                    try:
                        out[name] = float(val)
                    except ValueError:
                        pass
            return out
        except Exception:  # noqa: BLE001
            return None

    def _window_deltas(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """计数器差值（与上次巡检快照比），state 文件持久化，跨进程有效。"""
        prev: Dict[str, float] = {}
        if self._state_path.exists():
            try:
                prev = json.loads(self._state_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                prev = {}
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(metrics), encoding="utf-8")
        return {k: max(v - float(prev.get(k, 0)), 0) for k, v in metrics.items()}

    def _check_windowed(self, metrics: Dict[str, float]) -> List[dict]:
        alerts = []
        d = self._window_deltas(metrics)
        chat = d.get("fin_chat_total", 0)
        err = d.get("fin_chat_error", 0)
        if chat >= 5 and chat > 0 and err / chat > self.rules["error_rate_max"]:
            alerts.append(self._alert(
                "error_rate", "error",
                f"窗口内错误率 {err}/{chat} = {err / chat:.0%} > {self.rules['error_rate_max']:.0%}"))
        vp = d.get("fin_verify_pass", 0)
        vf = d.get("fin_verify_fail", 0)
        if vp + vf >= 10 and vp + vf > 0 and vf / (vp + vf) > self.rules["verify_fail_rate_max"]:
            alerts.append(self._alert(
                "verify_fail_rate", "warning",
                f"验证拦截率 {vf}/{vp + vf} = {vf / (vp + vf):.0%} > "
                f"{self.rules['verify_fail_rate_max']:.0%}"))
        return alerts

    def _check_latency(self, metrics: Dict[str, float]) -> List[dict]:
        p95 = metrics.get("fin_step_latency_ms_generate_p95")
        if p95 is None or p95 <= 0:
            return []
        baseline = None
        if self._baseline_path.exists():
            try:
                baseline = json.loads(self._baseline_path.read_text(encoding="utf-8")).get("generate_p95")
            except Exception:  # noqa: BLE001
                baseline = None
        if baseline is None:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(json.dumps({"generate_p95": p95}), encoding="utf-8")
            return []
        if p95 / baseline > self.rules["latency_p95_ratio_max"]:
            return [self._alert(
                "latency_p95", "warning",
                f"生成 p95 {p95:.0f}ms 相对基线 {baseline:.0f}ms 劣化 {p95 / baseline:.1f}x")]
        return []

    def _check_disk(self) -> List[dict]:
        usage = shutil.disk_usage(config.DATA_DIR)
        ratio = usage.used / usage.total
        if ratio > self.rules["disk_usage_max"]:
            return [self._alert("disk_usage", "error",
                                f"数据盘占用 {ratio:.0%} > {self.rules['disk_usage_max']:.0%}")]
        return []

    def _check_daily_cost(self) -> List[dict]:
        budget = self.rules.get("daily_cost_budget", 0)
        if budget <= 0:
            return []
        cost = self._sum_today_cost()
        alerts = []
        if cost >= budget * self.rules["daily_cost_max_ratio"]:
            alerts.append(self._alert("daily_cost", "error",
                                      f"今日 token 成本 {cost:,} ≥ 预算 {budget:,}（100%）"))
        elif cost >= budget * self.rules["daily_cost_warn_ratio"]:
            alerts.append(self._alert("daily_cost", "warning",
                                      f"今日 token 成本 {cost:,} 达预算 {budget:,} 的 "
                                      f"{cost / budget:.0%}"))
        return alerts

    def _sum_today_cost(self) -> int:
        """当日 S8 日志（含轮转文件）cost_tokens.total 求和。"""
        today = datetime.now().strftime("%Y-%m-%d")
        total = 0
        for p in sorted(config.S8_LOG_PATH.parent.glob("s8*.jsonl")):
            try:
                for line in open(p, encoding="utf-8"):
                    d = json.loads(line)
                    if d.get("ts", "").startswith(today):
                        total += (d.get("cost_tokens") or {}).get("total", 0)
            except Exception:  # noqa: BLE001
                continue
        return total

    # ---------- 投递 ----------

    def _alert(self, rule: str, level: str, message: str) -> dict:
        return {"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "rule": rule, "level": level, "message": message}

    def fire(self, alert: dict) -> None:
        """落盘 alerts.jsonl + 可选 webhook（webhook 失败静默，不阻塞巡检）。"""
        self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.alerts_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert, ensure_ascii=False) + "\n")
        if self.webhook:
            try:
                requests.post(self.webhook, json=alert, timeout=5)
            except Exception:  # noqa: BLE001
                pass
