# -*- coding: utf-8 -*-
"""P5 告警巡检入口（§13.5）：ECS crontab 每 5 分钟调用（独立进程，重启不丢告警历史）。

crontab（ECS）:
    */5 * * * * cd /data/financial && .venv/bin/python tools/alert_check.py >> data/logs/alert_check.log 2>&1
    5 0 * * *   cd /data/financial && .venv/bin/python tools/alert_check.py --daily >> data/logs/alert_check.log 2>&1

告警投递：data/logs/alerts.jsonl + 可选 webhook（config.ALERT_WEBHOOK_URL，空则禁用）。
用法: .venv/bin/python tools/alert_check.py [--daily] [--once]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
from server.alerts import AlertChecker  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="金融 RAG 告警巡检")
    ap.add_argument("--daily", action="store_true", help="凌晨日成本汇总模式")
    ap.add_argument("--once", action="store_true", help="手动触发模式（打印结果）")
    args = ap.parse_args()

    checker = AlertChecker()
    alerts = checker.check(daily_only=args.daily)
    for a in alerts:
        checker.fire(a)
        print(json.dumps(a, ensure_ascii=False), flush=True)
    if not alerts and args.once:
        print("无告警", flush=True)


if __name__ == "__main__":
    main()
