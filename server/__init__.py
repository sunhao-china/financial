# -*- coding: utf-8 -*-
"""P5 Web 服务包（大纲 §14）：FastAPI + SSE 流式 + 多轮对话 + 监控告警 + 降级收口。

模块职责：
    main.py           FastAPI app 与路由（/chat SSE、/sources、/health、/metrics）
    sse.py            SSEHub（线程→队列→生成器桥接）+ worker 线程入口 + done 组装
    hooks.py          ServiceHooks：agent 步骤 → SSE/S8/指标 分派
    chat_history.py   多轮会话历史（Redis + Memory 双实现、merge/trim）
    session_redis.py  S7 快照 Redis 实现 + RedisHealth 可用性探测
    logging_s8.py     S8 链路事件 JSONL 日志
    metrics.py        进程内 Prometheus 指标
    state.py          维护态状态机（探针自愈）
    alerts.py         告警规则检查（tools/alert_check.py cron 调用）
"""
