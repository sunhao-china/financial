# -*- coding: utf-8 -*-
"""全局配置：模型路由（大纲 §3）、语料路径与过滤规则（大纲 §7）。"""
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REPORT_DIR = DATA_DIR / "reports"

# ---------- 平台密钥（环境变量优先，服务器上从 .env 读取） ----------
_ENV_FILE = ROOT / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k, _v.strip())

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com"
DEEPSEEK_BASE = "https://api.deepseek.com"

# ---------- 模型路由（实测结论 2026-10-16，大纲 §3） ----------
MODELS = {
    # 百炼内部工序：快、便宜、严格 JSON
    "intent":       {"provider": "dashscope", "model": "qwen-plus"},   # qwen-turbo 限流，暂用 plus
    "query_expand": {"provider": "dashscope", "model": "qwen-plus"},
    "self_check":   {"provider": "dashscope", "model": "qwen-plus"},
    "verify":       {"provider": "dashscope", "model": "qwen-plus"},
    "embedding":    {"provider": "dashscope", "model": "text-embedding-v3"},
    "rerank":       {"provider": "dashscope", "model": "qwen3-rerank"},
    # DeepSeek 生成：按意图路由（大纲 §3/§11）
    "generate_fact":    {"provider": "deepseek", "model": "deepseek-flash"},   # 事实类，快
    "generate_complex": {"provider": "deepseek", "model": "deepseek-v4-pro"},  # 归纳/对比类，深推理
}

# ---------- 契约与提示词参数（大纲 §4/§6/§9/§10/§13.2） ----------
SCHEMA_VERSION = "1.0.0"       # 所有 JSON 结构顶层 schema_version（§9.1）
QUERY_EXPAND_COUNT = (3, 5)    # 查询扩展变体数 3~5（§4）
MAX_FOLLOWUP_QUERIES = 2       # 自检补检查询 ≤2（§10.3）
MAX_REFILL_ROUNDS = 2          # 补检轮数上限，pre_step 钩子强制（§6.1）
JSON_RETRY = 1                 # LLM JSON 解析失败重试次数（§13.2）

# ---------- 语料 ----------
SOURCE_FILES = ["rank_00045.parquet", "rank_00046.parquet", "rank_00047.parquet"]

# 质量过滤阈值（analyze_quality.py 报告生成后回填确认）
FILTER_RULES = {
    "quality_score_min": 3.5,
    "perplexity_max": 3000,
    "char_rep_ratio_max": 0.1,
    "flagged_words_ratio_max": 0.0,
}

# 去重参数（近重复用 MinHash + 精确 Jaccard，见 corpus/jaccard.py 注释）
JACCARD_MIN = 0.8          # 精确 Jaccard ≥ 0.8 视为近似重复（转载文）
MINHASH_PERM = 64          # minhash 置换数（仅用于排序近似）
JACCARD_SORT_KEYS = 12     # 排序用前 12 个 minhash 值
JACCARD_WINDOW = 100       # 排序后窗口 ±100 内比对

# 中间数据路径
FILTERED_PARQUET = DATA_DIR / "filtered.parquet"
DEDUPED_PARQUET = DATA_DIR / "deduped.parquet"
CLEANED_PARQUET = DATA_DIR / "cleaned.parquet"
# 文献名映射（§14.5 引用来源可读性）：doc_id → 首句规则提炼标题
# tools/build_title_map.py 离线生成；缺文件时 S5Hit.doc_title=None，前端回退「文章 #id」
TITLE_MAP_PARQUET = DATA_DIR / "title_map.parquet"

# ---------- P5 Web 服务（大纲 §14/§13.5） ----------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8001"))
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
# 前端构建产物目录（§14）：main.py 末尾条件挂载 StaticFiles，目录缺失则纯 API 模式
FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", str(ROOT / "frontend" / "dist")))
LOG_DIR = DATA_DIR / "logs"
S8_LOG_PATH = Path(os.environ.get("S8_LOG_PATH", str(LOG_DIR / "s8.jsonl")))
S8_LOG_STDOUT = os.environ.get("S8_LOG_STDOUT", "1") == "1"
ALERTS_PATH = Path(os.environ.get("ALERTS_PATH", str(LOG_DIR / "alerts.jsonl")))
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")

# 多轮对话（§14）：V1 不做 LLM 摘要（下游只消费末 3~6 轮），HISTORY_SUMMARIZE 为 V2 预留
HISTORY_MAX_TURNS = 10          # 会话保留轮数（1 轮 = 用户 + 助手 2 条）
HISTORY_MAX_CHARS = 12000       # 传给 agent 的历史总字符上限（从最旧裁剪）
HISTORY_SUMMARIZE = False

# ---------- Langfuse 追踪（§13.5，2026-10-20 加） ----------
# 自托管实例（ECS 同机 docker，数据不出境）。未启用/未装 SDK 时追踪为 no-op（llm_client._lf 降级），
# S7/S8/metrics 观测不受影响；一次问答 = 一条 trace（trace_id 由 query_id 确定性派生）。
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "0") == "1"
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")

# ---------- 用户系统（登录/会话持久化，2026-10-19 加） ----------
# MySQL 历史主存储（§14.5）；不可用降级 Memory（重启即丢），问答不挂
MYSQL_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "fin")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DB = os.environ.get("MYSQL_DB", "fin")
# token 签名密钥：env 优先，缺省生成后落盘持久（重启登录态不失效）
def _token_secret() -> str:  # noqa: E306
    from pathlib import Path as _P
    _f = _P(os.environ.get("FIN_TOKEN_SECRET_FILE", str(ROOT / ".token_secret")))
    _env = os.environ.get("FIN_TOKEN_SECRET", "")
    if _env:
        return _env
    if _f.exists():
        return _f.read_text(encoding="utf-8").strip()
    _s = secrets.token_hex(32)
    _f.write_text(_s, encoding="utf-8")
    return _s
TOKEN_SECRET = _token_secret()
TOKEN_TTL_SEC = int(os.environ.get("FIN_TOKEN_TTL", str(7 * 24 * 3600)))
# 预置管理员：首次启动自动创建（仅当不存在）；密码可经 env 覆盖
ADMIN_USERNAME = os.environ.get("FIN_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("FIN_ADMIN_PASSWORD", "admin123")
# 验证码：Pillow 自绘，Redis 存 5 分钟一次性；Redis 不可用降级内存 dict
CAPTCHA_TTL_SEC = int(os.environ.get("FIN_CAPTCHA_TTL", "300"))
CAPTCHA_LEN = int(os.environ.get("FIN_CAPTCHA_LEN", "5"))

# 登录安全加固（2026-10-22，§14.5）：防暴力破解三层（账号锁定 + IP 限流 + 验证码）
LOCKOUT_MAX_FAIL = int(os.environ.get("FIN_LOCKOUT_MAX_FAIL", "5"))    # 连续密码错误次数上限
LOCKOUT_SEC = int(os.environ.get("FIN_LOCKOUT_SEC", str(24 * 3600)))   # 锁定时长（默认 24h）
IP_FAIL_LIMIT = int(os.environ.get("FIN_IP_FAIL_LIMIT", "30"))         # 单 IP 窗口内失败上限（含验证码错）
IP_WINDOW_SEC = int(os.environ.get("FIN_IP_WINDOW_SEC", "900"))        # 失败计数窗口（15 分钟）
IP_BLOCK_SEC = int(os.environ.get("FIN_IP_BLOCK_SEC", "900"))          # IP 封禁时长
CAPTCHA_IP_LIMIT = int(os.environ.get("FIN_CAPTCHA_IP_LIMIT", "20"))   # 单 IP 每分钟验证码生成上限
PBKDF2_ROUNDS = int(os.environ.get("FIN_PBKDF2_ROUNDS", "310000"))     # 密码哈希迭代（旧哈希 10 万兼容）
PASSWORD_MIN_LEN = int(os.environ.get("FIN_PASSWORD_MIN_LEN", "8"))    # 注册密码最短长度

# 服务并发与探针（§14/§13.3）
CHAT_CONCURRENCY = int(os.environ.get("CHAT_CONCURRENCY", "4"))      # 并发信号量（workers=1 硬约束）
CHAT_SEMAPHORE_TIMEOUT = float(os.environ.get("CHAT_SEMAPHORE_TIMEOUT", "1"))
SSE_HEARTBEAT_SEC = int(os.environ.get("SSE_HEARTBEAT_SEC", "15"))   # SSE 空窗心跳
VECTOR_FAIL_STREAK = int(os.environ.get("VECTOR_FAIL_STREAK", "3"))  # 向量路连续失败转维护态
PROBE_INTERVAL_SEC = int(os.environ.get("PROBE_INTERVAL_SEC", "60")) # 探针自愈间隔

# ---------- Qdrant 向量库（P6 迁移，2026-10-21） ----------
# 向量路从进程内 hnswlib 迁至独立 Qdrant（大纲 §8）：286 万 chunk 向量 + 8 列元数据
# （含 text）全量入 collection，point id = chunk 行号（与 BM25 label 对齐契约不变）。
# Qdrant 不可达 → 启动 maintenance + 60s 探针重试；运行中失败 → vector_fail_streak
# 降级纯 BM25（§13.3，降级链与 hnswlib 时代完全一致）。
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "chunks")
QDRANT_TIMEOUT = float(os.environ.get("QDRANT_TIMEOUT", "5"))        # 秒，超时=向量路失败
QDRANT_RETRIES = int(os.environ.get("QDRANT_RETRIES", "1"))          # 单查询快速重试次数
QDRANT_SERVER_FILTER = os.environ.get("QDRANT_SERVER_FILTER", "1") == "1"  # 质量过滤下推
QDRANT_EF = int(os.environ.get("QDRANT_EF", "100"))                  # 对齐原 hnswlib set_ef(100)

# 告警规则阈值（tools/alert_check.py cron 巡检，§13.5）
ALERT_RULES = {
    "error_rate_max": 0.05,             # chat_error / chat_total > 5%（窗口内）
    "verify_fail_rate_max": 0.30,       # 验证拦截率 > 30%
    "latency_p95_ratio_max": 2.0,       # 生成 p95 相对基线劣化 > 2x
    "disk_usage_max": 0.85,             # 数据盘占用 > 85%
    "daily_cost_warn_ratio": 0.80,      # 日 token 成本达预算 80% 预警
    "daily_cost_max_ratio": 1.00,       # 达预算 100% 告警
    "daily_cost_budget": int(os.environ.get("DAILY_COST_BUDGET", "0")),  # 0 = 禁用日成本规则
}
