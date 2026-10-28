# -*- coding: utf-8 -*-
"""用户系统（2026-10-19 加）：登录/注册/验证码/会话列表/历史消息。

- token：自签 HMAC-SHA256（标准库，无 JWT 依赖）"v1.{user_id}.{exp}.{sig}"，7 天
- 密码：pbkdf2_hmac('sha256', PBKDF2_ROUNDS) + 随机盐（标准库，无 bcrypt 依赖）。
  2026-10-22 安全加固：轮数 10 万 → 31 万（config.PBKDF2_ROUNDS），盐编码
  "{rounds}${salt_hex}" 兼容旧哈希；登录成功时机会式重哈希升级（无感）。
- 验证码：见 server/captcha.py（Redis 一次性 5 分钟，内存降级）
- 防暴力破解：见 server/auth_guard.py（账号锁定 + IP 限流）；校验顺序
  IP 限流 → 验证码 → 账号锁定 → 密码。验证码错误不计数（防锁死他人账号）。
- 登录失败统一话术防用户枚举；用户不存在时跑 dummy pbkdf2 均衡耗时
- MySQL 不可用 → 503（问答本身经 chat_history 降级不受影响）
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

import config
from server.auth_guard import ip_throttle, lock_remaining, record_fail
from server.captcha import CaptchaStore
from server.db import db

router = APIRouter(prefix="/api")
captcha_store: Optional[CaptchaStore] = None
metrics = None  # main.py lifespan 注入（与 captcha_store 同款模式）

USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")

# ---------- 密码 ----------

_PBKDF2_ROUNDS = config.PBKDF2_ROUNDS
_LEGACY_ROUNDS = 100_000


def _split_salt(salt: str) -> tuple:
    """盐编码 "{rounds}${salt_hex}" → (rounds, salt_hex)；旧格式纯 hex → (10 万, hex)。"""
    if "$" in salt:
        rounds, _, hex_salt = salt.partition("$")
        try:
            return int(rounds), hex_salt
        except ValueError:  # noqa: BLE001 — 异常数据按旧轮数处理
            return _LEGACY_ROUNDS, hex_salt
    return _LEGACY_ROUNDS, salt


def hash_password(password: str, salt: Optional[str] = None) -> tuple:
    """→ (salt, hash_hex)。新哈希盐带轮数前缀，verify 按各自轮数验。"""
    rounds, hex_salt = _split_salt(salt) if salt else (_PBKDF2_ROUNDS, secrets.token_hex(16))
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), hex_salt.encode(), rounds)
    return f"{rounds}${hex_salt}", h.hex()


def verify_password(password: str, salt: str, pass_hash: str) -> bool:
    rounds, hex_salt = _split_salt(salt)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), hex_salt.encode(), rounds)
    return secrets.compare_digest(h.hex(), pass_hash)


def _dummy_verify() -> None:
    """用户不存在时跑一次等价 pbkdf2（固定假盐），均衡响应时间防用户名枚举。"""
    hashlib.pbkdf2_hmac("sha256", b"0", b"0" * 32, _PBKDF2_ROUNDS)


def _upgrade_password(user: dict, password: str) -> None:
    """用明文密码按新轮数重哈希并落库（机会式升级，登录成功时调用）。"""
    rounds, _ = _split_salt(user["salt"])
    if rounds >= _PBKDF2_ROUNDS:
        return
    salt, h = hash_password(password)
    db.update_password(user["id"], h, salt)
    if metrics:
        metrics.inc("rehash_upgraded")

# ---------- token ----------


def issue_token(user_id: int) -> str:
    exp = int(time.time()) + config.TOKEN_TTL_SEC
    body = f"v1.{user_id}.{exp}"
    sig = hmac.new(config.TOKEN_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def parse_token(token: str) -> Optional[int]:
    """→ user_id；无效/过期 → None。"""
    try:
        parts = token.split(".")
        if len(parts) != 4 or parts[0] != "v1":
            return None
        body = ".".join(parts[:3])
        sig = hmac.new(config.TOKEN_SECRET.encode(), body.encode(),
                       hashlib.sha256).hexdigest()
        if not secrets.compare_digest(sig, parts[3]):
            return None
        if int(parts[2]) < time.time():
            return None
        return int(parts[1])
    except Exception:  # noqa: BLE001
        return None

# ---------- 请求模型 ----------


class LoginReq(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
    captcha_id: str
    captcha_code: str


class RegisterReq(LoginReq):
    pass


# ---------- 依赖 ----------


def require_user(authorization: Optional[str] = Header(None)) -> dict:
    """Bearer token → users 行。未登录/失效 → 401。"""
    token = (authorization or "").removeprefix("Bearer ").strip()
    user_id = parse_token(token)
    if user_id is None:
        raise HTTPException(401, detail="请先登录")
    try:
        rows = db.query("SELECT id, username FROM users WHERE id=%s", (user_id,))
    except Exception:  # noqa: BLE001
        raise HTTPException(503, detail="数据库不可用，请稍后重试")
    if not rows:
        raise HTTPException(401, detail="请先登录")
    return rows[0]


def _check_captcha(req: LoginReq) -> None:
    ok = False
    if captcha_store is not None:
        try:
            ok = captcha_store.verify(req.captcha_id, req.captcha_code)
        except Exception:  # noqa: BLE001 — 验证码存储异常视为验证失败
            ok = False
    if not ok:
        raise HTTPException(401, detail="验证码错误或已过期，请点击图片刷新")


# ---------- 路由 ----------


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.get("/captcha")
def get_captcha(request: Request):
    ip_throttle.throttle_captcha(_client_ip(request))
    if captcha_store is None:
        raise HTTPException(503, detail="验证码服务不可用（缺少 Pillow）")
    cid, img = captcha_store.new()
    return {"captcha_id": cid, "image": img}


@router.post("/login")
def login(request: Request, req: LoginReq):
    ip = _client_ip(request)
    ip_throttle.throttle_login(ip)
    try:
        _check_captcha(req)
    except HTTPException:
        ip_throttle.note_login_fail(ip)   # 验证码错计 IP 失败，但绝不触账号锁定
        raise
    try:
        u = db.get_user(req.username)
    except Exception:  # noqa: BLE001
        raise HTTPException(503, detail="数据库不可用，请稍后重试")
    if u:
        locked = lock_remaining(u)
        if locked:
            if metrics:
                metrics.inc("login_locked")
            raise HTTPException(403, detail=locked)
        ok = verify_password(req.password, u["salt"], u["pass_hash"])
    else:
        _dummy_verify()                   # 不存在的用户也付一次 pbkdf2 的耗时
        ok = False
    if not ok:
        ip_throttle.note_login_fail(ip)
        if u:
            try:
                record_fail(u)
            except Exception:  # noqa: BLE001 — 记账失败不阻断统一话术
                pass
        if metrics:
            metrics.inc("login_fail")
        raise HTTPException(401, detail="用户名或密码错误")
    ip_throttle.note_login_ok(ip)
    try:
        db.reset_login_fail(u["id"])
        _upgrade_password(u, req.password)  # 老哈希机会式升级（一次性 ~0.2s）
    except Exception:  # noqa: BLE001 — 记账失败不阻断登录
        pass
    if metrics:
        metrics.inc("login_ok")
    return {"token": issue_token(u["id"]), "username": u["username"]}


def _password_ok(pw: str) -> bool:
    """注册密码策略：≥PASSWORD_MIN_LEN 位 且 同时含字母和数字（仅约束新注册）。"""
    return (len(pw) >= config.PASSWORD_MIN_LEN
            and re.search(r"[A-Za-z]", pw) is not None
            and re.search(r"\d", pw) is not None)


@router.post("/register")
def register(request: Request, req: RegisterReq):
    ip = _client_ip(request)
    ip_throttle.throttle_login(ip)
    try:
        _check_captcha(req)
    except HTTPException:
        ip_throttle.note_login_fail(ip)
        raise
    if not USERNAME_RE.match(req.username):
        raise HTTPException(400, detail="用户名需为 3-32 位字母、数字或下划线")
    if not _password_ok(req.password):
        raise HTTPException(400, detail=f"密码至少 {config.PASSWORD_MIN_LEN} 位，且需同时包含字母和数字")
    try:
        if db.get_user(req.username):
            raise HTTPException(409, detail="用户名已存在")
        salt, h = hash_password(req.password)
        uid = db.create_user(req.username, h, salt)
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        raise HTTPException(503, detail="数据库不可用，请稍后重试")
    ip_throttle.note_login_ok(ip)
    if metrics:
        metrics.inc("register_ok")
    return {"token": issue_token(uid), "username": req.username}


@router.get("/me")
def me(user: dict = Depends(require_user)):
    return {"username": user["username"]}


@router.get("/conversations")
def conversations(user: dict = Depends(require_user)):
    try:
        rows = db.list_conversations(user["id"])
    except Exception:  # noqa: BLE001
        raise HTTPException(503, detail="数据库不可用，请稍后重试")
    return {"conversations": [
        {"id": r["id"], "title": r["title"] or "新会话",
         "created_at": r["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
         "updated_at": r["updated_at"].strftime("%Y-%m-%d %H:%M:%S")}
        for r in rows]}


@router.get("/conversations/{conv_id}/messages")
def conv_messages(conv_id: str, user: dict = Depends(require_user)):
    try:
        own = db.query("SELECT id FROM conversations WHERE id=%s AND user_id=%s",
                       (conv_id, user["id"]))
        if not own:
            raise HTTPException(404, detail="会话不存在")
        return {"messages": db.get_messages(conv_id)}
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        raise HTTPException(503, detail="数据库不可用，请稍后重试")


def ensure_admin() -> None:
    """启动时预置管理员（仅当不存在）。MySQL 不可用时静默（登录暂不可用）。"""
    try:
        if not db.get_user(config.ADMIN_USERNAME):
            salt, h = hash_password(config.ADMIN_PASSWORD)
            db.create_user(config.ADMIN_USERNAME, h, salt)
    except Exception:  # noqa: BLE001
        pass
