# -*- coding: utf-8 -*-
"""图片验证码（登录/注册防爆破）：Pillow 自绘 + 逐字符随机旋转 + 干扰线噪点。

- captcha_id = secrets.token_hex(16)，code 存 Redis（TTL 5 分钟，一次性即删）
- Redis 不可用降级内存 dict（同 MemoryChatHistory 风格，单 worker 进程内有效）
- 字符集去掉易混淆的 0O1I 等；字符数 5（2026-10-22 安全加固，config.CAPTCHA_LEN）
- 逐字符独立图层随机旋转 ±25° 再合成：打散字符的规则排布，提高 OCR 分割成本
"""
from __future__ import annotations

import base64
import io
import random
import secrets
import threading
import time

import config

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # 本地无 Pillow：验证码功能不可用（部署时安装）
    Image = ImageDraw = ImageFont = None

try:
    import redis
except ImportError:
    redis = None

CHARS = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # 去 0O1I 等易混淆字符
CODE_LEN = config.CAPTCHA_LEN
IMG_W, IMG_H = 132, 44   # 与前端 CSS 120x40 等比（3:1），展示不变形


def _font(size: int):
    # 优先系统中文字体（验证码为字母数字，任意字体均可）
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "C:/Windows/Fonts/arialbd.ttf",
                 "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def render(code: str) -> str:
    """绘制 PNG → data URI（前端 <img src> 直接可挂）。"""
    img = Image.new("RGB", (IMG_W, IMG_H), (245, 247, 250))
    draw = ImageDraw.Draw(img)
    font = _font(28)
    step = (IMG_W - 20) // max(len(code), 1)
    for i, ch in enumerate(code):
        # 逐字符独立图层随机旋转后合成：打散规则排布，提高 OCR 分割成本
        layer = Image.new("RGBA", (46, 46), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.text((7, 6), ch, font=font,
                fill=(random.randint(0, 80), random.randint(0, 80), random.randint(0, 80)))
        layer = layer.rotate(random.randint(-25, 25),
                             resample=Image.BICUBIC, expand=False)
        x = 8 + i * step + random.randint(-2, 2)
        y = random.randint(-2, 4)
        img.paste(layer, (x, y), layer)
    for _ in range(8):  # 干扰线
        draw.line([(random.randint(0, IMG_W), random.randint(0, IMG_H)),
                   (random.randint(0, IMG_W), random.randint(0, IMG_H))],
                  fill=(random.randint(120, 200),) * 3, width=1)
    for _ in range(140):  # 噪点
        draw.point((random.randint(0, IMG_W), random.randint(0, IMG_H)),
                   fill=(random.randint(100, 180),) * 3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class CaptchaStore:
    """验证码存取。verify 一次性：无论对错即删（防试错）。"""

    def __init__(self, health=None):
        self._health = health          # RedisHealth（None = 纯内存模式）
        self._mem: dict = {}           # captcha_id -> (code, ts)
        self._lock = threading.Lock()
        self._r = None

    def _client(self):
        if self._r is None:
            self._r = redis.Redis.from_url(config.REDIS_URL,
                                           socket_connect_timeout=2, socket_timeout=2)
        return self._r

    @staticmethod
    def _key(captcha_id: str) -> str:
        return f"fin:captcha:{captcha_id}"

    def new(self) -> tuple:
        """生成一张 → (captcha_id, data_uri)。"""
        code = "".join(random.choices(CHARS, k=CODE_LEN))
        cid = secrets.token_hex(16)
        if self._health is not None and self._health.available:
            try:
                self._client().setex(self._key(cid), config.CAPTCHA_TTL_SEC, code)
            except Exception:  # noqa: BLE001 — Redis 挂：内存兜底
                self._r = None
                self._health.mark_down()
                self._mem_set(cid, code)
        else:
            self._mem_set(cid, code)
        return cid, render(code) if Image else ""

    def _mem_set(self, cid: str, code: str) -> None:
        with self._lock:
            self._mem[cid] = (code, time.time())
            # 惰性清理过期
            for k in [k for k, (_, ts) in self._mem.items()
                      if time.time() - ts > config.CAPTCHA_TTL_SEC]:
                del self._mem[k]

    def verify(self, captcha_id: str, code: str) -> bool:
        """校验并消费（一次性）。任何异常/缺失 → False。"""
        cid, got = (captcha_id or "").strip(), (code or "").strip().upper()
        if not cid or not got:
            return False
        want = None
        if self._health is not None and self._health.available:
            try:
                want = self._client().get(self._key(cid))
                self._client().delete(self._key(cid))
            except Exception:  # noqa: BLE001
                self._r = None
                self._health.mark_down()
        if want is None:
            with self._lock:
                ent = self._mem.pop(cid, None)
                if ent and time.time() - ent[1] <= config.CAPTCHA_TTL_SEC:
                    want = ent[0]
        if want is None:
            return False
        if isinstance(want, bytes):
            want = want.decode()
        return secrets.compare_digest(got, want.upper())
