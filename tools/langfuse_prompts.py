# -*- coding: utf-8 -*-
"""Langfuse 提示词注册/同步（2026-10-22，ECS 上运行）。

把 prompts/*.py 的 5 套 SYSTEM 模板文本注册为 Langfuse Prompts
（名 financial-rag/<name>，label=production），此后生产走远端版本、
本地 SYSTEM 自动降为兜底（prompts/remote.py）。

模式：
  .venv/bin/python tools/langfuse_prompts.py --list     列出本地与远端 production 版本比对
  .venv/bin/python tools/langfuse_prompts.py --register 同步（本地≠远端 production → 推新版本并打 production）
      [--force]   强制推新版本（即使文本一致，如远端配置/标签被误改）

设计要点：
- 比对以「文本」为准：本地 SYSTEM != 远端 production 文本 → create_prompt（新版本）
  并打 production 标签；一致 → 跳过。幂等，中断安全。
- 只迁模板文本本身；组装逻辑/FEWSHOT/INTENT_GUIDE/guard 正则/refusals 话术留代码。
- 推送只影响「prompts/remote.py 生效后」的进程（最多 5 分钟缓存延迟），
  老代码读本地 SYSTEM 不受任何影响 → 可先行注册再发代码。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402  （config.py 自带 /data/financial/.env 加载）

from prompts import remote  # noqa: E402

TAGS = ("prod", "rag", "financial")


def lf_client():
    """Langfuse 客户端（env 来自 .env；SDK 缺失/未启用时直接报错——本工具必须开启）。"""
    if not (config.LANGFUSE_ENABLED and config.LANGFUSE_PUBLIC_KEY):
        raise SystemExit("LANGFUSE_ENABLED/PUBLIC_KEY 未配置（查 /data/financial/.env）")
    from langfuse import Langfuse  # noqa: PLC0415 — 可选依赖
    return Langfuse(public_key=config.LANGFUSE_PUBLIC_KEY,
                    secret_key=config.LANGFUSE_SECRET_KEY,
                    host=config.LANGFUSE_HOST)


def remote_prod_text(lf, name: str):
    """远端 production 版文本；不存在/无标签 → None。"""
    try:
        p = lf.get_prompt(f"{remote.PREFIX}{name}", label="production",
                          fetch_timeout_seconds=remote.FETCH_TIMEOUT_SEC)
        text = getattr(p, "prompt", "")
        return text if text and text.strip() else None
    except Exception:  # noqa: BLE001 — 提示词不存在时 SDK 抛错
        return None


def list_mode() -> int:
    lf = lf_client()
    print(f"{'名称':<30} 本地字符  远端production  状态")
    for name in remote.NAMES:
        local = remote._default_text(name)
        rp = remote_prod_text(lf, name)
        if rp is None:
            status = "❌ 远端无 production 版（待 --register）"
        elif rp.strip() == local.strip():
            status = "= 一致"
        else:
            status = "≠ 不一致（待 --register 推新版）"
        print(f"{remote.PREFIX + name:<30} {len(local):>6}  "
              f"{len(rp) if rp else '-':>9}  {status}")
    return 0


def register(args) -> int:
    lf = lf_client()
    pushed, skipped = 0, 0
    for name in remote.NAMES:
        local = remote._default_text(name)
        rp = remote_prod_text(lf, name)
        if not args.force and rp is not None and rp.strip() == local.strip():
            print(f"  = {remote.PREFIX}{name} 与远端 production 一致，跳过", flush=True)
            skipped += 1
            continue
        # 文本不一致/远端缺失/--force → 推新版本（Langfuse 自动版本号 +1）并打 production
        p = lf.create_prompt(
            name=f"{remote.PREFIX}{name}", prompt=local, type="text",
            labels=["production"], tags=list(TAGS),
            config={"source": "prompts/" + name + ".py SYSTEM", "chars": len(local)})
        print(f"  + {remote.PREFIX}{name} → 版本 v{p.version}（production）", flush=True)
        pushed += 1
    print(f"完成：推送 {pushed} 套，跳过 {skipped} 套。"
          f"生产进程最多 {remote.CACHE_TTL_SEC // 60} 分钟后生效（SDK 缓存）。", flush=True)
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Langfuse 提示词注册/同步")
    ap.add_argument("--list", action="store_true", help="列出本地 vs 远端 production 比对")
    ap.add_argument("--register", action="store_true", help="同步（不一致才推新版本并打 production）")
    ap.add_argument("--force", action="store_true", help="强制推新版本（即使文本一致）")
    args = ap.parse_args()
    if args.list:
        return list_mode()
    if args.register:
        return register(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
