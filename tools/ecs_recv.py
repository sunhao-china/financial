# -*- coding: utf-8 -*-
"""ECS 上运行：从 AutoDL 拉取文件（exec base64 流式，绕开 AutoDL 代理 SFTP 阻断）。

AutoDL 入站 SFTP 被代理阻断，但入站 SSH exec 可用：本脚本在 ECS 上 paramiko 连接
AutoDL，执行 `base64 -w0 <remote>`，流式读 stdout 解码写盘（大文件不整读内存）。

用法（ECS 上）: .venv/bin/python tools/ecs_recv.py 远程路径 [本地路径]
"""
import base64
import sys
from pathlib import Path

import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 12324
USER = "root"
PWD = "zk6EPbpHhzxF"


def main():
    remote = sys.argv[1]
    local = sys.argv[2] if len(sys.argv) > 2 else Path(remote).name
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
    stdin, stdout, stderr = cli.exec_command(f"base64 -w0 {remote}")

    buf = ""
    total = 0
    with open(local, "wb") as f:
        while True:
            chunk = stdout.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk.decode("ascii", "ignore")
            # 按行切（base64 -w0 无换行；但保险处理任意空白）
            buf = "".join(buf.split())
            if len(buf) >= 1024 * 1024:
                take = len(buf) - (len(buf) % 4)
                f.write(base64.b64decode(buf[:take]))
                total += take // 4 * 3
                buf = buf[take:]
                print(f"\r{total / 1e9:.2f} GB", end="", flush=True)
        if buf:
            f.write(base64.b64decode(buf))
            total += len(buf) // 4 * 3
    err = stderr.read().decode("utf-8", "replace")
    cli.close()
    if err.strip():
        print("\nSTDERR:", err[:300])
        sys.exit(1)
    print(f"\n完成 {total:,} bytes → {local}")


if __name__ == "__main__":
    main()
