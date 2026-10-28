# -*- coding: utf-8 -*-
"""AutoDL SSH 客户端（自部署 embedding 用，2026-10-16 用户提供连接）。

用法: python tools/autodl.py "命令"
      python tools/autodl.py --put 本地文件 远程路径
      python tools/autodl.py --get 远程路径 本地文件
"""
import sys
from pathlib import Path

import paramiko

HOST = "connect.westb.seetacloud.com"
PORT = 12324
USER = "root"
PWD = "zk6EPbpHhzxF"


def connect() -> paramiko.SSHClient:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(3):
        try:
            cli.connect(HOST, port=PORT, username=USER, password=PWD, timeout=30)
            return cli
        except Exception as e:
            print(f"连接第 {attempt + 1} 次失败: {e}")
            if attempt == 2:
                raise
    return cli


def run(cmd: str, timeout: float = 600):
    cli = connect()
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    cli.close()
    if out:
        print(out, end="" if out.endswith("\n") else "\n")
    if err:
        print("STDERR:", err[:800])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) >= 3 and sys.argv[1] == "--put":
        local, remote = sys.argv[2], sys.argv[3]
        cli = connect()
        sftp = cli.open_sftp()
        sftp.put(str(Path(local)), remote)
        print(f"上传 {local} → {remote} OK")
        cli.close()
    elif len(sys.argv) >= 3 and sys.argv[1] == "--get":
        remote, local = sys.argv[2], sys.argv[3]
        cli = connect()
        sftp = cli.open_sftp()
        sftp.get(remote, str(Path(local)))
        print(f"下载 {remote} → {local} OK")
        cli.close()
    else:
        run(sys.argv[1] if len(sys.argv) > 1 else "uptime && nvidia-smi")


if __name__ == "__main__":
    main()
