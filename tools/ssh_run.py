# -*- coding: utf-8 -*-
"""远程执行服务器命令（paramiko 封装）。

用法: python tools/ssh_run.py "远程命令" [超时秒]
用法: python tools/ssh_put.py 本地文件 远程路径   （上传）
服务器密码在这里维护，初始化完成后可让用户改密。
"""
import os
import sys
import time

import paramiko

HOST = "8.137.192.231"
USER = "root"
try:
    from ssh_secret import PWD  # 本机凭据文件，不入库（.gitignore）
except ImportError:
    PWD = os.environ.get("FIN_SSH_PWD", "")  # 无本地文件时从环境变量取


def connect(host: str = None, user: str = None) -> paramiko.SSHClient:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for _ in range(3):  # 网络抖动重试
        try:
            cli.connect(host or HOST, username=user or USER, password=PWD, timeout=15)
            return cli
        except Exception as e:
            last = e
            time.sleep(3)
    raise last


def run(cmd: str, timeout: int = 300, host: str = None, user: str = None):
    cli = connect(host, user)
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    cli.close()
    return code, out, err


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = list(sys.argv[1:])
    host = user = None
    if "--host" in args:  # 可选目标主机（如香港轻量），缺省 ECS
        i = args.index("--host")
        host = args[i + 1]
        del args[i:i + 2]
    if "--user" in args:  # 可选登录用户（腾讯云 Ubuntu 镜像默认 ubuntu），缺省 root
        i = args.index("--user")
        user = args[i + 1]
        del args[i:i + 2]
    cmd = args[0]
    code, out, err = run(cmd, int(args[1]) if len(args) > 1 else 300, host=host, user=user)
    print(out, end="")
    if err:
        print("STDERR:", err, end="", file=sys.stderr)
    sys.exit(code)
