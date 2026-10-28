# -*- coding: utf-8 -*-
"""上传文件到服务器（paramiko SFTP）。

用法: python tools/ssh_put.py 本地文件 [远程路径]
远程路径缺省为 /data/financial/本地文件名
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ssh_run import connect

REMOTE_DIR = "/data/financial"


def put(local: str, remote: str = None, host: str = None, user: str = None):
    lp = Path(local)
    rp = remote or f"{REMOTE_DIR}/{lp.name}"
    cli = connect(host, user)
    sftp = cli.open_sftp()
    sftp.put(str(lp), rp)
    cli.close()
    size = lp.stat().st_size
    print(f"已上传 {local} ({size/1e6:.1f} MB) → {rp}")


if __name__ == "__main__":
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
    put(args[0], args[1] if len(args) > 1 else None, host=host, user=user)
