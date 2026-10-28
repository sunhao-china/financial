# -*- coding: utf-8 -*-
"""从服务器下载文件（paramiko SFTP）。

用法: python tools/ssh_get.py 远程文件 [本地路径]
远程文件按相对路径解析为 /data/financial/远程文件
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ssh_run import connect

REMOTE_DIR = "/data/financial"


def get(remote: str, local: str = None):
    rp = remote if remote.startswith("/") else f"{REMOTE_DIR}/{remote}"
    lp = Path(local) if local else Path(Path(rp).name)
    lp.parent.mkdir(parents=True, exist_ok=True)
    cli = connect()
    sftp = cli.open_sftp()
    sftp.get(rp, str(lp))
    cli.close()
    size = lp.stat().st_size
    print(f"已下载 {rp} ({size/1e6:.1f} MB) → {lp}")


if __name__ == "__main__":
    get(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
