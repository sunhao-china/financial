# -*- coding: utf-8 -*-
"""SFTP 不可用时的文件下载：远端 base64 -w0 整文件，一次读回解码。

用法: python tools/ssh_recv.py 远程路径 [本地文件]
"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from autodl import connect


def main():
    remote = sys.argv[1]
    local = sys.argv[2] if len(sys.argv) > 2 else Path(remote).name
    cli = connect()
    stdin, stdout, stderr = cli.exec_command(f"base64 -w0 {remote}")
    b64 = stdout.read()
    err = stderr.read().decode("utf-8", "replace")
    cli.close()
    if err.strip():
        print("STDERR:", err[:300])
        sys.exit(1)
    data = base64.b64decode(b64.strip())
    Path(local).write_bytes(data)
    print(f"{remote} ({len(data):,}B) → {local} OK")


if __name__ == "__main__":
    main()
