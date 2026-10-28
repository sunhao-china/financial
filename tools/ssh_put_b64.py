# -*- coding: utf-8 -*-
"""ECS SFTP 不可用时的上传：单 SSH 连接 + base64 分块（ssh_send.py 的 ECS 版）。

用法: python tools/ssh_put_b64.py 本地文件 远程路径
"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ssh_run import connect

CHUNK = 48 * 1024  # 原始字节/块（base64 后 64KB）


def main():
    local, remote = sys.argv[1], sys.argv[2]
    data = Path(local).read_bytes()
    cli = connect()

    def run_waited(cmd):
        """执行并等待完成（读尽 stdout/stderr，防止连接关闭时远端命令被 SIGHUP）。"""
        _, out, err = cli.exec_command(cmd)
        o, e = out.read(), err.read()
        if e.strip():
            print("STDERR:", e.decode("utf-8", "replace")[:300], flush=True)
        return o

    # 清理旧文件
    run_waited(f"rm -f {remote} {remote}.b64")
    b64 = base64.b64encode(data).decode()
    n_chunks = (len(b64) + CHUNK - 1) // CHUNK
    for i in range(n_chunks):
        part = b64[i * CHUNK:(i + 1) * CHUNK]
        stdin, stdout, stderr = cli.exec_command(f"echo {part} >> {remote}.b64")
        stdout.read(); stderr.read()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_chunks}", flush=True)
    run_waited(f"base64 -d {remote}.b64 > {remote} && rm {remote}.b64")
    size = int(run_waited(f"stat -c %s {remote}").decode().strip())
    ok = size == len(data)
    print(f"{local} ({len(data):,}B) → {remote} ({size:,}B) {'OK' if ok else 'SIZE MISMATCH'}")
    cli.close()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
