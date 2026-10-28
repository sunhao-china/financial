# -*- coding: utf-8 -*-
"""分片上传 parquet 到服务器（50MB/片，逐片重试，服务器端 pyarrow 拼装）。

用法: python tools/transfer_parquet.py data/cleaned.parquet [远程目录]
远程目录缺省 /data/financial/data
"""
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from ssh_run import connect

PART_MB = 50
REMOTE_DIR = "/data/financial/data"


def split(local: str, parts_dir: Path):
    """按 ~50MB 切成多个 parquet 分片（iter_batches 流式读，内存有界）。"""
    import pyarrow as pa
    parts_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(local)
    # 注意：不能用 pf.metadata.serialized_size（那是元数据大小，几 MB），
    # 要用文件实际字节数（实测此 bug 导致全表 1.92GB 进了单片）
    bytes_per_row = os.path.getsize(local) / max(pf.metadata.num_rows, 1)
    rows_per_part = max(1, int(PART_MB * 1e6 / max(bytes_per_row, 1)))
    files = []
    acc, acc_rows, part_idx = [], 0, 0
    for batch in pf.iter_batches(batch_size=50_000):
        acc.append(batch)
        acc_rows += batch.num_rows
        while acc_rows >= rows_per_part:
            table = pa.Table.from_batches(acc)
            part = table.slice(0, rows_per_part)
            rest = table.slice(rows_per_part)
            out = parts_dir / f"part_{part_idx:03d}.parquet"
            pq.write_table(part, out)
            files.append(out.name)
            print(f"切片 {out.name}: {part.num_rows:,} 行")
            acc = list(rest.to_batches())
            acc_rows = rest.num_rows
            part_idx += 1
    if acc_rows > 0:
        table = pa.Table.from_batches(acc)
        out = parts_dir / f"part_{part_idx:03d}.parquet"
        pq.write_table(table, out)
        files.append(out.name)
        print(f"切片 {out.name}: {table.num_rows:,} 行")
    return files


def upload(parts_dir: Path, files, remote_dir: str):
    """逐片上传，每片最多 5 次重试。"""
    cli = connect()
    sftp = cli.open_sftp()
    sftp.mkdir(remote_dir) if not _exists(sftp, remote_dir) else None
    for name in files:
        lp = parts_dir / name
        rp = f"{remote_dir}/{name}"
        for attempt in range(1, 6):
            try:
                sftp.put(str(lp), rp)
                print(f"上传 {name} ({lp.stat().st_size/1e6:.0f} MB) OK")
                break
            except Exception as e:
                print(f"上传 {name} 第 {attempt} 次失败: {e}")
                if attempt == 5:
                    raise
                time.sleep(5)
    cli.close()


def _exists(sftp, path) -> bool:
    try:
        sftp.stat(path)
        return True
    except IOError:
        return False


def reassemble(remote_dir: str, final_name: str):
    """服务器端用 pyarrow 拼装分片并校验行数。"""
    cli = connect()
    script = (
        "import glob\n"
        "import pyarrow as pa\n"
        "import pyarrow.parquet as pq\n"
        "files = sorted(glob.glob('part_*.parquet'))\n"
        "out = pa.concat_tables([pq.read_table(f) for f in files])\n"
        f"pq.write_table(out, '{final_name}')\n"
        "print('拼装完成:', len(files), '片 ->', f'{out.num_rows:,}', '行')\n"
    )
    cmd = (
        f"cd {remote_dir} && cat > _reassemble.py <<'PYEOF'\n{script}PYEOF\n"
        f"/data/financial/.venv/bin/python _reassemble.py && "
        f"rm _reassemble.py part_*.parquet"
    )
    stdin, stdout, stderr = cli.exec_command(cmd)
    print(stdout.read().decode("utf-8", "replace"))
    err = stderr.read().decode("utf-8", "replace")
    if err:
        print("STDERR:", err[:500])
    cli.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    local = sys.argv[1]
    remote_dir = sys.argv[2] if len(sys.argv) > 2 else REMOTE_DIR
    parts_dir = Path(local).parent / f"{Path(local).stem}_parts"

    print(f"== 1/3 切片: {local}")
    files = split(local, parts_dir)
    print(f"== 2/3 上传到 {remote_dir}")
    upload(parts_dir, files, remote_dir)
    print(f"== 3/3 服务器拼装 {Path(local).name}")
    reassemble(remote_dir, Path(local).name)


if __name__ == "__main__":
    main()
