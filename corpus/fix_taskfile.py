# -*- coding: utf-8 -*-
"""从 p4_eval.jsonl 删除指定 id 的条目（重跑前过滤）。用法: python corpus/fix_taskfile.py id1 id2 ..."""
import json
import sys

BAD = set(sys.argv[1:])
p = "tasks/p4_eval.jsonl"
keep = []
for line in open(p, encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("id") in BAD:
        continue
    keep.append(r)
with open(p, "w", encoding="utf-8") as f:
    for r in keep:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"保留 {len(keep)} 条，删除 {sorted(BAD)}")
