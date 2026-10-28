#!/bin/bash
# AutoDL 上模型下载脚本（nohup 后台执行）
export HF_ENDPOINT=https://hf-mirror.com
HF=/root/miniconda3/envs/emb/bin/hf
set -e
echo "== $(date) 开始下载 bge-m3"
$HF download BAAI/bge-m3 --local-dir /root/autodl-tmp/models/bge-m3
echo "== $(date) 开始下载 Qwen3-Embedding-0.6B"
$HF download Qwen/Qwen3-Embedding-0.6B --local-dir /root/autodl-tmp/models/qwen3-emb-0.6b
echo "== $(date) 全部完成"
echo ALL-DONE > /root/autodl-tmp/models/dl.done
