# -*- coding: utf-8 -*-
"""P0-1 指标分布分析（大纲 §7.2）。

只读数值列（不加载 text），输出：
- data/reports/quality_report.md —— 分位数表 + 各过滤规则保留率
- data/reports/charts/{metric}.png —— 每指标一张图（3 文件小倍数）
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# (列名, 展示名, 建议 bin 数)
METRICS = [
    ("quality_score", "quality_score 质量分", 40),
    ("perplexity", "perplexity 困惑度", 40),
    ("alnum_ratio", "alnum_ratio 字母数字占比", 40),
    ("avg_line_length", "avg_line_length 平均行长", 40),
    ("char_rep_ratio", "char_rep_ratio 字符重复率", 40),
    ("flagged_words_ratio", "flagged_words_ratio 标记词占比", 40),
    ("special_char_ratio", "special_char_ratio 特殊字符占比", 40),
    ("word_rep_ratio", "word_rep_ratio 词重复率", 40),
    ("max_line_length", "max_line_length 最长行长", 40),
    ("num_words", "num_words 词数", 40),
]
QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def load_metric_table(file: str) -> pa.Table:
    """只读数值列，不加载 text，内存友好。"""
    pf = pq.ParquetFile(config.ROOT / file)
    cols = [m[0] for m in METRICS]
    return pf.read_row_group(0, columns=cols)


def stats_table(table: pa.Table) -> dict:
    """各指标分位数表。"""
    out = {}
    for col, _, _ in METRICS:
        arr = table.column(col).to_numpy()
        arr = arr[~np.isnan(arr)]
        out[col] = {f"p{int(q*100)}": float(np.quantile(arr, q)) for q in QUANTILES}
        out[col]["n"] = len(arr)
    return out


def retention(col: np.ndarray, rule) -> float:
    """规则通过率。"""
    return float(np.mean(rule(col)))


def analyze():
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    chart_dir = config.REPORT_DIR / "charts"
    chart_dir.mkdir(exist_ok=True)

    files = config.SOURCE_FILES
    tables = {f: load_metric_table(f) for f in files}

    # 数值汇总
    lines = ["# 语料质量指标分布报告（P0）\n", f"分析时间：自动生成 · 三个文件共 "
             f"{sum(t.num_rows for t in tables.values()):,} 行\n"]
    for col, name, _ in METRICS:
        lines.append(f"\n## {name}\n")
        lines.append("| 分位数 | " + " | ".join(files) + " | 合并 |")
        lines.append("|---|" + "---|" * (len(files) + 1))
        merged = np.concatenate([tables[f].column(col).to_numpy() for f in files])
        merged = merged[~np.isnan(merged)]
        for q in QUANTILES:
            row = [f"p{int(q*100)}"]
            for f in files:
                arr = tables[f].column(col).to_numpy()
                arr = arr[~np.isnan(arr)]
                row.append(f"{np.quantile(arr, q):.4g}")
            row.append(f"{np.quantile(merged, q):.4g}")
            lines.append("| " + " | ".join(row) + " |")

    # 过滤规则保留率
    rules = {
        "quality_score ≥ 3.5": lambda a: a >= 3.5,
        "quality_score ≥ 4.0": lambda a: a >= 4.0,
        "perplexity ≤ 3000": lambda a: a <= 3000,
        "perplexity ≤ 2000": lambda a: a <= 2000,
        "char_rep_ratio ≤ 0.1": lambda a: a <= 0.1,
        "flagged_words_ratio = 0": lambda a: a == 0,
    }
    lines.append("\n## 各过滤规则保留率（用于定阈值，大纲 §7.2）\n")
    lines.append("| 规则 | " + " | ".join(files) + " | 合并 |")
    lines.append("|---|" + "---|" * (len(files) + 1))
    for name, rule in rules.items():
        col = name.split()[0]
        row = [name]
        for f in files:
            arr = tables[f].column(col).to_numpy()
            arr = arr[~np.isnan(arr)]
            row.append(f"{retention(arr, rule)*100:.1f}%")
        merged = np.concatenate([tables[f].column(col).to_numpy() for f in files])
        merged = merged[~np.isnan(merged)]
        row.append(f"{retention(merged, rule)*100:.1f}%")
        lines.append("| " + " | ".join(row) + " |")

    (config.REPORT_DIR / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")

    # 图表：每指标一张，3 文件小倍数（同一测度同一色相，身份靠子图标题）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
    BAR, ANNOT = "#2a78d6", "#104281"

    for col, name, bins in METRICS:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
        fig.patch.set_facecolor(SURFACE)
        fig.suptitle(name, color=INK, fontsize=12, y=1.02)
        for ax, f in zip(axes, files):
            arr = tables[f].column(col).to_numpy()
            arr = arr[~np.isnan(arr)]
            lo, hi = np.quantile(arr, [0.005, 0.995])
            ax.hist(arr, bins=bins, range=(lo, hi), color=BAR)
            for q, qs in [(0.05, "p5"), (0.50, "p50"), (0.95, "p95")]:
                v = np.quantile(arr, q)
                ax.axvline(v, color=ANNOT, lw=1, ls="--")
                ax.text(v, ax.get_ylim()[1] * 0.92, f" {qs} {v:.3g}", color=ANNOT, fontsize=8)
            ax.set_title(f.split("_")[1], color=INK, fontsize=10)
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ax.spines.values():
                s.set_color(GRID)
        fig.tight_layout()
        fig.savefig(chart_dir / f"{col}.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print(f"chart saved: {col}.png")


if __name__ == "__main__":
    analyze()
    print(f"报告: {config.REPORT_DIR / 'quality_report.md'}")
