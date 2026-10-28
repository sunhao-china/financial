# -*- coding: utf-8 -*-
"""P1 本地联调（大纲 §15 P1 交付标准）：四套提示词 + S2~S6 契约真机调用。

stage:
  intent    六类意图各一题 → qwen-plus 输出 S2，断言意图与预期一致
  generate  事实/归纳各一题 → deepseek-flash / deepseek-v4-pro；
            参考资料取自 cleaned.parquet 真实行（关键词匹配前 3 篇）；
            断言回答非空、引用编号全部存在于 S6
  selfcheck 充分/不足两例 → qwen-plus 输出 S3（充分例硬断言）
  verify    pass/fail 两例 → qwen-plus 输出 S4；掺入幻觉数字的答案必须判 fail

用法: python tests/test_p1_live.py [--stage intent|generate|selfcheck|verify|all]
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config
from llm_client import client
from prompts import generate as gen_p, intent_expand, self_check, verify
from questions import SMOKE
from schemas.models import S2IntentExpand, S3SelfCheck, S4Verify, S6Context, S6Reference
from schemas.parsing import fallback_s2, fallback_s3, fallback_s4, structured_call

CITE_RE = re.compile(r"\[(\d+)\]")
cost = {"total": 0}


def record(name, r):
    """打印一次调用的 token 用量并累计（§13.1）。"""
    if r is None:
        print(f"  [{name}] 两次解析均失败，已走规则兜底")
        return
    cost["total"] += r.usage.total_tokens
    print(f"  [{name}] {r.model} finish={r.finish_reason} "
          f"tokens={r.usage.total_tokens}（推理 {r.usage.reasoning_tokens}）")


def sample_refs(keyword: str, limit: int = 3, max_chars: int = 400):
    """从 cleaned.parquet 取含关键词的前 limit 篇真实文档做参考资料（S6Reference）。"""
    pf = pq.ParquetFile(config.CLEANED_PARQUET)
    refs = []
    for batch in pf.iter_batches(batch_size=50_000):
        t = pa.Table.from_batches([batch])
        mask = pc.match_substring(t.column("text"), keyword)
        if not pc.sum(mask).as_py():
            continue
        sel = t.filter(mask).slice(0, limit - len(refs))
        rows = sel.to_pydict()
        for i in range(sel.num_rows):
            refs.append(S6Reference(
                ref_no=len(refs) + 1,
                chunk_id=f"{rows['source_file'][i]}_{rows['_id'][i]}_c0",
                doc_id=str(rows["_id"][i]),
                quality_score=float(rows["quality_score"][i]),
                text=rows["text"][i][:max_chars],
            ))
        if len(refs) >= limit:
            break
    return refs


def stage_intent():
    print("== 意图识别+查询扩展（qwen-plus → S2）==")
    passed = True
    for expected, question in SMOKE.items():
        m, r = structured_call(intent_expand.build_intent_messages(question),
                               route="intent", model_cls=S2IntentExpand,
                               fallback=fallback_s2(question))
        record(f"intent:{expected}", r)
        mark = "✓" if m.intent == expected else "✗"
        if m.intent != expected:
            passed = False
        print(f"  {mark} [{expected}] {question!r} → {m.intent}（conf={m.confidence:.2f}）")
        print(f"      entities={m.entities} queries={m.queries} keywords={m.keywords}")
    return passed


def stage_generate():
    print("== RAG 生成（S6 上下文 → DeepSeek）==")
    passed = True
    cases = [
        ("fact_qa", "什么是存款准备金率？它有哪些作用？", "存款准备金率", "generate_fact", 4096),
        ("analysis", "分析人民币汇率贬值对中国经济的影响", "人民币汇率", "generate_complex", 8192),
    ]
    for intent, question, keyword, route, max_tokens in cases:
        refs = sample_refs(keyword)
        ctx = S6Context(question=question, references=refs)  # 组装 prompt 前校验（§9.4）
        messages = gen_p.build_generate_messages(question, refs, intent=intent)
        print(f"-- [{intent}] {question}（S6 校验通过，{len(refs)} 篇资料）")
        start = time.time()
        r = client.chat(messages, route=route, max_tokens=max_tokens, temperature=0.2)
        record(f"generate:{intent}", r)
        answer = r.content
        cited = sorted({int(n) for n in CITE_RE.findall(answer)})
        valid = {ref.ref_no for ref in refs}
        bad = [n for n in cited if n not in valid]
        mark = "✓" if answer and not bad else "✗"
        if not answer or bad:
            passed = False
        print(f"  {mark} 引用 {cited}（S6 编号 1~{len(refs)}，非法引用 {bad}）"
              f"耗时 {time.time() - start:.1f}s，正文 {len(answer)} 字")
        print(f"  预览: {answer[:200].replace(chr(10), ' / ')}")
    return passed


def stage_selfcheck():
    print("== 信息自检（qwen-plus → S3）==")
    passed = True
    q = "什么是存款准备金率？"
    cases = [
        ("充分（资料对题）", sample_refs("存款准备金率", limit=3), True),
        ("不足（资料无关）", sample_refs("新能源汽车", limit=3), False),
    ]
    for label, refs, expect in cases:
        m, r = structured_call(self_check.build_self_check_messages(q, ["存款准备金率是什么"], refs),
                               route="self_check", model_cls=S3SelfCheck, fallback=fallback_s3())
        record(f"selfcheck:{label[:2]}", r)
        mark = "✓" if m.sufficient == expect else "✗"
        # 充分例硬断言（资料与问题直接相关，判定应稳定）；不足例只报告（§12 评测细查）
        if expect and m.sufficient != expect:
            passed = False
        print(f"  {mark} {label}: sufficient={m.sufficient} missing={m.missing} followup={m.followup_queries}")
    return passed


def stage_verify():
    print("== 答案校验（qwen-plus → S4）==")
    passed = True
    question = "什么是存款准备金率？"
    refs = sample_refs("存款准备金率", limit=3)
    if not refs:
        print("  跳过：语料中未找到相关资料")
        return True
    # pass 例：先生成一个忠实回答（flash），再校验它自己 → 期望 pass/downgrade
    r = client.chat(gen_p.build_generate_messages(question, refs, intent="fact_qa"),
                    route="generate_fact", max_tokens=4096, temperature=0.2)
    record("generate(供校验)", r)
    answer_faithful = r.content
    m1, r1 = structured_call(verify.build_verify_messages(question, answer_faithful, refs),
                             route="verify", model_cls=S4Verify, fallback=fallback_s4())
    record("verify:pass例", r1)
    ok1 = m1.verdict in ("pass", "downgrade")
    if not ok1:
        passed = False
    print(f"  {'✓' if ok1 else '✗'} 忠实回答 → verdict={m1.verdict}"
          f" single={m1.single_source_items} unverified={m1.unverified_claims}")
    # fail 例：掺入幻觉数字（2023年/3.5%/10万亿，资料为 2022 年前旧闻）→ 必须判 fail
    answer_fake = ("根据资料，2023年央行将存款准备金率下调至3.5%，一次性释放资金约10万亿元[1]，"
                   "这是近年最大规模降准[2]。")
    m2, r2 = structured_call(verify.build_verify_messages(question, answer_fake, refs),
                             route="verify", model_cls=S4Verify, fallback=fallback_s4())
    record("verify:fail例", r2)
    ok2 = m2.verdict == "fail"
    if not ok2:
        passed = False
    print(f"  {'✓' if ok2 else '✗'} 幻觉数字答案 → verdict={m2.verdict} unverified={m2.unverified_claims}")
    return passed


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["intent", "generate", "selfcheck", "verify", "all"])
    args = ap.parse_args()
    stages = {"intent": stage_intent, "generate": stage_generate,
              "selfcheck": stage_selfcheck, "verify": stage_verify}
    names = list(stages) if args.stage == "all" else [args.stage]
    results = {}
    for name in names:
        try:
            results[name] = stages[name]()
        except Exception:
            import traceback
            traceback.print_exc()
            results[name] = False
    print(f"\n联调结果: {' / '.join(f'{k}={results[k]}' for k in names)}")
    print(f"累计 token: {cost['total']}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
