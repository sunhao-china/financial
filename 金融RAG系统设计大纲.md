# 金融问答 RAG 系统 — 设计大纲 v6

> 数据基础：3 个 parquet 文件（rank_00045/46/47），共约 61.4 万篇中文金融/经济文章，
> 每篇自带质量指标（perplexity、quality_score、各类 ratio），industry_type 均为「金融_经济」。
>
> v6 变更：生成模型从百炼 qwen-max 改为 **DeepSeek deepseek-v4-pro**（推理模型，已实测可用），
> deepseek-flash 作降级；系统升级为「百炼 + DeepSeek」双平台架构（内部工序走百炼，最终生成走 DeepSeek）。
> chunk 切分定版、embedding 全量调用、入向量库 —— 暂缓，待服务器确定后执行。

---

## 1. 目标与定位

- **定位**：基于语料库的中文金融领域问答系统（历史知识库型，非实时行情）。
- **问答类型**：事实检索类、归纳分析类、多文档综合类、对比类。
- **能力边界**：语料时效约到 2022 年；不答实时行情、不构成投资建议。
- **成功指标**：检索 recall@10 ≥ 0.8；答案忠实度 ≥ 0.9；端到端延迟 ≤ 3s。

## 2. 系统流程总览

```
用户提问
  → ① 意图识别（百炼 qwen-turbo，JSON 结构化输出）
       ├─ 实时/个股推荐 → 合规拒答（预留实时工具扩展点，见 §6.2）
       ├─ 闲聊/无关     → LLM 直接回答，不检索
       └─ 事实/分析/对比 → 进入 RAG
  → ② 多查询扩展（百炼 LLM 生成 3~5 个相似问题变体）
       + 关键词抽取（金融实体：公司名/政策名/指标名）
  → ③ 混合检索：每个变体 → 向量检索 top-50 + 关键词(BM25)检索 top-50
  → ④ 合并去重（按 chunk id）→ 百炼 rerank（gte-rerank）→ top-3~5
  → ⑤ 信息自检（轻量 Agent 循环，见 §6）：检索结果是否足够回答？
       ├─ 足够 → ⑥ 生成
       └─ 不足 → 生成补充查询 → 回到 ③（补检轮数由钩子强制限制，见 §6.1）
  → ⑥ LLM 生成（按意图路由：事实类 flash / 归纳对比类 v4-pro）：结论 + 依据 + 引用编号 [1][2]
  → ⑦ 答案验证关（见 §6.3）：来源可靠性 + 数字/日期可溯源校验，通过才回复
  → ⑧ SSE 流式返回，附来源文档列表
```

## 3. 模型接口方案（百炼 + DeepSeek 双平台，全链路云端 API）

> 关键决策：embedding / rerank / 内部工序 LLM 走百炼 API，**最终答案生成走 DeepSeek API**。
> 全部云端调用，**服务器不需要 GPU**，只需 CPU + 大内存 + 大磁盘承载向量库与检索服务。
> API 调通与封装**不依赖服务器**，本地开发机现在即可完成。

| 环节 | 模型 | 平台 | 用途 | 说明 |
|---|---|---|---|---|
| 意图识别 | qwen-turbo | 百炼 | 分类+路由 | 便宜快速，JSON 结构化输出（受限时用 qwen-plus 替代） |
| 查询扩展/关键词抽取 | qwen-plus | 百炼 | 生成查询变体+实体词 | 与意图识别可合并为一次调用 |
| 信息自检（Agent） | qwen-plus | 百炼 | 判断检索是否充分 | 见 §6 |
| Embedding | **bge-m3（MRL 截断 512 维）** | **自部署**（AutoDL 4090 离线全量 / ECS CPU 在线查询，同一模型文件） | 文档与查询向量化 | ✅ 2026-10-16 部署完成，模型文件 sha256 锁定（tools/model_lock.json），全量 GPU 实测 430/s；备选 qwen3-emb-0.6b（对比试验见 §13.1） |
| Embedding（备选） | qwen3-emb-0.6b（512 维） | 自部署 | 对比试验略胜 bge-m3（recall@10 +1.4p）但吞吐仅 149/s | 两模型文件均已在 AutoDL 锁定 |
| Rerank | **qwen3-rerank** | 百炼 | 精排 top-3~5 | ✅ 已实测可用；备选 gte-rerank-v2 |
| **生成（事实类）** | deepseek-flash | DeepSeek | fact_qa 默认生成 | ✅ 实测：简单合成 2.2s，质量与 v4-pro 持平 |
| **生成（归纳/对比类）** | deepseek-v4-pro | DeepSeek | 复杂多文档综合 | ✅ 实测：4.4s；需要深度推理时启用 |
| 生成降级 | deepseek-flash | DeepSeek | v4-pro 预算/故障时降级 | ✅ 已实测 |
| 内容安全 | 百炼内容安全接口 | 百炼 | 输入输出过滤 | 见 §13.4 |

- 统一封装一个 `llm_client.py`（双平台）：鉴权、重试、限流、超时、**usage（token 用量）回传**，全系统共用。
- 具体模型名与版本以各平台控制台实际可用为准，切换模型只改配置不改代码。
- **推理模型注意事项**（deepseek-v4-pro / deepseek-flash）：
  - 回答前消耗 reasoning tokens（**两个模型推理 token 量级相近，实测均约 210，换小模型省不了推理，省的是单价与速度**；
    flash 2.2s vs v4-pro 4.4s），成本与延迟都要把推理算进去；
  - **按意图路由生成模型**（实测结论）：fact_qa 事实类 → flash（快一倍、质量持平）；
    analysis/comparison 归纳对比类 → v4-pro（需要深度推理）；评测后可配置翻转；
  - SSE 流式时推理内容在 `reasoning_content` 字段，正文在 `content` 字段，前端可展示「思考中」；
  - max_tokens 要给足（推理+正文），否则推理吃光预算导致正文为空（实测 50 tokens 时正文为空）；
  - 推理模型不做内部工序（意图/自检/校验），这些环节要求快、便宜、严格 JSON 输出，继续用百炼 qwen-plus。
- 成本预估：全量 embedding 一次性成本为主（自部署后 ≈0，见 §13.1）；在线成本 = 百炼工序调用 + DeepSeek 生成（含推理 token）。

**Embedding 自部署（2026-10-16 落地，用户决策：为保证离线建库与在线查询版本一致）**：

| 项 | 内容 |
|---|---|
| 部署位置 | AutoDL RTX 4090 48G（离线全量）；在线查询用 ECS CPU + sentence-transformers 加载**同一份模型文件** |
| 模型文件 | bge-m3 + qwen3-emb-0.6b，ModelScope 下载，sha256 锁定于 `tools/model_lock.json` |
| 推理管线 | sentence-transformers 6.1.0 + torch fp16（GPU）/ fp32（CPU），`truncate_dim=512`（MRL 截断）+ normalize |
| 实测吞吐 | bge-m3 fp16 batch256 = 430~726 chunks/s（全量 297 万 ≈ 2 小时）；qwen3-emb = 149/s；vLLM 0.31 pooling = 423/s（XLM-R flash-attn 差，弃用） |
| 一致性约定 | 离线/在线同一模型文件（sha256 校验）+ 同一库 + 同一截断/归一化参数；换模型重灌时 `embedding_model` 字段区分 |
| 一致性实测 | GPU fp16 vs CPU fp32（同一文件，200 样本）：余弦 min=0.99986/中位=1.00018；top-10 检索 197/200 完全一致 → 离线 GPU 建库 + 在线 CPU 查询成立 |

**接口连通性测试结果（2026-10-16）**：

| 模型 | 平台 | 状态 | 说明 |
|---|---|---|---|
| text-embedding-v3 | 百炼 | ✅ 可用 | 实测 1024 维，与 §8 维度方案一致 |
| qwen-plus | 百炼 | ✅ 可用 | 内部工序主力模型 |
| qwen-turbo | 百炼 | ⚠️ 429 限流 | 意图识别先用 qwen-plus 替代 |
| qwen-max | 百炼 | ❌ 403 额度耗尽 | 不再使用（生成已改 DeepSeek） |
| **qwen3-rerank** | 百炼 | ✅ 可用 | **已选定为 rerank 模型** |
| gte-rerank-v2 | 百炼 | ✅ 可用 | 备选 |
| **deepseek-v4-pro** | DeepSeek | ✅ 可用 | **已选定为生成模型**；推理模型，实测简单问答 210 推理 token / 6.7s |
| deepseek-flash | DeepSeek | ✅ 可用 | 生成降级模型；实测 2.4s |

- 待办：无需任何控制台开通或充值，当前模型组合即可开发全链路。

## 4. 意图识别与路由

| 意图 | 示例 | 路由 |
|---|---|---|
| `fact_qa` 事实检索 | 「2021 年综合保税区进出口值是多少？」 | RAG |
| `analysis` 归纳分析 | 「总结央企海外并购的主要问题」 | RAG（多文档，常触发 Agent 补检） |
| `comparison` 对比 | 「人民币破7 与破8 有何不同影响？」 | RAG（常触发 Agent 补检） |
| `realtime` 时效性 | 「今天 A 股涨了多少？」 | V1 拒答（语料截止声明）；V2 走实时工具，见 §6.2 |
| `investment_advice` 个股推荐 | 「买什么股票好？」 | 合规拒答 + 免责声明 |
| `chitchat` 闲聊 | 「你好」 | 直接回答，不检索 |

- 实现：百炼 LLM few-shot 分类，`response_format: json_object` 输出，解析后按 S2 结构强制校验（见 §9）。
- 兜底规则：关键词正则（「今天/最新/涨跌/买入/推荐」）先行拦截，LLM 置信度低时默认走 RAG（宁检索不漏）。
- 意图识别与查询扩展可**合并为一次 LLM 调用**输出一个 JSON，省延迟省成本：
  ```json
  {
    "schema_version": "1.0.0",
    "intent": "fact_qa",
    "confidence": 0.95,
    "entities": ["综合保税区", "2021年", "进出口值"],
    "queries": ["2021年全国综合保税区进出口值是多少", "2021年综合保税区进出口数据", "综合保税区进出口额统计"],
    "keywords": ["综合保税区", "进出口值", "2021"]
  }
  ```

## 5. 检索层

1. **多查询扩展**：原始问题 + LLM 生成的 3~5 个相似变体，逐一检索，扩大召回面。
2. **混合检索**：
   - 向量检索：每个变体 top-50（embedding 走百炼）。**P6（2026-10-21）向量路由进程内 hnswlib 迁至独立 Qdrant 向量库**（286 万 chunk 向量 + 8 列元数据 payload 全量入 collection，point id = chunk 行号，融合对齐契约不变；查询侧 bge-m3 编码仍在进程内，cosine/HNSW m=16 ef_construct=200/查询 ef=100 与旧索引一致，64 题回归 recall@10 0.158/MRR 0.625 vs 基线 0.159/0.626）。
   - 关键词检索：抽取的实体词 → BM25（jieba 分词 + 金融自定义词典），top-50。
   - 融合：RRF 合并，按 chunk id 去重。
3. **重排序**：合并结果送百炼 gte-rerank 精排，取 top-3~5 进上下文。
4. **元数据过滤**：quality_score 下限、industry_type、data_version。
5. **热点缓存**：高频查询结果缓存（Redis）。

## 6. Agent 模式与实时数据预留

### 6.1 轻量 Agent 循环 + 钩子机制（要加，但不用完整 ReAct）

- **结论：加，但做成「受限两段式」，不做无限 ReAct 循环；限制条件通过钩子（hook）在代码层强制，不能只写在提示词里。**
- 理由：归纳/对比/多跳类问题单轮检索经常召回不全；但完整 ReAct 每步都带工具调用，
  延迟和成本翻几倍，还可能出现死循环。
- 设计：
  - **单轮快路径**：意图为事实检索且自检充分 → 直接生成。大多数查询走这里，延迟最低。
  - **慢路径**：自检不足 → 生成补充查询 → 再检索 → 合并去重 → 生成。
  - 自检实现：qwen-plus 一次调用，输出 S3 结构（§9.2），提示词见 §10.3。
- **钩子机制（对 ReAct 状态做代码级限制）**，循环每一步（检索/自检/补检/生成/验证）都经过钩子：

| 钩子 | 时机 | 强制规则 |
|---|---|---|
| `pre_step` | 每步执行前 | ① 补检轮数 > 2 → 强制终止进入生成；② 累计 token 成本超预算 → 终止；③ 新查询与历史查询重复（归一化后相似）→ 丢弃该步 |
| `post_step` | 每步执行后 | 记录步骤结果到会话记忆（S7 调用路径快照，见 §11）；更新轮数与成本计数 |
| `pre_reply` | 回复客户前 | 答案必须通过来源验证关（§6.3），未通过 → 不允许出站，走降级话术 |

- 每个 hook 返回 `{allow: bool, reason, action}`，`action ∈ {continue, terminate, fallback}`；
  任何一步被拦都走降级路径，保证**响应有界**（最坏情况：终止循环 → 用已有检索结果生成 → 验证失败则降级话术）。

### 6.2 实时数据：V1 不加，架构预留

- **V1 不加实时数据源**，理由：
  - 语料是离线历史知识库，混合实时源会让引用溯源复杂化（实时数据没有 doc_id / chunk 体系）；
  - 行情源（akshare / tushare / 付费 API）的稳定性、数据合规是独立工程；
  - 时效问题 V1 用话术解决：拒答 + 声明知识截止日期。
- **架构预留（V2 平滑接入）**：
  - 定义统一 tool 接口：`get_realtime_data(query) -> {content, source, timestamp}`，
    未来接 akshare 等只需实现一个适配器；
  - 意图路由的 `realtime` 分支升级为「有工具时走工具，无工具时拒答」；
  - Agent 循环（§6.1）天然支持 tool 调用，实时工具接入后自动生效。
- V2 接入顺序：先接免费源（akshare）验证 → 再按需上付费行情/资讯 API。

### 6.3 来源可靠性验证（回复客户前的强制关）

> 定位：生成答案后、出站回复前的最后一道闸门，由 `pre_reply` 钩子强制，
> 未通过一律不允许回复客户。金融数字说错比不答危害大。

三道检查（规则 + LLM 双重校验）：

1. **来源质量检查**：引用的 chunk 是否满足 `quality_score` 下限、`data_version` 为当前批次；
   剔除低质来源后重排。
2. **交叉验证**：关键结论（金额/比例/年份/政策）至少 2 个**不同 doc_id** 来源一致；
   只有单一来源时允许回复，但答案中必须明确标注「该信息仅来自单一来源，可靠性有限」。
3. **引用校验**：答案中每个数字/日期必须能从引用片段原文中定位
   （正则抽取数字 → LLM 校验归属），对不上的结论必须删除或标注。

- 校验实现：qwen-plus 一次调用，输出 S4 结构（§9.2），提示词见 §10.4。
- 处理策略：`pass` → 正常回复；`downgrade` → 附加可靠性说明后回复；
  `fail` → 重生成 1 次 → 仍失败 → 降级回复「现有资料不足以确认该信息」+ 返回相关片段。

## 7. 语料工程（数据层）

### 7.1 读取与合并
- pyarrow 逐文件读，统一 schema，加 `source_file` 来源列，输出合并中间表。

### 7.2 质量过滤（零成本，直接用现成指标）
- 先画各指标分布图再定阈值，初始规则建议：
  - `quality_score` ≥ 3.5（样本集中 4.0~4.3）
  - `perplexity` ≤ 3000、`char_rep_ratio` ≤ 0.1、`flagged_words_ratio` = 0
- 预计保留 80%~95%。

### 7.3 去重与清洗
- 三级流水（P0 已实现，dedup_corpus.py）：
  1. 精确去重（原文 md5）
  2. 规范去重（空白折叠后 md5）——转载最常见的差异就是空白/换行
  3. 近重复：字符 4-gram MinHash + **精确 Jaccard ≥ 0.8**（minhash 仅用于排序，窗口 ±100 比对，并查集合并保留最早文件）
- **为什么不用 simhash（P0 实测教训，两次翻车）**：① 裸词频下中文高频功能字锁死 bit，随机无关对海明均值只有 17.9（理论 32），大量误合并；② idf 加权 + bit 中心化后分布回到 32，但金融语料主题聚集，「真转载」与「同主题异文」在海明 1~3 区间混杂（实测 d=2/3 约半数误合并），阈值无法分离。Jaccard 方案实测：随机对 p99=0.009，真转载 ≥0.86，间隔清晰。
- **任何去重参数改动必须先跑 validate_jaccard.py**（3000 篇抽样：随机对 Jaccard 分布 + J≥0.8 边界对抽查 + 窗口召回率），通过才允许全量。
- 去文章来源头；**保留数字与单位**（金融数据是核心）；换行符、全半角规范化。

### 7.4 文档切分（P2 完成，切分器 v1.0）
- 策略：段落 → 超长按句边界，chunk 300~500 字，重叠 10%~15%。
- **切分器已实现并本地验证**（corpus/chunk_corpus.py + corpus/validate_chunking.py 验证关卡）：
  - 位置法切分（buf 只存原文区间，chunk 文本 = text[char_start:char_end] 精确切片，
    不变量全量验证：偏移精确 0 违例 / 覆盖完整 0 违例 / 相邻重叠恰好 50 字）；
  - 句边界含半角 `.` 后跟非数字（金融语料大量用 . 代句号；8.11 式小数点排除）；
  - 参数 v1.0：CHUNK_MAX=500 / CHUNK_MIN=300 / OVERLAP=50，chunking_method=paragraph_sentence_500_50。
  - 试验数据（前 3000 篇）：19,147 chunks，平均 6.4 chunk/篇，长度 min=58 / 中位=466 / p95=543 / max=549。
  - **任何切分参数改动必须先跑 validate_chunking.py**（与 §7.3 去重同一验证关要求）。
- 切分参数 vs 检索 recall 的对照试验，在服务器上建小索引后补做（见 P2）。
- **规模预估（切分器实测后更新）**：463,695 篇 × 6.4 ≈ **297 万 chunks**——
  比原估 140~185 万高约 70%（原按 3~4 chunk/篇，实际平均文本 2465 字）。
  1024 维 float32 向量约 12.2 GB，HNSW 索引 18~24 GB，**32G 内存已超预算**，方案见 §8 更新。

## 8. 向量库数据结构（S1，标准 JSON schema）

> 设计原则：**一 chunk 一条记录，`text` 即 chunk 原文片段（也就是转成向量前的文本）**，
> 不重复存整篇原文；需要上下文时用 `doc_id + chunk_index` 取相邻 chunk 扩展。
> 所有字段带版本标记，支持将来重切/换模型重灌时新旧数据共存对比。

```json
{
  "id": "rank_00045_200200000053_c0",
  "schema_version": "1.0.0",
  "text": "东方网4月10日消息:今天,人民币对美元汇率中间价今天报 6.9920,首次从“破7”进入“6”关口……（chunk 原文片段）",
  "embedding": [0.0123, -0.0045, "……1024 维"],
  "metadata": {
    "doc_id": "200200000053",
    "source_file": "rank_00045",
    "industry_type": "金融_经济",
    "quality_score": 4.109375,
    "perplexity": 1875.1,
    "num_words": 478,
    "chunk_index": 0,
    "chunk_count": 4,
    "char_start": 0,
    "char_end": 462,
    "chunking_method": "paragraph_sentence_500_overlap50",
    "chunking_version": "v1.0",
    "embedding_model": "text-embedding-v3",
    "embedding_dim": 1024,
    "data_version": "2026-10-16",
    "ingested_at": "2026-10-16T12:00:00+08:00"
  }
}
```

字段说明：

| 层级 | 字段 | 说明 |
|---|---|---|
| 顶层 | `id` | 唯一 chunk id：`{source_file}_{doc_id}_c{index}`，天然可溯源 |
| 顶层 | `schema_version` | JSON 结构版本，字段将来变更时的兼容开关 |
| 顶层 | `text` | chunk 原文片段（即转向量前的文本），检索与直接展示用 |
| 顶层 | `embedding` | 向量，入向量库时生成 |
| metadata | `doc_id` / `source_file` | 溯源到原文件与原文 |
| metadata | `industry_type` | 过滤维度（预留多行业扩展） |
| metadata | `quality_score` / `perplexity` | 质量指标，过滤与排序 |
| metadata | `chunk_index` / `chunk_count` | 分块定位；生成时可取相邻 chunk 扩上下文 |
| metadata | `char_start` / `char_end` | 在原文中的字符偏移 |
| metadata | `chunking_method` / `chunking_version` | 切分方案版本——换切分参数重切时新旧数据可区分对比 |
| metadata | `embedding_model` / `embedding_dim` | 向量版本——换 embedding 模型重灌时区分 |
| metadata | `data_version` | 语料批次版本（过滤规则变更时递增） |
| metadata | `ingested_at` | 入库时间戳 |

**维度选择**（2026-10-16 按切分实测 297 万 chunks 更新）：

| 方案 | 向量原始体积 | HNSW 索引内存 | 检索质量 |
|---|---|---|---|
| text-embedding-v3 **1024 维** | ≈ 12.2GB | ≈ 18~24GB | 基准 |
| text-embedding-v3 **768 维** | ≈ 9.1GB | ≈ 14~18GB | 略降 |
| text-embedding-v3 **512 维** | ≈ 6.1GB | ≈ 9~12GB | 略降 |
| 1024 维 + 量化（SQ8/PQ 或 diskann/mmap） | 12.2GB（索引压缩后 ~3~6GB） | 磁盘为主 | 略降 |

- 297 万 chunks 下，1024 维 + HNSW 全内存 ≈ 34~40GB（向量 12 + 索引 18~24 + 文本/元数据 ~4），
  **超 32G 服务器预算**。
- **V1 方案（推荐）：512 维 + HNSW**：6.1 + 9~12 + ~4 ≈ **19~22GB**，32G 内存余量充足。
  text-embedding-v3 原生支持 512 维输出（1024/768/512/256 可配），精度损失通常很小。
- 备选：1024 维 + 量化索引（Milvus SQ/PQ 或 Qdrant mmap+量化），内存 ≈ 16~20GB，
  精度略降；若 P2 检索评估显示 512 维 recall 不达标，再走这条升级路径
  （`embedding_dim` 版本字段为此预留，新旧数据可共存对比）。
- 2048 维（v4）在 32G 服务器上不再考虑；若后续上 64G 服务器，作为质量升级路径。

**存储估算**（512 维方案）：297 万 chunks ×（向量 2KB + text ≈ 1.2KB + 元数据 ≈ 0.5KB）≈ **11GB 数据**，
加索引与系统开销，单机 SSD + 32G 内存可承载。

**版本策略（要，但只留三个关键版本位）**：`schema_version`、`chunking_version`、`embedding_model`。
三者任一变更 → 重切/重灌时写入新值，检索时按最新版本过滤，旧数据保留可回滚对比。

## 9. 全链路 JSON 契约（关键节点标准结构）

> 核心原则：**记忆、中间状态、LLM 输入输出、存储、日志——所有关键节点的数据结构
> 一律集中定义、带版本、运行时强制校验。禁止各环节随意拼结构去存储或喂给大模型，
> 任何不合规的数据不得进入下一环节。**

### 9.1 管理方式

- **集中定义**：所有结构在 `schemas/` 目录统一维护（JSON Schema 格式），随代码入库版本管理。
- **强制校验**：运行时用 pydantic 模型校验（与 JSON Schema 双向同步），
  每个结构顶层都带 `schema_version` 字段。
- **拒绝带病流转**：解析/校验失败 → 拒绝进入下一环节 → 按 §13.2 异常处理走兜底路径；
  存储层读取时校验失败的数据标记为坏数据隔离，不得参与检索与生成。

### 9.2 结构清单（关键节点一览）

| # | 节点 | 结构定义处 | 校验时机 |
|---|---|---|---|
| S1 | 向量库入库记录 | §8 | 写入向量库前 |
| S2 | 意图识别+查询扩展输出 | §4 示例 | LLM 输出解析后 |
| S3 | 信息自检输出 | §6.1 | LLM 输出解析后 |
| S4 | 答案校验输出 | §6.3 | LLM 输出解析后 |
| S5 | 检索结果条目 | §9.3 | 检索服务返回时 |
| S6 | 喂给生成 LLM 的上下文 | §9.4 | 组装 prompt 前 |
| S7 | 会话记忆·调用路径快照 | §11 | post_step 钩子写入时 |
| S8 | 日志/链路事件 | §13.5 | 日志落盘前 |
| S9 | API 请求/响应与 SSE 事件 | §14 | 网关层 |

- S2/S3/S4 为 LLM 结构化输出：`response_format: json_object` + 顶层 `schema_version` 必填，
  解析后 pydantic 校验，失败重试 1 次 → 兜底（§13.2）。
- S7/S8 为存储类：写入与读取双向校验；坏数据隔离，禁止带病流转。

### 9.3 检索结果条目（S5）

```json
{
  "schema_version": "1.0.0",
  "chunk_id": "rank_00045_200200000358_c0",
  "doc_id": "200200000358",
  "text": "……（chunk 文本）",
  "score": 0.87,
  "rank_source": "fusion",
  "quality_score": 4.09,
  "chunk_index": 0,
  "chunk_count": 4
}
```

- `rank_source ∈ {vector, bm25, fusion}`；`score` 为融合后归一化分数。
- 检索服务所有路径（向量/关键词/合并/缓存命中）返回前统一转成 S5，下游只认这一种结构。

### 9.4 喂给生成 LLM 的上下文（S6）

```json
{
  "schema_version": "1.0.0",
  "question": "2021 年综合保税区进出口值是多少？",
  "references": [
    {"ref_no": 1, "chunk_id": "rank_00045_200200000358_c0", "doc_id": "200200000358", "quality_score": 4.09, "text": "……"},
    {"ref_no": 2, "chunk_id": "rank_00045_200200000612_c1", "doc_id": "200200000612", "quality_score": 4.17, "text": "……"}
  ],
  "history": [{"role": "user", "content": "……"}, {"role": "assistant", "content": "……"}]
}
```

- 规则：`ref_no` 从 1 连续编号；生成提示词要求 LLM **只允许引用存在的 ref_no**；
  pydantic 校验通过后才发送；校验器（§6.3）以同一 S6 结构核对答案引用。

## 10. 提示词工程

> 所有提示词集中管理在 `prompts/` 目录（YAML/模板文件），版本化、可评测，
> 每次修改跑离线评测集回归。

1. **意图识别 + 查询扩展提示词**（qwen-plus，一次调用）
   - 任务：给定对话历史与当前问题，输出 S2 结构（§9.2），顶层必含 `schema_version`。
   - few-shot：每类意图 1~2 个示例，示例取自真实语料问题。
   - 约束：queries 3~5 个、语义角度互不重复；keywords 只输出金融实体词。

2. **RAG 生成提示词**（按意图路由：flash / v4-pro，均为推理模型）
   - 系统人设：「你是严谨的金融信息助手」。
   - 硬约束：**只依据检索片段回答**；检索内容不足时明确说「现有资料中未提及」；禁止编造数字与日期。
   - 引用格式：每句结论后标 `[1][2]`，ref_no 与输入 S6 结构严格对应，不得引用不存在的编号。
   - 输出结构：结论先行 → 分点依据 → 来源引用 → 必要免责声明（不构成投资建议）。
   - 数字敏感性：涉及金额/比例/年份的结论必须能从片段原文直接定位。

3. **信息自检提示词**（qwen-plus，Agent 循环用）
   - 输入：原始问题 + 当前检索片段（S5 列表）。
   - 输出：S3 结构 `{schema_version, sufficient, missing, followup_queries}`；
     约束 followup_queries ≤ 2 个且必须与已检索查询不同。
   - 防误判：只要片段中存在直接相关的信息即判 sufficient，避免无意义补检。

4. **答案校验提示词**（qwen-plus，来源验证关用）
   - 输入：待回复答案 + S6 引用块列表。
   - 输出：S4 结构 `{schema_version, verdict, unverified_claims, single_source_items}`。
   - 约束：逐条核对答案中的数字/日期能否在片段原文定位；多来源一致性逐项标记；
     禁止放过任何无法溯源的数字结论。

5. **拒答话术模板**：实时行情类、个股推荐类、超语料范围类，各配固定话术。

6. **结构化输出**：全部 LLM 调用使用 `response_format: json_object`，解析失败重试 1 次，再失败走兜底。

## 11. 生成层与记忆管理

- LLM（按意图路由）：`fact_qa` → deepseek-flash（快一倍、质量持平）；`analysis`/`comparison` → deepseek-v4-pro（深度推理）；
  降级链 v4-pro → flash（见 §13.1）；均流式 SSE 输出
  （推理内容在 `reasoning_content` 字段、正文在 `content` 字段，前端可展示「思考中」）。
- 引用溯源：答案附来源文档（doc_id → 按 `doc_id` 拉取该文档全部 chunks 还原全文）。

### 记忆管理（V1 只做会话级，临时数据全部落在会话记忆）

| 层级 | V1 | 说明 |
|---|---|---|
| 会话级记忆 | ✅ 做 | 两块内容：① 对话历史（最近 10 轮 + 超长摘要压缩）；② **每次问答的完整调用路径快照（S7）**。支撑多轮追问、指代消解、话题切换、排障回溯 |
| 长期用户画像 | ❌ 不做 | 常问领域/偏好个性化，V1 收益不明显；V2 有用户反馈数据后再评估 |
| 知识回写 | ❌ 不做 | 用户纠错不自动改知识库；走人工审核流程，金融数据可控性优先 |

**调用路径快照（S7，问答的临时数据存会话记忆）**：
- 每次问答生成 `query_id`，全程中间数据由 `post_step` 钩子写入 Redis 会话记忆，
  **从问答开始一直保留到该问答结束，另加 7 天排障窗口**；写入前按 S7 结构校验：

```json
{
  "schema_version": "1.0.0",
  "query_id": "q_20261016_ab12cd",
  "session_id": "s_8f3e",
  "status": "answering | verified | degraded",
  "question": "2021 年综合保税区进出口值是多少？",
  "intent": "fact_qa",
  "entities": ["综合保税区", "2021年"],
  "queries": ["...3 个变体..."],
  "keywords": ["综合保税区", "进出口值"],
  "retrieval": [
    {"round": 1, "chunks": [{"id": "...", "doc_id": "...", "score": 0.87}], "source_count": 5},
    {"round": 2, "chunks": [...], "source_count": 3}
  ],
  "self_check": {"sufficient": true, "rounds": 1},
  "verification": {"verdict": "pass", "single_source_items": []},
  "answer": "……",
  "citations": ["rank_00045_200200000358_c0"],
  "timings_ms": {"intent": 180, "retrieval": 420, "generate": 1500, "verify": 260},
  "cost_tokens": {"total": 6200, "by_step": {"generate": 5200}}
}
```

- **与 §13 可观测性的分工**：问答进行中的**临时/热数据在会话记忆（Redis）**，结束即进入 7 天窗口后过期；
  聚合统计、长期日志与看板走日志系统（S8，query_id 两边对齐，可互查）。

## 12. 评估体系

- 检索评估：recall@k、MRR；生成评估：忠实度（LLM-as-judge + 人工抽检）、引用正确率。
- 评测集：50~100 条金融问题，覆盖事实/归纳/多文档/对比/拒答五类。
- Agent 专项评估：补检触发率、补检后答案改善率（对比单轮）、补检轮数分布、钩子拦截统计。
- 验证环节评估：拦截率、误拦截率（人工抽检，验证关不能把正确答案误杀）。
- 离线回归：每次改提示词、chunk 参数、检索参数后跑一遍。
- 在线监控：无答案率、拒答率、检索空结果率、API 成本与延迟。

## 13. 可观测性与韧性（辅助机制）

### 13.1 Token 消耗检测
- 全链路用量记录：每次 LLM 调用（百炼各工序 + DeepSeek 生成）的 prompt/completion tokens **含 reasoning tokens** 结构化落日志（S8），query_id 关联。
- 聚合看板：每 query 平均成本、每日/月总额、按环节拆分（生成环节单列推理 token 成本）。
- 预算告警：日/月阈值（80% 预警、100% 自动降级 deepseek-v4-pro → deepseek-flash）。
- 一次性成本（2026-10-16 更新：embedding 已改为自部署，**全量 embedding 成本 ≈0**）：
  AutoDL 4090 按小时计费（约 ¥2/小时），全量 297 万 chunks ≈ 2 小时 ≈ **¥4~8**。
  仅剩余 API 成本 = 在线工序（百炼 qwen-plus/rerank + DeepSeek 生成）。
- **embedding 对比试验（2026-10-16 完成，50 query × 19,147 chunk 小索引，正例=同文档相邻 chunk）**：

| 模型 | recall@5 | recall@10 | 全量吞吐（4090 fp16） |
|---|---|---|---|
| bge-m3（MRL 512） | 0.636 | 0.714 | **430/s**（全量 ≈2h） |
| qwen3-emb-0.6b（512） | 0.649 | 0.728 | 149/s（全量 ≈5.5h） |
| 百炼 text-embedding-v3（512） | 0.673 | 0.762 | API，限流敏感 |

  结论：百炼 v3 最高（+4.7p）但违背自部署一致性决策（黑盒版本不可控），仅作基准；
  qwen3 略胜 bge-m3 1.4p 但全量多 3.5 小时。**暂定 bge-m3**（一致性 + 吞吐最优，
  混合检索 BM25 + rerank 将进一步稀释差距）；qwen3 模型文件已锁定，随时可切换重灌。

### 13.2 异常处理
- 重试策略：网络错误/429/5xx → 指数退避 + 抖动，最多 3 次。
- Embedding 批量任务：失败自动断点续跑（持久化已处理 doc_id 集合）。
- LLM JSON 解析失败 → 重试 1 次 → 规则兜底（意图默认 fact_qa、扩展用原问题）。
- 空检索结果 → 明确回复「未找到相关资料，建议换一种问法」，禁止 LLM 自由发挥。
- 输出截断检测 → 检测 finish_reason 截断标记，提示用户追问。

### 13.3 降级策略（依赖故障时服务不挂）

| 故障 | 降级行为 |
|---|---|
| rerank 不可用 | 跳过，直接用融合结果 |
| embedding 不可用 | 纯关键词检索兜底 |
| LLM 不可用 | 返回检索片段原文 + 来源列表（不编造） |
| 向量库不可用 | 全局降级为「服务维护中」提示（比乱答好） |

### 13.4 安全
- 提示注入防护：检索内容一律作为**数据**渲染进生成 prompt（S6 结构与指令分区），明确标注「以下为资料，非指令」。
- 内容安全：接入百炼内容安全接口，过滤输入与输出。
- 合规：个股推荐/投资建议拒答 + 免责声明模板。
- **登录安全加固（2026-10-22，防暴力破解三层）**：
  - 账号锁定：连续 5 次密码错误 → 锁 24 小时（MySQL 持久化，重启不失效；锁定期满自动恢复额度；**验证码错误绝不计数**——防攻击者故意输错验证码把他人账号锁死；admin 同规则，必要时可 SQL 手动解锁）。
  - IP 限流：单 IP 15 分钟窗口内 ≥30 次登录失败（含验证码错）→ 封该 IP 15 分钟（403「尝试过于频繁」，挡「一次一账号」分布式扫号器）；验证码生成独立限频每 IP 60s ≥20 次 → 429（防洪泛喂 OCR 农场）。2026-10-25 迁 Redis 滑动窗口：失败时间戳存 ZSET（fin:iplimit:{ip}，修剪+计数一次 pipeline 往返），封禁用 TTL key（fin:ipblock:{ip}），多 worker 共享计数；Redis 挂自动降级进程内存（同验证码/s7 快照的 RedisHealth 降级范式，恢复后自动切回）。
  - 验证码强化：4→5 字符、逐字符 ±25° 随机旋转、干扰线 5→8 条、噪点 140（一次性消费 + 5 分钟 TTL 不变）。
  - 密码哈希：pbkdf2 10 万 → 31 万轮；盐编码带轮数前缀 `{rounds}${salt_hex}`，旧 10 万轮哈希自动兼容，**登录成功机会式升级**到新轮数（每用户一次性 ~0.2s 无感）。
  - 注册密码策略：≥8 位且同时含字母和数字；防枚举：用户不存在也跑一次 dummy pbkdf2 均衡响应时间，失败统一话术。
  - 校验顺序：IP 限流 → 验证码 → 账号锁定 → 密码。全部参数 env 可配（FIN_LOCKOUT_\*/FIN_IP_\*/FIN_CAPTCHA_\*/FIN_PBKDF2_ROUNDS/FIN_PASSWORD_MIN_LEN）。
  - 验证：单测 tests/test_auth.py 34/34（2026-10-25 加 Redis 滑动窗口 + 降级路径用例）、线上安全演练 14/14（策略拒弱密码/5 次锁定话术/验证码洪水 429/IP 封禁/DB 状态核对）、smoke_api 18/18；metrics 新增 login_fail/login_locked/login_ok/ip_blocked/captcha_blocked/rehash_upgraded。
  - 已知项：遗留弱密码账号 admin123（旧式 hash，暂保留待改密）；token 无撤销机制（面试项目取舍）。

### 13.5 监控与链路追踪
- query_id 贯穿全链路，每个 query 完整 trace 可查（意图/变体/各环节耗时/自检轮数/验证结果/生成耗时/成本）。
- 日志统一为 S8 事件结构（含 `schema_version`、`query_id`、`step`、`payload`、`ts`），落盘前校验。
- 调用路径临时数据存会话记忆（S7，§11），聚合统计与长期日志走日志系统，query_id 两边对齐可互查。
- 指标：各环节 p50/p95 延迟、无答案率、拒答率、空检索率、重试率、验证拦截率、token 成本。
- 告警：错误率、成本阈值、延迟劣化、验证拦截率突增、向量库磁盘水位。
- **Langfuse 可视化追踪（2026-10-20 加，官方 SaaS Cloud 版）**：
  - 方案：Langfuse Cloud（us.cloud.langfuse.com，账号在 US 区域；默认 cloud.langfuse.com 会 401）；原生 SDK 直连上报，不引入 LangChain。（曾试自托管 v3：事件管线要求 S3+Redis+worker 五容器，维护成本高，改回 SaaS。）
  - 接入点：`llm_client.py` 追踪层——一次问答 = 一条 trace（`trace_id = create_trace_id(seed=query_id)` 确定性派生，UI 可反查 S8 日志）；每道 LLM 工序一个子 span（intent/expand/self_check/generate/verify 以 route 为 span 名，embed/rerank 单列），记录 input/output/usage（input/output/total + reasoning）/model/延迟/失败 level=ERROR。
  - 归属：span 经 OTel contextvars 自动挂到当前 trace（workers=1 单请求单线程，传播安全）；`agent.run` 入口 `llm_trace()` 包裹整次问答，`chat_stream` 生成器在迭代期持有 span 上下文。
  - 维度绑定：UI 的 Session/User 列用 SDK 模块级 `propagate_attributes(session_id=, user_id=)`（trace_context 仅支持 trace_id，4.18.0/5.0.2 均不支持 session/user 键，会被静默忽略）；user_id 为登录用户名，由 /chat 路由 token 解析后一路传入（main.py → sse.run_chat_blocking → agent.run → llm_trace）。2026-10-20 上线，线上实测全 span 携带 user/session。
  - 容错（与 S7/S8 降级体系同原则）：未启用/未装 SDK/初始化失败/导出失败 → 全链路 no-op，问答主流程零影响；`_safe_update` 保证追踪异常不透传。
  - 配置：`.env` 的 `LANGFUSE_ENABLED/HOST/PUBLIC_KEY/SECRET_KEY`（config.py §P5 区块）；S8 日志、S7 快照、/metrics 观测不变（Langfuse 为补充视图，不替代）。
  - 回归：`tests/test_langfuse.py` 24 项（no-op 路径/确定性 trace_id/usage 记录/失败标注/流式 span/SDK 缺失降级），本地可跑（假 SDK 注入，无 hnswlib 依赖）。
- **Langfuse 提示词管理（2026-10-22 加，全套迁入）**：
  - 5 套 LLM 模板文本（intent_expand/generate/self_check/verify/judge）迁入 Langfuse Prompts（名 `financial-rag/<name>`，label=production）——UI 改版、版本历史、一键回滚；**组装逻辑/FEWSHOT/INTENT_GUIDE/guard 正则/拒答话术留代码**（含代码逻辑不入模板，模板只放纯文本）。
  - `prompts/remote.py`：`get_prompt` 复用 llm_client 同一 SDK 客户端（label=production，SDK 内存缓存 TTL 300s——UI 改完最多 5 分钟生效，fetch 超时 3s）；本地 SYSTEM 常量保留为兜底——远端不可用/未启用/空文本/异常 → 行为与迁移前完全一致；`FIN_LANGFUSE_PROMPTS=0` 整体关闭。
  - 版本联动：7 处调用点把 `get_prompt` 返回的 prompt 客户端经 `chat/chat_stream` 的 `prompt_client` 参数传入 span——UI 每条 LLM span 显示所用提示词名与版本，评测轮次可精确对比「哪版提示词产出哪个答案」。
  - 注册工具 `tools/langfuse_prompts.py`（`--list`/`--register` 本地≠远端 production 才推新版本并打 production 标签，幂等；`--force` 强制）。
  - 回归：`tests/test_prompts_remote.py` 27 项（未启用→本地兜底/开关=0→本地/远端正常→远端文本/缺失·空·异常→兜底/5 套文本一致/span 透传）。

## 14. 系统架构与工程化

```
[离线] parquet → 过滤/去重/清洗 → 切分 → 百炼embedding → 向量库（含元数据JSON）
[在线] 提问 → 意图识别 → 多查询+关键词 → 向量+BM25混合检索 → 自检(Agent) → 百炼rerank → qwen生成 → SSE
```

- **服务器选型建议**（云平台采购参考）：embedding/rerank/LLM 均走百炼 API，服务器**无需 GPU**。
  - CPU 16 核（8 核起步）/ **内存 32G（硬指标**，向量索引 12~16G + 服务余量）/ 系统盘 40G + 数据盘 200G SSD / 带宽 5Mbps / Ubuntu 22.04。
  - 开发期按量付费（16C32G 约 2~3 元/小时，不用即停），稳定后转包年；选 ECS 通用型，**不要轻量服务器**（内存上限不满足）。
  - 若预算允许上 64G，解锁 1024/2048 维全内存方案；32G 方案即 512 维（2026-10-16 按 297 万 chunks 更新，见 §8）。
- API（请求/响应与 SSE 事件均为 S9 契约结构，见 §9.2）：`POST /chat`（SSE）、`GET /sources/{query}`（检索调试）。
  - S9 结构已定稿（`schemas/models.py`）：请求体 `S9ChatRequest`；SSE 事件信封 `S9Event`，
    事件类型 `meta`（意图）/ `reasoning`（推理流）/ `delta`（正文流）/ `sources`（来源列表）/
    `done`（结束汇总：答案+引用+校验结果+耗时+token）/ `error`；`data` 按事件类型双向校验（`make_sse` 组装）。
  - P1 联调观测：deepseek-v4-pro 首次调用耗时 ~79s（推理 4143 token），fact_qa 走 flash 快路径
    （实测 10.8s / 推理 1624 token）的设计必要性再次印证；生产环境生成超时与 max_tokens 要按此量级配置。
- 技术栈：FastAPI + 向量库（Milvus/Qdrant，待定）+ ES/内置 BM25 + Redis（会话记忆/缓存）+ jieba。
- 监控：见 §13.5。
- **部署环境与凭据（2026-10-19 定，仅限项目内部使用）**：
  - ECS：8.137.192.231（2026-10-25 已转弹性公网 IP，原 183.17/174.74 均易主），root 密码已脱敏（本机 tools/ssh_secret.py，不入库；上传工具 ssh_put_b64.py）
  - MySQL（docker 容器 fin-mysql，仅绑 127.0.0.1，库名 fin）：root `JWRf7pLaJrZ8uByWn5wk`；业务账号 fin `VIaZ3YfLPTzvh6yvAutB`
  - Web 登录：admin / 密码见 .env `FIN_ADMIN_PASSWORD`（不入库；仅在账号不存在时生效——admin 已创建，改密需直接改 MySQL users 表或另注册账号）
  - 平台 API 密钥：百炼 DASHSCOPE_API_KEY、DeepSeek DEEPSEEK_API_KEY 存 `/data/financial/.env`（不入仓库、不随文档外传）
  - token 签名密钥：`/data/financial/.token_secret` 自动生成落盘（重启登录态不失效，勿删）

## 15. 实施路线图

| 阶段 | 前置条件 | 周期 | 交付物 |
|---|---|---|---|
| **P0 语料工程 + 接口封装** ✅ 2026-10-16 完成 | 无 | 1 周 | 过滤/去重/清洗脚本、指标分布报告、`llm_client.py` 双平台封装（含 usage 统计）、S1 结构定稿。产出 `cleaned.parquet` 463,695 篇 / 1.92GB |
| **P1 提示词原型** ✅ 2026-10-16 完成 | 百炼 API key | 1 周 | 四套提示词（`prompts/`：intent_expand / generate / self_check / verify + guard 正则兜底 + refusals 话术）+ `schemas/models.py` S2~S9 pydantic 契约 + `schemas/json/` 11 个 JSON Schema 导出。联调实测（`tests/test_p1_live.py`）：六类意图 6/6 命中、生成引用编号全部合法且「资料不足」时诚实声明、自检充分/不足判定正确、验证关拦截幻觉数字（verdict=fail 并列出无法溯源结论）。单 query 内部工序成本约 6~7k tokens（生成环节大头）。 |
| **P2 切分与索引** ✅ 2026-10-17 完成 | **服务器到位** | 1 周 | ✅ 切分器+验证关卡定版 v1.0；✅ AutoDL：bge-m3 下载锁定（模型文件 sha256 入 model_lock.json）+ GPU fp16/CPU fp32 管线一致性实测（余弦 ≥0.99986、top-10 检索 197/200 一致）；✅ 三模型对比 bge-m3 vs qwen3-emb-0.6b vs 百炼 → 定案 bge-m3（512 维 MRL，在线走 ECS CPU fp32 同管线）；✅ ECS 全量切分 118s → 2,863,611 chunks（验证关卡通过）；✅ AutoDL 全量 embedding 114 分钟（fp16）→ 回传 ECS（md5 一致）；✅ HNSW 索引（M=16, ef_construction=200，12 分钟，加载后须 set_ef(100)）；✅ 最终验证：模型锁 sha256 ✓、HNSW vs 暴力 top-20 重合率 100%（索引正确性）、查询延迟中位 343ms/p95 1008ms。全量库纯向量 recall@10≈0.12 为规模效应（286 万同质语料，正例排名中位 ~466，非 bug）——检索质量评估移至 P3 真实问答集 + 混合检索 + rerank |
| P3 检索链路 ✅ 2026-10-17 完成 | 服务器 | 1 周 | ✅ 评估集 80 条（65 文档类 + 15 拒答留 P4），LLM 出题 + evidence 逐字抄录子串校验接地（温度 0.6，消除 LLM 幻觉漂移，1 条手工补写）；✅ BM25 倒排索引（2,863,611 docs / 1,408,901 词 / 3.81 亿 postings，单遍低内存构建 11 分钟，查询 ~90ms）；**修复关键 bug**：postings 散落时花式赋值重复下标 → 每词每 worker 实际只写入 1 条（诊断：随机 1000 篇含词率 0.001，BM25 recall≈0.003 假象）→ 改为按 gid 排序 + 词内偏移唯一下标后含词率 1.000，并加写前槽位硬校验+抽样自检防回归；✅ 5 配置消融（recall@10 / MRR@10）：vector 0.073/0.244、vector_expand 0.062/0.170、bm25 0.132/0.474、hybrid 0.104/0.275、**hybrid_rerank 0.132/0.483 最优**（recall@5 0.118）；结论：证据锚定问题集偏词法匹配 → BM25 强于纯向量，RRF 等权融合会稀释 BM25 强信号（hybrid<bm25），P4 起可对 RRF 加权调优；✅ Agent 循环冒烟：快路径 1 轮 ~5-6s/3.4k tokens，慢路径补检 ≤2 轮（off-by-one 已修：轮数上限现在精确生效），钩子（token 预算/轮数/查询去重）全部触发验证，realtime 问题 14.8s/8k tokens 有界收敛；✅ 混合检索 + RRF(k=60) + 元数据过滤 + qwen3-rerank 全链路通；✅ 变体配额消融（10-19，64 题 × 6 配置）：hybrid_rerank_div（余弦≥0.85 每簇留 2）0.131/0.474 vs hybrid_rerank 0.136/0.476 —— **整体无提升（-0.005），不启用 diversify**；分题型：analysis +0.005（MRR +0.094）、multi_doc +0.005、fact_qa -0.017、comparison -0.008；检索瓶颈仍在 analysis 0.039 / comparison 0.069 归纳对比题，转向慢路径补检 + RRF 加权调优；✅ 生产检索深度 n=10→20（10-19，agent.py 快/慢路径 + assemble [:20]，冒烟 3 题全链路 pass，单题成本 +2~3k tokens）；✅ 融合消融定案（10-19，64 题全量）：**hybrid_rerank_max**（max 融合+等权+原问句入向量路+BM25 深度 300+精排 100）recall@10 **0.159** / hit@10 **0.78** / MRR **0.626** vs 旧 sum 基线 0.134 / 0.67 / 0.491 —— 全面胜出定为生产配置；analysis 类 recall 0.039→0.076 近翻倍、hit@10 0.56→0.88；comparison 类未改善（跨主题问句稀释单 chunk 匹配）；✅ 扩展消融定案（10-19 晚，64 题）：noexpand 0.156/0.610 vs 有扩展 0.159/0.626，逐题 62/64 一致、2 题有扩展更好（q062 MRR 0.14→1.00、q089 0→0.20，均 comparison）、0 题变差 —— **扩展保留不砍**（成本 ~1.2k tokens/查询，恰好救回最弱的 comparison 类）；❌ comparison 查询分解证伪（10-19 晚，diag_comp 三轮 + 64 题全量）：分侧检索+双路 rerank 并联在共用扩展的试点里 hit 0.55→0.64（救 q060），但全量独立扩展下 comparison MRR 0.325→**0.147**、q063 正例从第 1 被挤出前 10、q060 救援未复现、意图误分类泄漏伤 q051（multi_doc）——**分解依赖变体质量与意图分类稳定性，不启用**（decomp 分支保留在 retrieval.py 但默认关闭）；q065 确认为评估集噪声题（源文档无摩根大通内容） |
| P4 生成与评测 ✅ 2026-10-18 完成 | 服务器 | 1 周 | ✅ 生成链路（`generation.py`：按意图路由 + 流式 on_delta + 引用解析 + 空答案加倍重试）→ ✅ 验证关（`verify_gate.py`：越界编号规则硬校验 + qwen-plus JSON 三道检查，fail→flash 重生成 1 次→VERIFY_FAIL 话术）→ ✅ S7 会话记忆（`session_store.py` 快照 TTL 7 天）→ ✅ 评测集 79 条全链路评测：拒答准确率 **15/15 (100%)**、可回答率 **59/64 (92.2%)**、验证 pass 59 / fail 5、重生成改善 5/10、引用合法性 59/59 (100%)、证据覆盖 38/56 (67.9%)、judge 一致性 42/59 (71.2%)、慢路径率 19/64、成本 116 万 tokens/4006s → ✅ **flash vs v4-pro 对照实验（8 题深推理）**：judge 一致性持平 6/8=6/8、flash 平均 **14.6s vs 68.7s（快 4.7 倍）**、tokens 减半 → **全链路默认只用 deepseek-flash，弃用 v4-pro**（冒烟：comparison 题总耗时 33.2s）。遗留：judge 不一致主要源于检索召回不足（P3 已知 recall@10≈0.13）；DeepSeek API 偶发故障已重跑恢复 |
| P5 服务化上线 ✅ 2026-10-19 完成 | 服务器 | 1 周 | Web 服务 + 多轮对话 + 监控告警 + 降级机制。✅ server/ 包 10 模块（FastAPI + SSE 桥接 `SSEHub`（必达事件 deque + 增量队列 + done 不抢跑时序修复 + 15s 心跳 + 断连静默）/ `ServiceHooks`（step→meta/sources/S8/指标分派）/ Redis+Memory 双实现（会话历史 `fin:conv` + S7 快照 `fin:s7` + RedisHealth 60s 探测）/ S8 JSONL 日志（10MB 轮转 30 天）/ Prometheus 指标（有界直方图 p50/p95）/ 维护态状态机（ok/degraded_bm25/maintenance + 探针自愈））；✅ 存量改造：agent.py（query_id 前移贯穿 S7/S8/S9、on_reasoning 透传、_assemble 补 history 末 6 条、verify verdict 上报、空检索守卫修 latent bug）、generation.py（on_reasoning）、retrieval.py（**缓存加锁**修 4 并发 LRU 竞态、chunk_count 列加载、向量失败降级纯 BM25 + streak 计数）、llm_client.py（重试 before_sleep 回调→重试率指标）、refusals.py（MAINTENANCE）；✅ 回归：test_schemas 53/53、smoke_agent 3 题无回归、SSE 接线测试 16/16（抓出 done 抢跑 bug）；✅ 上线（ECS systemd + docker redis + crontab 告警巡检）：**smoke_api 14/14 通过**（SSE 时序 / sources 调试 / 多轮 fin:conv=4 / 拒答零生成 / **Redis 停容器降级演练 Memory 兜底** / metrics 非零 / S8 同 query_id 链路 6 条 / 4 并发 query_id 唯一）、告警演练阈值 0 触发落盘验证。✅ 前端（2026-10-19，Vue 3 + Vite + TS 单页，5 依赖无 UI 库，`frontend/` 14 文件）：`src/api/chat.ts` SSE 解析器（fetch+ReadableStream 逐帧、TextDecoder 单实例 stream:true 防中文断裂、心跳忽略、非 200 读 detail、done.answer 权威替换、abort/断流/单飞守卫）+ `useChat`（只发 session_id 不发 history，localStorage 持久）+ 消息流/引用面板组件（intent 标签 6 枚举、verdict 徽章、reasoning 折叠、citations 高亮联动）+ 深色金融风手写样式；`server/main.py` 末尾条件挂载 StaticFiles（`config.FRONTEND_DIST`，目录缺失纯 API 模式，同域零 CORS）。已验证：构建 vue-tsc 零错误（74.8KB）、解析器 20/20（断字/429/503/拒答/流内 error/abort/断流）、Vite 代理 SSE 透传无缓冲。已知观察：多轮第二问（「这个数据比上一年增长多少」）诚实拒答未绑定指代——指代消解质量调优留后续（V1 机制验收通过）。服务地址 http://8.137.174.74:8001（/ = Web UI、/chat SSE、/sources、/health、/metrics；安全组开放 8001；端口 2026-10-19 从 8000 改为 8001——config.API_PORT 连带 alerts.py 指标 URL 与 smoke_api 默认端口）。✅ 前端已上线：dist 原子替换 /data/financial/frontend/dist、systemd unit 更新+daemon-reload+重启、静态冒烟（/ 返回页面、assets 200、未知路径 404 不掩盖、metrics 正常）、**smoke_api 14/14 重跑通过**（8001）、公网直连 SSE 时序正确。纯前端后续更新走目录 swap 免重启（StaticFiles 每请求实时读）。✅ 引用面板改造（2026-10-19 晚，用户反馈：卡片不可点开原文/来源列表看不懂/正文 [n] 对应不上来源）：卡片整张可点展开全文、头部用户化标签「资料 n · 文章 #doc_id · 片段 x/y」、技术分（向量/BM25/score/质量分）隐到展开态小字；正文 [n] 金色可点 → 侧栏对应卡片滚动定位+金色闪烁 1.6s+自动展开，点旧消息的 [n] 侧栏切到该消息来源、新答案到达回最新；`utils/markers.ts` 与后端 CITE_PAT 逐字对齐解析 [n]/[1,2]，越界编号渲染纯文本防误点（[2012] 年份不被当引用）；联动正确性依据：hooks.py sources 事件与 agent.py _assemble 用同一 ctx.chunks[:20] 按序编 ref_no → 面板第 n 卡 = 资料 n，零后端改动。验证：解析器边界 13/13（含越界/年份/refCount=0）、vue-tsc+vite 零错误、远端原子替换免重启生效（新 hash index-Dsvm-qI1.js + index-rBQGg8nk.css）。✅ 用户系统 + 会话持久化（2026-10-19 深夜上线，§14.5）：**MySQL 主存储**（docker fin-mysql，127.0.0.1 只绑 + volume + utf8mb4；`users`/`conversations`/`messages` 三表启动自动建表；pymysql 单连接+重连，挂 → 历史降级 Memory 问答不挂）——多轮上下文改从 MySQL messages 读（Redis fin:conv 退役）；**登录墙**：/chat 需 Bearer token（自签 HMAC-SHA256 7 天，secret 落盘 .token_secret 防重启失效；密码 pbkdf2 10 万次；全标准库零安全依赖）；**图片验证码**：Pillow 自绘 4 字符+干扰（Redis 一次性 5 分钟，降级内存），登录/注册都要；**会话管理**：新建会话前端生成随机 id、首条消息懒落库（title=首问去空白截 20 字纯规则零 LLM 成本）、`/api` 六接口（captcha/login/register/me/conversations/{id}/messages，会话按 user_id 隔离越权 404）；预置 admin（FIN_ADMIN_PASSWORD env，缺省 admin123）。前端：LoginView（验证码点击换图+登录/注册切换）、SessionSidebar（新建/历史列表/退出）、useChat 重写（会话生命周期、401 自动回登录页、退出清空消息区）。验证：test_schemas 53/53、token/密码/验证码/标题逻辑单测全过、**smoke 升级 17/17**（新增：登录链路/未登录 401/MySQL messages=4 角色交错/标题断言）；dist 原子替换新 hash index-DR7KWpz_.js。遗留：注册接口开放（可自注册新账号）；admin 初始密码为部署时生成（.env FIN_ADMIN_PASSWORD）；时区+文献名修复（2026-10-19 深夜）：① 历史时间统一上海时间——db.py 显式 _now_cst() 写 conversations/messages（不再依赖 MySQL DEFAULT CURRENT_TIMESTAMP），fin-mysql 容器重建加 TZ=Asia/Shanghai（volume 数据保留），存量 conversations +8h 一次性修正；② 引用卡片加文献名 doc_title——tools/build_title_map.py 纯规则离线提炼（剥「XX网X月X日消息:」「电(记者 XX)」「(原标题:…)」等导语+首句截 44 字，98.7% 覆盖 46.4 万篇，14s）→ data/title_map.parquet，retrieval 加载 + S5Hit.doc_title 可选字段（前端回退「文章 #id」）；smoke 升级 18/18（新增 doc_title 断言），dist 新 hash index-DYX8Gsz5.js；回答结构去「来源列表」段（2026-10-19：用户反馈出处已由侧栏文献名承担，规则 4 只留 结论+分点依据）；对话记忆类问题修复（2026-10-19 深夜：问「我上一个问题问了什么」原被判 chitchat 短路返回引导话术——agent.py chitchat 分支加纯规则识别（上个/刚才/之前+问了什么）命中则从会话历史回显上一问，零 LLM 成本，线上实测 PASS）；Langfuse 可视化追踪（2026-10-20 上线，见 §13.5：llm_client.py 追踪层一次问答一条 trace、trace_id 由 query_id 确定性派生、每道工序一个 span 记录 usage/model/延迟、容错全 no-op，tests/test_langfuse.py 24/24；接入官方 Cloud（us.cloud.langfuse.com），线上问答实测 trace+span 落库成功）；✅ 空态示例问题 chips（2026-10-21：新建会话空态展示 4 道示例题——取自 P4 评估 verdict=pass，覆盖事实检索/分析归纳/跨文档对比三类意图，点击即经 send 流程提问（首问懒落库）；MessageList 增 SUGGESTIONS 常量 + ask 事件、App @ask="send" 接线、.suggest-chip 样式；构建 vue-tsc 零错误、dist 原子替换免重启上线、静态冒烟 ROOT/JS/CSS/HEALTH 全 200）；✅ 登录安全加固（2026-10-22，详见 §13.4）：防暴力破解三层上线——账号 5 次错误锁 24h（MySQL 持久化 + 幂等补列 `_ensure_column`；验证码错误不计数、登录成功清零）+ IP 限流（15 分钟 30 次失败封 15 分钟、验证码 20 次/分钟 429）+ 验证码强化（5 字符旋转 8 干扰线 140 噪点）；pbkdf2 10 万→31 万轮 + 盐带轮数前缀 + 登录成功机会式升级（线上 admin 实测升级成功，fin_rehash_upgraded=1）；注册密码 ≥8 位字母数字混合；dummy pbkdf2 防用户名枚举；全参数 env 可配。验证：tests/test_auth.py 28/28、线上安全演练 tools/security_drill.py 14/14（策略拒弱密码/5 次锁定话术含剩余时间/验证码洪水 429/IP 封禁/DB 状态与临时账号清理）、smoke_api 18/18 无回归。已知项：admin123 弱密码账号保留（用户决定先不动）；IP 限流进程内存态（workers=1 安全，>1 需迁 Redis）；✅ 提示词全套迁入 Langfuse（2026-10-22，见 §13.5）：5 套模板注册 v1+production（tools/langfuse_prompts.py --register 幂等）、prompts/remote.py 缓存 300s+3s 超时兜底（远端不可用行为与迁移前逐字一致）、7 处调用点经 prompt_client 透传 span 版本联动（SDK 走 OTel 保留属性 langfuse.observation.prompt.name/version，UI span 显示提示词 chip）；回归 test_prompts_remote 27/27 + test_langfuse 25/25 + smoke_api 18/18 |
| **P6 独立向量库（Qdrant）** ✅ 2026-10-21 完成 | 服务器 | 1 天 | 进程内 hnswlib → 独立 Qdrant 向量库（v1.19.2 容器 + qdrant-client 1.19.1；选型理由：单容器无 etcd/MinIO 依赖链、Rust 实现、mmap 向量 + on_disk payload，面试加分点「为什么不用 Milvus」）。✅ 数据面：286 万 chunk 向量 + 8 列元数据（含 text）全量入 collection（point id = chunk 行号 = BM25 文档号，融合对齐契约不变；HNSW m=16/ef_construct=200 与原索引一致、v1 不量化保召回对齐）；`build_qdrant.py` 灌库脚本（双 parquet 锁步逐批断言 id 一致 + checkpoint 断点续传 + 24 分钟灌完 + `--verify` 四段校验）。✅ 校验：计数一致 2,863,611；payload 抽样 200/200（含 meta.parquet 交叉验证）；召回对齐三重闸门——同 top-1 分数差 3.4e-07（数值等价）、ef=100 overlap@50 0.963（≥0.95，HNSW 图结构差异的低 ef 近似噪声）、ef=400 收敛 0.990（≥0.98，排除系统性偏差）；批量返回序==逐查询序；单查询延迟 p50 2ms/p95 3ms。✅ 踩坑（已修）：Qdrant 1.19 的 query_points 传「向量列表」被服务端当 multivector 输入 → 400，批量必须走 query_batch_points（requests=[QueryRequest…]，返回序==输入序）。✅ 64 题全量回归（与 hnswlib 基线逐配置对照）：生产配置 hybrid_rerank_max 0.158/0.625 vs 基线 0.159/0.626（-0.001，LLM 噪声 ±0.01 内）；纯向量 0.077/0.254 vs 0.074/0.247（+0.002）；BM25 不变。✅ 切换后指标：启动 90s→**3s**、uvicorn RSS 18.1G→**1.7G**、Qdrant 容器 651MiB/8GiB；smoke_api **18/18**。✅ 故障演练（docker stop fin-qdrant 实录）：宕机后 HTTP 200 纯 BM25 继续出答案（首请求懒加载列兜底 11.7s、后续 0.0s）、状态转 degraded_bm25；**抓出两个降级链漏洞并修复**：① payload 取回硬抛 → 宕机时 500（改懒加载列兜底，内存换可用性）② 同题缓存命中不触向量路 → streak 卡住状态滞后（degraded 态探针主动向量自检，**恢复后零流量 60s 自愈**验证通过）。遗留：量化压缩（scalar/TurboQuant）A/B、BM25 迁 Qdrant 稀疏向量、MemoryMax 22G→12G 待下一轮；hnsw.bin/chunks.parquet 保留（回滚 + 懒加载兜底用，回滚 = 恢复 5 个文件 + restart ≤5 分钟） |

> P0/P1 完全不需要服务器：百炼是云端 API，本地开发机即可调通全部接口、
> 完成提示词打磨与评测集构造；服务器到位后直接进入 P2。

## 16. 风险与注意事项

- **数据时效性**：语料约 2012~2022，回答带时间语境，系统声明知识截止日期。
- **金融合规**：个股推荐/买卖建议一律拒答，输出附免责声明。
- **幻觉风险**：数字错误危害大 → 引用溯源 + 无答案拒答 + 数字类问题重点评测。
- **Agent 循环成本**：自检+补检增加 LLM 调用 → 限制补检轮数、单轮快路径、自检用便宜模型。
- **提示注入**：语料可能含诱导性文本 → 生成 prompt 数据/指令分区（§13.4）。
- **API 成本与限流**：全量 embedding 一次性成本需预估配额；在线并发注意百炼限流与重试策略。
- **服务器采购**：向量库选型（Milvus vs Qdrant）建议在服务器到位后按内存/磁盘实测再定。
- **敏感内容**：flagged_words_ratio 全 0 说明已过滤，上线前再做一轮合规检查。
