# 金融问答 RAG 系统

基于 2012–2022 年 **46 万篇财经新闻语料（286 万文本块）** 自研的中文金融领域检索增强问答系统：语料清洗 → 切分向量化 → 混合检索 → 意图路由 → 流式生成 → 引用溯源，全链路自研，支持多用户会话与安全加固。

## 功能特性

- **混合检索**：BM25 + 稠密向量（bge-m3 512 维）双路召回，RRF max 融合 + rerank 精排，检索配置经 A/B 消融实验定稿
- **六类意图路由**：事实问答 / 分析归纳 / 对比 / 时效性拒答 / 投资建议拒答 / 闲聊；拒答类双保险（意图分类 + 高置信正则 guard，后者零 LLM 成本）
- **防幻觉双保险**：生成自检（证据不足自动补检）+ 验证关逐条溯源，失败重生成一次 → 诚实话术兜底
- **SSE 流式输出**：推理过程与正文双流独立，前端折叠展示
- **引用溯源**：回答带 `[n]` 标注，点击联动侧栏文献卡片（含文献名，覆盖 98.7%）
- **多用户系统**：注册/登录/图形验证码/会话持久化（MySQL 三表），token 鉴权按用户隔离防越权
- **安全加固**：防暴力破解三层（账号 5 次锁 24h / IP 滑动窗口限流 Redis / 验证码强化），pbkdf2 31 万轮机会式无感升级，防用户名枚举
- **高可用**：Redis/Qdrant 故障自动降级兜底链路，服务不中断；幂等 DDL 迁移兼容存量数据

## 量化成果

| 维度 | 结果 |
|---|---|
| 检索（64 题消融，生产 vs sum 融合基线） | recall@10 **+19%**（0.12→0.159）、hit@10 **+16%**（0.67→0.78）、MRR **+27%**（0.491→0.626） |
| 端到端（79 题五类评测） | 拒答准确率 **100%**（15/15）、可答问题回答率 **92.2%**（59/64）、引用编号合法性 **100%** |
| 性能（向量库架构迁移） | 启动 90s→3s（**-97%**）、服务内存 18.1G→1.7G（**-91%**） |
| 质量 | 单元测试 34/34、线上冒烟 18/18、生产安全演练通过 |

## 系统架构

```
[离线] parquet 语料 → 质量过滤 → MinHash 三级去重 → 句子级切分 → bge-m3 向量化 → Qdrant（286 万点）
[在线] 提问 → 意图识别 → 多查询扩展 → 向量+BM25 混合检索 → 自检(不足补检) → rerank 精排 → 生成(带 [n] 引用) → 验证关 → SSE 流式输出
```

## 技术栈

Python / FastAPI · Vue 3 + TypeScript · Qdrant · MySQL · Redis · Docker · nginx · Langfuse · bge-m3 · DeepSeek API

## 目录结构

```
config.py            全局配置（env 可覆盖，见 .env.example）
llm_client.py        LLM 客户端（DeepSeek，重试/并发）
corpus/              离线流水线：质量过滤/去重/切分/向量化/检索/评测
server/              在线服务：FastAPI/SSE/鉴权/限流/会话/监控
prompts/             5 套提示词模板（生成/自检/验证/意图/拒答）+ Langfuse 远端同步
schemas/             事件与数据模型（S9Event 等）
frontend/            Vue3 单页前端（SSE 解析、引用联动、会话栏）
tools/               部署/评测/演练工具（smoke、langfuse_eval、ssh 等）
tests/               单元测试
deploy/              systemd 单元文件
```

项目设计与实现细节见《金融RAG系统设计大纲.md》，面试叙事与指标口径见《金融RAG项目面试总结.md》。

## 评测体系

79 条人工标注评测集（`data/evals/qa_set.json`：事实问答 27% / 分析归纳 20% / 多文档 19% / 拒答 19% / 对比 15%，每题含标准答案与证据文档标注）。经 `tools/langfuse_eval.py` 注册为 Langfuse 数据集，每次改动跑一轮真实管线，UI 跨轮对比检索命中与 LLM-as-judge 打分。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env      # 填入 DEEPSEEK_API_KEY 等
python server/main.py     # 或参照 deploy/financial.service 用 systemd 运行
```

离线数据管线需自行准备语料 parquet（列：text / quality_score / perplexity / char_rep_ratio / flagged_words_ratio），依次运行 `corpus/filter_corpus.py → dedup_corpus.py → clean_corpus.py → chunk_corpus.py → build_qdrant.py`。

## 测试

```bash
pytest tests/                              # 单元测试
python tools/smoke_api.py --host 127.0.0.1 # 线上冒烟（需先起服务 + 本机可连 Redis 读验证码）
```

## 说明

- 在线演示已于 2026-11 随服务器释放下线，代码与文档完整保留
- 语料为公开财经新闻，仅用于学习研究，请勿商用
