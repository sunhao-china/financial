# -*- coding: utf-8 -*-
"""四套提示词集中管理（大纲 §10）：版本化、可评测，每次修改跑离线回归。

- intent_expand：意图识别+查询扩展（qwen-plus → S2）
- generate：RAG 生成（deepseek-flash / deepseek-v4-pro 按意图路由）
- self_check：信息自检（qwen-plus → S3）
- verify：答案校验（qwen-plus → S4）
- guard：意图兜底正则（§4 先行拦截）
- refusals：拒答与降级话术模板（§10.5/§13.3）
"""
