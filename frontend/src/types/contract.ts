// S9 契约 TS 镜像（schemas/models.py §9.2）：编译期契约校验，枚举与后端逐字对齐。
// 后端改契约时必须同步这里（否则 vue-tsc 报错提醒）。

export type Intent = 'fact_qa' | 'analysis' | 'comparison' | 'realtime' | 'investment_advice' | 'chitchat'
export type Verdict = 'pass' | 'downgrade' | 'fail'
export type RankSource = 'vector' | 'bm25' | 'fusion'
export type S9EventName = 'meta' | 'reasoning' | 'delta' | 'sources' | 'done' | 'error'

export const S9_EVENTS: readonly S9EventName[] =
  ['meta', 'reasoning', 'delta', 'sources', 'done', 'error']

/** S5Hit：检索命中统一出口（§9.3）。score 服务端已 clamp 到 0~1。 */
export interface S5Hit {
  schema_version: string
  chunk_id: string
  doc_id: string
  text: string
  score: number
  rank_source: RankSource
  quality_score: number
  chunk_index: number
  chunk_count: number
  /** §14.5：来源文章文献名（正文首句规则提炼）；缺 title_map 时为 null，展示回退「文章 #id」 */
  doc_title?: string | null
}

export interface S9Meta {
  intent: Intent
  entities: string[]
  queries: string[]
  keywords: string[]
}

export interface S9Sources {
  references: S5Hit[]
}

export interface S9Delta {
  delta: string
}

export interface S9Verification {
  verdict?: Verdict
  single_source_items: string[]
}

export interface S9Done {
  answer: string
  citations: string[]
  verification: S9Verification | null
  timings_ms: Record<string, number>
  cost_tokens: { total: number; by_step: Record<string, number> }
}

export interface S9Error {
  code: string
  message: string
}

/** SSE 信封：query_id/ts 在顶层，data 按 event 分型（§9.2 双向校验）。 */
export interface S9Envelope {
  schema_version: string
  query_id: string
  event: S9EventName
  data: unknown
  ts: string
}

export interface ChatTurn {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface S9ChatRequest {
  schema_version: string
  question: string
  session_id?: string
  history?: ChatTurn[]
}
