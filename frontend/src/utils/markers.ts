// 正文引用标记解析：与后端 prompts/generate.py 的 [n] / [1][2] / [1,2] 约定一致，
// 正则与 corpus/generation.py 的 CITE_PAT 逐字对齐。
// 仅编号落在 1..refCount 内的标记可点（后端 parse_citations 同样忽略越界编号，
// 且 verify 会把越界编号判 fail——合法答案中不应出现）；越界者原样渲染为纯文本。

export interface AnswerSeg {
  text?: string
  refs?: number[] // 合法引用编号（渲染为可点击标记）
}

const CITE_RE = /\[(\d+(?:,\d+)*)\]/g

export function parseAnswerSegments(content: string, refCount: number): AnswerSeg[] {
  const segs: AnswerSeg[] = []
  let last = 0
  // 相邻纯文本段合并，保持段列表最简（渲染结果不变）
  const pushText = (s: string): void => {
    const prev = segs[segs.length - 1]
    if (prev && prev.text !== undefined) prev.text += s
    else segs.push({ text: s })
  }
  for (const m of content.matchAll(CITE_RE)) {
    const start = m.index ?? 0
    const nums = m[1]!.split(',').map(Number).filter(n => n >= 1 && n <= refCount)
    if (start > last) pushText(content.slice(last, start))
    if (nums.length) segs.push({ refs: nums })
    else pushText(m[0]) // 越界/零编号 → 纯文本，不做可点误导
    last = start + m[0].length
  }
  if (last < content.length) pushText(content.slice(last))
  return segs
}
