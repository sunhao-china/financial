// S9 /chat SSE 客户端（对齐 server/sse.py）：fetch POST + ReadableStream 逐帧解析。
//
// 帧格式：`data: {信封 json}\n\n`（无 event: 行）；心跳注释行 `: ping`（15s 空窗）；
// 流结束即断流。done.answer 为最终权威（验证 fail 静默重生成 / 拒答路径 / delta 溢出
// 丢弃增量三种情况下，累加文本都可能不完整，必须无条件替换）。
import { S9_EVENTS } from '../types/contract'
import type {
  S9ChatRequest, S9Done, S9Envelope, S9Error, S9Meta, S9Sources,
} from '../types/contract'

export type ChatErrorKind = 'http' | 'network' | 'aborted' | 'parse' | 'sse' | 'busy'

export class ChatError extends Error {
  kind: ChatErrorKind
  status?: number // 非 200 的 HTTP 状态码（503 维护 / 429 繁忙）

  constructor(kind: ChatErrorKind, message: string, status?: number) {
    super(message)
    this.name = 'ChatError'
    this.kind = kind
    this.status = status
  }
}

export interface StreamHandlers {
  onMeta?(env: S9Envelope, m: S9Meta): void
  onSources?(env: S9Envelope, s: S9Sources): void
  onDelta?(text: string): void
  onReasoning?(text: string): void
  onDone?(env: S9Envelope, d: S9Done): void
  onError?(env: S9Envelope, e: S9Error): void
}

let inFlight = false // 单飞守卫：同一时刻只允许一个流式请求

function dispatch(handlers: StreamHandlers, env: S9Envelope): void {
  switch (env.event) {
    case 'meta':
      handlers.onMeta?.(env, env.data as S9Meta)
      break
    case 'sources':
      handlers.onSources?.(env, env.data as S9Sources)
      break
    case 'delta':
      handlers.onDelta?.((env.data as { delta: string }).delta)
      break
    case 'reasoning':
      handlers.onReasoning?.((env.data as { delta: string }).delta)
      break
    case 'done':
      handlers.onDone?.(env, env.data as S9Done)
      break
    case 'error':
      handlers.onError?.(env, env.data as S9Error)
      break
  }
}

export async function streamChat(
  req: S9ChatRequest,
  handlers: StreamHandlers,
  opts?: { signal?: AbortSignal; headers?: Record<string, string> },
): Promise<void> {
  if (inFlight) throw new ChatError('busy', '已有请求进行中，请先停止当前问题')
  inFlight = true
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(opts?.headers ?? {}) },
      body: JSON.stringify(req),
      signal: opts?.signal,
    })

    // 非 200 通道：503 维护 / 429 繁忙，body 为 {detail}（SSE 开始前返回）
    if (!res.ok) {
      let detail = ''
      try {
        const body: unknown = await res.json()
        detail = typeof (body as { detail?: unknown }).detail === 'string'
          ? (body as { detail: string }).detail
          : ''
      } catch { /* body 非 JSON 时忽略 */ }
      throw new ChatError('http', detail || `请求失败（HTTP ${res.status}）`, res.status)
    }
    if (!res.body) throw new ChatError('sse', '响应无流式内容')
    if (!res.headers.get('content-type')?.includes('text/event-stream')) {
      throw new ChatError('sse', '响应类型不是 SSE')
    }

    const reader = res.body.getReader()
    // 单实例 stream:true 跨 chunk 复用：中文多字节 UTF-8 序列可能跨网络 chunk 断裂，
    // 每次新建 decoder 会把断裂字变成 U+FFFD
    const dec = new TextDecoder('utf-8')
    let buf = ''
    let dataLines: string[] = []
    let terminal = false // 收到 done/error 才算正常结束（服务端断流未发终帧 = 异常）

    const flushFrame = (): void => {
      if (!dataLines.length) return
      const raw = dataLines.join('\n')
      dataLines = []
      let env: S9Envelope
      try {
        env = JSON.parse(raw) as S9Envelope
      } catch {
        throw new ChatError('parse', `SSE 帧解析失败：${raw.slice(0, 120)}`)
      }
      if (!S9_EVENTS.includes(env.event)) {
        throw new ChatError('parse', `未知 SSE 事件：${String(env.event)}`)
      }
      if (typeof env.data !== 'object' || env.data === null) {
        throw new ChatError('parse', `SSE ${env.event} 事件 data 为空`)
      }
      if (env.event === 'done' || env.event === 'error') terminal = true
      dispatch(handlers, env)
    }

    const consumeLine = (line: string): void => {
      if (line === '') {
        flushFrame() // 空行 = 一帧结束
      } else if (line.startsWith(':')) {
        // 心跳注释行 `: ping`，忽略
      } else if (line.startsWith('data:')) {
        // 剥 `data:` 前缀与单个空格（兼容多行 data，服务端只发单行）
        dataLines.push(line.slice(5).replace(/^ /, ''))
      }
      // event:/id:/其他行服务端不发，防御性忽略
    }

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      let nl: number
      while ((nl = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, nl).replace(/\r$/, '')
        buf = buf.slice(nl + 1)
        consumeLine(line)
      }
    }
    // 流结束冲刷：残行 + 未闭合帧（服务端恒以 \n\n 结尾，纯防御）
    buf += dec.decode()
    if (buf !== '') consumeLine(buf.replace(/\r$/, ''))
    flushFrame()
    if (!terminal) throw new ChatError('sse', '连接中断，未收到完成事件')
  } catch (e) {
    if (e instanceof ChatError) throw e
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new ChatError('aborted', '已停止')
    }
    if (e instanceof TypeError) throw new ChatError('network', '网络错误，请检查连接')
    throw e
  } finally {
    inFlight = false
  }
}
