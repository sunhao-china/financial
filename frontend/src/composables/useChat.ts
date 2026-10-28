// 聊天状态机：消息流 + SSE 回调落到「最后一条 assistant 消息」+ abort。
// 多轮约定（§14.5）：只发 session_id 不发 history（服务端自读 MySQL messages），
// session_id 由「新建会话」生成；历史会话切换时拉取历史消息本地渲染。
import { computed, reactive, ref } from 'vue'
import { auth, clearAuth, fetchConversations, fetchConvMessages } from '../api/auth'
import type { Conversation } from '../api/auth'
import { ChatError, streamChat } from '../api/chat'
import type { Intent, S5Hit, S9Done } from '../types/contract'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string // done 前为增量累加，done 后为权威替换
  streaming?: boolean
  intent?: Intent
  references?: S5Hit[]
  citations?: string[]
  verification?: S9Done['verification']
  costTokens?: number
  genMs?: number
  reasoning?: string
  error?: { kind: string; status?: number; detail?: string }
}

function genId(): string {
  // crypto.randomUUID 仅在安全上下文（https/localhost）可用——线上 http:// 直连时
  // 它是 undefined，直接调用会导致 send() 抛 TypeError、请求静默发不出去
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

// ---------- 引用焦点（跨组件共享，模块级单例） ----------
// MessageItem 点击正文 [n] → CitationPanel 滚动定位并高亮对应卡片。
// 单视图应用无 pinia，用模块级 ref 免去 App→MessageList→MessageItem 事件转发链；
// nonce 每次点击自增，同编号重复点击也能重触发闪烁。
export interface RefFocus { msgId: string; no: number; nonce: number }

export const focusRef = ref<RefFocus | null>(null)

export function jumpToRef(msgId: string, no: number): void {
  focusRef.value = { msgId, no, nonce: (focusRef.value?.nonce ?? 0) + 1 }
}

export function useRefFocus() {
  return { focusRef, jumpToRef }
}

export function useChat() {
  const messages = ref<ChatMessage[]>([])
  const sessionId = ref(genId()) // 新建会话即生成随机 id（§14.5），首问懒落库
  const conversations = ref<Conversation[]>([])
  const abortCtrl = ref<AbortController | null>(null)
  const streaming = computed(() => messages.value.some(m => m.streaming))

  async function loadConversations(): Promise<void> {
    try {
      conversations.value = await fetchConversations()
    } catch {
      conversations.value = [] // 列表拉取失败不阻塞聊天
    }
  }

  function newSession(): void {
    abortCtrl.value?.abort() // 正在流式的请求停止（服务端照常完成并落库）
    sessionId.value = genId()
    messages.value = []
  }

  async function openConversation(convId: string): Promise<void> {
    if (streaming.value) return
    try {
      const rows = await fetchConvMessages(convId)
      sessionId.value = convId
      messages.value = rows.map(m => ({
        id: genId(),
        role: m.role,
        content: m.content,
        // 历史消息无 references/verification：正文 [n] 不可点（面板亦隐藏），
        // 引用联动只对本次会话内的实时回答生效
      }))
    } catch {
      // 拉取失败保持当前会话不动
    }
  }

  async function send(question: string): Promise<void> {
    if (streaming.value || !auth.token) return
    // 必须 reactive：SSE 回调改的是闭包里的这个对象，若为裸对象则修改绕过
    // 响应式代理（经典 Vue 3 陷阱）——组件只会在 push 时渲染一次，流式增量、
    // done 落袋、streaming 状态切换全部不触发重渲染，UI 永远停在「分析中」
    const asstMsg = reactive<ChatMessage>({
      id: genId(), role: 'assistant', content: '', streaming: true,
    })
    messages.value.push(
      { id: genId(), role: 'user', content: question },
      asstMsg,
    )
    const ctrl = new AbortController()
    abortCtrl.value = ctrl

    try {
      await streamChat(
        { schema_version: '1.0.0', question, session_id: sessionId.value },
        {
          onMeta: (_env, m) => { asstMsg.intent = m.intent },
          onSources: (_env, s) => { asstMsg.references = s.references },
          onDelta: t => { asstMsg.content += t },
          onReasoning: t => { asstMsg.reasoning = (asstMsg.reasoning ?? '') + t },
          onDone: (_env, d) => {
            asstMsg.content = d.answer // done.answer 为最终权威（覆盖增量累加）
            asstMsg.citations = d.citations
            asstMsg.verification = d.verification
            asstMsg.costTokens = d.cost_tokens.total
            asstMsg.genMs = d.timings_ms.generate
          },
          onError: (_env, e) => { asstMsg.error = { kind: 'sse', detail: e.message } },
        },
        {
          signal: ctrl.signal,
          headers: { Authorization: `Bearer ${auth.token}` },
        },
      )
    } catch (e) {
      const err = e as ChatError
      if (err?.status === 401) {
        clearAuth() // token 失效/未登录 → 回登录页
        return
      }
      if (err?.kind === 'aborted') {
        if (!asstMsg.content) asstMsg.content = '（已停止）'
      } else {
        asstMsg.error = {
          kind: err?.kind ?? 'unknown',
          status: err?.status,
          detail: err?.message,
        }
      }
    } finally {
      asstMsg.streaming = false
      abortCtrl.value = null
      void loadConversations() // 会话列表刷新（新会话懒落库/updated_at 前移）
    }
  }

  function stop(): void {
    abortCtrl.value?.abort()
    // 服务端 GeneratorExit 后 worker 照常跑完并写历史（main.py），abort 无副作用
  }

  return {
    messages, sessionId, conversations, streaming,
    send, stop, newSession, openConversation, loadConversations,
  }
}
