// 用户系统客户端（§14.5）：token 管理 + /api 封装。
// token 存 localStorage（7 天，服务端签发）；auth 为模块级响应式状态，
// App.vue 以 auth.token 非空决定登录页/主界面切换；401 时 clearAuth 即回登录页。
import { reactive } from 'vue'

const TOKEN_KEY = 'fin_token'
const USER_KEY = 'fin_user'

export const auth = reactive<{ token: string; username: string }>({
  token: localStorage.getItem(TOKEN_KEY) ?? '',
  username: localStorage.getItem(USER_KEY) ?? '',
})

export interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
}

export interface ConvMessage {
  role: 'user' | 'assistant'
  content: string
  ts: string
}

export class ApiError extends Error {
  status: number

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
  }
}

function saveAuth(token: string, username: string): void {
  auth.token = token
  auth.username = username
  localStorage.setItem(TOKEN_KEY, token)
  localStorage.setItem(USER_KEY, username)
}

export function clearAuth(): void {
  auth.token = ''
  auth.username = ''
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}

async function post<T>(path: string, body: unknown, headers?: Record<string, string>): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(headers ?? {}) },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = `请求失败（HTTP ${res.status}）`
    try {
      const j = await res.json()
      if (typeof j.detail === 'string') detail = j.detail
    } catch { /* 非 JSON body 用默认话术 */ }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

async function get<T>(path: string, headers?: Record<string, string>): Promise<T> {
  const res = await fetch(path, { headers })
  if (!res.ok) {
    let detail = `请求失败（HTTP ${res.status}）`
    try {
      const j = await res.json()
      if (typeof j.detail === 'string') detail = j.detail
    } catch { /* ignore */ }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

const authHeaders = () => ({ Authorization: `Bearer ${auth.token}` })

// ---------- 接口 ----------

export async function fetchCaptcha(): Promise<{ captcha_id: string; image: string }> {
  return get('/api/captcha')
}

export async function apiLogin(
  username: string, password: string, captcha_id: string, captcha_code: string,
): Promise<void> {
  const r = await post<{ token: string; username: string }>('/api/login',
    { username, password, captcha_id, captcha_code })
  saveAuth(r.token, r.username)
}

export async function apiRegister(
  username: string, password: string, captcha_id: string, captcha_code: string,
): Promise<void> {
  const r = await post<{ token: string; username: string }>('/api/register',
    { username, password, captcha_id, captcha_code })
  saveAuth(r.token, r.username)
}

/** 启动时验证本地 token 是否仍有效；无效则清除并回登录页。 */
export async function apiMe(): Promise<void> {
  const r = await get<{ username: string }>('/api/me', authHeaders())
  auth.username = r.username
}

export async function fetchConversations(): Promise<Conversation[]> {
  const r = await get<{ conversations: Conversation[] }>('/api/conversations', authHeaders())
  return r.conversations
}

export async function fetchConvMessages(convId: string): Promise<ConvMessage[]> {
  const r = await get<{ messages: ConvMessage[] }>(
    `/api/conversations/${encodeURIComponent(convId)}/messages`, authHeaders())
  return r.messages
}
