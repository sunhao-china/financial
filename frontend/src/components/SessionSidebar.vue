<script setup lang="ts">
// 会话侧栏（§14.5）：新建会话按钮 + 历史会话列表（标题=首问截断，服务端懒落库）+ 退出登录。
import { computed } from 'vue'
import type { Conversation } from '../api/auth'
import { auth, clearAuth } from '../api/auth'

const props = defineProps<{
  conversations: Conversation[]
  currentId: string
  streaming: boolean
}>()

const emit = defineEmits<{ newSession: []; select: [id: string]; logout: [] }>()

// 列表未刷新前可能尚无当前会话行：把它放最前展示（标题「新会话」）
const rows = computed(() => {
  const cur = props.conversations.find(c => c.id === props.currentId)
  if (!cur) {
    return [{ id: props.currentId, title: '新会话', updated_at: '', created_at: '' },
      ...props.conversations]
  }
  return props.conversations
})

// 服务端契约（§14.5 时区修复）：所有历史时间 = 上海时间墙钟字符串（无时区标记），
// 显式标 +08:00 解析成绝对时刻，任意时区浏览器显示一致。
function shortTime(s: string): string {
  if (!s) return ''
  const t = s.replace(' ', 'T') + '+08:00'
  const d = new Date(t)
  if (isNaN(d.getTime())) return s.slice(5, 16)
  const now = new Date()
  if (d.toDateString() === now.toDateString()) {
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
  }
  return `${d.getMonth() + 1}/${d.getDate()}`
}
</script>

<template>
  <aside class="sessions">
    <button class="new-session" type="button" :disabled="streaming" @click="emit('newSession')">
      ＋ 新建会话
    </button>
    <nav class="session-list">
      <button
        v-for="c in rows"
        :key="c.id"
        type="button"
        class="session-item"
        :class="{ active: c.id === currentId }"
        :title="c.title"
        @click="emit('select', c.id)"
      >
        <span class="session-title">{{ c.title }}</span>
        <span v-if="shortTime(c.updated_at)" class="session-time">{{ shortTime(c.updated_at) }}</span>
      </button>
    </nav>
    <footer class="session-foot">
      <span class="who" :title="auth.username">{{ auth.username }}</span>
      <button type="button" class="logout" @click="clearAuth(); emit('logout')">退出</button>
    </footer>
  </aside>
</template>
