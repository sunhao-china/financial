<script setup lang="ts">
import { computed, onMounted, watch } from 'vue'
import { auth, apiMe, clearAuth } from './api/auth'
import { useChat } from './composables/useChat'
import CitationPanel from './components/CitationPanel.vue'
import Composer from './components/Composer.vue'
import LoginView from './components/LoginView.vue'
import MessageList from './components/MessageList.vue'
import SessionSidebar from './components/SessionSidebar.vue'

const { messages, sessionId, conversations, streaming, send, stop, newSession,
        openConversation, loadConversations } = useChat()

const authed = computed(() => auth.token !== '')

onMounted(async () => {
  if (!auth.token) return
  try {
    await apiMe() // 本地 token 有效性校验（7 天过期/失效 → 回登录页）
    await loadConversations()
  } catch {
    clearAuth()
  }
})

// 登录成功 → 拉会话列表；退出登录 → 清空消息区（App 组件实例不销毁，状态要显式重置）
watch(authed, (v) => {
  if (v) {
    void loadConversations()
  } else {
    newSession()
  }
})

function onNewSession(): void {
  newSession()
  void loadConversations()
}

function onSelect(convId: string): void {
  void openConversation(convId)
}
</script>

<template>
  <LoginView v-if="!authed" />
  <div v-else class="app">
    <header class="topbar">
      <h1>金融问答</h1>
      <span class="sub">2012–2022 财经新闻语料 · RAG</span>
    </header>
    <div class="body">
      <SessionSidebar
        :conversations="conversations"
        :current-id="sessionId"
        :streaming="streaming"
        @new-session="onNewSession"
        @select="onSelect"
      />
      <MessageList :messages="messages" @ask="send" />
      <CitationPanel :messages="messages" />
    </div>
    <Composer :streaming="streaming" @send="send" @stop="stop" />
  </div>
</template>
