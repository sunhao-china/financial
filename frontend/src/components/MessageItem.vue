<script setup lang="ts">
import { computed } from 'vue'
import type { ChatMessage } from '../composables/useChat'
import { jumpToRef } from '../composables/useChat'
import type { Intent, Verdict } from '../types/contract'
import { parseAnswerSegments } from '../utils/markers'

const props = defineProps<{ msg: ChatMessage }>()

const INTENT_LABEL: Record<Intent, string> = {
  fact_qa: '事实问答',
  analysis: '分析',
  comparison: '对比',
  realtime: '实时信息',
  investment_advice: '投资建议',
  chitchat: '闲聊',
}

const VERDICT_LABEL: Record<Verdict, string> = {
  pass: '校验通过',
  downgrade: '降级说明',
  fail: '校验未通过',
}

const verdict = computed(() => {
  const v = props.msg.verification?.verdict
  return v ? VERDICT_LABEL[v] : ''
})

const footnote = computed(() => {
  const parts: string[] = []
  if (props.msg.costTokens) parts.push(`${props.msg.costTokens} tokens`)
  if (props.msg.genMs) parts.push(`${(props.msg.genMs / 1000).toFixed(1)}s`)
  return parts.join(' · ')
})

// 正文引用标记：编号对应本消息的【资料 n】（面板第 n 张卡片，顺序由服务端保证）。
// 用户消息无 references → refCount 0 → 方括号内容一律原样纯文本。
const refCount = computed(() => props.msg.references?.length ?? 0)
const segments = computed(() => parseAnswerSegments(props.msg.content, refCount.value))
</script>

<template>
  <div class="msg" :class="msg.role">
    <div class="bubble">
      <div v-if="msg.role === 'assistant' && msg.intent" class="meta-row">
        <span class="intent-tag" :class="msg.intent">{{ INTENT_LABEL[msg.intent] }}</span>
      </div>

      <div v-if="msg.content" class="content">
        <template v-for="(seg, i) in segments" :key="i">
          <span v-if="seg.text !== undefined">{{ seg.text }}</span>
          <template v-else>
            <button
              v-for="n in seg.refs ?? []"
              :key="n"
              type="button"
              class="cite"
              :title="`查看资料 ${n} 原文`"
              @click="jumpToRef(msg.id, n)"
            >[{{ n }}]</button>
          </template>
        </template>
        <span v-if="msg.streaming" class="cursor"></span>
      </div>
      <div v-else-if="msg.streaming && !msg.error" class="content thinking">
        <span class="cursor"></span> 分析中…
      </div>

      <div v-if="msg.error" class="error-bubble">
        <span class="error-kind">{{ msg.error.status ? `HTTP ${msg.error.status}` : '出错了' }}</span>
        <p>{{ msg.error.detail || '请求失败，请稍后重试' }}</p>
      </div>

      <details v-if="msg.reasoning" class="reasoning">
        <summary>推理过程</summary>
        <p>{{ msg.reasoning }}</p>
      </details>

      <div class="meta-foot">
        <span v-if="verdict" class="verdict" :class="msg.verification?.verdict">{{ verdict }}</span>
        <span v-if="footnote" class="footnote">{{ footnote }}</span>
      </div>
    </div>
  </div>
</template>
