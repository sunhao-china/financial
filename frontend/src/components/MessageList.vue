<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import type { ChatMessage } from '../composables/useChat'
import MessageItem from './MessageItem.vue'

const props = defineProps<{ messages: ChatMessage[] }>()
const emit = defineEmits<{ ask: [q: string] }>()

// 空会话示例问题：取自 P4 评估中 verdict=pass 的题目（§14.5），覆盖事实检索 /
// 分析归纳 / 跨文档多银行对比三类意图，点击即经 send 流程提问（首问懒落库）
const SUGGESTIONS = [
  '2013年至2021年，中国GDP年均增长率为多少？',
  '2021年全国综合保税区进出口值是多少？',
  '为什么新能源汽车近几年的超高速增长主要依靠政策推动而非市场力量？',
  '2022年一季度末，农业银行、邮储银行、招商银行的拨备覆盖率分别是多少？',
]

const scroller = ref<HTMLElement | null>(null)
let pinned = true // 用户上滚后暂停自动滚底

function onScroll(): void {
  const el = scroller.value
  if (!el) return
  pinned = el.scrollHeight - el.scrollTop - el.clientHeight < 80
}

// deep：流式增量（消息对象内 content 追加）也要触发滚底
watch(() => props.messages, async () => {
  if (!pinned) return
  await nextTick()
  const el = scroller.value
  if (el) el.scrollTop = el.scrollHeight
}, { deep: true })
</script>

<template>
  <div ref="scroller" class="messages" @scroll="onScroll">
    <div v-if="!messages.length" class="empty">
      <p class="brand">金融问答助手</p>
      <p class="hint">基于 2012–2022 年财经新闻语料（约 286 万片段）<br />回答附引用来源，支持追问与指代</p>
      <div class="suggest">
        <p class="suggest-label">试试这样问：</p>
        <button v-for="q in SUGGESTIONS" :key="q" class="suggest-chip" @click="emit('ask', q)">
          {{ q }}
        </button>
      </div>
    </div>
    <MessageItem v-for="m in messages" :key="m.id" :msg="m" />
  </div>
</template>
