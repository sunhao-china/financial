<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { ChatMessage } from '../composables/useChat'
import { focusRef } from '../composables/useChat'
import type { RankSource } from '../types/contract'

const props = defineProps<{ messages: ChatMessage[] }>()

// 面板第 i 张卡片 = 生成上下文【资料 i+1】——顺序由服务端保证：
// server/hooks.py 在 assemble 步骤发 sources 事件（to_s5hits(ctx.chunks[:20])），
// corpus/agent.py _assemble 用同一列表按序编 ref_no=1..N，正文 [n] 与卡片一一对应。
// 拒答路径无 sources → 面板隐藏。
const lastWithRefs = computed<ChatMessage | null>(() => {
  for (let i = props.messages.length - 1; i >= 0; i--) {
    const m = props.messages[i]!
    if (m.role === 'assistant' && m.references?.length) return m
  }
  return null
})

// 点旧消息的 [n] 时面板临时切到该消息；新答案到达即跟随最新
const overrideMsgId = ref<string | null>(null)
watch(() => lastWithRefs.value?.id, () => { overrideMsgId.value = null })
watch(() => focusRef.value, f => { if (f) overrideMsgId.value = f.msgId })

const current = computed<ChatMessage | null>(() => {
  if (overrideMsgId.value) {
    const m = props.messages.find(x => x.id === overrideMsgId.value)
    if (m?.references?.length) return m
  }
  return lastWithRefs.value
})

const cited = computed(() => new Set(current.value?.citations ?? []))
const citedCount = computed(() =>
  current.value ? current.value.references?.filter(r => cited.value.has(r.chunk_id)).length ?? 0 : 0,
)
const refCount = computed(() => current.value?.references?.length ?? 0)

const RANK_LABEL: Record<RankSource, string> = { vector: '向量', bm25: 'BM25', fusion: '融合' }

// 展开/闪烁状态随数据源切换重置
const expanded = ref<Set<number>>(new Set())
const flashNo = ref(0)
let flashTimer: ReturnType<typeof setTimeout> | undefined
watch(() => current.value?.id, () => { expanded.value = new Set() })

// 正文 [n] 点击 → 展开 + 滚动定位 + 高亮闪烁（nonce 自增，同编号重复点击也重触发）
watch(() => focusRef.value, async f => {
  if (!f || f.msgId !== current.value?.id) return
  if (f.no < 1 || f.no > refCount.value) return
  expanded.value = new Set([...expanded.value, f.no])
  flashNo.value = f.no
  await nextTick()
  asideEl.value
    ?.querySelector(`[data-no="${f.no}"]`)
    ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  if (flashTimer) clearTimeout(flashTimer)
  flashTimer = setTimeout(() => { flashNo.value = 0 }, 1600)
})

const asideEl = ref<HTMLElement | null>(null)

function toggle(no: number): void {
  const s = new Set(expanded.value)
  if (s.has(no)) s.delete(no)
  else s.add(no)
  expanded.value = s
}

function pct(score: number): string {
  return `${(score * 100).toFixed(0)}%`
}
</script>

<template>
  <aside ref="asideEl" class="citations" :class="{ hidden: !current }">
    <template v-if="current">
      <h3>
        引用来源
        <span v-if="current.citations?.length" class="count">已引用 {{ citedCount }}/{{ refCount }}</span>
      </h3>
      <p class="panel-hint">点击正文中的 [n] 或卡片，可对照原文</p>
      <article
        v-for="(hit, i) in current.references"
        :key="hit.chunk_id"
        :data-no="i + 1"
        class="ref"
        :class="{ cited: cited.has(hit.chunk_id), flash: flashNo === i + 1 }"
        @click="toggle(i + 1)"
      >
        <header class="ref-head">
          <span class="ref-no">资料 {{ i + 1 }}</span>
          <span class="doc" :title="`来源文章编号 ${hit.doc_id}`">文章 #{{ hit.doc_id }}</span>
          <span class="frag">{{ hit.chunk_index + 1 }}/{{ hit.chunk_count }} 片段</span>
        </header>
        <p v-if="hit.doc_title" class="ref-title" :title="hit.doc_title">{{ hit.doc_title }}</p>
        <p class="ref-text" :class="{ clamp: !expanded.has(i + 1) }">{{ hit.text }}</p>
        <footer class="ref-foot">
          <span v-if="!expanded.has(i + 1)" class="hint">点击看全文</span>
          <span v-else class="tech">
            质量分 {{ hit.quality_score.toFixed(1) }} · {{ RANK_LABEL[hit.rank_source] }}检索 {{ pct(hit.score) }}
          </span>
        </footer>
      </article>
    </template>
  </aside>
</template>
