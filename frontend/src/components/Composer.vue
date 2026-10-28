<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ streaming: boolean }>()
const emit = defineEmits<{ send: [text: string]; stop: [] }>()

const text = ref('')
const composing = ref(false) // IME 组合期（中文输入法选词时 Enter 不发送）

function onSubmit(): void {
  if (props.streaming) {
    emit('stop')
    return
  }
  const q = text.value.trim()
  if (!q) return
  emit('send', q)
  text.value = ''
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Enter' && !e.shiftKey && !composing.value) {
    e.preventDefault()
    onSubmit()
  }
}
</script>

<template>
  <div class="composer">
    <textarea
      v-model="text"
      rows="2"
      placeholder="输入金融问题，Enter 发送，Shift+Enter 换行"
      @keydown="onKeydown"
      @compositionstart="composing = true"
      @compositionend="composing = false"
    ></textarea>
    <button class="send" :class="{ stop: streaming }" @click="onSubmit">
      {{ streaming ? '停止' : '发送' }}
    </button>
  </div>
</template>
