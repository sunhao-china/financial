<script setup lang="ts">
// 登录/注册视图（§14.5）：用户名 + 密码 + 图片验证码（点击换一张）。
// 成功后 auth.token 非空，App.vue 自动切到主界面。
import { onMounted, ref } from 'vue'
import { ApiError, apiLogin, apiRegister, fetchCaptcha } from '../api/auth'

const mode = ref<'login' | 'register'>('login')
const username = ref('')
const password = ref('')
const captchaCode = ref('')
const captchaId = ref('')
const captchaImg = ref('')
const busy = ref(false)
const error = ref('')

async function refreshCaptcha(): Promise<void> {
  try {
    const c = await fetchCaptcha()
    captchaId.value = c.captcha_id
    captchaImg.value = c.image
  } catch {
    error.value = '验证码加载失败，请刷新页面重试'
  }
}

async function submit(): Promise<void> {
  if (busy.value) return
  error.value = ''
  if (!username.value.trim() || !password.value) {
    error.value = '请输入用户名和密码'
    return
  }
  if (!captchaCode.value.trim()) {
    error.value = '请输入图片验证码'
    return
  }
  busy.value = true
  try {
    const fn = mode.value === 'login' ? apiLogin : apiRegister
    await fn(username.value.trim(), password.value, captchaId.value,
             captchaCode.value.trim().toUpperCase())
    // 成功：auth 已写入，App 自动切换；无需在此跳转
  } catch (e) {
    error.value = e instanceof ApiError ? e.message : '请求失败，请稍后重试'
    await refreshCaptcha() // 验证码一次性，失败后必须换新
    captchaCode.value = ''
  } finally {
    busy.value = false
  }
}

function switchMode(): void {
  mode.value = mode.value === 'login' ? 'register' : 'login'
  error.value = ''
  password.value = ''
  captchaCode.value = ''
  void refreshCaptcha()
}

onMounted(() => { void refreshCaptcha() })
</script>

<template>
  <div class="login">
    <form class="login-box" @submit.prevent="submit">
      <h1>金融问答</h1>
      <p class="sub">2012–2022 财经新闻语料 · RAG</p>

      <label class="field">
        <span>用户名</span>
        <input v-model="username" type="text" autocomplete="username"
               placeholder="3-32 位字母、数字或下划线" />
      </label>
      <label class="field">
        <span>密码</span>
        <input v-model="password" type="password" autocomplete="current-password"
               :placeholder="mode === 'register' ? '至少 8 位，需含字母和数字' : '请输入密码'" />
      </label>
      <label class="field captcha-row">
        <span>验证码</span>
        <div class="captcha-line">
          <input v-model="captchaCode" type="text" maxlength="5"
                 placeholder="图片中字符（不区分大小写）" autocomplete="off" />
          <img v-if="captchaImg" :src="captchaImg" alt="验证码"
               title="点击换一张" @click="refreshCaptcha" />
        </div>
      </label>

      <p v-if="error" class="login-error">{{ error }}</p>

      <button class="login-btn" type="submit" :disabled="busy">
        {{ busy ? '请稍候…' : (mode === 'login' ? '登 录' : '注 册') }}
      </button>
      <p class="switch">
        {{ mode === 'login' ? '没有账号？' : '已有账号？' }}
        <a href="#" @click.prevent="switchMode">
          {{ mode === 'login' ? '注册一个' : '去登录' }}
        </a>
      </p>
    </form>
  </div>
</template>
