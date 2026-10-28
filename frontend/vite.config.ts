import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// 开发期代理直连 ECS（本地无索引，后端跑在 8.137.183.17:8001）；
// SSE 由 http-proxy 默认透传，无需 buffering 配置。
// 本地 mock / 其他后端用 VITE_API_TARGET 切换：
//   VITE_API_TARGET=http://localhost:8010 npm run dev
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/chat': {
        target: process.env.VITE_API_TARGET ?? 'http://8.137.183.17:8001',
        changeOrigin: true,
      },
      '/api': { // 用户系统（§14.5）：登录/验证码/会话列表/历史消息
        target: process.env.VITE_API_TARGET ?? 'http://8.137.183.17:8001',
        changeOrigin: true,
      },
    },
  },
})
