import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  // loadEnv với prefix '' để đọc được tất cả biến (kể cả không có VITE_ prefix)
  const env = loadEnv(mode, process.cwd(), '')
  const chatTarget = env.CHAT_PROXY_TARGET || 'https://ai-demo-chat-service.nicesky-ba1698a3.southeastasia.azurecontainerapps.io'

  return {
    plugins: [react()],
    server: {
      proxy: {
        // ====================================================================
        // Proxy SSE streaming qua Node.js để tránh browser HTTP/2 buffering.
        // Browser gọi same-origin /api/v1/chat → Vite proxy (Node.js HTTP) → Azure.
        // Node.js không bị buffer như browser fetch (giống VS Code extension).
        // ====================================================================
        '/api/v1/chat': {
          target: chatTarget,
          changeOrigin: true,
          // Tắt buffer cho SSE streaming — quan trọng!
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              // Ép proxy không buffer response
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
              proxyRes.headers['x-accel-buffering'] = 'no'
            })
          },
        },
        '/api/v1/history': {
          target: chatTarget,
          changeOrigin: true,
        },
        '/api/v1/threads': {
          target: chatTarget,
          changeOrigin: true,
        },
      },
    },
  }
})