import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET || 'http://localhost:8000'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  // Keep the backend and browser configuration in the repository-root .env
  // so `npm run dev` configures both services from one file.
  envDir: '..',
  server: {
    proxy: {
      '/api': {
        target: apiProxyTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    // Don't attempt to delete existing dist files — the sandbox
    // file system may restrict unlinking files from previous builds.
    emptyOutDir: false,
  },
})
