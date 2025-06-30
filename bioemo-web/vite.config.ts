import { defineConfig, loadEnv } from 'vite'
import type { ConfigEnv, UserConfig } from 'vite'
import { resolve } from 'path'
import react from '@vitejs/plugin-react-swc'

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }: ConfigEnv): UserConfig => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [react()],
    // Allow using `@/` as an alias to the project src directory
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src')
      }
    },
    server: {
      port: 5174,
      strictPort: true,
      host: true,
      proxy: {
        // Proxy API requests to the backend server
        '/api': {
          target: env.VITE_API_SERVER_URL || 'http://localhost:8000',
          changeOrigin: true,
          secure: false,
          rewrite: (path: string) => path.replace(/^\/api/, '')
        }
      }
    },
    // Handle SPA routing - fallback to index.html for client-side routes
    appType: 'spa',
    build: {
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom'],
            router: ['react-router-dom'],
            ui: ['antd']
          }
        }
      }
    },
    // Preview server configuration (for production build)
    preview: {
      port: 5174,
      strictPort: true,
      host: true,
      // SPA fallback for preview mode
      open: true
    },
    define: {
      // Pass environment variables to the client
      __API_SERVER_URL__: JSON.stringify(env.VITE_API_SERVER_URL || 'http://localhost:8000')
    }
  }
})
