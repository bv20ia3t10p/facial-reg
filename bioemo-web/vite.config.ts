import { defineConfig, loadEnv, ConfigEnv, UserConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }: ConfigEnv): UserConfig => {
  // Load env file based on `mode` in the current working directory.
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
  plugins: [react()],
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
    define: {
      // Pass environment variables to the client
      __API_SERVER_URL__: JSON.stringify(env.VITE_API_SERVER_URL || 'http://localhost:8000')
    }
  }
})
