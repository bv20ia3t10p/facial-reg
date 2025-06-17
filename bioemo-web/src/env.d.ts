/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * The base URL for the API server
   * @default "http://localhost:8000"
   */
  readonly VITE_API_SERVER_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
} 