/**
 * Environment configuration utility
 * Provides typed access to environment variables
 */

export interface EnvConfig {
  /** Authentication API server base URL (time-sensitive) */
  authServerUrl: string;
  /** Emotion analysis API server base URL (less time-sensitive) */
  emotionServerUrl: string;
  /** Whether to use mock data instead of real API calls */
  useMockApi: boolean;
}

/**
 * Get environment configuration with type safety
 */
export function getEnvConfig(): EnvConfig {
  const authServerUrl = import.meta.env.VITE_AUTH_SERVER_URL || 'http://localhost:8001';
  const emotionServerUrl = import.meta.env.VITE_EMOTION_SERVER_URL || 'http://localhost:1235';
  const useMockApi = import.meta.env.VITE_USE_MOCK_API === 'true';
  
  if (!authServerUrl && !useMockApi) {
    throw new Error('VITE_AUTH_SERVER_URL environment variable is required when not using mock API');
  }

  if (!emotionServerUrl && !useMockApi) {
    throw new Error('VITE_EMOTION_SERVER_URL environment variable is required when not using mock API');
  }

  return {
    authServerUrl,
    emotionServerUrl,
    useMockApi
  };
}

/**
 * Environment configuration singleton
 */
export const envConfig = getEnvConfig(); 