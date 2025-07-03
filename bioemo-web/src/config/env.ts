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
  /** Confidence threshold for facial recognition (0.0 to 1.0) */
  confidenceThreshold: number;
  /** Low confidence threshold for verification requests (0.0 to 1.0) */
  lowConfidenceThreshold: number;
}

/**
 * Get environment configuration with type safety
 */
export function getEnvConfig(): EnvConfig {
  const authServerUrl = import.meta.env.VITE_AUTH_SERVER_URL || 'http://localhost:8001';
  const emotionServerUrl = import.meta.env.VITE_EMOTION_SERVER_URL || 'http://localhost:1235';
  const useMockApi = import.meta.env.VITE_USE_MOCK_API === 'true';
  
  // Parse confidence threshold with validation
  const thresholdStr = import.meta.env.VITE_CONFIDENCE_THRESHOLD;
  let confidenceThreshold = 0.9; // Default value
  
  if (thresholdStr) {
    const threshold = parseFloat(thresholdStr);
    if (!isNaN(threshold) && threshold >= 0.0 && threshold <= 1.0) {
      confidenceThreshold = threshold;
    } else {
      console.warn(`Invalid VITE_CONFIDENCE_THRESHOLD value: ${thresholdStr}. Using default: 0.7`);
    }
  }
  
  // Parse low confidence threshold with validation
  const lowThresholdStr = import.meta.env.VITE_LOW_CONFIDENCE_THRESHOLD;
  let lowConfidenceThreshold = 0.5; // Default value
  
  if (lowThresholdStr) {
    const threshold = parseFloat(lowThresholdStr);
    if (!isNaN(threshold) && threshold >= 0.0 && threshold <= 1.0) {
      lowConfidenceThreshold = threshold;
    } else {
      console.warn(`Invalid VITE_LOW_CONFIDENCE_THRESHOLD value: ${lowThresholdStr}. Using default: 0.3`);
    }
  }
  
  if (!authServerUrl && !useMockApi) {
    throw new Error('VITE_AUTH_SERVER_URL environment variable is required when not using mock API');
  }

  if (!emotionServerUrl && !useMockApi) {
    throw new Error('VITE_EMOTION_SERVER_URL environment variable is required when not using mock API');
  }

  return {
    authServerUrl,
    emotionServerUrl,
    useMockApi,
    confidenceThreshold,
    lowConfidenceThreshold
  };
}

/**
 * Environment configuration singleton
 */
export const envConfig = getEnvConfig(); 