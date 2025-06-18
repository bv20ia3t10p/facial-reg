// Type definitions
export interface EmotionPrediction {
  happiness: number;
  neutral: number;
  surprise: number;
  sadness: number;
  anger: number;
  disgust: number;
  fear: number;
}

export interface EmotionProbabilities {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  surprised: number;
  fearful: number;
  disgusted: number;
}

export interface EmotionData {
  emotion: string;
  confidence: number;
  probabilities: EmotionProbabilities;
  timestamp: string;
}

export interface NormalizedEmotionData extends EmotionData {
  normalized: EmotionPrediction;
}

export interface User {
  id: string;
  name: string;
  email?: string;
  role?: string;
  department?: string;
  joinDate?: string;
  lastAuthenticated?: string;
}

export interface AuthenticationAttempt {
  timestamp: string;
  success: boolean;
  confidence: number;
  emotion_data: EmotionData;
}

export interface AuthenticationResponse {
  success: boolean;
  user_id: string;
  confidence: number;
  emotions?: EmotionPrediction;
  error?: string;
  message?: string;
  authenticated_at?: string;
  capturedImage?: string;
}

export interface DashboardStats {
  totalAuthentications: number;
  averageConfidence: number;
  emotionBreakdown: {
    [date: string]: EmotionPrediction;
  };
  recentActivity: AuthenticationAttempt[];
}

export interface UserInfo {
  user_id: string;
  name: string;
  email: string;
  department: string;
  role: string;
  enrolled_at: string;
  last_authenticated: string;
  authentication_stats: {
    total_attempts: number;
    successful_attempts: number;
    success_rate: number;
    average_confidence: number;
  };
  recent_attempts: AuthenticationAttempt[];
  latest_auth: AuthenticationAttempt;
  emotional_state: EmotionData;
}

export interface EmotionAdvice {
  title: string;
  description: string;
  suggestions: string[];
}

export interface EmotionAdviceMap {
  [key: string]: EmotionAdvice;
}

