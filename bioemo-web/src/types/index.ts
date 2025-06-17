export interface EmotionPrediction {
  happiness: number;
  neutral: number;
  surprise: number;
  sadness: number;
  anger: number;
  disgust: number;
  fear: number;
  [key: string]: number;
}

export interface User {
  id: string;
  name: string;
  email: string;
  department: string;
  role: string;
  joinDate: string;
  lastAuthenticated: string;
  emotionHistory?: AuthenticationAttempt[];
}

export interface AuthenticationAttempt {
  success: boolean;
  confidence: number;
  timestamp: string;
  emotions?: EmotionPrediction;
  message?: string;
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
  last_login: string;
  latest_auth: {
    timestamp: string | null;
    confidence: number | null;
    device_info: string | null;
  };
  authentication_stats: {
    total_attempts: number;
    successful_attempts: number;
    success_rate: number;
    average_confidence: number;
  };
  recent_attempts: Array<{
    timestamp: string;
    success: boolean;
    confidence: number;
    device_info?: string;
  }>;
  emotional_state: {
    happiness: number;
    neutral: number;
    surprise: number;
    sadness: number;
    anger: number;
    disgust: number;
    fear: number;
  };
  last_updated: string;
} 