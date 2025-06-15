export interface EmotionPrediction {
  neutral: number;
  happiness: number;
  surprise: number;
  sadness: number;
  anger: number;
  disgust: number;
  fear: number;
  contempt: number;
}

export interface User {
  id: string;
  name: string;
  department: string;
  lastAuthenticated: string;
  emotionHistory: AuthenticationAttempt[];
}

export interface AuthenticationAttempt {
  success: boolean;
  confidence: number;
  timestamp: string;
  emotions?: EmotionPrediction;
  message?: string;
}

export interface DashboardStats {
  totalAuthentications: number;
  averageConfidence: number;
  emotionBreakdown: {
    [date: string]: EmotionPrediction;
  };
  recentActivity: AuthenticationAttempt[];
} 