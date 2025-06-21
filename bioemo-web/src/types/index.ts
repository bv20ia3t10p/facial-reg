// Type definitions
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

export interface EmotionProbabilities {
  neutral?: number;
  happiness?: number;
  sadness?: number;
  anger?: number;
  surprise?: number;
  fear?: number;
  disgust?: number;
}

export interface EmotionData {
  emotion?: string;
  confidence?: number;
  probabilities?: EmotionProbabilities;
  timestamp?: string;
  // New format fields
  neutral?: number;
  happiness?: number;
  surprise?: number;
  sadness?: number;
  anger?: number;
  disgust?: number;
  fear?: number;
  [key: string]: number | string | EmotionProbabilities | undefined;
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
  id: string;
  user_id: string;
  success: boolean;
  confidence: number;
  timestamp: string;
  emotion_data?: EmotionData;
  device_info?: string;
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
  token?: string;
  threshold?: number;
  meets_threshold?: boolean;
  department?: string;
  role?: string;
  name?: string;
  email?: string;
  user?: {
    id: string;
    name: string;
    email?: string;
    role?: string;
    department?: string;
  };
}

export interface DashboardStats {
  totalAuthentications: number;
  averageConfidence: number;
  emotionBreakdown: {
    [date: string]: EmotionPrediction;
  };
  recentActivity: AuthenticationAttempt[];
}

export interface AuthenticationStats {
  total_attempts: number;
  successful_attempts: number;
  success_rate: number;
  average_confidence: number;
}

export interface EmotionTrends {
  average: EmotionData;
  dominant: string;
}

export interface UserInfo {
  user_id: string;
  name: string;
  email: string;
  department: string;
  role: string;
  enrolled_at: string;
  last_authenticated: string;
  authentication_stats: AuthenticationStats;
  recent_attempts: AuthenticationAttempt[];
  latest_auth: AuthenticationAttempt | null;
  emotional_state: EmotionData;
  emotion_trends: EmotionTrends;
  stats?: {
    successRate: number;
    avgResponseTime: number;
    wellbeingScore: number;
    authenticationsCount: number;
  };
  recentActivity?: Array<{
    type: 'success' | 'warning' | 'info';
    description: string;
    timestamp: string;
  }>;
}

export interface EmotionAdvice {
  title: string;
  description: string;
  suggestions: string[];
}

export interface EmotionAdviceMap {
  [key: string]: EmotionAdvice;
}

export interface EmotionDistribution {
  [key: string]: number;
}

export interface AnalyticsStats {
  dailyAuthentications: number;
  averageConfidence: number;
  emotionDistribution: EmotionDistribution;
  recentAuthentications: AuthenticationAttempt[];
}

export interface Analytics {
  dailyAuthentications: number;
  averageConfidence: number;
  emotionDistribution: EmotionPrediction;
  emotionTrends: Array<{
    timestamp: string;
    emotions: EmotionPrediction;
  }>;
  recentAuthentications: AuthenticationAttempt[];
}

export interface VerificationRequest {
  id: string;
  employeeId: string;
  reason: string;
  additionalNotes?: string;
  capturedImage: string;
  confidence: number;
  status: 'pending' | 'approved' | 'rejected';
  submittedAt: string;
}

export interface PasswordVerificationProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: (user: User) => void;
}

