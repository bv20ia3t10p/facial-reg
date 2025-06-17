export interface User {
  id: string;
  name: string;
  email: string;
  department: string;
  role: string;
  joinDate?: string;
  lastAuthenticated?: string;
}

export interface EmotionPrediction {
  happiness: number;
  neutral: number;
  surprise: number;
  sadness: number;
  anger: number;
  disgust: number;
  fear: number;
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

export interface Authentication {
  id: string;
  timestamp: string;
  success: boolean;
  confidence: number;
  dominantEmotion: string;
}

export interface Stats {
  totalUsers: number;
  activeUsers: number;
  pendingVerifications: number;
}

export interface Analytics {
  dailyAuthentications: number;
  averageConfidence: number;
  emotionDistribution: EmotionPrediction;
}

export interface WellbeingMetrics {
  stressLevel: number;
  jobSatisfaction: number;
  emotionalBalance: number;
  wellbeingScore: number;
}

export interface DepartmentWellbeing {
  department: string;
  metrics: WellbeingMetrics;
  trendData: Array<{
    timestamp: string;
    metrics: WellbeingMetrics;
  }>;
}

export interface HRAnalytics {
  overallWellbeing: WellbeingMetrics;
  departmentAnalytics: DepartmentWellbeing[];
  recentEmotionalTrends: Array<{
    timestamp: string;
    emotionDistribution: Array<{
      emotion: string;
      percentage: number;
    }>;
  }>;
  alerts: Array<{
    id: string;
    type: 'stress' | 'satisfaction' | 'wellbeing';
    severity: 'low' | 'medium' | 'high';
    department: string;
    message: string;
    timestamp: string;
  }>;
}

export interface PasswordVerificationProps {
  onSubmit: (username: string, password: string) => Promise<void>;
  onCancel: () => void;
  loading?: boolean;
  visible: boolean;
  onClose: () => void;
  onSuccess: (user: User) => void;
}

export interface AuthenticationResponse {
  success: boolean;
  user_id: string;
  confidence: number;
  threshold: number;
  authenticated_at: string;
  emotions?: EmotionPrediction;
}

export interface UserInfo {
  user_id: string;
  enrolled_at: string;
  last_login: string;
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
    device_info: string | null;
  }>;
  latest_auth: {
    timestamp: string | null;
    confidence: number | null;
    device_info: string | null;
  };
  emotional_state: EmotionPrediction;
  last_updated: string;
}

export interface UserProfileProps {
  user: User;
  emotions: EmotionPrediction;
  userInfo?: UserInfo | null;
  isLoading?: boolean;
} 