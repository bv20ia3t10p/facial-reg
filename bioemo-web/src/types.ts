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
  processedAt?: string | null;
  user?: {
    id: string;
    name: string;
    department: string;
  }
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
  message?: string;
  error?: string;
  capturedImage?: string;
}

export interface EmotionProbabilities {
  neutral: number;
  happy: number;
  sad: number;
  angry: number;
  surprised: number;
  fearful: number;
  disgusted: number;
  [key: string]: number; // Allow for other emotion keys
}

export interface EmotionData {
  // Original format fields
  emotion?: string;
  confidence?: number;
  probabilities?: EmotionProbabilities;
  timestamp?: string;
  
  // RAF-DB format fields (direct properties)
  neutral?: number;
  happiness?: number;
  sadness?: number;
  anger?: number;
  surprise?: number;
  fear?: number;
  disgust?: number;
  
  // Allow for additional properties
  [key: string]: any;
}

export interface NormalizedEmotionData {
  emotion: string;
  confidence: number;
  probabilities: EmotionProbabilities;
  timestamp: string;
  normalized: EmotionPrediction;
}

export interface AuthenticationAttempt {
  timestamp: string;
  success: boolean;
  confidence: number;
  emotion_data: EmotionData | string; // Can be object or JSON string
}

export interface AuthenticationStats {
  total_attempts: number;
  successful_attempts: number;
  success_rate: number;
  average_confidence: number;
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
  latest_auth: AuthenticationAttempt;
  emotional_state: EmotionPrediction;
  emotion_trends?: {
    average: EmotionPrediction;
    dominant: string;
  };
}

export interface UserProfileProps {
  user: User;
  emotions: EmotionPrediction;
  userInfo?: UserInfo | null;
  isLoading?: boolean;
} 