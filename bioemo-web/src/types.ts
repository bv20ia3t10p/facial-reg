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