import type { User, DashboardStats, UserInfo, EmotionPrediction } from '../types';

export const mockUser: User = {
  id: "0000342",
  name: "Amy Smith",
  email: "amy.smith@company.com",
  department: "Marketing",
  role: "Manager",
  joinDate: "2024-03-10T12:06:02",
  lastAuthenticated: "2025-06-11T00:59:23"
};

export const mockUserInfo: UserInfo = {
  user_id: mockUser.id,
  name: mockUser.name,
  email: mockUser.email,
  department: mockUser.department,
  role: mockUser.role,
  enrolled_at: "2024-03-10T12:06:02",
  last_login: "2025-06-11T00:59:23",
  last_authenticated: "2025-06-11T00:59:23",
  authentication_stats: {
    total_attempts: 0,
    successful_attempts: 0,
    success_rate: 0,
    average_confidence: 0
  },
  recent_attempts: [],
  latest_auth: {
    timestamp: null,
    confidence: null,
    device_info: null
  },
  emotional_state: {
    happiness: 0.0,
    neutral: 1.0,
    surprise: 0.0,
    sadness: 0.0,
    anger: 0.0,
    disgust: 0.0,
    fear: 0.0
  },
  last_updated: new Date().toISOString()
};

export const mockDashboardStats: DashboardStats = {
  totalAuthentications: 157,
  averageConfidence: 0.89,
  emotionBreakdown: {
    [mockUser.lastAuthenticated]: {
      happiness: 0.0,
      neutral: 1.0,
      surprise: 0.0,
      sadness: 0.0,
      anger: 0.0,
      disgust: 0.0,
      fear: 0.0
    }
  },
  recentActivity: [
    {
      success: true,
      confidence: 0.95,
      timestamp: mockUser.lastAuthenticated,
      emotions: {
        happiness: 0.0,
        neutral: 1.0,
        surprise: 0.0,
        sadness: 0.0,
        anger: 0.0,
        disgust: 0.0,
        fear: 0.0
      },
      message: "Successfully authenticated"
    }
  ]
}; 