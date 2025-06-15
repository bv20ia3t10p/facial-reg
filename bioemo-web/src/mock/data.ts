import type { User, DashboardStats } from '../types';

export const mockUser: User = {
  id: "1",
  name: "John Doe",
  department: "Engineering",
  lastAuthenticated: new Date().toISOString(),
  emotionHistory: [
    { timestamp: new Date(Date.now() - 3600000).toISOString(), emotion: "neutral", confidence: 0.92 },
    { timestamp: new Date(Date.now() - 7200000).toISOString(), emotion: "happy", confidence: 0.85 },
    { timestamp: new Date(Date.now() - 10800000).toISOString(), emotion: "tired", confidence: 0.78 }
  ]
};

export const mockDashboardStats: DashboardStats = {
  totalAuthentications: 157,
  averageConfidence: 0.89,
  emotionBreakdown: {
    "neutral": 45,
    "happy": 35,
    "tired": 25,
    "stressed": 15
  },
  recentActivity: [
    {
      success: true,
      confidence: 0.95,
      timestamp: new Date(Date.now() - 1800000).toISOString(),
      emotion: "neutral",
      message: "Successfully authenticated"
    },
    {
      success: true,
      confidence: 0.88,
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      emotion: "happy",
      message: "Successfully authenticated"
    },
    {
      success: false,
      confidence: 0.45,
      timestamp: new Date(Date.now() - 5400000).toISOString(),
      message: "Authentication failed - low confidence"
    }
  ]
}; 