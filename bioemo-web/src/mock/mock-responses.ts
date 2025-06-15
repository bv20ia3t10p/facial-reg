// Mock data generation utilities
const generateTimestamp = (hoursAgo: number = 0) => {
  const date = new Date();
  date.setHours(date.getHours() - hoursAgo);
  return date.toISOString();
};

// Generate random emotion predictions that sum to 1
const generateEmotionPrediction = (dominantEmotion?: keyof EmotionPrediction) => {
  const emotions: EmotionPrediction = {
    neutral: Math.random() * 0.2,
    happiness: Math.random() * 0.2,
    surprise: Math.random() * 0.2,
    sadness: Math.random() * 0.2,
    anger: Math.random() * 0.2,
    disgust: Math.random() * 0.2,
    fear: Math.random() * 0.2,
    contempt: Math.random() * 0.2
  };

  // Normalize values to sum to 1
  const sum = Object.values(emotions).reduce((a, b) => a + b, 0);
  Object.keys(emotions).forEach(key => {
    emotions[key as keyof EmotionPrediction] /= sum;
  });

  // If a dominant emotion is specified, make it more prominent
  if (dominantEmotion) {
    const others = 1 - 0.6; // 60% for dominant emotion
    const factor = others / (1 - emotions[dominantEmotion]);
    Object.keys(emotions).forEach(key => {
      if (key !== dominantEmotion) {
        emotions[key as keyof EmotionPrediction] *= factor;
      }
    });
    emotions[dominantEmotion] = 0.6;
  }

  return emotions;
};

import type { EmotionPrediction } from '../types';

// Mock user data
export const mockUser = {
  id: "user123",
  name: "John Doe",
  department: "Engineering",
  lastAuthenticated: generateTimestamp(1),
  emotionHistory: [
    { 
      timestamp: generateTimestamp(1),
      success: true,
      confidence: 0.92,
      emotions: generateEmotionPrediction('neutral')
    },
    { 
      timestamp: generateTimestamp(2),
      success: true,
      confidence: 0.85,
      emotions: generateEmotionPrediction('happiness')
    },
    { 
      timestamp: generateTimestamp(3),
      success: true,
      confidence: 0.78,
      emotions: generateEmotionPrediction('sadness')
    }
  ]
};

// Mock user settings
export const mockUserSettings = {
  notifications: {
    emailAlerts: true,
    pushNotifications: false
  },
  privacy: {
    shareEmotionData: true,
    anonymizeReports: false
  },
  authentication: {
    requireSecondaryVerification: true,
    sessionTimeout: 3600
  }
};

// Mock dashboard statistics
export const mockDashboardStats = {
  totalAuthentications: 157,
  averageConfidence: 0.89,
  emotionBreakdown: {
    [generateTimestamp(0)]: generateEmotionPrediction('neutral'),
    [generateTimestamp(24)]: generateEmotionPrediction('happiness'),
    [generateTimestamp(48)]: generateEmotionPrediction('surprise')
  },
  recentActivity: [
    {
      success: true,
      confidence: 0.95,
      timestamp: generateTimestamp(1),
      emotions: generateEmotionPrediction('neutral'),
      message: "Successfully authenticated"
    },
    {
      success: true,
      confidence: 0.88,
      timestamp: generateTimestamp(2),
      emotions: generateEmotionPrediction('happiness'),
      message: "Successfully authenticated"
    },
    {
      success: false,
      confidence: 0.45,
      timestamp: generateTimestamp(3),
      message: "Authentication failed - low confidence"
    }
  ]
};

// Mock emotion trends
export const mockEmotionTrends = {
  dailyBreakdown: [
    {
      date: "2024-03-20",
      emotions: generateEmotionPrediction('happiness'),
      averageConfidence: 0.88
    },
    {
      date: "2024-03-19",
      emotions: generateEmotionPrediction('neutral'),
      averageConfidence: 0.91
    }
  ],
  trends: {
    dominantEmotions: [
      {
        emotion: 'happiness',
        percentage: 35
      },
      {
        emotion: 'neutral',
        percentage: 30
      },
      {
        emotion: 'surprise',
        percentage: 15
      }
    ],
    emotionShifts: [
      {
        from: "neutral",
        to: "sadness",
        count: 12,
        timeOfDay: "afternoon"
      }
    ],
    peakStressTimes: [
      {
        dayOfWeek: "Monday",
        timeOfDay: "morning",
        emotions: generateEmotionPrediction('anger')
      }
    ]
  }
};

// Mock authentication history
export const mockAuthHistory = {
  total: 157,
  page: 1,
  limit: 10,
  data: [
    {
      id: "auth123",
      timestamp: generateTimestamp(1),
      success: true,
      confidence: 0.95,
      emotions: generateEmotionPrediction('neutral'),
      location: "Main Entrance",
      device: "Mobile-QR",
      duration: 2.5
    }
  ],
  summary: {
    successRate: 0.92,
    averageConfidence: 0.89,
    averageDuration: 2.8,
    dominantEmotions: generateEmotionPrediction('neutral')
  }
};

// Mock department overview
export const mockDepartmentOverview = {
  departmentId: "eng123",
  name: "Engineering",
  metrics: {
    totalEmployees: 45,
    activeToday: 38,
    emotionalState: {
      overall: generateEmotionPrediction('happiness'),
      breakdown: {
        morning: generateEmotionPrediction('neutral'),
        afternoon: generateEmotionPrediction('happiness'),
        evening: generateEmotionPrediction('sadness')
      }
    },
    authenticationStats: {
      successRate: 0.95,
      averageConfidence: 0.88,
      failureReasons: {
        lowConfidence: 8,
        systemError: 2,
        other: 1
      }
    }
  },
  trends: {
    weeklyMood: [
      {
        day: "Monday",
        emotions: generateEmotionPrediction('neutral'),
        stressLevel: 0.3
      },
      {
        day: "Tuesday",
        emotions: generateEmotionPrediction('happiness'),
        stressLevel: 0.2
      }
    ],
    peakTimes: {
      authentication: {
        busiest: "09:00-10:00",
        quietest: "15:00-16:00"
      },
      stress: {
        highest: "Monday 09:00-11:00",
        lowest: "Friday 15:00-17:00"
      }
    }
  }
};

// Error responses
export const mockErrors = {
  unauthorized: {
    error: "unauthorized",
    message: "Invalid or expired token",
    code: "AUTH001"
  },
  forbidden: {
    error: "forbidden",
    message: "Insufficient permissions to access this resource",
    code: "AUTH002"
  },
  notFound: {
    error: "not_found",
    message: "Requested resource not found",
    code: "REQ001"
  },
  internalError: {
    error: "internal_error",
    message: "An unexpected error occurred",
    code: "SRV001"
  }
}; 