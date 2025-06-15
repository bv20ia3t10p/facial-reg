import type { EmotionPrediction, User, Analytics as AnalyticsType } from '../types';
import type { UploadFile } from 'antd/es/upload/interface';
import { calculateAllMetrics } from './wellbeingCalculator';

// Helper function to create a complete emotion prediction
const createEmotionPrediction = (values: Partial<EmotionPrediction>): EmotionPrediction => ({
  happiness: 0,
  neutral: 0,
  surprise: 0,
  sadness: 0,
  anger: 0,
  disgust: 0,
  fear: 0,
  ...values,
});

export interface Authentication {
  id: string;
  timestamp: string;
  success: boolean;
  confidence: number;
  dominantEmotion: string;
  user?: User;
}

export interface LocalStats {
  totalAuthentications: number;
  successRate: number;
  avgResponseTime: number;
  recentAuthentications: Authentication[];
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

export interface UserForm {
  name: string;
  email: string;
  department: string;
  role: string;
  images: UploadFile[];
}

export interface AuthenticationResponse {
  success: boolean;
  confidence: number;
  message: string;
  emotions?: EmotionPrediction;
  user?: User;
  capturedImage?: string;
}

export interface LocalVerificationRequest {
  id: string;
  employeeId: string;
  reason: string;
  additionalNotes?: string;
  capturedImage: string;
  confidence: number;
  status: 'pending' | 'approved' | 'rejected';
  submittedAt: string;
}

export interface VerificationRequestDetails extends LocalVerificationRequest {
  user?: User;
  emotions?: EmotionPrediction;
}

export interface VerificationReviewResponse {
  success: boolean;
  message?: string;
  request?: VerificationRequestDetails;
}

// Mock data
const mockUser: User = {
  id: "user123",
  name: "John Doe",
  email: "john.doe@example.com",
  department: "Engineering",
  role: "employee",
  joinDate: new Date().toISOString(),
  lastAuthenticated: new Date().toISOString()
};

const mockStats: LocalStats = {
  totalAuthentications: 1234,
  successRate: 95.5,
  avgResponseTime: 250,
  recentAuthentications: [
    {
      id: '1',
      timestamp: new Date().toISOString(),
      success: true,
      confidence: 0.95,
      dominantEmotion: 'happiness',
      user: mockUser,
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      success: true,
      confidence: 0.88,
      dominantEmotion: 'neutral',
      user: mockUser,
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 7200000).toISOString(),
      success: false,
      confidence: 0.45,
      dominantEmotion: 'surprise',
    },
  ],
};

const mockAnalytics: AnalyticsType = {
  dailyAuthentications: 150,
  averageConfidence: 0.85,
  emotionDistribution: createEmotionPrediction({
    happiness: 0.4,
    neutral: 0.3,
    surprise: 0.1,
    sadness: 0.1,
    anger: 0.05,
    disgust: 0.03,
    fear: 0.02,
  }),
};

const mockEmotions = createEmotionPrediction({
  happiness: 0.5,
  neutral: 0.3,
  anger: 0.1,
  sadness: 0.1,
});

const mockHRAnalytics: HRAnalytics = {
  overallWellbeing: calculateAllMetrics(mockEmotions),
  departmentAnalytics: [
    {
      department: 'Engineering',
      metrics: calculateAllMetrics(createEmotionPrediction({
        happiness: 0.3,
        neutral: 0.3,
        anger: 0.2,
        sadness: 0.1,
        fear: 0.1,
      })),
      trendData: [
        {
          timestamp: new Date(Date.now() - 7 * 24 * 3600000).toISOString(),
          metrics: calculateAllMetrics(createEmotionPrediction({
            happiness: 0.35,
            neutral: 0.35,
            anger: 0.15,
            sadness: 0.1,
            fear: 0.05,
          })),
        },
        {
          timestamp: new Date().toISOString(),
          metrics: calculateAllMetrics(createEmotionPrediction({
            happiness: 0.3,
            neutral: 0.3,
            anger: 0.2,
            sadness: 0.1,
            fear: 0.1,
          })),
        },
      ],
    },
    {
      department: 'Marketing',
      metrics: calculateAllMetrics(createEmotionPrediction({
        happiness: 0.6,
        neutral: 0.2,
        surprise: 0.1,
        sadness: 0.1,
      })),
      trendData: [
        {
          timestamp: new Date(Date.now() - 7 * 24 * 3600000).toISOString(),
          metrics: calculateAllMetrics(createEmotionPrediction({
            happiness: 0.55,
            neutral: 0.25,
            surprise: 0.1,
            sadness: 0.1,
          })),
        },
        {
          timestamp: new Date().toISOString(),
          metrics: calculateAllMetrics(createEmotionPrediction({
            happiness: 0.6,
            neutral: 0.2,
            surprise: 0.1,
            sadness: 0.1,
          })),
        },
      ],
    },
  ],
  recentEmotionalTrends: [
    {
      timestamp: new Date(Date.now() - 7 * 24 * 3600000).toISOString(),
      emotionDistribution: [
        { emotion: 'happiness', percentage: 40 },
        { emotion: 'neutral', percentage: 30 },
        { emotion: 'surprise', percentage: 15 },
        { emotion: 'sadness', percentage: 10 },
        { emotion: 'anger', percentage: 5 },
      ],
    },
    {
      timestamp: new Date().toISOString(),
      emotionDistribution: [
        { emotion: 'happiness', percentage: 35 },
        { emotion: 'neutral', percentage: 35 },
        { emotion: 'surprise', percentage: 15 },
        { emotion: 'sadness', percentage: 10 },
        { emotion: 'anger', percentage: 5 },
      ],
    },
  ],
  alerts: [
    {
      id: '1',
      type: 'stress',
      severity: 'high',
      department: 'Engineering',
      message: 'High stress levels detected in Engineering department',
      timestamp: new Date().toISOString(),
    },
    {
      id: '2',
      type: 'satisfaction',
      severity: 'medium',
      department: 'Marketing',
      message: 'Moderate decrease in job satisfaction',
      timestamp: new Date(Date.now() - 24 * 3600000).toISOString(),
    },
  ],
};

// Mock verification requests
const mockVerificationRequests: LocalVerificationRequest[] = [
  {
    id: '1',
    employeeId: 'EMP001',
    reason: 'Recent appearance change',
    additionalNotes: 'Got a new haircut',
    capturedImage: 'mock-image-url-1',
    confidence: 0.45,
    status: 'pending',
    submittedAt: new Date(Date.now() - 3600000).toISOString(),
  },
  {
    id: '2',
    employeeId: 'EMP002',
    reason: 'System not recognizing face',
    capturedImage: 'mock-image-url-2',
    confidence: 0.38,
    status: 'approved',
    submittedAt: new Date(Date.now() - 7200000).toISOString(),
  },
];

// Simulated API delay
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Mock image upload function
const simulateImageUpload = async (file: UploadFile): Promise<string> => {
  await delay(1000); // Simulate upload delay
  return URL.createObjectURL(file.originFileObj as Blob); // Create a temporary URL for preview
};

interface AdditionalVerificationResponse {
  success: boolean;
  message?: string;
}

interface PasswordVerificationResponse {
  success: boolean;
  message?: string;
  user?: User;
}

// Mock credentials for testing
const mockCredentials = {
  'john.doe': 'password123',
  'jane.smith': 'password456',
};

// Mock authentication scenarios based on random confidence scores
const generateMockAuthResult = (): AuthenticationResponse => {
  const confidence = Math.random() * 0.7 + 0.3;
  
  if (confidence > 0.9) {
    return {
      success: true,
      confidence,
      message: 'High confidence match - Access granted',
      emotions: createEmotionPrediction({
        happiness: 0.8,
        neutral: 0.15,
        surprise: 0.05,
      }),
      user: mockUser,
    };
  }
  
  if (confidence > 0.6) {
    return {
      success: false,
      confidence,
      message: 'Medium confidence match - Additional verification required',
      emotions: createEmotionPrediction({
        neutral: 0.6,
        happiness: 0.2,
        surprise: 0.2,
      }),
    };
  }
  
  return {
    success: false,
    confidence,
    message: 'Low confidence match - Access denied',
    emotions: createEmotionPrediction({
      neutral: 0.4,
      surprise: 0.3,
      fear: 0.3,
    }),
  };
};

// API functions
export const api = {
  getCurrentUser: async (): Promise<User> => {
    await delay(1000);
    return mockUser;
  },

  getStats: async (): Promise<LocalStats> => {
    await delay(1000);
    return mockStats;
  },

  getAnalytics: async (): Promise<AnalyticsType> => {
    await delay(1000);
    return mockAnalytics;
  },

  authenticate: async (imageSrc: string): Promise<AuthenticationResponse> => {
    await delay(2000);
    const confidence = Math.random();
    if (confidence > 0.9) {
      return {
        success: true,
        confidence,
        message: 'Authentication successful',
        emotions: createEmotionPrediction({
          happiness: 0.7,
          neutral: 0.2,
          surprise: 0.1,
          sadness: 0,
          anger: 0,
          disgust: 0,
          fear: 0,
        }),
        user: mockUser,
        capturedImage: imageSrc,
      };
    }
    if (confidence > 0.6) {
      return {
        success: false,
        confidence,
        message: 'Additional verification required',
        emotions: createEmotionPrediction({
          neutral: 0.6,
          happiness: 0.2,
          surprise: 0.2,
          sadness: 0,
          anger: 0,
          disgust: 0,
          fear: 0,
        }),
        capturedImage: imageSrc,
      };
    }
    return {
      success: false,
      confidence,
      message: 'Authentication failed',
      emotions: createEmotionPrediction({
        neutral: 0.4,
        surprise: 0.3,
        fear: 0.3,
        happiness: 0,
        sadness: 0,
        anger: 0,
        disgust: 0,
      }),
      capturedImage: imageSrc,
    };
  },

  verifyCredentials: async (username: string, password: string): Promise<{
    success: boolean;
    user?: User;
  }> => {
    await delay(1000);
    if (username === 'demo' && password === 'password') {
      return {
        success: true,
        user: {
          id: '1',
          name: 'Demo User',
          email: 'demo@example.com',
          department: 'Engineering',
          role: 'employee',
        },
      };
    }
    return { success: false };
  },

  getVerificationRequests: async (): Promise<LocalVerificationRequest[]> => {
    await delay(1000);
    return [
      {
        id: '1',
        employeeId: 'EMP001',
        reason: 'System did not recognize me',
        capturedImage: '',
        confidence: 0.4,
        status: 'pending',
        submittedAt: new Date().toISOString(),
      },
    ];
  },

  submitManualVerification: async (request: Omit<LocalVerificationRequest, 'id' | 'status' | 'submittedAt'>) => {
    await delay(1000);
    return {
      success: true,
      requestId: '123',
    };
  },

  approveVerificationRequest: async (requestId: string) => {
    await delay(1000);
    return { success: true };
  },

  rejectVerificationRequest: async (requestId: string) => {
    await delay(1000);
    return { success: true };
  },

  getVerificationRequestDetails: async (requestId: string): Promise<VerificationRequestDetails> => {
    await delay(1000);
    const request = mockVerificationRequests.find(r => r.id === requestId);
    if (!request) {
      throw new Error('Verification request not found');
    }

    return {
      ...request,
      user: mockUser,
      emotions: createEmotionPrediction({
        neutral: 0.4,
        surprise: 0.3,
        fear: 0.2,
        sadness: 0.1,
      }),
    };
  },

  reviewVerificationRequest: async (
    requestId: string,
    decision: 'approve' | 'reject',
    notes?: string
  ): Promise<VerificationReviewResponse> => {
    await delay(1000);
    const request = mockVerificationRequests.find(r => r.id === requestId);
    if (!request) {
      return {
        success: false,
        message: 'Verification request not found',
      };
    }

    // In a real implementation, this would update the database
    request.status = decision === 'approve' ? 'approved' : 'rejected';

    return {
      success: true,
      message: `Request ${decision}d successfully`,
      request: {
        ...request,
        user: mockUser,
        emotions: createEmotionPrediction({
          neutral: 0.4,
          surprise: 0.3,
          fear: 0.2,
          sadness: 0.1,
        }),
      },
    };
  },

  addUser: async (userData: UserForm): Promise<User> => {
    await delay(1000);
    // Simulate uploading images
    const imageUrls = await Promise.all(userData.images.map(simulateImageUpload));
    
    // Simulate adding a new user
    return {
      id: Math.random().toString(36).substring(7),
      name: userData.name,
      email: userData.email,
      department: userData.department,
      role: userData.role,
      joinDate: new Date().toISOString(),
      lastAuthenticated: new Date().toISOString(),
    };
  },

  getUser: async (userId: string): Promise<User> => {
    await delay(500);
    return mockUser;
  },

  getHRAnalytics: async (): Promise<HRAnalytics> => {
    await delay(1000);
    return mockHRAnalytics;
  },

  getDepartmentWellbeing: async (department: string): Promise<DepartmentWellbeing> => {
    await delay(800);
    const deptData = mockHRAnalytics.departmentAnalytics.find(d => d.department === department);
    return deptData || mockHRAnalytics.departmentAnalytics[0];
  },
};