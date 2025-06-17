import type { EmotionPrediction, User, Analytics as AnalyticsType, UserInfo, AuthenticationResponse } from '../types';
import type { UploadFile } from 'antd/es/upload/interface';
import { calculateAllMetrics } from './wellbeingCalculator';
import { getEnvConfig } from '../config/env';

const { authServerUrl, emotionServerUrl, useMockApi } = getEnvConfig();
const API_PREFIX = '/api';  // API prefix for all endpoints

const EMO_API_URL = 'http://localhost:1236'; // Update this if the emo-api runs on a different port

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
  const timestamp = new Date().toISOString();
  return {
    success: true,
    confidence: 0.95,
    message: 'High confidence match - Access granted',
    user_id: "0000342",
    authenticated_at: timestamp,
    emotions: {
      happiness: 0.0,
      neutral: 1.0,
      surprise: 0.0,
      sadness: 0.0,
      anger: 0.0,
      disgust: 0.0,
      fear: 0.0
    },
    user: {
      id: "0000342",
      name: "Amy Smith",
      email: "amy.smith@company.com",
      department: "Marketing",
      role: "Manager",
      joinDate: "2024-03-10T12:06:02",
      lastAuthenticated: timestamp
    }
  };
};

// Helper function to handle API responses
const handleResponse = async (response: Response) => {
  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'An error occurred' }));
    throw new Error(error.message || 'API request failed');
  }
  return response.json();
};

// Helper function to simulate API delay
const simulateApiCall = async <T>(mockData: T, delayMs: number = 500): Promise<T> => {
  await delay(delayMs);
  return mockData;
};

// API endpoints

// Authentication API
export const authenticateUser = async (imageData: string | Blob | File): Promise<AuthenticationResponse> => {
  if (useMockApi) {
    console.log('Using mock API - skipping image processing');
    return simulateApiCall(generateMockAuthResult());
  }
  
  try {
    console.log('Processing image for authentication...');
    console.log('Image data type:', typeof imageData);
    if (typeof imageData !== 'string') {
      if ('type' in imageData) {
        const fileType = 'name' in imageData ? 'File' : 'Blob';
        console.log(`Image is a ${fileType}:`, {
          size: imageData.size,
          type: imageData.type,
          ...(fileType === 'File' ? { name: (imageData as File).name } : {})
        });
      }
    } else {
      console.log('Image is a string, length:', imageData.length);
      console.log('Image data starts with:', imageData.substring(0, 50));
    }

    const formData = new FormData();
    
    // Handle different types of image data
    if (typeof imageData === 'string') {
      console.log('Converting base64 string to Blob...');
      try {
        // If it's a base64 string, convert it to a Blob
        const parts = imageData.split(',');
        const base64Data = parts[1] || imageData;
        const mimeMatch = parts[0]?.match(/:(.*?);/);
        const mimeString = mimeMatch ? mimeMatch[1] : 'image/jpeg';
        
        console.log('MIME type:', mimeString);
        console.log('Base64 data length:', base64Data.length);
        
        const byteString = atob(base64Data);
        console.log('Decoded byte string length:', byteString.length);
        
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
          ia[i] = byteString.charCodeAt(i);
        }
        const blob = new Blob([ab], { type: mimeString });
        console.log('Created Blob:', blob.size, 'bytes,', blob.type);
        
        formData.append('image', blob, 'image.jpg');
        console.log('Added Blob to FormData');
      } catch (e) {
        console.error('Error processing base64 image:', e);
        throw new Error('Invalid image data format');
      }
    } else {
      // If it's already a Blob or File, use it directly
      formData.append('image', imageData, 'image.jpg');
      console.log('Added Blob/File directly to FormData');
    }
    
    console.log('Sending request to:', `${authServerUrl}/authenticate`);
    const response = await fetch(`${authServerUrl}/authenticate`, {
      method: 'POST',
      body: formData,
    });
    
    console.log('Response status:', response.status);
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Authentication failed:', errorText);
      throw new Error(errorText || 'Authentication request failed');
    }
    
    const result = await response.json();
    console.log('Authentication response:', result);
    return result;
  } catch (error) {
    console.error('Authentication error:', error);
    throw error;
  }
};

// Get analytics stats
export const getAnalyticsStats = async (): Promise<LocalStats> => {
  if (useMockApi) {
    return simulateApiCall(mockStats);
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/analytics/stats`);
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to fetch analytics stats:', error);
    throw error;
  }
};

// Get authentication analytics
export const getAuthAnalytics = async (): Promise<AnalyticsType> => {
  if (useMockApi) {
    return simulateApiCall(mockAnalytics);
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/analytics/auth`);
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to fetch auth analytics:', error);
    throw error;
  }
};

// Get HR analytics
export const getHRAnalytics = async (): Promise<HRAnalytics> => {
  if (useMockApi) {
    return simulateApiCall(mockHRAnalytics);
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/analytics/hr`);
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to fetch HR analytics:', error);
    throw error;
  }
};

// Get verification requests
export const getVerificationRequests = async (): Promise<VerificationRequestDetails[]> => {
  if (useMockApi) {
    return simulateApiCall([
      {
        id: 'vr1',
        employeeId: 'user123',
        reason: 'System upgrade verification',
        capturedImage: 'mock-image-data',
        confidence: 0.95,
        status: 'pending',
        submittedAt: new Date().toISOString(),
        user: mockUser,
        emotions: mockEmotions
      }
    ]);
  }

  const response = await fetch(`${authServerUrl}/verification-requests`);
  return handleResponse(response);
};

// Get user profile
export const getUserProfile = async (userId: string): Promise<User> => {
  if (useMockApi) {
    return simulateApiCall(mockUser);
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/users/${userId}`);
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to fetch user profile:', error);
    throw error;
  }
};

// Password verification
export const verifyCredentials = async (username: string, password: string): Promise<{ success: boolean; user?: User }> => {
  if (useMockApi) {
    // Simulate password verification
    await delay(500);
    if (username === 'admin' && password === 'admin') {
      return {
        success: true,
        user: mockUser
      };
    }
    return { success: false };
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/auth/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to verify credentials:', error);
    throw error;
  }
};

// Update verification request status
export const updateVerificationRequestStatus = async (
  requestId: string,
  status: 'approved' | 'rejected'
): Promise<VerificationReviewResponse> => {
  if (useMockApi) {
    return simulateApiCall({
      success: true,
      message: `Request ${requestId} has been ${status}`,
      request: {
        id: requestId,
        employeeId: 'user123',
        reason: 'System upgrade verification',
        capturedImage: 'mock-image-data',
        confidence: 0.95,
        status: status,
        submittedAt: new Date().toISOString(),
        user: mockUser,
        emotions: mockEmotions
      }
    });
  }

  const response = await fetch(`${authServerUrl}/verification-requests/${requestId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ status })
  });
  return handleResponse(response);
};

// Get user info
export async function getUserInfo(userId: string): Promise<UserInfo> {
  if (useMockApi) {
    return simulateApiCall({
      user_id: userId,
      name: "Alex Thompson",
      email: "alex.thompson@company.com",
      department: "Research & Development",
      role: "Senior Research Engineer",
      enrolled_at: new Date().toISOString(),
      last_authenticated: new Date().toISOString(),
      last_login: new Date().toISOString(),
      latest_auth: {
        timestamp: new Date().toISOString(),
        confidence: 0.732,
        device_info: "Chrome on Windows"
      },
      authentication_stats: {
        total_attempts: 10,
        successful_attempts: 9,
        success_rate: 0.90,
        average_confidence: 0.732
      },
      recent_attempts: [
        {
          timestamp: new Date().toISOString(),
          success: true,
          confidence: 0.697,
          device_info: "Chrome on Windows"
        },
        {
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          success: true,
          confidence: 0.736,
          device_info: "Chrome on Windows"
        }
      ],
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
    });
  }

  try {
    console.log('Fetching user info for:', userId);
    const response = await fetch(`${authServerUrl}/api/users/${userId}`);
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to fetch user info:', error);
    throw error;
  }
}

// Export all API functions as a single object
export const api = {
  authenticateUser,
  getUserProfile,
  getAnalyticsStats,
  getAuthAnalytics,
  getHRAnalytics,
  getVerificationRequests,
  verifyCredentials,
  updateVerificationRequestStatus,
  getUserInfo,
  // Add other API functions here as needed
};

export async function predictEmotions(imageData: string): Promise<EmotionPrediction> {
  try {
    // Convert base64 to blob
    const base64Response = await fetch(`data:image/jpeg;base64,${imageData}`);
    const blob = await base64Response.blob();
    
    // Create form data
    const formData = new FormData();
    formData.append('file', blob, 'image.jpg');
    
    const response = await fetch(`${EMO_API_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error('Failed to predict emotions');
    }
    
    const data = await response.json();
    
    // The API now returns the emotions directly in the correct format
    const emotions: EmotionPrediction = {
      happiness: data.happiness || 0,
      neutral: data.neutral || 0,
      surprise: data.surprise || 0,
      sadness: data.sadness || 0,
      anger: data.anger || 0,
      disgust: data.disgust || 0,
      fear: data.fear || 0
    };

    // Normalize the probabilities to ensure they sum to 1
    const total = Object.values(emotions).reduce((sum, val) => sum + val, 0);
    if (total > 0) {
      Object.keys(emotions).forEach(key => {
        emotions[key] = emotions[key] / total;
      });
    }

    console.log('Predicted emotions:', emotions);
    return emotions;
  } catch (error) {
    console.error('Error predicting emotions:', error);
    // Return neutral state on error
    return {
      happiness: 0,
      neutral: 1,
      surprise: 0,
      sadness: 0,
      anger: 0,
      disgust: 0,
      fear: 0
    };
  }
}