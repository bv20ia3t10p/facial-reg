import type { EmotionPrediction, User, Analytics, UserInfo, AuthenticationResponse, AuthenticationAttempt } from '../types';
import type { UploadFile } from 'antd/es/upload/interface';
import { calculateAllMetrics } from './wellbeingCalculator';
import { getEnvConfig } from '../config/env';
import { getAuthToken, setAuthToken, removeAuthToken } from './auth';

const { authServerUrl, emotionServerUrl, useMockApi } = getEnvConfig();
const API_PREFIX = '/api';  // API prefix for all endpoints


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
  user?: {
    id: string;
    name: string;
    email?: string;
    role?: string;
    department?: string;
  };
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
      user: {
        id: mockUser.id,
        name: mockUser.name,
        email: mockUser.email,
        role: mockUser.role,
        department: mockUser.department
      }
    },
    {
      id: '2',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      success: true,
      confidence: 0.88,
      dominantEmotion: 'neutral',
      user: {
        id: mockUser.id,
        name: mockUser.name,
        email: mockUser.email,
        role: mockUser.role,
        department: mockUser.department
      }
    },
    {
      id: '3',
      timestamp: new Date(Date.now() - 7200000).toISOString(),
      success: false,
      confidence: 0.45,
      dominantEmotion: 'surprise'
    },
  ],
};

const mockAnalytics: Analytics = {
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
  })
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
  const userId = "0000342";
  
  // Create a mock JWT token
  const mockToken = `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.${btoa(JSON.stringify({
    sub: userId,
    name: "Alex Thompson",
    email: "alex.thompson@company.com",
    department: "Human Resources",
    role: "HR Manager",
    exp: Math.floor(Date.now() / 1000) + 86400 // 24 hours
  }))}.MOCK_SIGNATURE`;
  
  return {
    success: true,
    confidence: 0.95,
    message: 'High confidence match - Access granted',
    user_id: userId,
    authenticated_at: timestamp,
    threshold: 0.7,
    token: mockToken,
    department: "Human Resources",
    role: "HR Manager",
    name: "Alex Thompson",
    email: "alex.thompson@company.com",
    emotions: {
      happiness: 0.0,
      neutral: 1.0,
      surprise: 0.0,
      sadness: 0.0,
      anger: 0.0,
      disgust: 0.0,
      fear: 0.0
    }
  };
};

// Helper function to get headers with auth token
const getHeaders = (contentType?: string) => {
  const headers: Record<string, string> = {};
  const token = getAuthToken();
  
  if (token) {
    // Check if it's a JWT token (contains dots) or mock token
    if (token.includes('.')) {
      // It's a JWT token
      headers['Authorization'] = `Bearer ${token}`;
    } else if (token.startsWith('mock-token-')) {
      // It's a mock token, extract the user ID
      const userId = token.replace('mock-token-', '');
      headers['X-User-ID'] = userId;
    }
  }
  
  if (contentType) {
    headers['Content-Type'] = contentType;
  }
  
  return headers;
};

// Helper function to handle API responses
const handleResponse = async (response: Response) => {
  if (!response.ok) {
    if (response.status === 401) {
      // Remove invalid token
      removeAuthToken();
      throw new Error('Authentication required. Please log in again.');
    }
    if (response.status === 403) {
      throw new Error('You do not have permission to access this resource.');
    }
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
        console.log('Added Blob to FormData with field name "image"');
      } catch (e) {
        console.error('Error processing base64 image:', e);
        throw new Error('Invalid image data format');
      }
    } else {
      // If it's already a Blob or File, use it directly
      formData.append('image', imageData, 'image.jpg');
      console.log('Added Blob/File directly to FormData with field name "image"');
    }
    
    const response = await fetch(`${authServerUrl}${API_PREFIX}/auth/authenticate`, {
      method: 'POST',
      body: formData,
      headers: getHeaders()
    });
    
    console.log('Response status:', response.status);
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Authentication failed:', errorText);
      throw new Error(errorText || 'Authentication request failed');
    }
    
    const result = await response.json();
    console.log('Authentication response:', result);
    
    // Store the token if it's in the response
    if (result.token) {
      setAuthToken(result.token);
    }
    
    return result;
  } catch (error) {
    console.error('Authentication error:', error);
    throw error;
  }
};

// Get analytics stats
export const getAnalyticsStats = async (): Promise<Analytics> => {
  try {
    if (useMockApi) {
      return simulateApiCall(mockAnalytics);
    }

    const response = await fetch(`${authServerUrl}${API_PREFIX}/analytics/auth`, {
      headers: getHeaders(),
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to fetch analytics stats:', error);
    throw error;
  }
};

// Get authentication analytics
export const getAuthAnalytics = async (timeRange?: string): Promise<Analytics> => {
  if (useMockApi) {
    return simulateApiCall(mockAnalytics);
  }

  try {
    const url = timeRange 
      ? `${authServerUrl}${API_PREFIX}/analytics/auth?timeRange=${timeRange}` 
      : `${authServerUrl}${API_PREFIX}/analytics/auth`;
      
    const response = await fetch(url, {
      headers: getHeaders()
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to fetch auth analytics:', error);
    throw error;
  }
};

// Get HR analytics
export const getHRAnalytics = async (timeRange?: string): Promise<HRAnalytics> => {
  if (useMockApi) {
    await delay(500);
    return mockHRAnalytics;
  }

  try {
    const url = timeRange 
      ? `${authServerUrl}${API_PREFIX}/analytics/hr?timeRange=${timeRange}` 
      : `${authServerUrl}${API_PREFIX}/analytics/hr`;
    
    const response = await fetch(url, {
      headers: getHeaders(),
    });
    
    return await handleResponse(response);
  } catch (error) {
    console.error('Failed to fetch HR analytics:', error);
    return mockHRAnalytics;
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

  const response = await fetch(`${authServerUrl}${API_PREFIX}/verification/requests`, {
    headers: getHeaders()
  });
  return handleResponse(response);
};

// Get user profile
export const getUserProfile = async (userId: string): Promise<User> => {
  if (useMockApi) {
    return simulateApiCall(mockUser);
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/users/${userId}`, {
      headers: getHeaders()
    });
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
        user: {
          ...mockUser,
          department: "Human Resources",
          role: "HR Manager"
        }
      };
    }
    return { success: false };
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/auth/verify`, {
      method: 'POST',
      headers: getHeaders('application/json'),
      body: JSON.stringify({ username, password })
    });
    const result = await handleResponse(response);
    
    // Store the token if it's in the response
    if (result.token) {
      setAuthToken(result.token);
    }
    
    return result;
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

  const response = await fetch(`${authServerUrl}${API_PREFIX}/verification/requests/${requestId}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      ...getHeaders()
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
      department: "Human Resources",
      role: "HR Manager",
      enrolled_at: new Date().toISOString(),
      last_authenticated: new Date().toISOString(),
      latest_auth: {
        id: "auth123",
        user_id: userId,
        timestamp: new Date().toISOString(),
        confidence: 0.732,
        success: true,
        device_info: "Chrome on Windows",
        emotion_data: createEmotionPrediction({
          happiness: 0.2,
          neutral: 0.7,
          surprise: 0.1
        })
      },
      authentication_stats: {
        total_attempts: 10,
        successful_attempts: 9,
        success_rate: 0.90,
        average_confidence: 0.732
      },
      recent_attempts: [
        {
          id: "auth123",
          user_id: userId,
          timestamp: new Date().toISOString(),
          success: true,
          confidence: 0.697,
          device_info: "Chrome on Windows",
          emotion_data: createEmotionPrediction({
            happiness: 0.2,
            neutral: 0.7,
            surprise: 0.1
          })
        },
        {
          id: "auth124",
          user_id: userId,
          timestamp: new Date(Date.now() - 3600000).toISOString(),
          success: true,
          confidence: 0.736,
          device_info: "Chrome on Windows",
          emotion_data: createEmotionPrediction({
            happiness: 0.3,
            neutral: 0.6,
            surprise: 0.1
          })
        }
      ],
      emotional_state: createEmotionPrediction({
        happiness: 0.0,
        neutral: 1.0,
        surprise: 0.0,
        sadness: 0.0,
        anger: 0.0,
        disgust: 0.0,
        fear: 0.0
      }),
      emotion_trends: {
        average: createEmotionPrediction({
          happiness: 0.2,
          neutral: 0.7,
          surprise: 0.1
        }),
        dominant: "neutral"
      }
    });
  }

  try {
    console.log('Fetching user info for:', userId);
    const response = await fetch(`${authServerUrl}/api/users/${userId}`, {
      headers: getHeaders()
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to fetch user info:', error);
    throw error;
  }
}

// Submit verification request
export const submitVerificationRequest = async (request: {
  employeeId: string;
  reason: string;
  additionalNotes?: string;
  capturedImage: string;
  confidence: number;
}): Promise<{ success: boolean; message: string }> => {
  if (useMockApi) {
    await delay(1000);
    console.log('Mock verification request submitted:', request);
    return {
      success: true,
      message: 'Verification request submitted successfully'
    };
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/verification-requests`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getHeaders()
      },
      body: JSON.stringify(request)
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to submit verification request:', error);
    throw error;
  }
};

export const predictEmotions = async (imageData: string): Promise<EmotionPrediction> => {
  if (useMockApi) {
    await delay(500);
    return mockEmotions;
  }

  try {
    // Convert base64 image data to a Blob for FormData
    let imageBlob: Blob;
    if (typeof imageData === 'string' && imageData.startsWith('data:')) {
      const parts = imageData.split(',');
      const base64Data = parts[1] || imageData;
      const mimeMatch = parts[0]?.match(/:(.*?);/);
      const mimeString = mimeMatch ? mimeMatch[1] : 'image/jpeg';
      
      const byteString = atob(base64Data);
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);
      for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
      }
      imageBlob = new Blob([ab], { type: mimeString });
    } else {
      // If it's not a base64 string, try to use it as is
      imageBlob = new Blob([imageData], { type: 'image/jpeg' });
    }

    // Create FormData
    const formData = new FormData();
    formData.append('file', imageBlob, 'image.jpg');

    // First try the emotion API directly
    try {
      console.log('Sending request to emotion API...');
      const response = await fetch(`${emotionServerUrl}/predict`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Emotion API response:', data);
        
        // Return the emotion data directly as it should already be in the correct format
        return data;
      } else {
        console.warn(`Emotion API returned status ${response.status}`);
        const errorText = await response.text();
        console.warn('Error response:', errorText);
      }
    } catch (e) {
      console.warn('Emotion API failed, falling back to auth server:', e);
    }
    
    // Fallback to the auth server
    console.log('Trying auth server emotion endpoint...');
    const response = await fetch(`${authServerUrl}${API_PREFIX}/emotions/predict`, {
      method: 'POST',
      headers: getHeaders(),
      body: formData
    });

    if (response.ok) {
      const data = await response.json();
      return data.emotions || createEmotionPrediction({});
    }

    throw new Error('Failed to predict emotions');
  } catch (error) {
    console.error('Failed to predict emotions:', error);
    return mockEmotions;
  }
};

export interface OTPRequestData {
  user_id: string;
  reason: string;
  additional_notes?: string;
  capturedImage: string;
  confidence: number;
}

export interface OTPVerifyData {
  user_id: string;
  otp: string;
}

export interface CredentialsVerifyData {
  user_id: string;
  password: string;
}

export const requestVerificationOTP = async (data: OTPRequestData): Promise<{ success: boolean; message: string; otp?: string }> => {
  if (useMockApi) {
    await delay(1000);
    const otp = Math.floor(100000 + Math.random() * 900000).toString();
    console.log('Mock OTP generated:', otp);
    return {
      success: true,
      message: 'OTP sent successfully',
      otp: otp // Only for testing
    };
  }

  try {
    // Make sure confidence is a number to avoid serialization issues
    const payload = {
      ...data,
      confidence: typeof data.confidence === 'number' ? data.confidence : parseFloat(data.confidence as any) || 0
    };

    const response = await fetch(`${authServerUrl}${API_PREFIX}/verification/request-otp`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getHeaders()
      },
      body: JSON.stringify(payload)
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to request OTP:', error);
    throw error;
  }
};

export const verifyVerificationOTP = async (data: OTPVerifyData): Promise<{ success: boolean; message: string; user_id: string; user?: any }> => {
  if (useMockApi) {
    await delay(1000);
    console.log('Mock OTP verified:', data.otp);
    return {
      success: true,
      message: 'OTP verified successfully',
      user_id: data.user_id,
      user: mockUser
    };
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/verification/verify-otp`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getHeaders()
      },
      body: JSON.stringify(data)
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to verify OTP:', error);
    throw error;
  }
};

export const verifyUserCredentials = async (data: CredentialsVerifyData): Promise<{ success: boolean; message: string; otp?: string }> => {
  if (useMockApi) {
    await delay(1000);
    if (data.password === 'demo') {
      const otp = Math.floor(100000 + Math.random() * 900000).toString();
      console.log('Mock OTP generated after credential verification:', otp);
      return {
        success: true,
        message: 'Credentials verified, OTP sent',
        otp: otp // Only for testing
      };
    }
    return {
      success: false,
      message: 'Invalid credentials'
    };
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/verification/verify-credentials`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...getHeaders()
      },
      body: JSON.stringify(data)
    });
    return handleResponse(response);
  } catch (error) {
    console.error('Failed to verify credentials:', error);
    throw error;
  }
};

export const registerUser = async (formData: FormData): Promise<{ success: boolean; message: string; user_id?: string }> => {
  if (useMockApi) {
    await delay(2000); // Simulate network latency
    return {
      success: true,
      message: 'User registered successfully with default password: demo. Model training started in the background.',
      user_id: 'mock-user-id'
    };
  }

  try {
    const response = await fetch(`${authServerUrl}${API_PREFIX}/users/register`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${getAuthToken()}`
      },
      body: formData,
    });

    const data = await response.json();
    return {
      success: response.ok,
      message: data.message || (response.ok ? 'User registered successfully with default password: demo' : 'Failed to register user'),
      user_id: data.user_id
    };
  } catch (error) {
    console.error('Error registering user:', error);
    return {
      success: false,
      message: 'Network error while registering user'
    };
  }
};

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
  submitVerificationRequest,
  requestVerificationOTP,
  verifyVerificationOTP,
  verifyUserCredentials,
  predictEmotions,
  registerUser
};