import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import type { UserInfo, User, EmotionPrediction, AuthenticationResponse } from '../types';
import { WebcamCapture } from '../components/WebcamCapture';
import { AuthenticationResult } from '../components/AuthenticationResult';
import { authenticateUser, predictEmotions, submitVerificationRequest, requestVerificationOTP } from '../services/api';
import { setAuthToken, getAuthToken } from '../services/auth';
import { toast } from 'react-hot-toast';
import { useUser } from '../contexts/UserContext';
import { useTheme } from '../contexts/ThemeContext';

// Using Ant Design components since they're already in the project
import { Card, Typography, Space, Row, Col, Spin, Steps, Divider, Alert } from 'antd';
import { 
  ScanOutlined, 
  UserOutlined, 
  SafetyOutlined, 
  CheckCircleOutlined,
  LoadingOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { Step } = Steps;

const Auth: React.FC = () => {
  const navigate = useNavigate();
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [authResult, setAuthResult] = useState<AuthenticationResponse | null>(null);
  const [currentEmotions, setCurrentEmotions] = useState<EmotionPrediction | null>(null);
  const { isDarkMode } = useTheme();
  const { refreshUserData } = useUser();

  const handleCapture = useCallback(async (imageSrc: string) => {
    setCapturedImage(imageSrc);
    setIsLoading(true);
    setAuthResult(null);
    try {
      // First authenticate the user
      const response = await authenticateUser(imageSrc);
      console.log('Auth response:', response);
      
      // Update the auth result with the captured image
      const updatedResponse: AuthenticationResponse = {
        ...response,
        capturedImage: imageSrc
      };
      console.log('Setting auth result:', updatedResponse);
      setAuthResult(updatedResponse);
      
      if (response.success) {
        // Store the token if it's in the response
        if (response.token) {
          console.log('Saving authentication token to localStorage');
          setAuthToken(response.token);
        } else {
          // Create a mock token for authentication persistence only if no real token
          console.log('No token in response, creating mock token');
          const mockToken = `mock-token-${response.user_id}`;
          setAuthToken(mockToken);
        }
        
        // Refresh user context data immediately after successful authentication
        await refreshUserData();
        
        // Get emotions from the emotion API if not provided in response
        if (!response.emotions) {
          try {
            const emotions = await predictEmotions(imageSrc);
            console.log('Setting emotions:', emotions);
            setCurrentEmotions(emotions);
          } catch (emotionErr) {
            console.error('Failed to get emotions:', emotionErr);
          }
        } else {
          console.log('Setting emotions from response:', response.emotions);
          setCurrentEmotions(response.emotions);
        }
      }
    } catch (err) {
      console.error('Authentication error:', err);
      const errorResponse: AuthenticationResponse = {
        success: false,
        confidence: 0,
        message: 'Authentication failed. Please try again.',
        error: err instanceof Error ? err.message : 'Authentication failed',
        user_id: 'unknown',
        authenticated_at: new Date().toISOString(),
        capturedImage: imageSrc,
        threshold: 0.7
      };
      setAuthResult(errorResponse);
      toast.error('Authentication failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [refreshUserData]);

  const handleRetry = useCallback(() => {
    setAuthResult(null);
    setCapturedImage(null);
    setCurrentEmotions(null);
  }, []);

  const handleNotMe = useCallback(() => {
    toast('Please try authentication again');
    handleRetry();
  }, [handleRetry]);

  const handlePasswordVerification = useCallback(async (password: string) => {
    if (!authResult || !authResult.user_id) {
      console.error('Cannot verify: missing user ID');
      return false;
    }

    try {
      // In a real app, we would verify against the actual user ID
      // For demo purposes, we'll just check if password is "demo"
      if (password === 'demo') {
        toast.success('Password verified successfully');
        
        // If this was a successful verification, we should use the new data to train the model
        // This would be handled by the backend in a real application
        console.log('Training model with new data for user:', authResult.user_id);
        
        return true;
      }
      
      toast.error('Invalid password');
      return false;
    } catch (error) {
      console.error('Password verification error:', error);
      toast.error('Verification failed');
      return false;
      } 
  }, [authResult]);

  const handleRequestOTP = useCallback(async () => {
    if (!authResult || !capturedImage) {
      console.error('Cannot request OTP: missing data');
      return;
    }

    try {
      // Check if capturedImage is a data URL and extract just the base64 part if needed
      let imageData = capturedImage;
      if (capturedImage.startsWith('data:image')) {
        // Extract base64 data from data URL (removing the prefix like 'data:image/jpeg;base64,')
        imageData = capturedImage.split(',')[1];
      }
      
      console.log('Requesting OTP for user:', authResult.user_id);
      
      // Call the API to request OTP verification with explicit typing
      const otpRequestData = {
        user_id: authResult.user_id,
        reason: 'Low confidence authentication',
        capturedImage: imageData,
        confidence: typeof authResult.confidence === 'number' ? authResult.confidence : 0,
        additional_notes: 'Verification requested from authentication screen'
      };
      
      const response = await requestVerificationOTP(otpRequestData);
      
      toast.success('Verification request sent to HR');
      
      if (response.otp) {
        // In development/demo mode, show the OTP
        console.log('Demo OTP:', response.otp);
        toast.success(`Demo OTP: ${response.otp}`);
      }
      
    } catch (error) {
      console.error('Failed to request OTP:', error);
      toast.error('Failed to request verification');
    }
  }, [authResult, capturedImage]);

  const handleConfirm = useCallback(async () => {
    console.log('handleConfirm called with:', {
      authResult,
      currentEmotions,
      capturedImage
    });

    if (!authResult) {
      console.error('Cannot confirm: authResult is null');
      return;
    }

    if (!authResult.success) {
      console.error('Cannot confirm: authentication was not successful');
      return;
    }

    // Token should already be set in handleCapture, but ensure it's set
    if (authResult.token) {
      console.log('Setting token in handleConfirm');
      setAuthToken(authResult.token);
    } else if (!getAuthToken()) {
      // Only create a mock token if we don't already have one
      console.log('No token found, creating mock token');
      const mockToken = `mock-token-${authResult.user_id}`;
      setAuthToken(mockToken);
    }

    try {
      // Refresh user context to get the latest user data including department and role
      await refreshUserData();
      toast.success('Welcome back!');
      
      // Navigate to profile with authentication data
      navigate(`/profile/${authResult.user_id}`, { 
        state: { 
          capturedImage,
          emotions: currentEmotions,
          authResult
        },
        replace: true
      });
    } catch (error) {
      console.error('Failed to refresh user data:', error);
      toast.error('Authentication error');
    }
  }, [navigate, authResult, currentEmotions, capturedImage, refreshUserData]);

  // Current step in the authentication process
  const getCurrentStep = () => {
    if (authResult?.success) return 2;
    if (capturedImage) return 1;
    return 0;
  };

  const cardStyle = {
    borderRadius: '12px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
    overflow: 'hidden',
  };

  return (
    <Row justify="center" style={{ padding: '24px' }}>
      <Col xs={24} md={20} lg={16} xl={12}>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <div style={{ 
            background: isDarkMode ? 'rgba(29, 161, 242, 0.1)' : 'rgba(29, 161, 242, 0.05)', 
            padding: '24px',
            borderRadius: '12px',
            marginBottom: '12px'
          }}>
            <Title level={2} style={{ margin: 0 }}>Face Recognition Authentication</Title>
            <Paragraph style={{ marginBottom: 0 }}>
              Secure access with biometric facial recognition
            </Paragraph>
          </div>

          <Steps 
            current={getCurrentStep()} 
            items={[
              {
                title: 'Prepare',
                description: 'Position your face',
                icon: <UserOutlined />
              },
              {
                title: 'Scan',
                description: 'Processing image',
                icon: isLoading ? <LoadingOutlined /> : <ScanOutlined />
              },
              {
                title: 'Verify',
                description: 'Authentication result',
                icon: <SafetyOutlined />
              },
            ]}
            style={{ marginBottom: '24px' }}
          />

          <Card style={cardStyle}>
            <Space direction="vertical" align="center" style={{ width: '100%' }}>
              {getCurrentStep() === 0 && (
                <Alert
                  message="Authentication Instructions"
                  description={
                    <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                      <li>Ensure your face is well-lit and clearly visible</li>
                      <li>Look directly at the camera</li>
                      <li>Remove glasses or accessories that may obstruct facial features</li>
                      <li>Keep a neutral expression for best results</li>
                    </ul>
                  }
                  type="info"
                  showIcon
                  style={{ marginBottom: '16px', width: '100%' }}
                />
              )}

              {isLoading ? (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '40px', 
                  background: isDarkMode ? 'rgba(0, 0, 0, 0.05)' : '#f7f9fa',
                  borderRadius: '8px',
                  width: '100%'
                }}>
                  <Spin size="large" indicator={<LoadingOutlined style={{ fontSize: 48, color: '#1DA1F2' }} spin />} />
                  <Text style={{ display: 'block', marginTop: '16px', fontSize: '16px' }}>
                    Verifying your identity...
                  </Text>
                </div>
              ) : (
                <>
                  {!authResult && (
                    <Text type="secondary" style={{ marginBottom: '16px', textAlign: 'center' }}>
                    Please position your face in front of the camera and click the camera button to capture
                  </Text>
                  )}
                  <WebcamCapture
                    onCapture={handleCapture}
                    isScanning={isLoading}
                  />
                </>
              )}
            </Space>
          </Card>
        </Space>
      </Col>

      {authResult && (
        <AuthenticationResult
          result={authResult}
          onAdditionalVerification={() => {}}
          onRetry={handleRetry}
          onConfirm={handleConfirm}
          onNotMe={handleNotMe}
          onPasswordVerification={handlePasswordVerification}
          onRequestOTP={handleRequestOTP}
        />
      )}
    </Row>
  );
};

export default Auth; 