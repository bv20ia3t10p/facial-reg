import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import type { UserInfo, User, EmotionPrediction, AuthenticationResponse } from '../types';
import { WebcamCapture } from '../components/WebcamCapture';
import { AuthenticationResult } from '../components/AuthenticationResult';
import { authenticateUser, getUserInfo, predictEmotions } from '../services/api';
import { toast } from 'react-hot-toast';

// Using Ant Design components since they're already in the project
import { Card, Typography, Space, Row, Col, Spin } from 'antd';

const { Title, Text } = Typography;

const Auth: React.FC = () => {
  const [authResult, setAuthResult] = useState<AuthenticationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [currentEmotions, setCurrentEmotions] = useState<EmotionPrediction | null>(null);
  const navigate = useNavigate();

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
        
        // Fetch detailed user info
        try {
          const info = await getUserInfo(response.user_id);
          console.log('Setting user info:', info);
          setUserInfo(info);
        } catch (infoErr) {
          console.error('Failed to fetch user info:', infoErr);
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
        capturedImage: imageSrc
      };
      setAuthResult(errorResponse);
      toast.error('Authentication failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleRetry = useCallback(() => {
    setAuthResult(null);
    setCapturedImage(null);
    setCurrentEmotions(null);
    setUserInfo(null);
  }, []);

  const handlePasswordVerification = useCallback(() => {
    if (!authResult) return;
    
    navigate('/password-verification', { 
      state: { 
        authResult,
        capturedImage,
        userInfo,
        emotions: currentEmotions
      } 
    });
  }, [navigate, authResult, capturedImage, userInfo, currentEmotions]);

  const handleConfirm = useCallback(() => {
    console.log('handleConfirm called with:', {
      authResult,
      userInfo,
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

    if (!userInfo) {
      console.error('Cannot confirm: userInfo is null');
      return;
    }

    // Create default emotions if none exist
    const emotions = currentEmotions || {
      happiness: 0.0,
      neutral: 1.0,
      surprise: 0.0,
      sadness: 0.0,
      anger: 0.0,
      disgust: 0.0,
      fear: 0.0
    };

    // Create a user object from userInfo
    const user: User = {
      id: userInfo.user_id,
      name: userInfo.name,
      email: userInfo.email,
      department: userInfo.department,
      role: userInfo.role,
      joinDate: userInfo.enrolled_at,
      lastAuthenticated: userInfo.last_authenticated
    };

    console.log('Navigating to profile with:', {
      user,
      userInfo,
      emotions
    });

    toast.success('Welcome back!');
    
    // Navigate to profile with all available data
    navigate(`/profile/${authResult.user_id}`, { 
      state: { 
        user,
        userInfo,
        capturedImage,
        emotions,
        authResult
      },
      replace: true
    });
  }, [navigate, authResult, userInfo, currentEmotions, capturedImage]);

  console.log('Current state:', {
    authResult,
    isLoading,
    userInfo,
    currentEmotions,
    capturedImage
  });

  return (
    <Row justify="center" style={{ padding: '24px' }}>
      <Col xs={24} md={20} lg={16} xl={12}>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Card>
            <Space direction="vertical" align="center" style={{ width: '100%' }}>
              <Title level={2}>Face Recognition Authentication</Title>

              {isLoading ? (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <Spin size="large" />
                  <Text style={{ display: 'block', marginTop: '16px' }}>
                    Verifying your identity...
                  </Text>
                </div>
              ) : (
                <>
                  <Text type="secondary">
                    Please position your face in front of the camera and click the camera button to capture
                  </Text>
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
          onAdditionalVerification={handlePasswordVerification}
          onRetry={handleRetry}
          onConfirm={handleConfirm}
        />
      )}
    </Row>
  );
};

export default Auth; 