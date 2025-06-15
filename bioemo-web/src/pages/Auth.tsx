import { Card, Typography, Button, Space, Row, Col } from 'antd';
import { useState, useCallback } from 'react';
import { CameraOutlined } from '@ant-design/icons';
// import { useMutation } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { api } from '../services/api';
import type { EmotionPrediction, User } from '../types';
import { AuthenticationResult } from '../components/AuthenticationResult';
import { PasswordVerification } from '../components/PasswordVerification';
import { UserProfile } from '../components/UserProfile';
import { WebcamCapture } from '../components/WebcamCapture';
import { LoginForm } from '../components/LoginForm';

const { Title, Text } = Typography;

// const emotionColors: Record<string, string> = {
//   happiness: '#52c41a',
//   neutral: '#8c8c8c',
//   surprise: '#faad14',
//   sadness: '#1677ff',
//   anger: '#f5222d',
//   disgust: '#722ed1',
//   fear: '#eb2f96',
// };

type AuthenticationResponse = Awaited<ReturnType<typeof api.authenticate>>;

export function Auth() {
  const [scanning, setScanning] = useState(false);
  const [emotions, setEmotions] = useState<EmotionPrediction | null>(null);
  const [showResult, setShowResult] = useState(false);
  const [showPasswordVerification, setShowPasswordVerification] = useState(false);
  const [isVerifying, setIsVerifying] = useState(false);
  const [lastAuthResult, setLastAuthResult] = useState<AuthenticationResponse | null>(null);
  const [authenticatedUser, setAuthenticatedUser] = useState<User | null>(null);
  const [showLoginForm, setShowLoginForm] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);

  const handleCapture = useCallback((imageSrc: string) => {
    setCapturedImage(imageSrc);
    handleVerify(imageSrc);
  }, []);

  const handleVerify = async (imageSrc?: string) => {
    setScanning(true);
    try {
      const result = await api.authenticate(imageSrc || capturedImage || '');
      setLastAuthResult(result);
      setShowResult(true);
      if (result.success && result.emotions) {
        setEmotions(result.emotions);
        if (result.user) {
          setAuthenticatedUser(result.user);
        } else if (result.confidence > 0.6) {
          setShowLoginForm(true);
        }
      }
    } catch (error) {
      toast.error('Error verifying identity');
    } finally {
      setScanning(false);
    }
  };

  const handleLogin = async (values: { username: string; password: string }) => {
    setIsVerifying(true);
    try {
      const result = await api.verifyCredentials(values.username, values.password);
      if (result.success && result.user) {
        setAuthenticatedUser(result.user);
        setShowLoginForm(false);
      } else {
        toast.error('Invalid credentials');
      }
    } catch (error) {
      toast.error('Error verifying credentials');
    } finally {
      setIsVerifying(false);
    }
  };

  const handleAdditionalVerification = async () => {
    setShowResult(false);
    setShowPasswordVerification(true);
  };

  const handlePasswordVerification = async (username: string, password: string) => {
    try {
      const result = await api.verifyCredentials(username, password);
      if (result.success && result.user) {
        setAuthenticatedUser(result.user);
        setShowPasswordVerification(false);
      } else {
        throw new Error('Invalid credentials');
      }
    } catch (error) {
      throw error;
    }
  };

  if (authenticatedUser && emotions) {
    return <UserProfile user={authenticatedUser} emotions={emotions} />;
  }

  return (
    <Row justify="center" style={{ padding: '24px' }}>
      <Col xs={24} md={20} lg={16} xl={12}>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Card>
            <Space direction="vertical" align="center" style={{ width: '100%' }}>
              <Title level={2}>Face Recognition Authentication</Title>
              <Text type="secondary">
                Please position your face in front of the camera and click the button below
              </Text>
              
              <WebcamCapture
                onCapture={handleCapture}
                isScanning={scanning}
              />

              <Button
                type="primary"
                size="large"
                icon={<CameraOutlined />}
                onClick={() => handleVerify()}
                loading={scanning}
                style={{
                  height: '48px',
                  padding: '0 32px',
                  fontSize: '16px',
                  borderRadius: '24px',
                  marginTop: '24px',
                }}
              >
                {scanning ? 'Verifying...' : 'Verify Identity'}
              </Button>
            </Space>
          </Card>

          {showLoginForm && (
            <Card title="Additional Verification Required">
              <LoginForm onSubmit={handleLogin} loading={isVerifying} />
            </Card>
          )}

          {showResult && lastAuthResult && (
            <AuthenticationResult
              result={lastAuthResult}
              onAdditionalVerification={handleAdditionalVerification}
            />
          )}

          {showPasswordVerification && (
            <PasswordVerification
              visible={showPasswordVerification}
              onClose={() => setShowPasswordVerification(false)}
              onCancel={() => setShowPasswordVerification(false)}
              onSuccess={(user) => {
                setAuthenticatedUser(user);
                setShowPasswordVerification(false);
              }}
              onSubmit={handlePasswordVerification}
              loading={isVerifying}
            />
          )}
        </Space>
      </Col>
    </Row>
  );
} 