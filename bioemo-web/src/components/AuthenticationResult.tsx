import { Modal, Typography, Progress, Button, Space, Alert, Steps, Divider, Input, Form } from 'antd';
import { CheckCircleOutlined, WarningOutlined, CloseCircleOutlined, CameraOutlined, UserOutlined, LockOutlined } from '@ant-design/icons';
import type { AuthenticationResponse } from '../types';
import { useState } from 'react';
import { ManualVerificationForm } from './ManualVerificationForm';
import { EmotionDisplay } from './EmotionDisplay';
import { envConfig } from '../config/env';

const { Text, Title, Paragraph } = Typography;
const { Step } = Steps;

export interface AuthenticationResultProps {
  result: AuthenticationResponse;
  onAdditionalVerification: () => void;
  onRetry: () => void;
  onConfirm: () => void;
  onNotMe: () => void;
  onPasswordVerification: (password: string) => Promise<boolean>;
  onRequestOTP: () => Promise<void>;
}

export function AuthenticationResult({ 
  result, 
  onAdditionalVerification, 
  onRetry, 
  onConfirm,
  onNotMe,
  onPasswordVerification,
  onRequestOTP
}: AuthenticationResultProps) {
  const { success, confidence, message, error, threshold = envConfig.confidenceThreshold, emotions } = result;
  const { lowConfidenceThreshold } = envConfig;
  const [showManualForm, setShowManualForm] = useState(false);
  const [showPasswordForm, setShowPasswordForm] = useState(false);
  const [showLowConfidenceForm, setShowLowConfidenceForm] = useState(false);
  const [password, setPassword] = useState("");
  const [userId, setUserId] = useState("");
  const [passwordError, setPasswordError] = useState("");
  const [isVerifying, setIsVerifying] = useState(false);
  const [otpRequested, setOtpRequested] = useState(false);
  const [otp, setOtp] = useState("");
  const [isSubmittingOtp, setIsSubmittingOtp] = useState(false);
  const [hrRequestSent, setHrRequestSent] = useState(false);

  const getStatusIcon = () => {
    if (success) {
      return <CheckCircleOutlined style={{ fontSize: 48, color: '#52c41a' }} />;
    }
    if (confidence > threshold * 0.85) { // Close to threshold
      return <WarningOutlined style={{ fontSize: 48, color: '#faad14' }} />;
    }
    return <CloseCircleOutlined style={{ fontSize: 48, color: '#f5222d' }} />;
  };

  const getTitle = () => {
    if (success) {
      return 'Authentication Successful';
    }
    if (confidence > threshold * 0.85) {
      return 'Additional Verification Required';
    }
    if (confidence < 0.1) {
      return 'Face Not Detected';
    }
    if (error?.includes('Unknown user')) {
      return 'User Not Recognized';
    }
    return 'Authentication Failed';
  };

  const getTroubleshootingSteps = () => {
    if (confidence < 0.1) {
      return [
        'Ensure your face is well-lit and centered in the frame',
        'Remove any face coverings or accessories',
        'Make sure you\'re looking directly at the camera',
        'Try adjusting your distance from the camera'
      ];
    }
    if (error?.includes('Unknown user')) {
      return [
        'Verify that you\'ve been enrolled in the system',
        'Try adjusting your position or lighting',
        'Consider updating your enrollment photos',
        'Contact HR if the issue persists'
      ];
    }
    return [
      'Ensure proper lighting conditions',
      'Adjust your position relative to the camera',
      'Remove any face coverings or accessories',
      'Try a different authentication method if needed'
    ];
  };

  const getProgressStatus = () => {
    if (success) return 'success';
    if (confidence > threshold * 0.85) return 'normal';
    if (confidence < 0.1) return 'exception';
    return 'exception';
  };

  const getCurrentStep = () => {
    if (success) return 2;
    if (confidence > threshold * 0.85) return 1;
    return 0;
  };

  const handlePasswordSubmit = async () => {
    if (!password) {
      setPasswordError("Please enter your password");
      return;
    }
    
    setIsVerifying(true);
    try {
      const success = await onPasswordVerification(password);
      if (success) {
        setShowPasswordForm(false);
        onConfirm();
      } else {
        setPasswordError("Incorrect password. Please try again.");
      }
    } catch (error) {
      setPasswordError("Verification failed. Please try again.");
    } finally {
      setIsVerifying(false);
    }
  };

  const handleLowConfidenceSubmit = async () => {
    if (!userId) {
      setPasswordError("Please enter your user ID");
      return;
    }
    
    if (!password) {
      setPasswordError("Please enter your password");
      return;
    }
    
    setIsVerifying(true);
    try {
      // For demo purposes, we'll just check if password is "demo"
      if (password === "demo") {
        // Send verification request to HR
        await onRequestOTP();
        setShowLowConfidenceForm(false);
        setHrRequestSent(true);
        setOtpRequested(true);
      } else {
        setPasswordError("Incorrect password. Please try again.");
      }
    } catch (error) {
      setPasswordError("Verification failed. Please try again.");
    } finally {
      setIsVerifying(false);
    }
  };

  const handleRequestOTP = async () => {
    try {
      await onRequestOTP();
      setOtpRequested(true);
    } catch (error) {
      console.error("Failed to request OTP:", error);
    }
  };

  const handleSubmitOTP = async () => {
    if (!otp) {
      return;
    }
    
    setIsSubmittingOtp(true);
    try {
      // In a real app, you would verify the OTP here
      // For demo purposes, we'll just accept any OTP
      onConfirm();
    } catch (error) {
      console.error("Failed to verify OTP:", error);
    } finally {
      setIsSubmittingOtp(false);
    }
  };

  const renderAuthenticationOptions = () => {
    // High confidence success
    if (success && confidence >= threshold) {
      // Get user information with better fallbacks
      const userId = result.user_id;
      
      return (
        <Space direction="vertical" align="center" style={{ width: '100%' }}>
          <Alert
            message={`Welcome!`}
            description={
              <div>
                <Text strong>You are being authenticated as:</Text>
                <div style={{ padding: '10px', margin: '10px 0', background: '#f0f2f5', borderRadius: '4px' }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div><Text strong>User ID:</Text> {userId}</div>
                    {result.user && <div><Text strong>Name:</Text> {result.user.name}</div>}
                  </Space>
                </div>
                <Text type="secondary">If this is not you, please click "Not Me" to try again.</Text>
              </div>
            }
            type="success"
            showIcon
            style={{ width: '100%', marginBottom: 16 }}
          />
          <Space>
            <Button type="primary" onClick={onConfirm} size="large">
              Continue to Profile
            </Button>
            <Button onClick={onNotMe}>
              Not Me
            </Button>
          </Space>
        </Space>
      );
    }
    
    // Medium confidence (below threshold but above low threshold)
    if (confidence < threshold && confidence >= lowConfidenceThreshold) {
      return showPasswordForm ? (
        <Space direction="vertical" style={{ width: '100%' }}>
          <Alert
            message="Password Verification Required"
            description={`We need additional verification. Please enter the password for ${result.user?.name || 'this account'}.`}
            type="info"
            showIcon
            style={{ width: '100%', marginBottom: 16 }}
          />
          <Form layout="vertical">
            <Form.Item 
              validateStatus={passwordError ? "error" : ""}
              help={passwordError || ""}
            >
              <Input.Password 
                placeholder="Enter password (use 'demo' for testing)" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onPressEnter={handlePasswordSubmit}
              />
            </Form.Item>
            <Space>
              <Button type="primary" onClick={handlePasswordSubmit} loading={isVerifying}>
                Verify
              </Button>
              <Button onClick={onNotMe}>
                Not Me
              </Button>
              <Button onClick={() => setShowPasswordForm(false)}>
                Cancel
              </Button>
            </Space>
          </Form>
        </Space>
      ) : (
        <Space direction="vertical" style={{ width: '100%' }}>
          <Alert
            message="Additional Verification Required"
            description={
              <div>
                <Text strong>We've identified you with moderate confidence as:</Text>
                <div style={{ padding: '10px', margin: '10px 0', background: '#f0f2f5', borderRadius: '4px' }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div><Text strong>User ID:</Text> {result.user_id}</div>
                    {result.user && <div><Text strong>Name:</Text> {result.user.name}</div>}
                  </Space>
                </div>
                <Text type="secondary">Please verify your identity with your password or try again if this is not you.</Text>
              </div>
            }
            type="warning"
            showIcon
            icon={<WarningOutlined />}
            style={{ width: '100%', marginBottom: 16 }}
          />
          <Space>
            <Button type="primary" onClick={() => setShowPasswordForm(true)}>
              Enter Password
            </Button>
            <Button onClick={onNotMe}>
              Not Me
            </Button>
            <Button onClick={onRetry}>
              Try Again
            </Button>
          </Space>
        </Space>
      );
    }
    
    // Very low confidence (below low threshold)
    if (confidence < lowConfidenceThreshold) {
      // If HR request has been sent and OTP is requested, show OTP form
      if (hrRequestSent && otpRequested) {
        return (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Alert
              message="OTP Verification Required"
              description="A verification request has been sent to HR. Please enter the OTP provided by HR."
              type="info"
              showIcon
              style={{ width: '100%', marginBottom: 16 }}
            />
            <Form layout="vertical">
              <Form.Item>
                <Input 
                  placeholder="Enter OTP" 
                  value={otp}
                  onChange={(e) => setOtp(e.target.value)}
                  onPressEnter={handleSubmitOTP}
                />
              </Form.Item>
              <Space>
                <Button type="primary" onClick={handleSubmitOTP} loading={isSubmittingOtp}>
                  Verify OTP
                </Button>
                <Button onClick={onNotMe}>
                  Not Me
                </Button>
              </Space>
            </Form>
          </Space>
        );
      }
      
      // Show low confidence form for ID/password entry
      if (showLowConfidenceForm) {
        return (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Alert
              message="Additional Verification Required"
              description="Please enter your user ID and password for verification."
              type="error"
              showIcon
              style={{ width: '100%', marginBottom: 16 }}
            />
            <Form layout="vertical">
              <Form.Item 
                label="User ID"
                validateStatus={!userId && passwordError ? "error" : ""}
                help={!userId && passwordError ? "Please enter your user ID" : ""}
              >
                <Input 
                  placeholder="Enter your user ID" 
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                />
              </Form.Item>
              <Form.Item 
                label="Password"
                validateStatus={passwordError ? "error" : ""}
                help={passwordError || "Use 'demo' for testing"}
              >
                <Input.Password 
                  placeholder="Enter password" 
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onPressEnter={handleLowConfidenceSubmit}
                />
              </Form.Item>
              <Space>
                <Button type="primary" onClick={handleLowConfidenceSubmit} loading={isVerifying}>
                  Submit for Verification
                </Button>
                <Button onClick={() => setShowLowConfidenceForm(false)}>
                  Cancel
                </Button>
              </Space>
            </Form>
          </Space>
        );
      }
      
      return (
        <Space direction="vertical" style={{ width: '100%' }}>
          <Alert
            message="Authentication Failed"
            description={
              <div>
                <Text strong>We couldn't verify your identity with sufficient confidence.</Text>
                {result.user_id !== 'unknown' && (
                  <div style={{ padding: '10px', margin: '10px 0', background: '#f0f2f5', borderRadius: '4px' }}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div><Text strong>Possible match:</Text> User ID: {result.user_id}</div>
                    </Space>
                  </div>
                )}
                <Text type="secondary">If this is you, please verify your identity with your credentials.</Text>
              </div>
            }
            type="error"
            showIcon
            icon={<CloseCircleOutlined />}
            style={{ width: '100%', marginBottom: 16 }}
          />
          <Space>
            <Button type="primary" onClick={() => setShowLowConfidenceForm(true)}>
              Verify with Credentials
            </Button>
            <Button onClick={onNotMe}>
              Not Me
            </Button>
            <Button onClick={onRetry}>
              Try Again
            </Button>
          </Space>
        </Space>
      );
    }
    
    // Fallback
    return (
      <Space>
        <Button type="primary" onClick={onRetry}>
          Try Again
        </Button>
        <Button danger onClick={() => setShowManualForm(true)}>
          Request Manual Verification
        </Button>
      </Space>
    );
  };

  return (
    <>
      <Modal
        open={true}
        footer={null}
        closable={false}
        centered
        width={600}
      >
        <Space direction="vertical" align="center" style={{ width: '100%', textAlign: 'center' }}>
          {getStatusIcon()}
          <Title level={3}>{getTitle()}</Title>
          
          <Steps current={getCurrentStep()} style={{ maxWidth: 400, margin: '20px 0' }}>
            <Step title="Face Detection" icon={<CameraOutlined />} />
            <Step title="Recognition" icon={<UserOutlined />} />
            <Step title="Verified" icon={<LockOutlined />} />
          </Steps>

          {!success && confidence >= threshold * 0.85 && (
            <Alert
              message={error || message}
              type="warning"
              showIcon
              style={{ width: '100%', marginBottom: 16 }}
            />
          )}
          
          <div style={{ width: '100%', padding: '16px 0' }}>
            <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
              <Text strong>Confidence Score</Text>
              <Text type="secondary">Threshold: {(threshold * 100).toFixed(1)}%</Text>
            </Space>
            <Progress
              percent={confidence * 100}
              status={getProgressStatus()}
              format={(percent) => `${percent?.toFixed(1)}%`}
              strokeWidth={12}
            />
          </div>

          {emotions && (
            <>
              <Divider>Emotional State</Divider>
              <EmotionDisplay emotions={emotions} />
            </>
          )}

          {!success && confidence < threshold * 0.85 && !showLowConfidenceForm && !otpRequested && (
            <>
              <Divider>Troubleshooting Tips</Divider>
              <ul style={{ textAlign: 'left' }}>
                {getTroubleshootingSteps().map((step, index) => (
                  <li key={index}>
                    <Paragraph>{step}</Paragraph>
                  </li>
                ))}
              </ul>
            </>
          )}

          <Space size="middle" style={{ marginTop: 24 }}>
            {renderAuthenticationOptions()}
          </Space>
        </Space>
      </Modal>

      <ManualVerificationForm
        visible={showManualForm}
        onClose={() => setShowManualForm(false)}
        capturedImage={result.capturedImage}
        confidence={confidence}
      />
    </>
  );
} 