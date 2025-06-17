import { Modal, Typography, Progress, Button, Space, Alert, Steps, Divider } from 'antd';
import { CheckCircleOutlined, WarningOutlined, CloseCircleOutlined, CameraOutlined, UserOutlined, LockOutlined } from '@ant-design/icons';
import type { AuthenticationResponse } from '../services/api';
import { useState } from 'react';
import { ManualVerificationForm } from './ManualVerificationForm';

const { Text, Title, Paragraph } = Typography;
const { Step } = Steps;

export interface AuthenticationResultProps {
  result: AuthenticationResponse;
  onAdditionalVerification: () => void;
  onRetry: () => void;
  onConfirm: () => void;
}

export function AuthenticationResult({ result, onAdditionalVerification, onRetry, onConfirm }: AuthenticationResultProps) {
  const { success, confidence, message, error, threshold = 0.7 } = result;
  const [showManualForm, setShowManualForm] = useState(false);

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

          {!success && (
            <Alert
              message={error || message}
              type={confidence > threshold * 0.85 ? "warning" : "error"}
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

          {!success && (
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
            {success ? (
              <Button type="primary" onClick={onConfirm} size="large">
                Continue to Profile
              </Button>
            ) : confidence > threshold * 0.85 ? (
              <Button type="primary" onClick={onAdditionalVerification}>
                Proceed with Password Verification
              </Button>
            ) : (
              <>
                <Button type="primary" onClick={onRetry}>
                  Try Again
                </Button>
                <Button danger onClick={() => setShowManualForm(true)}>
                  Request Manual Verification
                </Button>
              </>
            )}
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