import { Modal, Typography, Progress, Button, Space } from 'antd';
import { CheckCircleOutlined, WarningOutlined, CloseCircleOutlined } from '@ant-design/icons';
import type { AuthenticationResponse } from '../services/api';
import { useState } from 'react';
import { ManualVerificationForm } from './ManualVerificationForm';

const { Text, Title } = Typography;

export interface AuthenticationResultProps {
  result: AuthenticationResponse;
  onAdditionalVerification: () => void;
}

export function AuthenticationResult({ result, onAdditionalVerification }: AuthenticationResultProps) {
  const { success, confidence, message } = result;
  const [showManualForm, setShowManualForm] = useState(false);

  const getStatusIcon = () => {
    if (success) {
      return <CheckCircleOutlined style={{ fontSize: 48, color: '#52c41a' }} />;
    }
    if (confidence > 0.6) {
      return <WarningOutlined style={{ fontSize: 48, color: '#faad14' }} />;
    }
    return <CloseCircleOutlined style={{ fontSize: 48, color: '#f5222d' }} />;
  };

  const getTitle = () => {
    if (success) {
      return 'Authentication Successful';
    }
    if (confidence > 0.6) {
      return 'Additional Verification Required';
    }
    return 'Authentication Failed';
  };

  return (
    <>
      <Modal
        open={true}
        footer={null}
        closable={false}
        centered
      >
        <Space direction="vertical" align="center" style={{ width: '100%', textAlign: 'center' }}>
          {getStatusIcon()}
          <Title level={3}>{getTitle()}</Title>
          <Text>{message}</Text>
          
          <div style={{ width: '100%', padding: '24px 0' }}>
            <Text>Confidence Score</Text>
            <Progress
              percent={confidence * 100}
              status={success ? 'success' : confidence > 0.6 ? 'normal' : 'exception'}
              format={(percent) => `${percent?.toFixed(1)}%`}
            />
          </div>

          {confidence > 0.6 && !success && (
            <Button type="primary" onClick={onAdditionalVerification}>
              Proceed with Password Verification
            </Button>
          )}

          {confidence <= 0.6 && !success && (
            <Button type="primary" danger onClick={() => setShowManualForm(true)}>
              Request Manual Verification
            </Button>
          )}
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