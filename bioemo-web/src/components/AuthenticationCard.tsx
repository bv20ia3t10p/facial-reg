import React from 'react';
import { Card, Typography, Space, Tag } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';
import type { Authentication } from '../services/api';
import { useTheme } from '../contexts/ThemeContext';

const { Text } = Typography;

export interface AuthenticationCardProps {
  authentication: Authentication;
  style?: React.CSSProperties;
}

export const AuthenticationCard: React.FC<AuthenticationCardProps> = ({ 
  authentication,
  style 
}) => {
  const { isDarkMode } = useTheme();

  return (
    <Card
      size="small"
      style={{
        borderRadius: '12px',
        background: isDarkMode ? '#192734' : '#ffffff',
        ...style
      }}
    >
      <Space direction="vertical" size="small" style={{ width: '100%' }}>
        <Space size="middle" style={{ width: '100%', justifyContent: 'space-between' }}>
          <Space>
            {authentication.success ? (
              <CheckCircleOutlined style={{ color: '#52c41a', fontSize: '16px' }} />
            ) : (
              <CloseCircleOutlined style={{ color: '#ff4d4f', fontSize: '16px' }} />
            )}
            <Text strong style={{ color: isDarkMode ? '#ffffff' : '#000000' }}>
              {authentication.user?.name || 'Unknown User'}
            </Text>
          </Space>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {new Date(authentication.timestamp).toLocaleString()}
          </Text>
        </Space>
        <Space size="small">
          <Tag color="blue">
            Confidence: {(authentication.confidence * 100).toFixed(1)}%
          </Tag>
          <Tag color="purple">
            Emotion: {authentication.dominantEmotion}
          </Tag>
        </Space>
      </Space>
    </Card>
  );
}; 