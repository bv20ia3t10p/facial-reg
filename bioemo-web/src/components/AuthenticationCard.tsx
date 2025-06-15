import { Card, Space, Typography, Tag } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons';

const { Text } = Typography;

interface Authentication {
  id: string;
  timestamp: string;
  success: boolean;
  confidence: number;
  dominantEmotion: string;
}

interface AuthenticationCardProps {
  auth: Authentication;
}

export function AuthenticationCard({ auth }: AuthenticationCardProps) {
  return (
    <Card>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Space>
          <Tag color={auth.success ? 'success' : 'error'}>
            {auth.success ? (
              <><CheckCircleOutlined /> Success</>
            ) : (
              <><CloseCircleOutlined /> Failed</>
            )}
          </Tag>
          <Text type="secondary">{new Date(auth.timestamp).toLocaleString()}</Text>
        </Space>
        <Space>
          <Text>Confidence: {(auth.confidence * 100).toFixed(1)}%</Text>
          <Text>•</Text>
          <Text>Emotion: <Text strong style={{ textTransform: 'capitalize' }}>{auth.dominantEmotion}</Text></Text>
        </Space>
      </Space>
    </Card>
  );
} 