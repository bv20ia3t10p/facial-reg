import { Space, Typography, Row, Col } from 'antd';
import type { EmotionPrediction } from '../types';

const { Text } = Typography;

interface EmotionDisplayProps {
  emotions: EmotionPrediction;
}

export function EmotionDisplay({ emotions }: EmotionDisplayProps) {
  return (
    <Row gutter={[16, 16]}>
      {Object.entries(emotions).map(([emotion, value]) => (
        <Col key={emotion} span={12}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Text strong style={{ textTransform: 'capitalize' }}>{emotion}</Text>
            <div style={{ width: '100%', background: '#f0f0f0', borderRadius: 4, overflow: 'hidden' }}>
              <div
                style={{
                  width: `${value * 100}%`,
                  height: 8,
                  background: '#1677ff',
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
            <Text>{(value * 100).toFixed(1)}%</Text>
          </Space>
        </Col>
      ))}
    </Row>
  );
} 