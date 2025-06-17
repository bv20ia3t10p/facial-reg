import { Typography, Row, Col, Card, Space, Spin } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { getAuthAnalytics } from '../services/api';
import type { EmotionPrediction, Analytics as AnalyticsType } from '../types';

const { Title, Text } = Typography;

function getEmotionColor(emotion: keyof EmotionPrediction): string {
  const colors: Record<string, string> = {
    happiness: '#52c41a',
    neutral: '#8c8c8c',
    surprise: '#faad14',
    sadness: '#1677ff',
    anger: '#f5222d',
    disgust: '#722ed1',
    fear: '#eb2f96',
  };
  return colors[emotion] || '#1677ff';
}

export function Analytics() {
  const { data: analytics, isLoading } = useQuery({
    queryKey: ['authAnalytics'],
    queryFn: () => getAuthAnalytics(),
  });

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Title level={2}>Analytics</Title>

      <Row gutter={[16, 16]}>
        <Col span={8}>
          <Card>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text type="secondary">Daily Authentications</Text>
              {isLoading ? <Spin /> : <Title level={3}>{analytics?.dailyAuthentications || 0}</Title>}
            </Space>
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text type="secondary">Average Confidence</Text>
              {isLoading ? <Spin /> : <Title level={3}>{((analytics?.averageConfidence || 0) * 100).toFixed(1)}%</Title>}
            </Space>
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text type="secondary">Emotion Balance</Text>
              {isLoading ? (
                <Spin />
              ) : (
                <Title level={3}>
                  {analytics?.emotionDistribution && 
                    (analytics.emotionDistribution.happiness > analytics.emotionDistribution.sadness ? 'Positive' : 'Needs Attention')}
                </Title>
              )}
            </Space>
          </Card>
        </Col>
      </Row>

      <Card title="Emotion Distribution">
        <Row gutter={[16, 16]}>
          {isLoading ? (
            <Col span={24}><Spin /></Col>
          ) : (
            analytics?.emotionDistribution && Object.entries(analytics.emotionDistribution).map(([emotion, value]) => (
              <Col key={emotion} span={12}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text strong style={{ textTransform: 'capitalize' }}>{emotion}</Text>
                  <div style={{ width: '100%', background: '#f0f0f0', borderRadius: 4, overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${(value * 100)}%`,
                        height: 8,
                        background: getEmotionColor(emotion as keyof EmotionPrediction),
                        transition: 'width 0.3s ease',
                      }}
                    />
                  </div>
                  <Text>{(value * 100).toFixed(1)}%</Text>
                </Space>
              </Col>
            ))
          )}
        </Row>
      </Card>
    </Space>
  );
} 