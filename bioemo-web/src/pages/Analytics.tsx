import { Typography, Row, Col, Card, Space, Spin, Radio } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { getAuthAnalytics } from '../services/api';
import type { EmotionPrediction, Analytics } from '../types';
import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

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
  const [timeRange, setTimeRange] = useState<string>('24h');
  
  const { data: analytics, isLoading } = useQuery<Analytics>({
    queryKey: ['authAnalytics', timeRange],
    queryFn: () => getAuthAnalytics(timeRange),
  });

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <Title level={2}>Analytics</Title>
        <Radio.Group 
          value={timeRange} 
          onChange={(e) => setTimeRange(e.target.value)}
          buttonStyle="solid"
        >
          <Radio.Button value="24h">Last 24 Hours</Radio.Button>
          <Radio.Button value="7d">Last 7 Days</Radio.Button>
          <Radio.Button value="30d">Last 30 Days</Radio.Button>
        </Radio.Group>
      </div>

      <Row gutter={[16, 16]}>
        <Col span={8}>
          <Card>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text type="secondary">
                {timeRange === '24h' ? 'Daily' : timeRange === '7d' ? 'Weekly' : 'Monthly'} Authentications
              </Text>
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

      {/* Emotion Trends Chart */}
      {analytics?.emotionTrends && analytics.emotionTrends.length > 0 && (
        <Card title={`Emotion Trends (${timeRange === '24h' ? 'Last 24 Hours' : timeRange === '7d' ? 'Last 7 Days' : 'Last 30 Days'})`}>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={analytics.emotionTrends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(timestamp) => new Date(timestamp).toLocaleTimeString([], { 
                  hour: '2-digit', 
                  minute: '2-digit',
                  day: timeRange !== '24h' ? '2-digit' : undefined,
                  month: timeRange !== '24h' ? 'short' : undefined
                })}
              />
              <YAxis 
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
              />
              <Tooltip 
                labelFormatter={(timestamp) => new Date(timestamp).toLocaleString()}
                formatter={(value: number) => [`${(value * 100).toFixed(1)}%`]}
              />
              <Legend />
              {analytics.emotionTrends[0].emotions && Object.keys(analytics.emotionTrends[0].emotions).map((emotion) => (
                <Line
                  key={emotion}
                  type="monotone"
                  dataKey={`emotions.${emotion}`}
                  name={emotion}
                  stroke={getEmotionColor(emotion as keyof EmotionPrediction)}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
      )}

      <Card title={`Emotion Distribution (${timeRange === '24h' ? 'Last 24 Hours' : timeRange === '7d' ? 'Last 7 Days' : 'Last 30 Days'})`}>
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