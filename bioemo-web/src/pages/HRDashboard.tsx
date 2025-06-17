import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Progress, Tabs, Timeline } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { HRAnalytics, DepartmentWellbeing } from '../types';
import { getHRAnalytics } from '../services/api';

const { TabPane } = Tabs;

interface WellbeingCardProps {
  title: string;
  value: number;
  color: string;
}

const WellbeingCard: React.FC<WellbeingCardProps> = ({ title, value, color }) => (
  <Card variant="outlined">
    <Statistic
      title={title}
      value={value}
      suffix="%"
      valueStyle={{ color }}
    />
    <Progress percent={value} strokeColor={color} />
  </Card>
);

interface EmotionTrendChartProps {
  data: HRAnalytics['recentEmotionalTrends'];
}

const EmotionTrendChart: React.FC<EmotionTrendChartProps> = ({ data }) => (
  <ResponsiveContainer width="100%" height={300}>
    <LineChart data={data}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis 
        dataKey="timestamp" 
        tickFormatter={(timestamp: string) => new Date(timestamp).toLocaleDateString()}
      />
      <YAxis />
      <Tooltip 
        labelFormatter={(timestamp: string) => new Date(timestamp).toLocaleString()}
        formatter={(value: number) => [`${value.toFixed(1)}%`]}
      />
      <Legend />
      {data[0]?.emotionDistribution.map(({ emotion }) => (
        <Line
          key={emotion}
          type="monotone"
          dataKey={(item: typeof data[0]) => 
            item.emotionDistribution.find((e: { emotion: string }) => e.emotion === emotion)?.percentage
          }
          name={emotion}
          stroke={getEmotionColor(emotion)}
        />
      ))}
    </LineChart>
  </ResponsiveContainer>
);

interface DepartmentMetricsProps {
  department: DepartmentWellbeing;
}

const DepartmentMetrics: React.FC<DepartmentMetricsProps> = ({ department }) => (
  <Card variant="outlined" title={department.department}>
    <Row gutter={[16, 16]}>
      <Col span={12}>
        <WellbeingCard
          title="Stress Level"
          value={department.metrics.stressLevel}
          color={getStressColor(department.metrics.stressLevel)}
        />
      </Col>
      <Col span={12}>
        <WellbeingCard
          title="Job Satisfaction"
          value={department.metrics.jobSatisfaction}
          color="#52c41a"
        />
      </Col>
      <Col span={12}>
        <WellbeingCard
          title="Emotional Balance"
          value={department.metrics.emotionalBalance}
          color="#1890ff"
        />
      </Col>
      <Col span={12}>
        <WellbeingCard
          title="Wellbeing Score"
          value={department.metrics.wellbeingScore}
          color="#722ed1"
        />
      </Col>
    </Row>
  </Card>
);

interface AlertsTimelineProps {
  alerts: HRAnalytics['alerts'];
}

const AlertsTimeline: React.FC<AlertsTimelineProps> = ({ alerts }) => (
  <Timeline>
    {alerts.map(alert => (
      <Timeline.Item
        key={alert.id}
        color={getSeverityColor(alert.severity)}
      >
        <p><strong>{alert.department}</strong> - {alert.message}</p>
        <small>{new Date(alert.timestamp).toLocaleString()}</small>
      </Timeline.Item>
    ))}
  </Timeline>
);

// Helper functions for colors
const getEmotionColor = (emotion: string): string => {
  const colors: Record<string, string> = {
    happiness: '#52c41a',
    neutral: '#1890ff',
    sadness: '#597ef7',
    anger: '#f5222d',
    surprise: '#faad14'
  };
  return colors[emotion] || '#666';
};

const getStressColor = (level: number): string => {
  if (level < 30) return '#52c41a';
  if (level < 60) return '#faad14';
  return '#f5222d';
};

const getSeverityColor = (severity: 'low' | 'medium' | 'high'): string => {
  const colors: Record<string, string> = {
    low: '#52c41a',
    medium: '#faad14',
    high: '#f5222d'
  };
  return colors[severity];
};

const HRDashboard: React.FC = () => {
  const [analytics, setAnalytics] = useState<HRAnalytics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await getHRAnalytics();
        setAnalytics(data);
      } catch (error) {
        console.error('Error fetching HR analytics:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    // Refresh data every 5 minutes
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  if (loading || !analytics) {
    return <Card variant="outlined" loading />;
  }

  return (
    <div style={{ padding: '24px' }}>
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card variant="outlined" title="Overall Workplace Wellbeing">
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <WellbeingCard
                  title="Overall Stress Level"
                  value={analytics.overallWellbeing.stressLevel}
                  color={getStressColor(analytics.overallWellbeing.stressLevel)}
                />
              </Col>
              <Col span={6}>
                <WellbeingCard
                  title="Job Satisfaction"
                  value={analytics.overallWellbeing.jobSatisfaction}
                  color="#52c41a"
                />
              </Col>
              <Col span={6}>
                <WellbeingCard
                  title="Emotional Balance"
                  value={analytics.overallWellbeing.emotionalBalance}
                  color="#1890ff"
                />
              </Col>
              <Col span={6}>
                <WellbeingCard
                  title="Wellbeing Score"
                  value={analytics.overallWellbeing.wellbeingScore}
                  color="#722ed1"
                />
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={16}>
          <Card variant="outlined" title="Emotional Trends">
            <EmotionTrendChart data={analytics.recentEmotionalTrends} />
          </Card>
        </Col>

        <Col span={8}>
          <Card variant="outlined" title="Recent Alerts">
            <AlertsTimeline alerts={analytics.alerts} />
          </Card>
        </Col>

        <Col span={24}>
          <Tabs defaultActiveKey="1">
            {analytics.departmentAnalytics.map((dept, index) => (
              <TabPane tab={dept.department} key={String(index + 1)}>
                <DepartmentMetrics department={dept} />
              </TabPane>
            ))}
          </Tabs>
        </Col>
      </Row>
    </div>
  );
};

export default HRDashboard; 