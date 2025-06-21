import React, { useEffect, useState } from 'react';
import { Card, Row, Col, Statistic, Progress, Tabs, Timeline, message, Select } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { useNavigate } from 'react-router-dom';
import type { HRAnalytics, DepartmentWellbeing } from '../types';
import { getHRAnalytics } from '../services/api';
import { isAuthenticated } from '../services/auth';

const { TabPane } = Tabs;
const { Option } = Select;

interface WellbeingCardProps {
  title: string;
  value: number;
  color: string;
}

const WellbeingCard: React.FC<WellbeingCardProps> = ({ title, value, color }) => (
  <Card variant="outlined">
    <Statistic
      title={title}
      value={value.toFixed(2)}
      suffix="%"
      valueStyle={{ color }}
    />
    <Progress percent={Number(value.toFixed(2))} strokeColor={color} />
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
      <YAxis 
        tickFormatter={(value: number) => value.toFixed(2)}
      />
      <Tooltip 
        labelFormatter={(timestamp: string) => new Date(timestamp).toLocaleString()}
        formatter={(value: number) => [`${Number(value).toFixed(2)}%`]}
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
  <div style={{ height: '300px', overflowY: 'auto', padding: '0 12px' }}>
  <Timeline>
    {alerts.map(alert => (
      <Timeline.Item
        key={alert.id}
        color={getSeverityColor(alert.severity)}
          style={{ 
            padding: '12px 0',
            height: '90px',  // Fixed height for each alert
            marginBottom: '8px'
          }}
      >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ 
              fontWeight: 'bold', 
              fontSize: '14px',
              color: getSeverityColor(alert.severity)
            }}>
              {alert.department}
            </div>
            <div style={{ fontSize: '14px' }}>{alert.message}</div>
            <div style={{ 
              fontSize: '12px', 
              color: 'rgba(0, 0, 0, 0.45)'
            }}>
              {new Date(alert.timestamp).toLocaleString()}
            </div>
          </div>
      </Timeline.Item>
    ))}
  </Timeline>
  </div>
);

const DepartmentComparisonChart: React.FC<{ departments: DepartmentWellbeing[] }> = ({ departments }) => (
  <ResponsiveContainer width="100%" height={300}>
    <BarChart data={departments}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="department" />
      <YAxis tickFormatter={(value: number) => value.toFixed(2)} />
      <Tooltip 
        formatter={(value: number) => [value.toFixed(2) + '%']}
      />
      <Legend />
      <Bar dataKey="metrics.wellbeingScore" name="Wellbeing Score" fill="#722ed1" />
      <Bar dataKey="metrics.jobSatisfaction" name="Job Satisfaction" fill="#52c41a" />
      <Bar dataKey="metrics.stressLevel" name="Stress Level" fill="#f5222d" />
    </BarChart>
  </ResponsiveContainer>
);

const EmotionDistributionPie: React.FC<{ data: HRAnalytics['recentEmotionalTrends'][0] }> = ({ data }) => {
  const pieData = data?.emotionDistribution || [];
  const COLORS = ['#52c41a', '#1890ff', '#597ef7', '#f5222d', '#faad14', '#13c2c2', '#722ed1'];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={pieData}
          dataKey="percentage"
          nameKey="emotion"
          cx="50%"
          cy="50%"
          outerRadius={100}
          label={(entry) => `${entry.emotion}: ${Number(entry.percentage).toFixed(2)}%`}
        >
          {pieData.map((entry, index) => (
            <Cell key={entry.emotion} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip 
          formatter={(value: number) => [`${Number(value).toFixed(2)}%`]}
        />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
};

// Helper functions for colors
const getEmotionColor = (emotion: string): string => {
  const colors: Record<string, string> = {
    happiness: '#52c41a',
    neutral: '#1890ff',
    sadness: '#597ef7',
    anger: '#f5222d',
    surprise: '#faad14',
    fear: '#13c2c2',
    disgust: '#722ed1'
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
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is authenticated
    if (!isAuthenticated()) {
      message.error('Please log in to access the HR dashboard');
      navigate('/authentication');
      return;
    }

    const fetchData = async () => {
      try {
        // Pass the selected time range to the API
        const data = await getHRAnalytics(selectedTimeRange);
        setAnalytics(data);
      } catch (error) {
        console.error('Error fetching HR analytics:', error);
        if (error instanceof Error) {
          message.error(error.message);
          if (error.message.includes('Authentication required')) {
            navigate('/authentication');
          }
        }
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    // Refresh data every 5 minutes
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [navigate, selectedTimeRange]); // Add selectedTimeRange to dependencies

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
          <Card 
            variant="outlined" 
            title={
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>Emotional Trends</span>
                <Select 
                  value={selectedTimeRange} 
                  onChange={setSelectedTimeRange}
                  style={{ width: 120 }}
                >
                  <Option value="24h">Last 24 Hours</Option>
                  <Option value="7d">Last 7 Days</Option>
                  <Option value="30d">Last 30 Days</Option>
                </Select>
              </div>
            }
          >
            <EmotionTrendChart data={analytics.recentEmotionalTrends} />
          </Card>
        </Col>

        <Col span={8}>
          <Card variant="outlined" title="Recent Alerts">
            <AlertsTimeline alerts={analytics.alerts} />
          </Card>
        </Col>

        <Col span={12}>
          <Card variant="outlined" title="Department Comparison">
            <DepartmentComparisonChart departments={analytics.departmentAnalytics} />
          </Card>
        </Col>

        <Col span={12}>
          <Card variant="outlined" title="Current Emotion Distribution">
            <EmotionDistributionPie data={analytics.recentEmotionalTrends[0]} />
          </Card>
        </Col>

        <Col span={24}>
          <Card variant="outlined">
          <Tabs defaultActiveKey="1">
            {analytics.departmentAnalytics.map((dept, index) => (
              <TabPane tab={dept.department} key={String(index + 1)}>
                <DepartmentMetrics department={dept} />
              </TabPane>
            ))}
          </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default HRDashboard; 