import React from 'react';
import { Card, Row, Col, Statistic, Typography, Space, Button, List, Tag, Spin, Skeleton } from 'antd';
import { useQuery } from '@tanstack/react-query';
import {
  SafetyOutlined,
  SmileOutlined,
  TeamOutlined,
  SecurityScanOutlined,
  LockOutlined,
  DatabaseOutlined
} from '@ant-design/icons';
import { getAnalyticsStats } from '../services/api';
import { isAuthenticated } from '../services/auth';
import type { Analytics, EmotionPrediction } from '../types';

const { Title, Paragraph } = Typography;

interface SecurityFeature {
  title: string;
  description: string;
  icon: React.ReactNode;
}

const Home: React.FC = () => {
  const { data: analyticsStats, isLoading } = useQuery<Analytics>({
    queryKey: ['analyticsStats'],
    queryFn: getAnalyticsStats
  });

  const cardStyle = {
    background: 'linear-gradient(135deg, #1890ff 0%, #722ed1 100%)',
    color: 'white'
  };

  const featureCardStyle = {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between'
  } as const;

  const getEmotionColor = (emotion: string): string => {
    const emotionColors: Record<string, string> = {
      happiness: '#52c41a',
      neutral: '#faad14',
      sadness: '#f5222d',
      anger: '#ff4d4f',
      surprise: '#1890ff',
      fear: '#722ed1',
      disgust: '#eb2f96'
    };
    return emotionColors[emotion] || '#1890ff';
  };

  const renderEmotionDistribution = (distribution: EmotionPrediction) => {
    return Object.entries(distribution)
      .sort(([, a], [, b]) => b - a)
      .map(([emotion, value]) => (
        <Tag key={emotion} color={getEmotionColor(emotion)}>
          {emotion}: {(value * 100).toFixed(1)}%
        </Tag>
      ));
  };

  const securityFeatures: SecurityFeature[] = [
    {
      title: 'Authentication Security',
      description: 'Face recognition with confidence threshold and liveness detection',
      icon: <SecurityScanOutlined style={{ color: '#1890ff' }} />
    },
    {
      title: 'Data Protection',
      description: 'Encrypted storage of biometric data and authentication logs',
      icon: <LockOutlined style={{ color: '#52c41a' }} />
    },
    {
      title: 'Access Control',
      description: 'Role-based access control with department segregation',
      icon: <SafetyOutlined style={{ color: '#722ed1' }} />
    },
    {
      title: 'Audit Logging',
      description: 'Comprehensive logging of all authentication attempts',
      icon: <DatabaseOutlined style={{ color: '#faad14' }} />
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Row gutter={[16, 16]}>
        {/* Hero Section */}
        <Col span={24}>
          <Card style={cardStyle}>
            <Title level={2} style={{ color: 'white', marginBottom: 0 }}>
              BioEmo Authentication System
            </Title>
            <Paragraph style={{ color: 'rgba(255, 255, 255, 0.85)' }}>
              Secure biometric authentication with real-time emotion analysis
            </Paragraph>
          </Card>
        </Col>

        {/* System Stats */}
        <Col span={24}>
          <Card title="System Overview">
            <Row gutter={[16, 16]}>
              <Col xs={24} md={8}>
                {isLoading ? (
                  <Skeleton active paragraph={{ rows: 1 }} />
                ) : (
                  <Statistic
                    title="Daily Authentications"
                    value={analyticsStats?.dailyAuthentications || 0}
                    prefix={<TeamOutlined />}
                  />
                )}
              </Col>
              <Col xs={24} md={8}>
                {isLoading ? (
                  <Skeleton active paragraph={{ rows: 1 }} />
                ) : (
                  <Statistic
                    title="Average Confidence"
                    value={analyticsStats?.averageConfidence ? (analyticsStats.averageConfidence * 100).toFixed(2) : 0}
                    suffix="%"
                    prefix={<SecurityScanOutlined />}
                  />
                )}
              </Col>
              <Col xs={24} md={8}>
                <Card bordered={false}>
                  <Title level={4}>Emotion Distribution</Title>
                  {isLoading ? (
                    <div style={{ display: 'flex', justifyContent: 'center', padding: '20px 0' }}>
                      <Spin />
                    </div>
                  ) : (
                    <Space wrap>
                      {analyticsStats?.emotionDistribution && renderEmotionDistribution(analyticsStats.emotionDistribution)}
                    </Space>
                  )}
                </Card>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Implemented Features */}
        <Col span={24}>
          <Card title="Implemented Features">
            <Row gutter={[16, 16]}>
              <Col xs={24} md={8}>
                <Card style={featureCardStyle}>
                  <div>
                    <SafetyOutlined style={{ fontSize: '24px', color: '#1890ff', marginBottom: '16px' }} />
                    <Title level={4}>Biometric Authentication</Title>
                    <ul>
                      <li>Facial recognition with confidence scoring</li>
                      <li>Real-time authentication processing</li>
                      <li>Secure image capture and validation</li>
                    </ul>
                  </div>
                </Card>
              </Col>
              <Col xs={24} md={8}>
                <Card style={featureCardStyle}>
                  <div>
                    <SmileOutlined style={{ fontSize: '24px', color: '#52c41a', marginBottom: '16px' }} />
                    <Title level={4}>Emotion Analysis</Title>
                    <ul>
                      <li>Real-time emotion detection</li>
                      <li>Seven basic emotion classifications</li>
                      <li>Emotion trend tracking</li>
                    </ul>
                  </div>
                </Card>
              </Col>
              <Col xs={24} md={8}>
                <Card style={featureCardStyle}>
                  <div>
                    <DatabaseOutlined style={{ fontSize: '24px', color: '#722ed1', marginBottom: '16px' }} />
                    <Title level={4}>Data Management</Title>
                    <ul>
                      <li>Secure biometric data storage</li>
                      <li>Authentication history tracking</li>
                      <li>Department-based access control</li>
                    </ul>
                  </div>
                </Card>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* Security Features */}
        <Col span={24}>
          <Card title="Security Implementation">
            {isLoading ? (
              <div style={{ padding: '20px 0' }}>
                <Skeleton active paragraph={{ rows: 4 }} />
              </div>
            ) : (
              <List
                grid={{ gutter: 16, column: 2 }}
                dataSource={securityFeatures}
                renderItem={item => (
                  <List.Item>
                    <Card>
                      <List.Item.Meta
                        avatar={item.icon}
                        title={item.title}
                        description={item.description}
                      />
                    </Card>
                  </List.Item>
                )}
              />
            )}
          </Card>
        </Col>

        {!isAuthenticated() && (
          <Col span={24}>
            <Card>
              <Space direction="vertical" align="center" style={{ width: '100%' }}>
                <Title level={3}>Get Started with BioEmo</Title>
                <Button type="primary" size="large" href="/auth">
                  Login to Access Dashboard
                </Button>
              </Space>
            </Card>
          </Col>
        )}
      </Row>
    </div>
  );
};

export { Home };