import React from 'react';
import { Card, Row, Col, Statistic } from 'antd';
import { useQuery } from '@tanstack/react-query';
import {
  UserOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { getAnalyticsStats } from '../services/api';
import { AuthenticationCard } from '../components/AuthenticationCard';
import { useTheme } from '../contexts/ThemeContext';

export function Home() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['analyticsStats'],
    queryFn: () => getAnalyticsStats(),
  });

  const { isDarkMode } = useTheme();

  return (
    <div>
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={24} md={8}>
          <Card 
            loading={isLoading}
            className="dashboard-card"
            style={{
              background: isDarkMode ? '#1DA1F2' : '#1DA1F2',
              borderRadius: '16px',
            }}
          >
            <Statistic
              title={<span style={{ color: '#ffffff' }}>Total Authentications</span>}
              value={stats?.totalAuthentications || 0}
              valueStyle={{ color: '#ffffff' }}
              prefix={<UserOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={24} md={8}>
          <Card 
            loading={isLoading}
            className="dashboard-card"
            style={{
              background: isDarkMode ? '#4CAF50' : '#4CAF50',
              borderRadius: '16px',
            }}
          >
            <Statistic
              title={<span style={{ color: '#ffffff' }}>Success Rate</span>}
              value={stats?.successRate || 0}
              precision={1}
              valueStyle={{ color: '#ffffff' }}
              prefix={<CheckCircleOutlined />}
              suffix="%"
            />
          </Card>
        </Col>
        <Col xs={24} sm={24} md={8}>
          <Card 
            loading={isLoading}
            className="dashboard-card"
            style={{
              background: isDarkMode ? '#9C27B0' : '#9C27B0',
              borderRadius: '16px',
            }}
          >
            <Statistic
              title={<span style={{ color: '#ffffff' }}>Average Response Time</span>}
              value={stats?.avgResponseTime || 0}
              valueStyle={{ color: '#ffffff' }}
              prefix={<ClockCircleOutlined />}
              suffix="ms"
            />
          </Card>
        </Col>
      </Row>

      <div style={{ marginTop: '24px' }}>
        <Card
          title="Recent Authentications"
          style={{
            borderRadius: '16px',
            background: isDarkMode ? '#15202B' : '#ffffff',
          }}
        >
          {stats?.recentAuthentications.map((auth, index) => (
            <AuthenticationCard
              key={auth.id || index}
              authentication={auth}
              style={{ marginBottom: index < stats.recentAuthentications.length - 1 ? 16 : 0 }}
            />
          ))}
        </Card>
      </div>
    </div>
  );
} 