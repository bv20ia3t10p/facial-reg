import { Typography, Row, Col, Card, Space, Spin, Statistic } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { 
  CheckCircleOutlined, 
  FieldTimeOutlined, 
  TeamOutlined 
} from '@ant-design/icons';
import { api } from '../services/api';
import { AuthenticationCard } from '../components/AuthenticationCard';

const { Title } = Typography;

export function Home() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['stats'],
    queryFn: () => api.getStats(),
  });

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Title level={2} style={{ margin: 0, fontWeight: 600 }}>Dashboard</Title>

      <Row gutter={[24, 24]}>
        <Col xs={24} md={8}>
          <Card 
            bordered={false}
            style={{ 
              borderRadius: '16px',
              background: 'linear-gradient(135deg, #1677ff 0%, #4096ff 100%)',
            }}
          >
            <Statistic 
              title={<span style={{ color: '#fff', fontSize: '16px' }}>Total Authentications</span>}
              value={isLoading ? '-' : stats?.totalAuthentications}
              valueStyle={{ color: '#fff', fontSize: '36px', fontWeight: 600 }}
              prefix={<TeamOutlined style={{ fontSize: '24px' }} />}
            />
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card 
            bordered={false}
            style={{ 
              borderRadius: '16px',
              background: 'linear-gradient(135deg, #52c41a 0%, #73d13d 100%)',
            }}
          >
            <Statistic 
              title={<span style={{ color: '#fff', fontSize: '16px' }}>Success Rate</span>}
              value={isLoading ? '-' : stats?.successRate}
              suffix="%"
              valueStyle={{ color: '#fff', fontSize: '36px', fontWeight: 600 }}
              prefix={<CheckCircleOutlined style={{ fontSize: '24px' }} />}
            />
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card 
            bordered={false}
            style={{ 
              borderRadius: '16px',
              background: 'linear-gradient(135deg, #722ed1 0%, #9254de 100%)',
            }}
          >
            <Statistic 
              title={<span style={{ color: '#fff', fontSize: '16px' }}>Average Response Time</span>}
              value={isLoading ? '-' : stats?.avgResponseTime}
              suffix="ms"
              valueStyle={{ color: '#fff', fontSize: '36px', fontWeight: 600 }}
              prefix={<FieldTimeOutlined style={{ fontSize: '24px' }} />}
            />
          </Card>
        </Col>
      </Row>

      <Card 
        title={<Title level={4} style={{ margin: 0 }}>Recent Authentications</Title>}
        bordered={false}
        style={{ borderRadius: '16px' }}
      >
        <Row gutter={[16, 16]}>
          {isLoading ? (
            Array(4).fill(null).map((_, i) => (
              <Col key={i} xs={24} md={12}>
                <Card loading bordered={false} />
              </Col>
            ))
          ) : (
            stats?.recentAuthentications.map((auth) => (
              <Col key={auth.id} xs={24} md={12}>
                <AuthenticationCard auth={auth} />
              </Col>
            ))
          )}
        </Row>
      </Card>
    </Space>
  );
} 