import { Card, Typography, Space, Table, Button, Row, Col, Statistic, Tabs, Badge } from 'antd';
import { UserOutlined, SmileOutlined, BarChartOutlined, BellOutlined } from '@ant-design/icons';
import { useState, useEffect } from 'react';
import { api } from '../services/api';
import type { User, VerificationRequest } from '../types';
import { AddUserModal } from './AddUserModal';
import { VerificationRequestsTable } from './VerificationRequestsTable';

const { Title } = Typography;
const { TabPane } = Tabs;

export function HRDashboard() {
  const [isAddUserModalVisible, setIsAddUserModalVisible] = useState(false);
  const [pendingRequests, setPendingRequests] = useState<VerificationRequest[]>([]);
  const [activeTab, setActiveTab] = useState('verification');
  const [loading, setLoading] = useState(false);

  // Load verification requests
  const loadVerificationRequests = async () => {
    try {
      setLoading(true);
      const requests = await api.getVerificationRequests();
      setPendingRequests(requests.filter(req => req.status === 'pending'));
    } catch (error) {
      console.error('Failed to load verification requests:', error);
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    loadVerificationRequests();
  }, []);

  // Poll for new requests every 30 seconds
  useEffect(() => {
    const interval = setInterval(loadVerificationRequests, 30000);
    return () => clearInterval(interval);
  }, []);

  // Mock data for demonstration
  const stats = {
    totalEmployees: 156,
    avgWellbeingScore: 85,
    recentVerifications: pendingRequests.length
  };

  const handleTabChange = (key: string) => {
    setActiveTab(key);
  };

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <Row justify="space-between" align="middle">
        <Col>
          <Title level={2}>HR Dashboard</Title>
        </Col>
        <Col>
          <Space>
            <Badge count={pendingRequests.length} offset={[-8, 0]}>
              <Button 
                icon={<BellOutlined />}
                onClick={() => setActiveTab('verification')}
              >
                Pending Requests
              </Button>
            </Badge>
            <Button 
              type="primary" 
              icon={<UserOutlined />}
              onClick={() => setIsAddUserModalVisible(true)}
            >
              Add New User
            </Button>
          </Space>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col xs={24} md={8}>
          <Card>
            <Statistic 
              title="Total Employees"
              value={stats.totalEmployees}
              prefix={<UserOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card>
            <Statistic 
              title="Average Wellbeing Score"
              value={stats.avgWellbeingScore}
              suffix="%"
              prefix={<SmileOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Badge count={pendingRequests.length} offset={[-30, 0]}>
            <Card>
              <Statistic 
                title="Pending Verifications"
                value={stats.recentVerifications}
                prefix={<BarChartOutlined />}
              />
            </Card>
          </Badge>
        </Col>
      </Row>

      <Card>
        <Tabs 
          activeKey={activeTab} 
          onChange={handleTabChange}
        >
          <TabPane 
            tab={
              <Badge count={pendingRequests.length} size="small">
                <span>Verification Requests</span>
              </Badge>
            } 
            key="verification"
          >
            <VerificationRequestsTable 
              onRequestProcessed={loadVerificationRequests}
              loading={loading}
              requests={pendingRequests}
            />
          </TabPane>
          <TabPane tab="Employee Activity" key="activity">
            <Table 
              columns={[
                {
                  title: 'Name',
                  dataIndex: 'name',
                  key: 'name',
                },
                {
                  title: 'Department',
                  dataIndex: 'department',
                  key: 'department',
                },
                {
                  title: 'Last Authentication',
                  dataIndex: 'lastAuthenticated',
                  key: 'lastAuthenticated',
                  render: (date: string) => new Date(date).toLocaleString(),
                },
                {
                  title: 'Dominant Emotion',
                  dataIndex: 'dominantEmotion',
                  key: 'dominantEmotion',
                },
              ]}
              dataSource={[]} // This would be populated with real data from the API
              rowKey="id"
            />
          </TabPane>
        </Tabs>
      </Card>

      <AddUserModal 
        visible={isAddUserModalVisible}
        onClose={() => setIsAddUserModalVisible(false)}
      />
    </Space>
  );
} 