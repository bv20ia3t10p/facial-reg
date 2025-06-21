import { useState } from 'react';
import { Card, Button, Typography, Space, Row, Col, Divider, Badge } from 'antd';
import { UserAddOutlined, TeamOutlined, SafetyOutlined, ScanOutlined, FileImageOutlined } from '@ant-design/icons';
import { Layout } from '../components/Layout';
import { AddUserModal } from '../components/AddUserModal';
import { useUser } from '../contexts/UserContext';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';

const { Title, Text, Paragraph } = Typography;

export function AddUser() {
  const [modalVisible, setModalVisible] = useState(false);
  const { isHRDepartment } = useUser();
  const navigate = useNavigate();
  const { isDarkMode } = useTheme();

  // Redirect non-HR users
  if (!isHRDepartment) {
    navigate('/');
    return null;
  }

  const cardStyle = {
    borderRadius: '12px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
    height: '100%',
    transition: 'all 0.3s ease',
  };

  const iconStyle = {
    fontSize: '36px',
    color: '#1DA1F2',
    marginBottom: '16px'
  };

  const stepStyle = {
    padding: '16px',
    background: isDarkMode ? 'rgba(29, 161, 242, 0.1)' : 'rgba(29, 161, 242, 0.05)',
    borderRadius: '8px',
    marginBottom: '16px',
    display: 'flex',
    alignItems: 'flex-start',
  };

  const stepNumberStyle = {
    width: '28px',
    height: '28px',
    borderRadius: '50%',
    background: '#1DA1F2',
    color: 'white',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 'bold',
    marginRight: '12px',
    flexShrink: 0,
  };

  return (
    <Layout>
      <Row gutter={[0, 24]}>
        <Col span={24}>
          <div style={{ 
            background: isDarkMode ? 'rgba(29, 161, 242, 0.1)' : 'rgba(29, 161, 242, 0.05)', 
            padding: '24px',
            borderRadius: '12px',
            marginBottom: '24px'
          }}>
            <Title level={2} style={{ margin: 0 }}>User Management</Title>
            <Paragraph style={{ marginBottom: 0 }}>
              Register new employees for facial recognition authentication
            </Paragraph>
          </div>
        </Col>
      </Row>

      <Row gutter={[24, 24]}>
        <Col xs={24} lg={16}>
          <Card 
            title={
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <UserAddOutlined style={{ fontSize: '24px', marginRight: '12px', color: '#1DA1F2' }} />
                <span>Register New User</span>
              </div>
            }
            bordered={false}
            style={cardStyle}
            extra={
              <Button 
                type="primary" 
                icon={<UserAddOutlined />} 
                onClick={() => setModalVisible(true)}
                size="large"
                style={{ borderRadius: '8px', height: '40px' }}
              >
                Add User
              </Button>
            }
          >
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <div>
                <Paragraph>
                  Register a new user with their personal information and face images for biometric authentication.
                </Paragraph>
                <Text type="secondary">
                  The system will automatically train the facial recognition model with the new user data.
                </Text>
              </div>
              
              <Divider orientation="left">Registration Process</Divider>
              
              <div style={stepStyle}>
                <div style={stepNumberStyle}>1</div>
                <div>
                  <Text strong>Enter User Information</Text>
                  <Paragraph style={{ margin: 0 }}>
                    Provide name, email, department, role, and create a password for the new user.
                  </Paragraph>
                </div>
              </div>
              
              <div style={stepStyle}>
                <div style={stepNumberStyle}>2</div>
                <div>
                  <Text strong>Upload Face Images</Text>
                  <Paragraph style={{ margin: 0 }}>
                    Upload 3-5 clear face images from different angles for optimal recognition.
                  </Paragraph>
                </div>
              </div>
              
              <div style={stepStyle}>
                <div style={stepNumberStyle}>3</div>
                <div>
                  <Text strong>Automatic Model Training</Text>
                  <Paragraph style={{ margin: 0 }}>
                    The system will automatically process the images and train the facial recognition model.
                  </Paragraph>
                </div>
              </div>
            </Space>
          </Card>
        </Col>
        
        <Col xs={24} lg={8}>
          <Row gutter={[0, 24]}>
            <Col span={24}>
              <Card 
                bordered={false} 
                style={cardStyle}
                bodyStyle={{ padding: '24px' }}
              >
                <TeamOutlined style={iconStyle} />
                <Title level={4}>User Management</Title>
                <Paragraph>
                  Manage user enrollment and authentication for your organization's biometric system.
                </Paragraph>
              </Card>
            </Col>
            
            <Col span={24}>
              <Card 
                bordered={false} 
                style={cardStyle}
                bodyStyle={{ padding: '24px' }}
              >
                <SafetyOutlined style={iconStyle} />
                <Title level={4}>Best Practices</Title>
                <Space direction="vertical" size="small">
                  <Badge status="processing" text="Use high-quality, well-lit photos" />
                  <Badge status="processing" text="Include different facial angles" />
                  <Badge status="processing" text="Ensure clear facial features" />
                  <Badge status="processing" text="Avoid heavy makeup or accessories" />
                </Space>
              </Card>
            </Col>
          </Row>
        </Col>
      </Row>

      <AddUserModal 
        visible={modalVisible} 
        onClose={() => setModalVisible(false)} 
      />
    </Layout>
  );
} 