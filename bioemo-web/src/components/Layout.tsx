import { Layout as AntLayout, Menu, Avatar, Typography, Card } from 'antd';
import { 
  HomeOutlined, 
  ScanOutlined, 
  BarChartOutlined, 
  SettingOutlined,
  UserOutlined,
  UserAddOutlined,
  HeartOutlined,
  SafetyOutlined
} from '@ant-design/icons';
import { Link, useLocation } from 'react-router-dom';
import React from 'react';

const { Header, Sider, Content } = AntLayout;
const { Text } = Typography;

const mainLinks = [
  { icon: HomeOutlined, label: 'Home', key: '/' },
  { icon: ScanOutlined, label: 'Authentication', key: '/auth' },
  { icon: BarChartOutlined, label: 'Analytics', key: '/analytics' },
  { icon: HeartOutlined, label: 'HR Dashboard', key: '/hr-dashboard' },
  { icon: SafetyOutlined, label: 'Verification Requests', key: '/verification-requests' },
  { icon: UserAddOutlined, label: 'Add User', key: '/add-user' },
  { icon: SettingOutlined, label: 'Settings', key: '/settings' },
];

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation();

  return (
    <AntLayout style={{ minHeight: '100vh', background: '#fff' }}>
      <Sider
        width={250}
        style={{
          background: '#fff',
          borderRight: '1px solid #f0f0f0',
          position: 'fixed',
          left: 0,
          top: 0,
          bottom: 0,
          zIndex: 1000,
        }}
      >
        <div style={{ 
          padding: '16px 24px',
          borderBottom: '1px solid #f0f0f0',
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <Avatar 
            size={32}
            style={{ background: '#1677ff' }}
            icon={<UserOutlined />}
          />
          <Text strong style={{ fontSize: '18px' }}>BioEmo</Text>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[location.pathname]}
          style={{ 
            border: 'none',
            padding: '8px',
          }}
          items={mainLinks.map(link => ({
            key: link.key,
            icon: <link.icon style={{ fontSize: '20px' }} />,
            label: <Link to={link.key} style={{ fontSize: '16px' }}>{link.label}</Link>,
          }))}
        />
      </Sider>
      <AntLayout style={{ marginLeft: 250, background: '#fff' }}>
        <Content style={{ 
          padding: '24px',
          maxWidth: '1200px',
          margin: '0 auto',
          width: '100%'
        }}>
          {children}
        </Content>
      </AntLayout>
    </AntLayout>
  );
}

export const GradientCard = ({ children, style = {}, ...props }: any) => (
  <Card
    style={{
      background: 'linear-gradient(135deg, #ffffff 0%, #f0f2f5 100%)',
      borderRadius: '16px',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
      border: '1px solid rgba(0, 0, 0, 0.06)',
      overflow: 'hidden',
      ...style
    }}
    {...props}
  >
    {children}
  </Card>
);

export const InfoRow = ({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode }) => (
  <div style={{ display: 'flex', alignItems: 'center', marginBottom: '16px' }}>
    <div style={{ 
      width: '40px', 
      height: '40px', 
      borderRadius: '12px', 
      background: 'linear-gradient(135deg, #f0f2f5 0%, #e6e9ed 100%)',
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      marginRight: '16px',
      boxShadow: '0 2px 6px rgba(0, 0, 0, 0.04)'
    }}>
      {icon}
    </div>
    <div>
      <Text type="secondary" style={{ fontSize: '14px', display: 'block', marginBottom: '4px' }}>{label}</Text>
      <Text strong style={{ fontSize: '16px', color: '#1a1a1a' }}>{value || '-'}</Text>
    </div>
  </div>
); 