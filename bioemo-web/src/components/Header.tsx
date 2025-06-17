import React, { useState } from 'react';
import { Layout, Button, Space, Avatar, Typography, Switch, Drawer, Menu } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  HomeOutlined,
  UserOutlined,
  BarChartOutlined,
  SafetyCertificateOutlined,
  TeamOutlined,
  SettingOutlined,
  BulbOutlined,
  BulbFilled,
  MenuOutlined,
} from '@ant-design/icons';
import { useTheme } from '../contexts/ThemeContext';

const { Header: AntHeader } = Layout;
const { Text } = Typography;

// Custom hamburger menu icon component
const MenuIcon: React.FC<{ isDarkMode: boolean }> = ({ isDarkMode }) => (
  <div style={{ 
    display: 'flex', 
    flexDirection: 'column', 
    gap: '4px',
    padding: '4px'
  }}>
    {[1, 2, 3].map(i => (
      <div
        key={i}
        style={{
          width: '20px',
          height: '2px',
          backgroundColor: isDarkMode ? '#ffffff' : '#000000',
          borderRadius: '2px',
          transition: 'all 0.3s ease'
        }}
      />
    ))}
  </div>
);

export const Header: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { isDarkMode, toggleTheme } = useTheme();
  const [drawerVisible, setDrawerVisible] = useState(false);

  const menuItems = [
    { key: '/', icon: <HomeOutlined />, text: 'Home' },
    { key: '/authentication', icon: <SafetyCertificateOutlined />, text: 'Authentication' },
    { key: '/analytics', icon: <BarChartOutlined />, text: 'Analytics' },
    { key: '/hr-dashboard', icon: <TeamOutlined />, text: 'HR Dashboard' },
    { key: '/verification-requests', icon: <UserOutlined />, text: 'Verification' },
  ];

  const handleMenuClick = (path: string) => {
    navigate(path);
    setDrawerVisible(false);
  };

  const MobileMenu = () => (
    <Menu
      mode="vertical"
      selectedKeys={[location.pathname]}
      style={{
        background: 'transparent',
        border: 'none',
      }}
    >
      {menuItems.map(item => (
        <Menu.Item
          key={item.key}
          icon={item.icon}
          onClick={() => handleMenuClick(item.key)}
          style={{
            color: location.pathname === item.key 
              ? '#1DA1F2' 
              : isDarkMode ? '#ffffff' : '#536471',
          }}
        >
          {item.text}
        </Menu.Item>
      ))}
      <Menu.Item
        key="settings"
        icon={<SettingOutlined />}
        onClick={() => handleMenuClick('/settings')}
      >
        Settings
      </Menu.Item>
      <Menu.Item
        key="theme"
        icon={isDarkMode ? <BulbFilled /> : <BulbOutlined />}
        onClick={toggleTheme}
      >
        {isDarkMode ? 'Light Mode' : 'Dark Mode'}
      </Menu.Item>
    </Menu>
  );

  return (
    <>
      <AntHeader style={{
        position: 'fixed',
        zIndex: 1,
        width: '100%',
        padding: '0 16px',
        background: isDarkMode ? '#15202B' : '#ffffff',
        borderBottom: `1px solid ${isDarkMode ? '#38444d' : '#ebeef0'}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '64px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <Avatar
            size={32}
            style={{ 
              backgroundColor: '#1DA1F2',
              marginRight: 8 
            }}
          >
            BE
          </Avatar>
          <Text strong style={{ 
            fontSize: '1.2em',
            color: isDarkMode ? '#ffffff' : '#000000',
          }}>
            BioEmo
          </Text>
        </div>

        {/* Desktop Menu */}
        <Space size={32} className="desktop-menu">
          {menuItems.map(item => (
            <Button
              key={item.key}
              type={location.pathname === item.key ? 'text' : 'link'}
              icon={item.icon}
              onClick={() => navigate(item.key)}
              style={{
                color: location.pathname === item.key 
                  ? '#1DA1F2' 
                  : isDarkMode ? '#ffffff' : '#536471',
                fontWeight: location.pathname === item.key ? 'bold' : 'normal',
              }}
            >
              <span className="menu-text">{item.text}</span>
            </Button>
          ))}
          <Switch
            checkedChildren={<BulbFilled />}
            unCheckedChildren={<BulbOutlined />}
            checked={isDarkMode}
            onChange={toggleTheme}
          />
        </Space>

        {/* Mobile Menu Button */}
        <Button
          type="text"
          icon={<MenuIcon isDarkMode={isDarkMode} />}
          onClick={() => setDrawerVisible(true)}
          className="mobile-menu-button"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '40px',
            height: '40px',
            padding: 0,
            border: 'none',
            background: 'transparent'
          }}
        />
      </AntHeader>

      {/* Mobile Drawer */}
      <Drawer
        title="Menu"
        placement="right"
        onClose={() => setDrawerVisible(false)}
        open={drawerVisible}
        bodyStyle={{
          padding: 0,
          background: isDarkMode ? '#15202B' : '#ffffff',
        }}
        headerStyle={{
          background: isDarkMode ? '#15202B' : '#ffffff',
          borderBottom: `1px solid ${isDarkMode ? '#38444d' : '#ebeef0'}`,
        }}
        style={{
          background: isDarkMode ? '#15202B' : '#ffffff',
        }}
      >
        <MobileMenu />
      </Drawer>
    </>
  );
}; 