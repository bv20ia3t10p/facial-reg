import React, { useState } from 'react';
import { Layout, Button, Space, Avatar, Typography, Switch, Drawer, Menu, message } from 'antd';
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
  LogoutOutlined,
  UserAddOutlined,
} from '@ant-design/icons';
import { useTheme } from '../contexts/ThemeContext';
import { logout, isAuthenticated } from '../services/auth';
import { useUser } from '../contexts/UserContext';

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
  const { isHRDepartment, clearUserData } = useUser();
  
  const handleLogout = () => {
    logout();
    clearUserData(); // Clear user context data
    message.success('You have been logged out');
    navigate('/authentication', { replace: true }); // Use replace to ensure history is updated
    setDrawerVisible(false);
  };

  // Define base menu items (available to all authenticated users)
  const baseMenuItems = [
    { key: '/', icon: <HomeOutlined />, text: 'Home' },
    { key: '/authentication', icon: <SafetyCertificateOutlined />, text: 'Authentication' },
    { key: '/analytics', icon: <BarChartOutlined />, text: 'Analytics' },
  ];
  
  // Define HR-specific menu items (only visible to HR department users)
  const hrMenuItems = [
    { key: '/hr-dashboard', icon: <TeamOutlined />, text: 'HR Dashboard' },
    { key: '/verification-requests', icon: <UserOutlined />, text: 'Verification' },
    { key: '/add-user', icon: <UserAddOutlined />, text: 'Add User' },
  ];
  
  // Combine menu items based on user role
  const menuItems = [...baseMenuItems, ...(isHRDepartment ? hrMenuItems : [])];

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
      {menuItems.map(item => {
        const isActive = location.pathname === item.key;
        return (
          <Menu.Item
            key={item.key}
            icon={item.icon}
            onClick={() => handleMenuClick(item.key)}
            style={{
              color: isActive ? '#1DA1F2' : isDarkMode ? '#ffffff' : '#536471',
              position: 'relative',
              backgroundColor: isActive 
                ? (isDarkMode ? 'rgba(29, 161, 242, 0.1)' : 'rgba(29, 161, 242, 0.05)')
                : 'transparent',
              borderRadius: '8px',
              margin: '4px 0',
              padding: '10px 16px',
            }}
          >
            {item.text}
            {isActive && (
              <div
                style={{
                  position: 'absolute',
                  left: 0,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  width: '3px',
                  height: '70%',
                  background: '#1DA1F2',
                  borderRadius: '0 2px 2px 0',
                  transition: 'all 0.3s ease',
                }}
              />
            )}
          </Menu.Item>
        );
      })}
      <Menu.Item
        key="settings"
        icon={<SettingOutlined />}
        onClick={() => handleMenuClick('/settings')}
        style={{
          color: isDarkMode ? '#ffffff' : '#536471',
          position: 'relative',
          borderRadius: '8px',
          margin: '4px 0',
          padding: '10px 16px',
        }}
      >
        Settings
      </Menu.Item>
      <Menu.Item
        key="theme"
        icon={isDarkMode ? <BulbFilled /> : <BulbOutlined />}
        onClick={toggleTheme}
        style={{
          color: isDarkMode ? '#ffffff' : '#536471',
          position: 'relative',
          borderRadius: '8px',
          margin: '4px 0',
          padding: '10px 16px',
        }}
      >
        {isDarkMode ? 'Light Mode' : 'Dark Mode'}
      </Menu.Item>
      {isAuthenticated() && (
        <Menu.Item
          key="logout"
          icon={<LogoutOutlined />}
          onClick={handleLogout}
          style={{
            color: '#ff4d4f',
            position: 'relative',
            borderRadius: '8px',
            margin: '4px 0',
            padding: '10px 16px',
          }}
        >
          Logout
        </Menu.Item>
      )}
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
          {menuItems.map(item => {
            const isActive = location.pathname === item.key;
            return (
              <Button
                key={item.key}
                type="link"
                icon={item.icon}
                onClick={() => navigate(item.key)}
                style={{
                  color: isActive
                    ? '#1DA1F2' 
                    : isDarkMode ? '#ffffff' : '#536471',
                  fontWeight: isActive ? '500' : 'normal',
                  position: 'relative',
                  padding: '4px 0',
                  border: 'none',
                  background: 'transparent',
                  boxShadow: 'none',
                  outline: 'none',
                }}
              >
                <span className="menu-text">{item.text}</span>
                {isActive && (
                  <div
                    style={{
                      position: 'absolute',
                      bottom: -8,
                      left: 0,
                      width: '100%',
                      height: '3px',
                      background: '#1DA1F2',
                      borderRadius: '2px',
                      transition: 'all 0.3s ease',
                    }}
                  />
                )}
              </Button>
            );
          })}
          <Switch
            checkedChildren={<BulbFilled />}
            unCheckedChildren={<BulbOutlined />}
            checked={isDarkMode}
            onChange={toggleTheme}
          />
          {isAuthenticated() && (
            <Button
              type="text"
              danger
              icon={<LogoutOutlined />}
              onClick={handleLogout}
            >
              <span className="menu-text">Logout</span>
            </Button>
          )}
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