import React from 'react';
import { Layout, Space, Typography } from 'antd';
import { useTheme } from '../contexts/ThemeContext';

const { Footer: AntFooter } = Layout;
const { Link, Text } = Typography;

export const Footer: React.FC = () => {
  const { isDarkMode } = useTheme();
  const currentYear = new Date().getFullYear();

  const footerLinks = [
    { text: 'About', href: '#' },
    { text: 'Privacy Policy', href: '#' },
    { text: 'Terms of Service', href: '#' },
    { text: 'Contact', href: '#' },
  ];

  return (
    <AntFooter
      style={{
        textAlign: 'center',
        background: isDarkMode ? '#15202B' : '#ffffff',
        borderTop: `1px solid ${isDarkMode ? '#38444d' : '#ebeef0'}`,
        padding: '24px',
      }}
    >
      <Space split={<Text type="secondary" style={{ padding: '0 8px' }}>·</Text>}>
        {footerLinks.map((link, index) => (
          <Link
            key={index}
            href={link.href}
            style={{
              color: isDarkMode ? '#8899A6' : '#536471',
              fontSize: '14px',
            }}
          >
            {link.text}
          </Link>
        ))}
      </Space>
      <div style={{ marginTop: '16px' }}>
        <Text
          type="secondary"
          style={{
            fontSize: '14px',
            color: isDarkMode ? '#8899A6' : '#536471',
          }}
        >
          © {currentYear} BioEmo. All rights reserved.
        </Text>
      </div>
      <div style={{ marginTop: '8px' }}>
        <Text
          type="secondary"
          style={{
            fontSize: '12px',
            color: isDarkMode ? '#8899A6' : '#536471',
          }}
        >
          Powered by EmotionNet & Privacy-First Technology
        </Text>
      </div>
    </AntFooter>
  );
}; 