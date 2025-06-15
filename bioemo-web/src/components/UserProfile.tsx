import { Card, Typography, Space, Progress, List, Avatar, Divider } from 'antd';
import { UserOutlined, CalendarOutlined, BankOutlined } from '@ant-design/icons';
import type { User, EmotionPrediction } from '../types';
import { generateAdvice } from '../utils/emotionAdvice';
import { emotionColors } from '../constants/colors';

const { Title, Text } = Typography;

interface UserProfileProps {
  user: User;
  emotions: EmotionPrediction;
}

export function UserProfile({ user, emotions }: UserProfileProps) {
  const advice = generateAdvice(emotions);

  return (
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      {/* User Info Card */}
      <Card>
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Space align="center" size="large">
            <Avatar size={64} icon={<UserOutlined />} />
            <Space direction="vertical" size={0}>
              <Title level={3} style={{ margin: 0 }}>{user.name}</Title>
              <Space size="middle">
                <Text type="secondary">
                  <BankOutlined /> {user.department}
                </Text>
                <Text type="secondary">
                  <CalendarOutlined /> Last login: {user.lastAuthenticated ? new Date(user.lastAuthenticated).toLocaleString() : 'Never'}
                </Text>
              </Space>
            </Space>
          </Space>
        </Space>
      </Card>

      {/* Current Emotions Card */}
      <Card title={<Title level={4}>Current Emotional State</Title>}>
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          {Object.entries(emotions).map(([emotion, value]) => (
            <div key={emotion}>
              <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 8 }}>
                <Text strong style={{ textTransform: 'capitalize' }}>{emotion}</Text>
                <Text>{(value * 100).toFixed(1)}%</Text>
              </Space>
              <Progress 
                percent={value * 100} 
                showInfo={false}
                strokeColor={emotionColors[emotion] || '#1677ff'}
                strokeLinecap="round"
                size="small"
              />
            </div>
          ))}
        </Space>
      </Card>

      {/* Advice Card */}
      <Card title={<Title level={4}>{advice.title}</Title>}>
        <Space direction="vertical" size="middle" style={{ width: '100%' }}>
          <Text>{advice.description}</Text>
          <Divider orientation="left">Suggestions</Divider>
          <List
            size="small"
            dataSource={advice.suggestions}
            renderItem={(item) => (
              <List.Item>
                <Text>{item}</Text>
              </List.Item>
            )}
          />
        </Space>
      </Card>
    </Space>
  );
} 