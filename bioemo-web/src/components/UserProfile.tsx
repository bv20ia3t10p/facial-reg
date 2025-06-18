import React from 'react';
import { Card, Typography, Space, List, Avatar, Spin, Statistic, Row, Col, Tag } from 'antd';
import {
  UserOutlined,
  CalendarOutlined, CheckCircleOutlined,
  ClockCircleOutlined, TrophyOutlined,
  SafetyOutlined,
  MailOutlined,
  TeamOutlined
} from '@ant-design/icons';
import type {
  User,
  EmotionPrediction,
  UserInfo,
  AuthenticationAttempt,
  EmotionData,
  NormalizedEmotionData,
  EmotionProbabilities
} from '../types';
import { generateAdvice } from '../utils/emotionAdvice';
import { emotionColors } from '../constants/colors';
import { EmotionDisplay } from './EmotionDisplay';
import { EmotionIcon } from './EmotionDisplay';
import { InfoRow } from './Layout';

const { Title, Text, Paragraph } = Typography;

// Using EmotionProbabilities from types

interface UserProfileProps {
  user: User;
  emotions?: EmotionData | null;
  userInfo?: UserInfo | null;
  isLoading?: boolean;
}

const defaultEmotionProbabilities: EmotionProbabilities = {
  neutral: 1,
  happy: 0,
  sad: 0,
  angry: 0,
  surprised: 0,
  fearful: 0,
  disgusted: 0
};

const defaultEmotionData: EmotionData = {
  emotion: 'neutral',
  confidence: 0,
  probabilities: defaultEmotionProbabilities,
  timestamp: new Date().toISOString()
};

const normalizeEmotionData = (emotionData: EmotionData | null | undefined): NormalizedEmotionData => {
  const data = emotionData || defaultEmotionData;
  const probabilities = data?.probabilities || defaultEmotionProbabilities;
  
  return {
    emotion: data?.emotion || defaultEmotionData.emotion,
    confidence: data?.confidence || defaultEmotionData.confidence,
    probabilities: probabilities,
    timestamp: data?.timestamp || defaultEmotionData.timestamp,
    normalized: {
      happiness: Math.max(0, probabilities.happy || 0),
      neutral: Math.max(0, probabilities.neutral || 0),
      surprise: Math.max(0, probabilities.surprised || 0),
      sadness: Math.max(0, probabilities.sad || 0),
      anger: Math.max(0, probabilities.angry || 0),
      disgust: Math.max(0, probabilities.disgusted || 0),
      fear: Math.max(0, probabilities.fearful || 0)
    }
  };
};

const getDominantEmotion = (emotions: EmotionPrediction): string => {
  return Object.entries(emotions).reduce((a, b) => a[1] > b[1] ? a : b)[0];
};

const RecentAuthentication: React.FC<{ attempt: AuthenticationAttempt }> = ({ attempt }) => {
  const normalizedData = normalizeEmotionData(attempt.emotion_data);
  const dominantEmotion = normalizedData.emotion;
  const dominantEmotionColor = emotionColors[dominantEmotion.toLowerCase()] || '#1677ff';

  return (
    <Card 
      size="small" 
      style={{ 
        marginBottom: '12px',
        borderLeft: `4px solid ${dominantEmotionColor}`,
        borderRadius: '12px',
        background: `linear-gradient(135deg, ${dominantEmotionColor}05 0%, #ffffff 100%)`,
        transition: 'all 0.3s ease',
      }}
      hoverable
    >
      <Row gutter={16} align="middle">
        <Col span={6}>
          <Text strong style={{ fontSize: '13px' }}>
            {new Date(attempt.timestamp).toLocaleDateString()}
          </Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {new Date(attempt.timestamp).toLocaleTimeString()}
          </Text>
        </Col>
        <Col span={6}>
          <Tag color={attempt.success ? 'success' : 'error'} style={{ minWidth: '60px', textAlign: 'center' }}>
            {attempt.success ? 'Success' : 'Failed'}
          </Tag>
        </Col>
        <Col span={6} style={{ textAlign: 'center' }}>
          <Text type="secondary" style={{ fontSize: '12px' }}>Confidence</Text>
          <br />
          <Text strong>{(attempt.confidence * 100).toFixed(1)}%</Text>
        </Col>
        <Col span={6} style={{ textAlign: 'right' }}>
          <Tag color={dominantEmotionColor} style={{ minWidth: '70px', textAlign: 'center' }}>
            {dominantEmotion}
          </Tag>
        </Col>
      </Row>
    </Card>
  );
};

export function UserProfile({ user, emotions, userInfo, isLoading = false }: UserProfileProps) {
  const normalizedData = normalizeEmotionData(userInfo?.emotional_state || null);
  const adviceData = generateAdvice(normalizedData.normalized);
  const dominantEmotion = normalizedData.emotion;
  const dominantEmotionColor = emotionColors[dominantEmotion.toLowerCase()] || '#1677ff';

  if (isLoading) {
    return (
      <div style={{ 
        textAlign: 'center', 
        padding: '48px',
        background: 'linear-gradient(135deg, #ffffff 0%, #f0f2f5 100%)',
        borderRadius: '16px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.08)',
        backdropFilter: 'blur(8px)'
      }}>
        <Spin size="large" />
        <Text style={{ display: 'block', marginTop: '16px', fontSize: '16px' }}>Loading user information...</Text>
      </div>
    );
  }

  return (
    <Row gutter={[24, 24]} style={{ padding: '24px' }}>
      {/* Left Column */}
      <Col span={16}>
        {/* User Info Card */}
        <Card
          style={{
            borderRadius: '20px',
            background: 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
            border: 'none',
            overflow: 'hidden',
            marginBottom: '24px'
          }}
        >
          <Row gutter={24} align="middle">
            <Col span={8}>
              <div style={{
                textAlign: 'center',
                padding: '24px',
                background: `linear-gradient(135deg, ${dominantEmotionColor}15 0%, #ffffff 100%)`,
                borderRadius: '16px',
                transition: 'all 0.3s ease'
              }}>
                <div style={{
                  position: 'relative',
                  display: 'inline-block',
                  margin: '0 auto 20px',
                }}>
                  <Avatar 
                    size={140} 
                    icon={<UserOutlined />} 
                    style={{ 
                      backgroundColor: dominantEmotionColor,
                      boxShadow: `0 8px 24px ${dominantEmotionColor}40`,
                      transition: 'all 0.3s ease',
                    }}
                  />
                  <div style={{
                    position: 'absolute',
                    bottom: -5,
                    right: -5,
                    background: '#fff',
                    borderRadius: '50%',
                    padding: '4px',
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                  }}>
                    <EmotionIcon emotion={dominantEmotion} />
                  </div>
                </div>
                <Title level={3} style={{ margin: '0 0 8px', color: '#2c3e50' }}>{user.name}</Title>
                <Tag color={dominantEmotionColor} style={{ padding: '4px 12px', borderRadius: '12px', fontSize: '14px' }}>
                  {user.role}
                </Tag>
              </div>
            </Col>
            <Col span={16}>
              <Space direction="vertical" size="middle" style={{ width: '100%', padding: '0 16px' }}>
                <InfoRow 
                  label="User ID" 
                  value={userInfo?.user_id || user.id} 
                  icon={<UserOutlined style={{ color: '#1677ff', fontSize: '20px' }} />} 
                />
                <InfoRow 
                  label="Email" 
                  value={userInfo?.email || user.email || '-'} 
                  icon={<MailOutlined style={{ color: '#52c41a', fontSize: '20px' }} />} 
                />
                <InfoRow 
                  label="Department" 
                  value={userInfo?.department || user.department || '-'} 
                  icon={<TeamOutlined style={{ color: '#722ed1', fontSize: '20px' }} />} 
                />
                <InfoRow 
                  label="Enrolled At" 
                  value={new Date(userInfo?.enrolled_at || user.joinDate || Date.now()).toLocaleString()} 
                  icon={<CalendarOutlined style={{ color: '#faad14', fontSize: '20px' }} />} 
                />
                <InfoRow 
                  label="Last Authentication" 
                  value={new Date(userInfo?.last_authenticated || user.lastAuthenticated || Date.now()).toLocaleString()} 
                  icon={<ClockCircleOutlined style={{ color: '#13c2c2', fontSize: '20px' }} />} 
                />
              </Space>
            </Col>
          </Row>
        </Card>

        {/* Stats Cards */}
        <Row gutter={16} style={{ marginBottom: '24px' }}>
          <Col span={8}>
            <Card 
              hoverable
              style={{ 
                borderRadius: '16px', 
                textAlign: 'center',
                background: 'linear-gradient(135deg, #f6ffed 0%, #ffffff 100%)',
                border: 'none',
                boxShadow: '0 4px 12px rgba(82, 196, 26, 0.1)',
                transition: 'all 0.3s ease',
              }}
            >
              <Statistic
                title={<Text strong style={{ fontSize: '16px', color: '#52c41a' }}>Total Attempts</Text>}
                value={userInfo?.authentication_stats.total_attempts || 0}
                prefix={<TrophyOutlined style={{ color: '#52c41a', fontSize: '24px' }} />}
                valueStyle={{ color: '#52c41a', fontSize: '28px' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card 
              hoverable
              style={{ 
                borderRadius: '16px', 
                textAlign: 'center',
                background: 'linear-gradient(135deg, #fff7e6 0%, #ffffff 100%)',
                border: 'none',
                boxShadow: '0 4px 12px rgba(250, 173, 20, 0.1)',
                transition: 'all 0.3s ease',
              }}
            >
              <Statistic
                title={<Text strong style={{ fontSize: '16px', color: '#faad14' }}>Success Rate</Text>}
                value={userInfo?.authentication_stats.success_rate || 0}
                precision={1}
                suffix="%"
                prefix={<CheckCircleOutlined style={{ color: '#faad14', fontSize: '24px' }} />}
                valueStyle={{ color: '#faad14', fontSize: '28px' }}
              />
            </Card>
          </Col>
          <Col span={8}>
            <Card 
              hoverable
              style={{ 
                borderRadius: '16px', 
                textAlign: 'center',
                background: 'linear-gradient(135deg, #e6f7ff 0%, #ffffff 100%)',
                border: 'none',
                boxShadow: '0 4px 12px rgba(22, 119, 255, 0.1)',
                transition: 'all 0.3s ease',
              }}
            >
              <Statistic
                title={<Text strong style={{ fontSize: '16px', color: '#1677ff' }}>Average Confidence</Text>}
                value={(userInfo?.authentication_stats.average_confidence || 0) * 100}
                precision={1}
                suffix="%"
                prefix={<SafetyOutlined style={{ color: '#1677ff', fontSize: '24px' }} />}
                valueStyle={{ color: '#1677ff', fontSize: '28px' }}
              />
            </Card>
          </Col>
        </Row>

        {/* Emotional State Card */}
        <Card
          style={{
            borderRadius: '16px',
            background: 'linear-gradient(135deg, #fafafa 0%, #ffffff 100%)',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.06)',
            border: 'none',
          }}
        >
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <div>
              <Title level={4} style={{ margin: '0 0 16px', color: '#2c3e50' }}>
                Current Emotional State
              </Title>
              <EmotionDisplay emotions={normalizedData.normalized} />
            </div>
            
            <div style={{
              background: `${dominantEmotionColor}10`,
              padding: '16px',
              borderRadius: '12px',
              marginTop: '16px'
            }}>
              <Text strong style={{ fontSize: '16px', color: dominantEmotionColor }}>
                {adviceData.title}
              </Text>
              <Paragraph style={{ margin: '8px 0 0', color: '#595959' }}>
                {adviceData.description}
              </Paragraph>
              <List
                style={{ marginTop: '16px' }}
                dataSource={adviceData.suggestions}
                renderItem={(suggestion: string) => (
                  <List.Item>
                    <Space>
                      <CheckCircleOutlined style={{ color: dominantEmotionColor }} />
                      <Text>{suggestion}</Text>
                    </Space>
                  </List.Item>
                )}
              />
            </div>
          </Space>
        </Card>
      </Col>

      {/* Right Column - Recent Authentication History */}
      <Col span={8}>
        <Card
          title={
            <Space>
              <ClockCircleOutlined style={{ color: '#1677ff' }} />
              <Text strong>Recent Activity</Text>
            </Space>
          }
          style={{
            borderRadius: '16px',
            background: 'linear-gradient(135deg, #fafafa 0%, #ffffff 100%)',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.06)',
            border: 'none',
            height: '100%',
          }}
          styles={{
            body: { 
              maxHeight: 'calc(100vh - 250px)', 
              overflowY: 'auto',
              padding: '16px',
            }
          }}
        >
          <List
            dataSource={userInfo?.recent_attempts || []}
            renderItem={(attempt: AuthenticationAttempt) => (
              <RecentAuthentication attempt={attempt} />
            )}
            locale={{ 
              emptyText: (
                <div style={{ textAlign: 'center', padding: '24px' }}>
                  <Text type="secondary">No recent authentication attempts</Text>
                </div>
              ) 
            }}
          />
        </Card>
      </Col>
    </Row>
  );
} 