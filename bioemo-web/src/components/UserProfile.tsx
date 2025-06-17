import React from 'react';
import { Card, Typography, Space, Progress, List, Avatar, Divider, Spin, Statistic, Row, Col, Tag, Tooltip } from 'antd';
import { 
  UserOutlined, 
  CalendarOutlined, 
  BankOutlined, 
  CheckCircleOutlined, 
  ClockCircleOutlined, 
  HeartOutlined, 
  ThunderboltOutlined, 
  SmileOutlined, 
  TrophyOutlined, 
  SafetyOutlined,
  MailOutlined,
  TeamOutlined,
  DashboardOutlined
} from '@ant-design/icons';
import type { User, EmotionPrediction, UserInfo } from '../types';
import { generateAdvice, getDominantEmotion } from '../utils/emotionAdvice';
import { emotionColors } from '../constants/colors';

const { Title, Text, Paragraph } = Typography;

interface UserProfileProps {
  user: User;
  emotions: EmotionPrediction;
  userInfo?: UserInfo | null;
  isLoading?: boolean;
}

const EmotionIcon = ({ emotion }: { emotion: string }) => {
  const iconStyle = { fontSize: '24px' };
  switch (emotion.toLowerCase()) {
    case 'happiness': return <SmileOutlined style={iconStyle} />;
    case 'neutral': return <SafetyOutlined style={iconStyle} />;
    case 'sadness': return <HeartOutlined style={iconStyle} />;
    case 'anger': return <ThunderboltOutlined style={iconStyle} />;
    default: return <SmileOutlined style={iconStyle} />;
  }
};

const GradientCard = ({ children, style = {}, ...props }: any) => (
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

const InfoRow = ({ label, value, icon }: { label: string; value: string | number; icon: React.ReactNode }) => (
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

export function UserProfile({ user, emotions, userInfo, isLoading = false }: UserProfileProps) {
  const advice = generateAdvice(emotions);
  const dominantEmotion = getDominantEmotion(emotions);
  const dominantEmotionColor = emotionColors[dominantEmotion] || '#1677ff';

  if (isLoading) {
    return (
      <div style={{ 
        textAlign: 'center', 
        padding: '48px',
        background: 'linear-gradient(135deg, #ffffff 0%, #f0f2f5 100%)',
        borderRadius: '16px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)'
      }}>
        <Spin size="large" />
        <Text style={{ display: 'block', marginTop: '16px' }}>Loading user information...</Text>
      </div>
    );
  }

  return (
    <Space direction="vertical" size="large" style={{ width: '100%', padding: '24px' }}>
      {/* User Info Card */}
      <GradientCard>
        <Row gutter={[24, 24]} align="middle">
          <Col span={8}>
            <Space direction="vertical" size="middle" align="center" style={{ width: '100%' }}>
              <div style={{
                padding: '8px',
                background: `linear-gradient(135deg, ${dominantEmotionColor}20 0%, ${dominantEmotionColor}40 100%)`,
                borderRadius: '50%',
                boxShadow: `0 4px 12px ${dominantEmotionColor}30`
              }}>
                <Avatar 
                  size={120} 
                  icon={<UserOutlined style={{ fontSize: '64px' }} />} 
                  style={{ 
                    backgroundColor: dominantEmotionColor,
                    boxShadow: `0 4px 12px ${dominantEmotionColor}40`
                  }} 
                />
              </div>
              <Title level={3} style={{ margin: 0, color: '#1a1a1a', textAlign: 'center' }}>
                {userInfo?.name || user.name || `User ${userInfo?.user_id || user.id}`}
              </Title>
              <Space wrap style={{ justifyContent: 'center' }}>
                <Tag icon={<BankOutlined />} color="blue" style={{ padding: '6px 16px', borderRadius: '99px', fontSize: '14px' }}>
                  {userInfo?.department || user.department || 'No Department'}
                </Tag>
                <Tag icon={<TrophyOutlined />} color="gold" style={{ padding: '6px 16px', borderRadius: '99px', fontSize: '14px' }}>
                  {userInfo?.role || user.role || 'Employee'}
                </Tag>
              </Space>
            </Space>
          </Col>
          <Col span={16}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
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
      </GradientCard>

      {/* Stats Cards */}
      <Row gutter={[24, 24]}>
        <Col span={6}>
          <Card 
            style={{ 
              borderRadius: '16px', 
              textAlign: 'center', 
              height: '100%',
              background: 'linear-gradient(135deg, #e6f4ff 0%, #ffffff 100%)',
              border: '1px solid #91caff'
            }}
          >
            <Statistic
              title={<Text strong style={{ fontSize: '16px', color: '#1677ff' }}>Total Attempts</Text>}
              value={userInfo?.authentication_stats.total_attempts || 0}
              prefix={<DashboardOutlined style={{ color: '#1677ff' }} />}
              valueStyle={{ color: '#1677ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card 
            style={{ 
              borderRadius: '16px', 
              textAlign: 'center', 
              height: '100%', 
              background: 'linear-gradient(135deg, #f6ffed 0%, #ffffff 100%)',
              border: '1px solid #b7eb8f'
            }}
          >
            <Statistic
              title={<Text strong style={{ fontSize: '16px', color: '#52c41a' }}>Success Rate</Text>}
              value={userInfo?.authentication_stats.success_rate || 0}
              precision={1}
              suffix="%"
              prefix={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card 
            style={{ 
              borderRadius: '16px', 
              textAlign: 'center', 
              height: '100%', 
              background: 'linear-gradient(135deg, #fff7e6 0%, #ffffff 100%)',
              border: '1px solid #ffd591'
            }}
          >
            <Statistic
              title={<Text strong style={{ fontSize: '16px', color: '#faad14' }}>Successful Attempts</Text>}
              value={userInfo?.authentication_stats.successful_attempts || 0}
              prefix={<ThunderboltOutlined style={{ color: '#faad14' }} />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card 
            style={{ 
              borderRadius: '16px', 
              textAlign: 'center', 
              height: '100%', 
              background: 'linear-gradient(135deg, #e6f7ff 0%, #ffffff 100%)',
              border: '1px solid #91d5ff'
            }}
          >
            <Statistic
              title={<Text strong style={{ fontSize: '16px', color: '#1677ff' }}>Avg Confidence</Text>}
              value={(userInfo?.authentication_stats.average_confidence || 0) * 100}
              precision={1}
              suffix="%"
              prefix={<SafetyOutlined style={{ color: '#1677ff' }} />}
              valueStyle={{ color: '#1677ff' }}
            />
          </Card>
        </Col>
      </Row>

      {/* Emotions and Advice */}
      <Row gutter={24}>
        <Col span={16}>
          <GradientCard
            title={
              <Space>
                <EmotionIcon emotion={dominantEmotion} />
                <Title level={4} style={{ margin: 0 }}>Emotional State</Title>
              </Space>
            }
            style={{ height: '100%' }}
          >
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              {Object.entries(emotions).map(([emotion, value]) => (
                <div key={emotion}>
                  <Space style={{ width: '100%', justifyContent: 'space-between', marginBottom: 8 }}>
                    <Text strong style={{ 
                      textTransform: 'capitalize', 
                      fontSize: '16px',
                      color: emotionColors[emotion] || '#1677ff'
                    }}>
                      <EmotionIcon emotion={emotion} /> {emotion}
                    </Text>
                    <Text>{(value * 100).toFixed(1)}%</Text>
                  </Space>
                  <Tooltip title={`${(value * 100).toFixed(1)}% ${emotion}`}>
                    <Progress 
                      percent={value * 100} 
                      showInfo={false}
                      strokeColor={{
                        '0%': emotionColors[emotion] || '#1677ff',
                        '100%': `${emotionColors[emotion]}90` || '#1677ff90'
                      }}
                      strokeLinecap="round"
                      style={{ 
                        marginBottom: '16px', 
                        height: 12,
                        background: `${emotionColors[emotion]}10` || '#1677ff10'
                      }}
                    />
                  </Tooltip>
                </div>
              ))}
            </Space>
          </GradientCard>
        </Col>

        <Col span={8}>
          <GradientCard 
            title={
              <Space>
                <SmileOutlined style={{ fontSize: '24px' }} />
                <Title level={4} style={{ margin: 0 }}>Wellbeing Insights</Title>
              </Space>
            }
            style={{ height: '100%' }}
            bodyStyle={{ height: 'calc(100% - 58px)', display: 'flex', flexDirection: 'column' }}
          >
            <Space direction="vertical" size="middle" style={{ width: '100%', flex: 1 }}>
              <Title level={5} style={{ 
                color: dominantEmotionColor,
                fontSize: '18px',
                textAlign: 'center',
                padding: '16px',
                background: `linear-gradient(135deg, ${dominantEmotionColor}10 0%, ${dominantEmotionColor}20 100%)`,
                borderRadius: '12px',
                margin: 0,
                border: `1px solid ${dominantEmotionColor}30`
              }}>
                {advice.title}
              </Title>
              <Paragraph style={{ fontSize: '16px', color: '#666' }}>{advice.description}</Paragraph>
              <Divider orientation="left">Recommendations</Divider>
              <List
                size="large"
                dataSource={advice.suggestions}
                renderItem={(item) => (
                  <List.Item>
                    <Space align="start">
                      <div style={{
                        width: '32px',
                        height: '32px',
                        borderRadius: '50%',
                        background: '#f6ffed',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: '1px solid #b7eb8f'
                      }}>
                        <CheckCircleOutlined style={{ color: '#52c41a', fontSize: '20px' }} />
                      </div>
                      <Text style={{ fontSize: '16px', flex: 1 }}>{item}</Text>
                    </Space>
                  </List.Item>
                )}
              />
            </Space>
          </GradientCard>
        </Col>
      </Row>

      {/* Recent Activity */}
      {userInfo?.recent_attempts && userInfo.recent_attempts.length > 0 && (
        <GradientCard 
          title={
            <Space>
              <ClockCircleOutlined style={{ fontSize: '24px' }} />
              <Title level={4} style={{ margin: 0 }}>Recent Activity</Title>
            </Space>
          }
        >
          <List
            size="large"
            dataSource={userInfo.recent_attempts}
            renderItem={(attempt) => (
              <List.Item>
                <Space size="middle" style={{ width: '100%', justifyContent: 'space-between' }}>
                  <Space>
                    {attempt.success ? (
                      <Tag color="success" icon={<CheckCircleOutlined />} style={{ padding: '6px 16px', borderRadius: '99px', fontSize: '14px' }}>
                        Success
                      </Tag>
                    ) : (
                      <Tag color="warning" icon={<ClockCircleOutlined />} style={{ padding: '6px 16px', borderRadius: '99px', fontSize: '14px' }}>
                        Failed
                      </Tag>
                    )}
                    <Text strong style={{ fontSize: '16px' }}>
                      {new Date(attempt.timestamp).toLocaleString()}
                    </Text>
                  </Space>
                  <Space>
                    <Tag color="blue" style={{ padding: '6px 16px', borderRadius: '99px', fontSize: '14px' }}>
                      Confidence: {(attempt.confidence * 100).toFixed(1)}%
                    </Tag>
                    {attempt.device_info && (
                      <Tag color="purple" style={{ padding: '6px 16px', borderRadius: '99px', fontSize: '14px' }}>
                        {attempt.device_info}
                      </Tag>
                    )}
                  </Space>
                </Space>
              </List.Item>
            )}
          />
        </GradientCard>
      )}
    </Space>
  );
} 